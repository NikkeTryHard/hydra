mod integrity;

use std::{
    collections::{HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Condvar, Mutex,
    },
};

use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, Subcommand};
use crossbeam_channel::bounded;
use fs2::available_space;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use integrity::{
    inspect_output_verified, load_manifest, packager_config_hash, packager_identity,
    publish_manifest, sweep_stale_temps, OutputFacts, OutputState, PackagedObjectRow, SourceFacts,
    SourceKind,
};
use rayon::prelude::*;
use tar::Archive;
use walkdir::WalkDir;

const COPY_BUFFER: usize = 1024 * 1024;
const DEFAULT_THREADS: usize = 16;
const DEFAULT_MAX_ITEM: u64 = 512 * 1024 * 1024;
const DEFAULT_MEMORY_LIMIT: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Parser)]
#[command(about = "Restart-safe MJAI JSON zstd dataset packager")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Inspect all inputs, reject collisions, and check output capacity without writing.
    Preflight(Args),
    /// Perform conversion after the same complete preflight.
    Convert(Args),
}

#[derive(clap::Args, Clone)]
struct Args {
    /// Input file or directory. Directories are traversed recursively.
    input: PathBuf,
    /// Output root. Relative input paths are preserved below this directory.
    output: PathBuf,
    #[arg(long, default_value_t = DEFAULT_THREADS)]
    threads: usize,
    /// Zstd level for raw JSON (1 is fastest).
    #[arg(long, default_value_t = 1)]
    level: i32,
    /// Reject archive members larger than this many bytes.
    #[arg(long, default_value_t = DEFAULT_MAX_ITEM)]
    max_item_bytes: u64,
    /// Maximum uncompressed archive payload bytes queued or being compressed.
    #[arg(long, default_value_t = DEFAULT_MEMORY_LIMIT)]
    memory_limit_bytes: u64,
    /// Transport manifest for authoritative PackagedObjectRows (WP-00B).
    /// Hidden operational flag: accepted by both subcommands, consumed by
    /// `convert`. Kept out of --help so the recorded help bytes and the
    /// no-flag behavior remain identical.
    #[arg(long, hide = true)]
    manifest: Option<PathBuf>,
}

#[derive(Clone, Debug)]
enum Source {
    Raw(PathBuf),
    Zstd(PathBuf),
    TarZstd {
        archive: PathBuf,
        member: PathBuf,
        size: u64,
    },
}

#[derive(Clone, Debug)]
struct Job {
    source: Source,
    output: PathBuf,
    estimate: u64,
    complete: bool,
}

struct ProgressState {
    completed: AtomicU64,
    skipped: AtomicU64,
    bytes: AtomicU64,
    finished: AtomicBool,
    archive: Mutex<String>,
}

struct Plan {
    jobs: Vec<Job>,
    ignored_files: u64,
    ignored_bytes: u64,
    required_bytes: u64,
    available_bytes: u64,
}

struct MemoryLimiter {
    limit: u64,
    used: Mutex<u64>,
    changed: Condvar,
}

struct MemoryPermit {
    limiter: Arc<MemoryLimiter>,
    bytes: u64,
}

impl MemoryLimiter {
    fn new(limit: u64) -> Self {
        Self {
            limit,
            used: Mutex::new(0),
            changed: Condvar::new(),
        }
    }

    fn acquire(self: &Arc<Self>, bytes: u64) -> MemoryPermit {
        let mut used = self.used.lock().expect("memory limiter poisoned");
        while *used > self.limit - bytes {
            used = self.changed.wait(used).expect("memory limiter poisoned");
        }
        *used += bytes;
        MemoryPermit {
            limiter: Arc::clone(self),
            bytes,
        }
    }
}

impl Drop for MemoryPermit {
    fn drop(&mut self) {
        let mut used = self.limiter.used.lock().expect("memory limiter poisoned");
        *used -= self.bytes;
        self.limiter.changed.notify_all();
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Preflight(args) => {
            let plan = preflight(&args)?;
            print_plan(&plan, &args.output, false);
        }
        Command::Convert(args) => {
            let plan = preflight(&args)?;
            print_plan(&plan, &args.output, true);
            convert(plan.jobs, &args)?;
        }
    }
    Ok(())
}

fn preflight(args: &Args) -> Result<Plan> {
    ensure!(args.threads > 0, "--threads must be greater than zero");
    ensure!(
        args.max_item_bytes > 0,
        "--max-item-bytes must be greater than zero"
    );
    ensure!(
        args.memory_limit_bytes >= args.max_item_bytes,
        "--memory-limit-bytes must be at least --max-item-bytes"
    );
    ensure!(
        args.input.exists(),
        "input does not exist: {}",
        args.input.display()
    );

    let input = fs::canonicalize(&args.input).context("canonicalizing input")?;
    let output_abs = absolute_path(&args.output)?;
    reject_symlink_components(&output_abs)?;
    if input.is_dir() {
        ensure!(
            output_abs != input && !output_abs.starts_with(&input),
            "output must not be inside input"
        );
    }

    let mut jobs = Vec::new();
    let mut ignored_files = 0u64;
    let mut ignored_bytes = 0u64;
    if input.is_file() {
        plan_file(
            &input,
            Path::new(""),
            &args.output,
            args.max_item_bytes,
            &mut jobs,
        )?;
    } else {
        let authoritative = authoritative_directories(&input)?;
        for entry in WalkDir::new(&input).follow_links(false) {
            let entry = entry.context("walking input")?;
            if entry.file_type().is_symlink() {
                bail!(
                    "symbolic links are not supported: {}",
                    entry.path().display()
                );
            }
            if !entry.file_type().is_file() {
                continue;
            }
            let rel = entry
                .path()
                .strip_prefix(&input)
                .expect("walked below root");
            if authoritative
                .iter()
                .any(|directory| rel.starts_with(directory))
            {
                if is_raw_mjai_name(&entry.file_name().to_string_lossy()) {
                    ignored_files += 1;
                    ignored_bytes += entry.metadata()?.len();
                }
                continue;
            }
            let parent = rel.parent().unwrap_or_else(|| Path::new(""));
            plan_file(
                entry.path(),
                parent,
                &args.output,
                args.max_item_bytes,
                &mut jobs,
            )?;
        }
    }

    let mut destinations = HashMap::<PathBuf, String>::new();
    for job in &jobs {
        let description = source_name(&job.source);
        if let Some(first) = destinations.insert(job.output.clone(), description.clone()) {
            bail!(
                "duplicate output collision at {}: {first} and {description}",
                job.output.display()
            );
        }
    }
    let mut required_bytes = 0u64;
    for job in &mut jobs {
        job.complete = inspect_output(&job.output)?;
        if !job.complete {
            required_bytes = required_bytes
                .checked_add(job.estimate)
                .context("required capacity overflow")?;
        }
    }
    let probe = existing_ancestor(&args.output)?;
    let available_bytes = available_space(&probe)
        .with_context(|| format!("checking capacity at {}", probe.display()))?;
    ensure!(available_bytes >= required_bytes, "insufficient output capacity: need at most {required_bytes} bytes, have {available_bytes} bytes at {}", probe.display());
    Ok(Plan {
        jobs,
        ignored_files,
        ignored_bytes,
        required_bytes,
        available_bytes,
    })
}

fn reject_symlink_components(path: &Path) -> Result<()> {
    let mut current = Some(path);
    while let Some(component) = current {
        match fs::symlink_metadata(component) {
            Ok(metadata) => ensure!(
                !metadata.file_type().is_symlink(),
                "output path contains symlink: {}",
                component.display()
            ),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("inspecting output component {}", component.display())
                })
            }
        }
        current = component.parent();
    }
    Ok(())
}

fn authoritative_directories(root: &Path) -> Result<HashSet<PathBuf>> {
    let mut directories = HashSet::new();
    for entry in WalkDir::new(root).min_depth(1).follow_links(false) {
        let entry = entry.context("finding authoritative archives")?;
        if !entry.file_type().is_file() {
            continue;
        }
        let rel = entry.path().strip_prefix(root).expect("walked below root");
        let Some(name) = rel.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(stem) = name.strip_suffix(".tar.zst") else {
            continue;
        };
        let directory = rel.parent().unwrap_or_else(|| Path::new("")).join(stem);
        if root.join(&directory).is_dir() {
            directories.insert(directory);
        }
    }
    Ok(directories)
}

fn plan_file(
    path: &Path,
    parent: &Path,
    output: &Path,
    max_item: u64,
    jobs: &mut Vec<Job>,
) -> Result<()> {
    let name = path
        .file_name()
        .and_then(|v| v.to_str())
        .context("non-UTF-8 input filename")?;
    if is_raw_mjai_name(name) {
        let size = path.metadata()?.len();
        ensure!(
            size <= max_item,
            "input exceeds --max-item-bytes: {}",
            path.display()
        );
        let output_name = normalized_output_name(name)?;
        jobs.push(Job {
            source: Source::Raw(path.to_owned()),
            output: output.join(parent).join(output_name),
            estimate: zstd_bound(size)?,
            complete: false,
        });
    } else if name.ends_with(".mjai.json.zst") {
        let size = path.metadata()?.len();
        validate_zstd(path)?;
        jobs.push(Job {
            source: Source::Zstd(path.to_owned()),
            output: output.join(parent).join(name),
            estimate: size,
            complete: false,
        });
    } else if name.ends_with(".tar.zst") {
        let archive_stem = name.strip_suffix(".tar.zst").expect("suffix checked");
        plan_archive(path, &parent.join(archive_stem), output, max_item, jobs)?;
    }
    Ok(())
}

fn is_raw_mjai_name(name: &str) -> bool {
    name.ends_with(".mjai.json") || name.ends_with(".mjson")
}

fn normalized_output_name(name: &str) -> Result<String> {
    if name.ends_with(".mjai.json") {
        Ok(format!("{name}.zst"))
    } else if let Some(stem) = name.strip_suffix(".mjson") {
        Ok(format!("{stem}.mjai.json.zst"))
    } else {
        bail!("unsupported MJAI filename: {name}")
    }
}

fn plan_archive(
    path: &Path,
    parent: &Path,
    output: &Path,
    max_item: u64,
    jobs: &mut Vec<Job>,
) -> Result<()> {
    let file = File::open(path)?;
    let decoder = zstd::stream::read::Decoder::new(BufReader::with_capacity(COPY_BUFFER, file))?;
    let mut archive = Archive::new(decoder);
    let mut members = HashSet::new();
    let first_job = jobs.len();
    for entry in archive.entries().context("reading tar entries")? {
        let entry = entry?;
        let member = entry
            .path()
            .context("reading tar member path")?
            .into_owned();
        validate_relative(&member)?;
        if entry.header().entry_type().is_dir() {
            continue;
        }
        ensure!(
            entry.header().entry_type().is_file(),
            "unsupported non-regular archive member: {}:{}",
            path.display(),
            member.display()
        );
        let member_name = member.file_name().and_then(|v| v.to_str()).unwrap_or("");
        ensure!(
            is_raw_mjai_name(member_name),
            "unsupported regular archive member: {}:{}",
            path.display(),
            member.display()
        );
        ensure!(
            entry.size() <= max_item,
            "archive member exceeds --max-item-bytes: {}:{}",
            path.display(),
            member.display()
        );
        ensure!(
            members.insert(member.clone()),
            "duplicate archive member: {}:{}",
            path.display(),
            member.display()
        );
        let output_path = output
            .join(parent)
            .join(&member)
            .with_file_name(normalized_output_name(member_name)?);
        jobs.push(Job {
            source: Source::TarZstd {
                archive: path.to_owned(),
                member,
                size: entry.size(),
            },
            output: output_path,
            estimate: zstd_bound(entry.size())?,
            complete: false,
        });
    }
    ensure!(
        jobs.len() > first_job,
        "archive contains no MJAI JSON files: {}",
        path.display()
    );
    Ok(())
}

fn zstd_bound(bytes: u64) -> Result<u64> {
    let bytes = usize::try_from(bytes).context("member size exceeds platform usize")?;
    u64::try_from(zstd_safe::compress_bound(bytes)).context("zstd bound exceeds u64")
}

fn convert(jobs: Vec<Job>, args: &Args) -> Result<()> {
    fs::create_dir_all(&args.output)?;
    let identity = packager_identity();
    let config_hash = packager_config_hash(args.level, args.max_item_bytes);

    // Manifest mode only: remove interrupted temporaries of planned outputs
    // so a resumed run starts from a clean slate. Legacy runs keep their
    // exact prior filesystem behavior.
    if args.manifest.is_some() {
        let mut swept_parents: HashSet<PathBuf> = HashSet::new();
        for job in &jobs {
            let Some(parent) = job.output.parent() else {
                continue;
            };
            if !swept_parents.insert(parent.to_owned()) {
                continue;
            }
            let names: HashSet<String> = jobs
                .iter()
                .filter(|other| other.output.parent() == Some(parent))
                .filter_map(|other| {
                    other
                        .output
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                })
                .collect();
            sweep_stale_temps(parent, &names)?;
        }
    }

    let loaded_rows = match &args.manifest {
        Some(path) => load_manifest(path)?,
        None => Vec::new(),
    };
    let ctx = IntegrityCtx {
        output_root: absolute_path(&args.output)?,
        manifest_path: args.manifest.clone(),
        identity,
        config_hash,
        level: args.level,
        max_item: args.max_item_bytes,
        threads: args.threads,
        memory_limit: args.memory_limit_bytes,
        stored: loaded_rows
            .iter()
            .map(|row| (row.compressed_path.clone(), row.clone()))
            .collect(),
        container_cache: Mutex::new(HashMap::new()),
        rows: Mutex::new(Vec::new()),
    };

    let total_items = jobs.len() as u64;
    let total_bytes = jobs.iter().map(job_source_bytes).sum();
    let progress = Arc::new(ProgressState {
        completed: AtomicU64::new(0),
        skipped: AtomicU64::new(0),
        bytes: AtomicU64::new(0),
        finished: AtomicBool::new(false),
        archive: Mutex::new(String::from("individual files")),
    });
    let renderer = spawn_progress(Arc::clone(&progress), total_items, total_bytes);
    let result = convert_inner(&jobs, args, &progress, &ctx);
    progress.finished.store(true, Ordering::Release);
    renderer
        .join()
        .map_err(|_| anyhow::anyhow!("progress renderer panicked"))?;
    result?;

    // Every output is verified at this point: publish the authoritative
    // transport manifest (crash-safe; new rows supersede stale ones by
    // `compressed_path`).
    if let Some(manifest_path) = &args.manifest {
        let mut merged: HashMap<String, PackagedObjectRow> = loaded_rows
            .into_iter()
            .map(|row| (row.compressed_path.clone(), row))
            .collect();
        for row in ctx.rows.lock().expect("row sink poisoned").drain(..) {
            merged.insert(row.compressed_path.clone(), row);
        }
        publish_manifest(manifest_path, merged.into_values().collect())?;
    }

    eprintln!(
        "completed={}, skipped={}",
        progress.completed.load(Ordering::Relaxed),
        progress.skipped.load(Ordering::Relaxed)
    );
    Ok(())
}

fn convert_inner(
    jobs: &[Job],
    args: &Args,
    progress: &ProgressState,
    ctx: &IntegrityCtx,
) -> Result<()> {
    let regular: Vec<&Job> = jobs
        .iter()
        .filter(|job| !matches!(job.source, Source::TarZstd { .. }))
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    pool.install(|| {
        regular.par_iter().try_for_each(|job| -> Result<()> {
            execute_job(job, None, ctx, progress)
        })
    })?;
    // Every job whose bytes may have been (re)published is synced; a job can
    // be rebuilt even when preflight saw a magic-valid file.
    sync_output_directories(jobs.iter())?;
    let mut grouped: HashMap<&Path, Vec<&Job>> = HashMap::new();
    for job in jobs {
        if let Source::TarZstd { archive, .. } = &job.source {
            grouped.entry(archive).or_default().push(job);
        }
    }
    let mut archives: Vec<_> = grouped.into_iter().collect();
    archives.sort_unstable_by(|left, right| left.0.cmp(right.0));
    for (archive, archive_jobs) in archives {
        *progress.archive.lock().expect("progress archive poisoned") = archive
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        convert_archive(archive, &archive_jobs, ctx, progress)?;
        sync_output_directories(archive_jobs.into_iter())?;
    }
    Ok(())
}

fn sync_output_directories<'a>(jobs: impl Iterator<Item = &'a Job>) -> Result<()> {
    let mut parents = HashSet::new();
    for job in jobs {
        if let Some(parent) = job.output.parent() {
            parents.insert(parent.to_owned());
        }
    }
    for parent in parents {
        // A parent that does not exist yet has nothing published to flush;
        // archive stages create their parents lazily during publication.
        match fs::symlink_metadata(&parent) {
            Ok(_) => sync_dir(&parent)?,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("inspecting {}", parent.display()))
            }
        }
    }
    Ok(())
}

fn spawn_progress(
    state: Arc<ProgressState>,
    total_items: u64,
    total_bytes: u64,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let bar =
            ProgressBar::with_draw_target(Some(total_items), ProgressDrawTarget::stderr_with_hz(5));
        bar.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {wide_bar:.cyan/blue} {pos}/{len} {per_sec} ETA {eta_precise} | {msg}").expect("static progress template"));
        while !state.finished.load(Ordering::Acquire) {
            update_progress(&bar, &state, total_bytes);
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        update_progress(&bar, &state, total_bytes);
        bar.finish_and_clear();
    })
}

fn update_progress(bar: &ProgressBar, state: &ProgressState, total_bytes: u64) {
    let completed = state.completed.load(Ordering::Relaxed);
    let skipped = state.skipped.load(Ordering::Relaxed);
    let bytes = state.bytes.load(Ordering::Relaxed);
    let archive = state.archive.lock().expect("progress archive poisoned");
    bar.set_position(completed + skipped);
    bar.set_message(format!(
        "done {completed} | skipped {skipped} | {bytes}/{total_bytes} B | {archive}"
    ));
}

fn job_source_bytes(job: &Job) -> u64 {
    match &job.source {
        Source::Raw(path) | Source::Zstd(path) => {
            path.metadata().map(|meta| meta.len()).unwrap_or(0)
        }
        Source::TarZstd { size, .. } => *size,
    }
}

fn convert_archive(
    archive_path: &Path,
    jobs: &[&Job],
    ctx: &IntegrityCtx,
    progress: &ProgressState,
) -> Result<()> {
    let wanted: Vec<Job> = jobs.iter().map(|job| Job::clone(job)).collect();
    let limiter = Arc::new(MemoryLimiter::new(ctx.memory_limit));
    let (sender, receiver) = bounded::<(Job, Vec<u8>, MemoryPermit)>(ctx.threads * 2);
    std::thread::scope(|scope| -> Result<()> {
        let mut workers = Vec::with_capacity(ctx.threads);
        for _ in 0..ctx.threads {
            let receiver = receiver.clone();
            workers.push(scope.spawn(move || -> Result<()> {
                for (job, data, _permit) in receiver {
                    execute_job(&job, Some(data), ctx, progress)?;
                }
                Ok(())
            }));
        }
        drop(receiver);
        let producer_result = (|| -> Result<()> {
            let file = File::open(archive_path)?;
            let decoder =
                zstd::stream::read::Decoder::new(BufReader::with_capacity(COPY_BUFFER, file))?;
            let mut archive = Archive::new(decoder);
            let mut remaining = wanted.len();
            for entry in archive.entries()? {
                let entry = entry?;
                if !entry.header().entry_type().is_file() {
                    continue;
                }
                let member = entry.path()?.into_owned();
                let Some(index) = wanted.iter().position(|job| match &job.source {
                    Source::TarZstd { member: want, .. } => want == &member,
                    _ => false,
                }) else {
                    continue;
                };
                let job = wanted[index].clone();
                let expected = match job.source {
                    Source::TarZstd { size, .. } => size,
                    _ => unreachable!(),
                };
                ensure!(
                    entry.size() == expected,
                    "archive changed since preflight: {}",
                    archive_path.display()
                );
                let permit = limiter.acquire(expected);
                let mut data = Vec::with_capacity(expected as usize);
                entry.take(expected + 1).read_to_end(&mut data)?;
                ensure!(
                    data.len() as u64 == expected,
                    "truncated archive member: {}:{}",
                    archive_path.display(),
                    member.display()
                );
                sender
                    .send((job.clone(), data, permit))
                    .context("archive worker stopped")?;
                remaining -= 1;
            }
            ensure!(
                remaining == 0,
                "archive members disappeared after preflight: {}",
                archive_path.display()
            );
            Ok(())
        })();
        drop(sender);
        for worker in workers {
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("archive worker panicked"))??;
        }
        producer_result
    })
}

/// Context threaded through the WP-00B integrity authority.
struct IntegrityCtx {
    output_root: PathBuf,
    manifest_path: Option<PathBuf>,
    identity: String,
    config_hash: String,
    level: i32,
    max_item: u64,
    threads: usize,
    memory_limit: u64,
    /// Authoritative rows loaded from an existing manifest, keyed by
    /// `compressed_path`.
    stored: HashMap<String, PackagedObjectRow>,
    /// Memoized SHA-256 of archive containers (`source_container_sha256`).
    container_cache: Mutex<HashMap<PathBuf, String>>,
    /// Rows recorded during this run, published only after every output is
    /// verified.
    rows: Mutex<Vec<PackagedObjectRow>>,
}

impl IntegrityCtx {
    fn manifest_mode(&self) -> bool {
        self.manifest_path.is_some()
    }
}

fn rel_output_path(ctx: &IntegrityCtx, job: &Job) -> String {
    // Planned outputs are built under the (possibly relative) CLI root; make
    // both sides absolute so the manifest stores root-relative paths.
    let absolute = absolute_path(&job.output).unwrap_or_else(|_| job.output.clone());
    match absolute.strip_prefix(&ctx.output_root) {
        Ok(relative) => relative.to_string_lossy().into_owned(),
        Err(_) => job.output.to_string_lossy().into_owned(),
    }
}

fn container_hash(ctx: &IntegrityCtx, archive: &Path) -> Result<String> {
    let mut cache = ctx
        .container_cache
        .lock()
        .expect("container cache poisoned");
    if let Some(hash) = cache.get(archive) {
        return Ok(hash.clone());
    }
    let facts = integrity::sha256_file(archive)
        .with_context(|| format!("hashing archive container {}", archive.display()))?;
    cache.insert(archive.to_owned(), facts.sha256.clone());
    Ok(facts.sha256)
}

fn expected_source_fields(
    job: &Job,
    ctx: &IntegrityCtx,
) -> Result<(SourceKind, Option<String>, Option<String>)> {
    Ok(match &job.source {
        Source::Raw(_) => (SourceKind::Raw, None, None),
        Source::Zstd(_) => (SourceKind::Precompressed, None, None),
        Source::TarZstd { archive, member, .. } => (
            SourceKind::ArchiveMember,
            Some(container_hash(ctx, archive)?),
            Some(member.to_string_lossy().into_owned()),
        ),
    })
}

fn source_facts_for(job: &Job, member_data: Option<&[u8]>) -> Result<SourceFacts> {
    match (member_data, &job.source) {
        (Some(data), _) => Ok(SourceFacts {
            sha256: integrity::sha256_hex(data),
            length: data.len() as u64,
        }),
        (None, Source::TarZstd { .. }) => {
            bail!("archive jobs require the bounded archive pipeline")
        }
        (None, Source::Raw(path)) | (None, Source::Zstd(path)) => integrity::sha256_file(path),
    }
}

/// A stored transport row authorizes reuse only when it agrees with the
/// freshly measured source bytes, the fully decoded output, this packager's
/// identity and config hash, and the expected provenance triple.
fn row_authorizes_reuse(
    row: &PackagedObjectRow,
    facts: &OutputFacts,
    source: &SourceFacts,
    kind: SourceKind,
    container: &Option<String>,
    member: &Option<String>,
    rel: &str,
    ctx: &IntegrityCtx,
) -> bool {
    row.packager_identity == ctx.identity
        && row.packager_config_hash == ctx.config_hash
        && row.compressed_path == rel
        && row.source_kind == kind
        && row.source_container_sha256 == *container
        && row.source_member_path == *member
        && row.source_bytes_sha256 == source.sha256
        && row.source_bytes_length == source.length
        && row.compressed_bytes_sha256 == facts.compressed_sha256
        && row.compressed_bytes_length == facts.compressed_length
        && row.decoded_bytes_sha256 == facts.decoded_sha256
        && row.decoded_bytes_length == facts.decoded_length
        && row.record_count == facts.jsonl.record_count
        && row.canonical_jsonl == facts.jsonl.canonical_jsonl
}

/// Converts one job: honors an authorized reuse, or rebuilds through the
/// verified pipeline and records a fresh authoritative row.
fn execute_job(
    job: &Job,
    member_data: Option<Vec<u8>>,
    ctx: &IntegrityCtx,
    progress: &ProgressState,
) -> Result<()> {
    let source_bytes = job_source_bytes(job);
    let rel = rel_output_path(ctx, job);

    if job.complete {
        // Magic bytes alone never authorize reuse: fully decode first.
        if let OutputState::Verified(facts) = inspect_output_verified(&job.output)? {
            if !ctx.manifest_mode() {
                progress.skipped.fetch_add(1, Ordering::Relaxed);
                progress.bytes.fetch_add(source_bytes, Ordering::Relaxed);
                return Ok(());
            }
            if let Some(row) = ctx.stored.get(&rel) {
                let source = source_facts_for(job, member_data.as_deref())?;
                let (kind, container, member) = expected_source_fields(job, ctx)?;
                if row_authorizes_reuse(
                    row, &facts, &source, kind, &container, &member, &rel, ctx,
                ) {
                    ctx.rows.lock().expect("row sink poisoned").push(row.clone());
                    progress.skipped.fetch_add(1, Ordering::Relaxed);
                    progress.bytes.fetch_add(source_bytes, Ordering::Relaxed);
                    return Ok(());
                }
            }
        }
    }

    let source = source_facts_for(job, member_data.as_deref())?;
    let facts = write_verified(job, member_data, ctx)?;
    let (kind, container, member) = expected_source_fields(job, ctx)?;
    let mut row = PackagedObjectRow {
        packaged_object_id: String::new(),
        source_kind: kind,
        source_container_sha256: container,
        source_member_path: member,
        source_bytes_sha256: source.sha256,
        source_bytes_length: source.length,
        compressed_path: rel,
        compressed_bytes_sha256: facts.compressed_sha256,
        compressed_bytes_length: facts.compressed_length,
        decoded_bytes_sha256: facts.decoded_sha256,
        decoded_bytes_length: facts.decoded_length,
        record_count: facts.jsonl.record_count,
        canonical_jsonl: facts.jsonl.canonical_jsonl,
        packager_identity: ctx.identity.clone(),
        packager_config_hash: ctx.config_hash.clone(),
        created_at_utc: integrity::utc_now(),
    };
    row.seal();
    ctx.rows.lock().expect("row sink poisoned").push(row);
    progress.completed.fetch_add(1, Ordering::Relaxed);
    progress.bytes.fetch_add(source_bytes, Ordering::Relaxed);
    Ok(())
}

/// Writes the compressed output to a unique same-directory temp (O_EXCL),
/// verifies it by FULL zstd decode plus hashing before publication, then
/// renames atomically. On any error only the owned temp is removed.
fn write_verified(
    job: &Job,
    member_data: Option<Vec<u8>>,
    ctx: &IntegrityCtx,
) -> Result<OutputFacts> {
    let parent = job.output.parent().context("output has no parent")?;
    fs::create_dir_all(parent)?;
    let temp = temp_path(&job.output);
    let result = (|| -> Result<OutputFacts> {
        match (&job.source, member_data) {
            (Source::TarZstd { .. }, Some(data)) => {
                compress(io::Cursor::new(data), &temp, ctx.level, ctx.max_item)?
            }
            (_, Some(_)) => unreachable!("buffered payloads only occur inside archives"),
            (Source::Raw(path), None) => compress(
                BufReader::with_capacity(COPY_BUFFER, File::open(path)?),
                &temp,
                ctx.level,
                ctx.max_item,
            )?,
            (Source::Zstd(path), None) => copy_file(path, &temp)?,
            (Source::TarZstd { .. }, None) => {
                bail!("archive jobs require the bounded archive pipeline")
            }
        }
        let facts = match inspect_output_verified(&temp)? {
            OutputState::Verified(facts) => facts,
            _ => bail!(
                "freshly written output failed verification: {}",
                temp.display()
            ),
        };
        fs::rename(&temp, &job.output)
            .with_context(|| format!("atomically publishing {}", job.output.display()))?;
        Ok(facts)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn compress<R: Read>(mut input: R, temp: &Path, level: i32, max_item: u64) -> Result<()> {
    let file = create_temp(temp)?;
    let writer = BufWriter::with_capacity(COPY_BUFFER, file);
    let mut encoder = zstd::stream::write::Encoder::new(writer, level)?;
    let mut limited = (&mut input).take(max_item + 1);
    let copied = io::copy(&mut limited, &mut encoder)?;
    ensure!(
        copied <= max_item,
        "input grew beyond --max-item-bytes during conversion"
    );
    let mut writer = encoder.finish()?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

fn copy_file(source: &Path, temp: &Path) -> Result<()> {
    let mut input = BufReader::with_capacity(COPY_BUFFER, File::open(source)?);
    let mut output = BufWriter::with_capacity(COPY_BUFFER, create_temp(temp)?);
    io::copy(&mut input, &mut output)?;
    output.flush()?;
    output.get_ref().sync_all()?;
    Ok(())
}

fn create_temp(path: &Path) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("creating temporary output {}", path.display()))
}

fn temp_path(output: &Path) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let name = output.file_name().unwrap_or_default().to_string_lossy();
    output.with_file_name(format!(".{name}.tmp.{}.{id}", std::process::id()))
}

fn inspect_output(path: &Path) -> Result<bool> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error).with_context(|| format!("inspecting {}", path.display())),
    };
    ensure!(
        !metadata.file_type().is_symlink(),
        "refusing symlink output: {}",
        path.display()
    );
    ensure!(
        metadata.is_file(),
        "output path is not a regular file: {}",
        path.display()
    );
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("opening existing output {}", path.display()))?;
    if metadata.len() < 8 {
        return Ok(false);
    }
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    Ok(magic == [0x28, 0xb5, 0x2f, 0xfd])
}

fn validate_zstd(path: &Path) -> Result<()> {
    let file = File::open(path)?;
    let mut decoder =
        zstd::stream::read::Decoder::new(BufReader::with_capacity(COPY_BUFFER, file))?;
    io::copy(&mut decoder, &mut io::sink())
        .with_context(|| format!("validating {}", path.display()))?;
    Ok(())
}

fn validate_relative(path: &Path) -> Result<()> {
    ensure!(!path.as_os_str().is_empty(), "empty archive path");
    for component in path.components() {
        ensure!(
            matches!(component, Component::Normal(_) | Component::CurDir),
            "unsafe archive path: {}",
            path.display()
        );
    }
    Ok(())
}

fn existing_ancestor(path: &Path) -> Result<PathBuf> {
    let mut current = absolute_path(path)?;
    while !current.exists() {
        current = current
            .parent()
            .context("output path has no existing ancestor")?
            .to_owned();
    }
    Ok(current)
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    Ok(if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    })
}

fn source_name(source: &Source) -> String {
    match source {
        Source::Raw(path) | Source::Zstd(path) => path.display().to_string(),
        Source::TarZstd {
            archive, member, ..
        } => format!("{}:{}", archive.display(), member.display()),
    }
}

fn sync_dir(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn print_plan(plan: &Plan, output: &Path, converting: bool) {
    eprintln!(
        "mode={}, items={}, conservative_bytes={}, available_bytes={}, ignored_extracted_files={}, ignored_extracted_bytes={}, output={}",
        if converting { "convert" } else { "preflight" },
        plan.jobs.len(), plan.required_bytes, plan.available_bytes,
        plan.ignored_files, plan.ignored_bytes, output.display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_traversal() {
        assert!(validate_relative(Path::new("../game.mjai.json")).is_err());
        assert!(validate_relative(Path::new("/game.mjai.json")).is_err());
        assert!(validate_relative(Path::new("safe/game.mjai.json")).is_ok());
    }

    #[test]
    fn archive_takes_precedence_over_matching_extracted_directory() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("input");
        let output = temp.path().join("output");
        fs::create_dir_all(input.join("dataset"))?;
        fs::write(input.join("dataset/partial.mjai.json"), b"partial")?;
        fs::write(input.join("unrelated.mjai.json"), b"raw")?;
        let archive_file = File::create(input.join("dataset.tar.zst"))?;
        let encoder = zstd::stream::write::Encoder::new(archive_file, 1)?;
        let mut tar = tar::Builder::new(encoder);
        let payload = b"archive";
        let mut header = tar::Header::new_gnu();
        header.set_size(payload.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        tar.append_data(&mut header, "complete.mjai.json", &payload[..])?;
        let encoder = tar.into_inner()?;
        encoder.finish()?.sync_all()?;
        let args = Args {
            input,
            output,
            threads: 16,
            level: 1,
            max_item_bytes: 1024,
            memory_limit_bytes: 4096,
            manifest: None,
        };
        let plan = preflight(&args)?;
        assert_eq!(plan.ignored_files, 1);
        assert_eq!(plan.ignored_bytes, 7);
        assert_eq!(plan.jobs.len(), 2);
        assert!(plan
            .jobs
            .iter()
            .any(|job| job.output.ends_with("dataset/complete.mjai.json.zst")));
        assert!(plan
            .jobs
            .iter()
            .any(|job| job.output.ends_with("unrelated.mjai.json.zst")));
        Ok(())
    }

    #[test]
    fn memory_limiter_blocks_until_weight_is_released() {
        let limiter = Arc::new(MemoryLimiter::new(10));
        let first = limiter.acquire(7);
        let (sender, receiver) = std::sync::mpsc::channel();
        let other = Arc::clone(&limiter);
        let worker = std::thread::spawn(move || {
            let permit = other.acquire(4);
            sender.send(()).unwrap();
            permit
        });
        assert!(receiver
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err());
        drop(first);
        receiver
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        drop(worker.join().unwrap());
    }

    #[test]
    fn zstd_capacity_bound_covers_encoded_payload() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let output = temp.path().join("payload.zst");
        let payload = vec![0x5a; 1_000_000];
        compress(io::Cursor::new(&payload), &output, 1, payload.len() as u64)?;
        assert!(output.metadata()?.len() <= zstd_bound(payload.len() as u64)?);
        Ok(())
    }

    #[test]
    fn malformed_writable_output_is_replaceable() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let output = temp.path().join("bad.mjai.json.zst");
        fs::write(&output, b"not-zstd")?;
        assert!(!inspect_output(&output)?);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn symlink_output_is_rejected() -> Result<()> {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir()?;
        let target = temp.path().join("target");
        fs::write(&target, b"not-zstd")?;
        let output = temp.path().join("output");
        symlink(&target, &output)?;
        assert!(inspect_output(&output).is_err());
        Ok(())
    }
}
