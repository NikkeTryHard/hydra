//! feed_duel — BENCH-ONLY hydra2 RUST feed wall on the frozen F10 duel corpus.
//!
//! BENCH-ONLY driver (one example file, zero hot-path edits, zero new deps):
//! reads the 8 F10 `.zst` files → explicit zstd decode (same `zstd` dep the
//! feed uses) → per-game chunk split ([`file_chunk_ranges`], the production
//! staging seam) → one [`StageJob`] per game (all F10 games) → [`stage_batch`]
//! on the explicit pool (per-game parallelism, bounded channel, ordered) →
//! ordered [`commit_batch`] into [`Scratch`] → [`fill_pinned`] into
//! caller-owned `Vec<u8>` plane buffers (13 planes). NO Python, NO GPU,
//! NO ring — pure feed wall, directly comparable to hydra1's `raw_mjai_stream`
//! CLI wall.
//!
//! Wall discipline: `.zst` bytes are read ONCE before timing (8.6 KB total,
//! page-resident; disk read excluded and labeled). Every timed pass decodes,
//! chunks, stages, commits and fills from those resident bytes, so the
//! wall covers exactly zstd + chunk + stage + commit + fill.
//!
//! Walled path: games stage with the static marker digest on the [`StageJob`]
//! wall path (the production `stage_one` core tests `is_walled()` only); the
//! digest CONTENT is never read hot, and the feed forbids sha2-hot, so cold
//! sha is outside the feed wall by construction. Labeled, not hidden.
//!
//! ns attribution (report shape frozen for the harness parser): zstd = serial
//! decode, frame = serial chunk-split prep, walk = fused parallel
//! [`stage_batch`] wall (frame+gate+walk per game), gate = 0 by construction,
//! commit = [`commit_batch`], fill = [`fill_pinned`].
//!
//! Protocol: cold = fresh process with `--runs 1` (run 3x); warm = one
//! process with `--runs 5` (median of 5). Per-run arrays + medians printed,
//! rows/s reported under BOTH numerators: expected corpus decisions
//! (`--expect-rows`, default 432) and native staged rows. Per-pass counts
//! assert against `--expect-rows` / `--expect-games` (defaults 432/96)
//! with zero rejects.
//!
//! [`file_chunk_ranges`]: hydra_feed::ingest::file_chunk_ranges
//! [`StageJob`]: hydra_feed::fill::StageJob
//! [`stage_batch`]: hydra_feed::fill::stage_batch
//! [`commit_batch`]: hydra_feed::fill::Scratch::commit_batch
//! [`Scratch`]: hydra_feed::fill::Scratch
//! [`fill_pinned`]: hydra_feed::fill::Scratch::fill_pinned

use hydra_feed::fill::{FeedPool, N_PLANES, Scratch, StageJob, pick_t_len, row_bytes, stage_batch};
use hydra_feed::gate::reason_name;
use hydra_feed::ingest::file_chunk_ranges;
use std::io::Read as _;
use std::sync::Arc;
use std::time::Instant;

/// Default expected staged rows: F10 oracle corpus decisions
/// (8 files x 54 tsumo-decisions, duel reference; `--expect-rows`).
const DEFAULT_EXPECT_ROWS: usize = 432;
/// Default expected staged games: F10 kyoku-games (`--expect-games`).
const DEFAULT_EXPECT_GAMES: usize = 96;
/// Caller-buffer row headroom (>> staged rows; whole-stage single fill).
const ROWS_CAP: usize = 4096;
/// Widest history bucket: caller buffers sized once, reused every pass.
const T_MAX: usize = 256;
/// Static walled-mode marker (content unread hot; see module docs).
const WALL_MARKER: &str = "feed_duel/bench-static-marker";
/// hydra1 leg-1 reference: B2048/Q8/t20 warm-median native rows/s
/// (source: Hydra1DuelLeg report; 384 dahai-rows numerator, ~12.0 ms wall).
const H1_LEG1_NATIVE: f64 = 32000.6;
/// hydra1 leg-1 normalized to 432 corpus decisions.
const H1_LEG1_NORM: f64 = 36000.0;
/// hydra1 leg-2 reference: B32/Q128/t20 warm-median native rows/s (~7.6 ms).
const H1_LEG2_NATIVE: f64 = 50527.6;
/// hydra1 leg-2 normalized to 432 corpus decisions.
const H1_LEG2_NORM: f64 = 56842.1;

struct Config {
    dir: String,
    threads: usize,
    runs: usize,
    expect_rows: usize,
    expect_games: usize,
}

struct FileInput {
    name: String,
    zst: Vec<u8>,
    object_id: u32,
    base_game_idx: u32,
}

/// Per-file reusable buffers (outside the wall; capacities kept across runs).
/// Single arena: chunk ranges are copied into owned per-game `StageJob`
/// bytes immediately, so no borrow escapes and no double-buffer flip is needed.
struct FileBufs {
    decoded: Vec<u8>,
    arena: Vec<u8>,
}

/// One reject record for quarantine accounting.
/// stage: 0 = chunk/frame err, 1 = gate err (fused away, see `reject_stage`),
/// 2 = gate skip, 3 = walk err.
struct RejectRec {
    stage: u8,
    reason: u8,
}

struct PassOut {
    wall_ns: u64,
    zstd_ns: u64,
    frame_ns: u64,
    gate_ns: u64,
    walk_ns: u64,
    commit_ns: u64,
    fill_ns: u64,
    staged_rows: usize,
    staged_games: usize,
    rejects: usize,
    max_hist: u32,
    t_len: usize,
}

fn usage() -> ! {
    eprintln!("usage: feed_duel [--dir <corpus>] [--threads <n>] [--runs <n>] [--expect-rows <n>] [--expect-games <n>]");
    eprintln!("  defaults: --dir /home/cachybtw/dev/hydra2/bench/corpus/f10-decisions --threads 20 --runs 5 --expect-rows 432 --expect-games 96");
    // NOTE: bench-only default corpus path above; override with `--dir`.
    std::process::exit(2);
}

fn parse_config() -> Config {
    let mut dir = String::from("/home/cachybtw/dev/hydra2/bench/corpus/f10-decisions");
    let mut threads: usize = 20;
    let mut runs: usize = 5;
    let mut expect_rows: usize = DEFAULT_EXPECT_ROWS;
    let mut expect_games: usize = DEFAULT_EXPECT_GAMES;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "--dir" {
            i += 1;
            if i >= args.len() {
                usage();
            }
            dir = args[i].clone();
        } else if a == "--threads" {
            i += 1;
            if i >= args.len() {
                usage();
            }
            match args[i].parse::<usize>() {
                Ok(n) => threads = n,
                Err(_) => usage(),
            }
        } else if a == "--runs" {
            i += 1;
            if i >= args.len() {
                usage();
            }
            match args[i].parse::<usize>() {
                Ok(n) => runs = n,
                Err(_) => usage(),
            }
        } else if a == "--expect-rows" {
            i += 1;
            if i >= args.len() {
                usage();
            }
            match args[i].parse::<usize>() {
                Ok(n) => expect_rows = n,
                Err(_) => usage(),
            }
        } else if a == "--expect-games" {
            i += 1;
            if i >= args.len() {
                usage();
            }
            match args[i].parse::<usize>() {
                Ok(n) => expect_games = n,
                Err(_) => usage(),
            }
        } else {
            usage();
        }
        i += 1;
    }
    if threads < 1 || runs < 1 || expect_rows < 1 || expect_games < 1 {
        usage();
    }
    Config { dir, threads, runs, expect_rows, expect_games }
}

fn load_corpus(dir: &str) -> Result<Vec<FileInput>, String> {
    let rd = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) => return Err(format!("read_dir {}: {}", dir, e)),
    };
    let mut names: Vec<String> = Vec::new();
    for ent in rd {
        let ent = match ent {
            Ok(e) => e,
            Err(e) => return Err(format!("dir entry: {}", e)),
        };
        let ft = match ent.file_type() {
            Ok(f) => f,
            Err(e) => return Err(format!("file_type: {}", e)),
        };
        if !ft.is_file() {
            continue;
        }
        let name = ent.file_name();
        let s = match name.to_str() {
            Some(s) => s,
            None => return Err("non-utf8 file name".to_string()),
        };
        if s.ends_with(".zst") {
            names.push(s.to_string());
        }
    }
    names.sort();
    if names.is_empty() {
        return Err(format!("no .zst files in {}", dir));
    }
    let mut out: Vec<FileInput> = Vec::new();
    let mut oid: u32 = 0;
    for n in names.iter() {
        let path = format!("{}/{}", dir, n);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => return Err(format!("read {}: {}", path, e)),
        };
        // base_game_idx placeholder (12 kyoku-games/file); reconciled from
        // pass-0 staged counts so lineage keys stay file-order disjoint.
        out.push(FileInput {
            name: n.clone(),
            zst: bytes,
            object_id: oid,
            base_game_idx: oid.saturating_mul(12),
        });
        oid = oid.saturating_add(1);
    }
    Ok(out)
}

/// Fused-stage reject bucket for the quarantine ledger.
///
/// `stage_batch` runs the production per-game core (frame → gate → walk
/// fused), so a fused reject carries only its reason: gate skip (6–8) and
/// walk (10+) bucket exactly; the framing family (1–5) is shared by chunk,
/// frame and gate-Err verdicts and attributes to stage 0. The F10 corpus
/// stages 0 rejects, so the ledger is identical regardless.
fn reject_stage(reason: u8) -> u8 {
    match reason {
        6 | 7 | 8 => 2,
        10.. => 3,
        _ => 0,
    }
}

fn median_u64(v: &[u64]) -> u64 {
    let mut s = v.to_vec();
    s.sort_unstable();
    let n = s.len();
    if n == 0 {
        return 0;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        s[n / 2 - 1].saturating_add(s[n / 2]) / 2
    }
}

fn median_f64(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

fn rate_per_s(numer: f64, wall_ns: u64) -> f64 {
    if wall_ns == 0 {
        return 0.0;
    }
    numer * 1_000_000_000.0 / (wall_ns as f64)
}

fn cpu_model() -> String {
    let text = match std::fs::read_to_string("/proc/cpuinfo") {
        Ok(t) => t,
        Err(_) => return "unknown".to_string(),
    };
    for line in text.lines() {
        if line.starts_with("model name") {
            if let Some(v) = line.splitn(2, ':').nth(1) {
                return v.trim().to_string();
            }
        }
    }
    "unknown".to_string()
}

fn cmd_output(prog: &str, args: &[&str]) -> String {
    let mut cmd = std::process::Command::new(prog);
    let mut k = 0usize;
    while k < args.len() {
        cmd.arg(args[k]);
        k += 1;
    }
    let out = match cmd.output() {
        Ok(o) => o,
        Err(_) => return "unknown".to_string(),
    };
    if !out.status.success() {
        return "unknown".to_string();
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        "unknown".to_string()
    } else {
        s
    }
}

fn print_report(
    cfg: &Config,
    cmdline: &[String],
    inputs: &[FileInput],
    passes: &[PassOut],
    q_counts: &[u64; 1024],
) {
    let mut cmd = String::new();
    for (k, a) in cmdline.iter().enumerate() {
        if k > 0 {
            cmd.push(' ');
        }
        cmd.push_str(a);
    }
    let nproc = match std::thread::available_parallelism() {
        Ok(n) => n.get().to_string(),
        Err(_) => "unknown".to_string(),
    };
    println!("=== feed_duel: hydra2 RUST feed wall (F10 duel corpus, bench-only) ===");
    println!("command: {}", cmd);
    println!("profile: {}", if cfg!(debug_assertions) { "debug" } else { "release" });
    println!("--- env fingerprint ---");
    println!("cpu: {}", cpu_model());
    println!("kernel: {}", cmd_output("uname", &["-r"]));
    println!("nproc: {}", nproc);
    println!("threads: {} (explicit FeedPool, never rayon default; no taskset)", cfg.threads);
    println!("rustc: {}", cmd_output("rustc", &["--version"]));
    println!("git_head: {}", cmd_output("git", &["rev-parse", "--short", "HEAD"]));
    println!("--- corpus (disk reads outside wall; resident bytes timed) ---");
    println!("dir: {}", cfg.dir);
    let mut zst_total: usize = 0;
    for inp in inputs.iter() {
        println!("file: {} zst_bytes={} object_id={}", inp.name, inp.zst.len(), inp.object_id);
        zst_total = zst_total.saturating_add(inp.zst.len());
    }
    println!("files={} zst_total_bytes={}", inputs.len(), zst_total);
    println!("wall_path: walled-mode (static marker digest; digest content unread hot; cold sha outside feed wall per feed no-sha rule)");
    println!("history: hot-13 planes, T buckets [32,64,128,256]; fill buckets staged t_len per pass");
    println!("--- per-run (wall covers zstd+frame+gate+walk+commit+fill) ---");
    for (k, p) in passes.iter().enumerate() {
        let rora = rate_per_s(cfg.expect_rows as f64, p.wall_ns);
        let rnat = rate_per_s(p.staged_rows as f64, p.wall_ns);
        println!(
            "run={} wall_ms={:.3} staged_rows={} staged_games={} rejects={} max_hist={} t_len={} rows_s_oracle{}={:.1} rows_s_native={:.1} ns(zstd={} frame={} gate={} walk={} commit={} fill={})",
            k,
            (p.wall_ns as f64) / 1_000_000.0,
            p.staged_rows,
            p.staged_games,
            p.rejects,
            p.max_hist,
            p.t_len,
            cfg.expect_rows,
            rora,
            rnat,
            p.zstd_ns,
            p.frame_ns,
            p.gate_ns,
            p.walk_ns,
            p.commit_ns,
            p.fill_ns,
        );
    }
    // Medians + arrays.
    let walls: Vec<u64> = passes.iter().map(|p| p.wall_ns).collect();
    let zstds: Vec<u64> = passes.iter().map(|p| p.zstd_ns).collect();
    let frames: Vec<u64> = passes.iter().map(|p| p.frame_ns).collect();
    let gates: Vec<u64> = passes.iter().map(|p| p.gate_ns).collect();
    let walks: Vec<u64> = passes.iter().map(|p| p.walk_ns).collect();
    let commits: Vec<u64> = passes.iter().map(|p| p.commit_ns).collect();
    let fills: Vec<u64> = passes.iter().map(|p| p.fill_ns).collect();
    let rora: Vec<f64> = passes.iter().map(|p| rate_per_s(cfg.expect_rows as f64, p.wall_ns)).collect();
    let rnat: Vec<f64> = passes
        .iter()
        .map(|p| rate_per_s(p.staged_rows as f64, p.wall_ns))
        .collect();
    println!("--- medians (median-of-{}; arrays below) ---", passes.len());
    println!("wall_ms_median={:.3}", (median_u64(&walls) as f64) / 1_000_000.0);
    println!("rows_s_oracle{}_median={:.1}", cfg.expect_rows, median_f64(&rora));
    println!("rows_s_native_median={:.1}", median_f64(&rnat));
    println!(
        "ns_median(zstd={} frame={} gate={} walk={} commit={} fill={})",
        median_u64(&zstds),
        median_u64(&frames),
        median_u64(&gates),
        median_u64(&walks),
        median_u64(&commits),
        median_u64(&fills),
    );
    print!("arr_wall_ms=[");
    for (k, p) in passes.iter().enumerate() {
        if k > 0 {
            print!(",");
        }
        print!("{:.3}", (p.wall_ns as f64) / 1_000_000.0);
    }
    println!("]");
    print!("arr_rows_s_oracle{}=[", cfg.expect_rows);
    for (k, v) in rora.iter().enumerate() {
        if k > 0 {
            print!(",");
        }
        print!("{:.1}", v);
    }
    println!("]");
    print!("arr_rows_s_native=[");
    for (k, v) in rnat.iter().enumerate() {
        if k > 0 {
            print!(",");
        }
        print!("{:.1}", v);
    }
    println!("]");
    println!("--- quarantine ledger (pass-stable counts) ---");
    let stages = ["frame", "gate_err", "gate_skip", "walk"];
    let mut any = false;
    let mut s = 0usize;
    while s < 4 {
        let mut r = 0usize;
        while r < 256 {
            let c = q_counts[s.saturating_mul(256).saturating_add(r)];
            if c > 0 {
                println!("quarantine stage={} reason={}({}) count={}", stages[s], r, reason_name(r as u8), c);
                any = true;
            }
            r += 1;
        }
        s += 1;
    }
    if !any {
        println!("quarantine: none (0 rejects across all passes)");
    }
    println!("--- hydra1 reference (Hydra1DuelLeg, same box+corpus, pure-Rust CLI) ---");
    println!("hydra1 leg1 B2048/Q8/t20: native={:.1} rows/s, normalized432={:.1} rows/s", H1_LEG1_NATIVE, H1_LEG1_NORM);
    println!("hydra1 leg2 B32/Q128/t20: native={:.1} rows/s, normalized432={:.1} rows/s", H1_LEG2_NATIVE, H1_LEG2_NORM);
    println!("note: numerators differ by construction (hydra1 native=384 dahai-rows over 8 file-streams; hydra2 native=staged walk rows over 96 kyoku-games); compare normalized432 columns.");
}

fn main() {
    let cfg = parse_config();
    let cmdline: Vec<String> = std::env::args().collect();
    let mut inputs = match load_corpus(&cfg.dir) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("feed_duel: {}", e);
            std::process::exit(1);
        }
    };
    // Explicit pool (never the rayon default): the SOLE hot pool, pinned count.
    let pool = match FeedPool::build(cfg.threads) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("feed_duel: pool build (threads={}): {}", cfg.threads, e);
            std::process::exit(1);
        }
    };
    // Per-file reusable buffers (decoded bytes + chunk arena; caps kept).
    let mut bufpool: Vec<FileBufs> = Vec::new();
    for _ in 0..inputs.len() {
        bufpool.push(FileBufs {
            decoded: Vec::new(),
            arena: Vec::new(),
        });
    }
    // Cold walled-mode marker digest (feed forbids sha2-hot): one allocation,
    // shared by every StageJob via Arc clone (content unread hot).
    let wall_arc: Arc<str> = Arc::from(WALL_MARKER);
    // Caller-owned plane buffers (pinned-memory stand-ins): sized once for
    // ROWS_CAP rows at T_MAX, reused every pass; allocation outside the wall.
    let mut caller: [Vec<u8>; N_PLANES] = Default::default();
    {
        let mut i = 0usize;
        while i < N_PLANES {
            caller[i] = vec![0u8; ROWS_CAP.saturating_mul(row_bytes(i, T_MAX))];
            i += 1;
        }
    }
    let mut caps: [usize; N_PLANES] = [0usize; N_PLANES];
    {
        let mut i = 0usize;
        while i < N_PLANES {
            caps[i] = caller[i].len();
            i += 1;
        }
    }
    let mut passes: Vec<PassOut> = Vec::new();
    let mut q_counts: [u64; 1024] = [0u64; 1024];
    let mut pass_idx = 0usize;
    while pass_idx < cfg.runs {
        let wall_t0 = Instant::now();
        // (prep, serial) decode + chunk-split into one StageJob per game.
        // zstd = explicit decode via the feed's own zstd dep; frame = the
        // production chunk seam plus owned per-game copies.
        let mut jobs: Vec<StageJob> = Vec::new();
        let mut file_advance: Vec<usize> = vec![0usize; inputs.len()];
        let mut zstd_ns: u64 = 0;
        let mut frame_ns: u64 = 0;
        let mut rejects_pre: Vec<RejectRec> = Vec::new();
        let mut failed: Option<String> = None;
        let mut fi = 0usize;
        while fi < inputs.len() {
            let inp = &inputs[fi];
            let bufs = &mut bufpool[fi];
            let t0 = Instant::now();
            bufs.decoded.clear();
            let mut dec = match zstd::Decoder::new(&inp.zst[..]) {
                Ok(d) => d,
                Err(e) => {
                    failed = Some(format!("{}: zstd decoder: {}", inp.name, e));
                    break;
                }
            };
            match dec.read_to_end(&mut bufs.decoded) {
                Ok(_) => {}
                Err(e) => {
                    failed = Some(format!("{}: zstd decode: {}", inp.name, e));
                    break;
                }
            }
            zstd_ns = zstd_ns.saturating_add(t0.elapsed().as_nanos() as u64);
            let t0 = Instant::now();
            let chunked = file_chunk_ranges(&mut bufs.arena, &bufs.decoded, inp.base_game_idx);
            let ranges = match chunked {
                Ok(r) => r,
                Err(e) => {
                    rejects_pre.push(RejectRec {
                        stage: 0,
                        reason: e.stub().reason,
                    });
                    file_advance[fi] = 1;
                    frame_ns = frame_ns.saturating_add(t0.elapsed().as_nanos() as u64);
                    fi += 1;
                    continue;
                }
            };
            let mut k = 0usize;
            while k < ranges.len() {
                let (s, e) = ranges[k];
                let mut bytes = Vec::with_capacity(e.saturating_sub(s));
                bytes.extend_from_slice(&bufs.arena[s..e]);
                jobs.push(StageJob {
                    game_idx: inp.base_game_idx.saturating_add(k as u32),
                    object_id: inp.object_id,
                    bytes,
                    wall_digest: Some(wall_arc.clone()),
                });
                k += 1;
            }
            file_advance[fi] = if ranges.is_empty() { 1 } else { ranges.len() };
            frame_ns = frame_ns.saturating_add(t0.elapsed().as_nanos() as u64);
            fi += 1;
        }
        if let Some(e) = failed {
            eprintln!("feed_duel: pass {}: {}", pass_idx, e);
            std::process::exit(1);
        }
        // (stage, parallel) per-game jobs on the explicit pool: bounded
        // channel, ordered outcomes (production orchestration). The fused
        // frame+gate+walk wall reports under walk_ns; gate_ns is 0 by
        // construction (report shape kept for the harness parser).
        let s0 = Instant::now();
        let (games, rejects) = stage_batch(&pool, &jobs);
        let walk_ns: u64 = s0.elapsed().as_nanos() as u64;
        let gate_ns: u64 = 0;
        for r in rejects_pre.iter() {
            let slot = (r.stage as usize).saturating_mul(256).saturating_add(r.reason as usize);
            if slot < q_counts.len() {
                q_counts[slot] = q_counts[slot].saturating_add(1);
            }
        }
        for r in rejects.iter() {
            let stage = reject_stage(r.stub.reason);
            let slot = (stage as usize).saturating_mul(256).saturating_add(r.stub.reason as usize);
            if slot < q_counts.len() {
                q_counts[slot] = q_counts[slot].saturating_add(1);
            }
        }
        let rejects_total: usize = rejects_pre.len().saturating_add(rejects.len());
        // Reconcile per-file base_game_idx from pass-0 staged counts so
        // lineage keys stay file-order disjoint (rejects occupy keyspace).
        if pass_idx == 0 {
            let mut base: u32 = 0;
            let mut k = 0usize;
            while k < inputs.len() {
                let adv = file_advance[k] as u32;
                inputs[k].base_game_idx = base;
                base = base.saturating_add(if adv == 0 { 1 } else { adv });
                k += 1;
            }
        }
        // (commit) ordered whole-game commits into Scratch (production order).
        let mut scratch = Scratch::new();
        let c0 = Instant::now();
        scratch.commit_batch(&games);
        let commit_ns = c0.elapsed().as_nanos() as u64;
        let mut staged_total: usize = 0;
        let mut gi = 0usize;
        while gi < games.len() {
            staged_total = staged_total.saturating_add(games[gi].rows as usize);
            gi += 1;
        }
        let games_total: usize = games.len();
        let max_hist = scratch.max_hist();
        let t_len = match pick_t_len(max_hist) {
            Some(t) => t,
            None => {
                eprintln!("feed_duel: pass {}: max_hist {} exceeds buckets", pass_idx, max_hist);
                std::process::exit(1);
            }
        };
        // (fill) memcpy staged rows into caller buffers at this pass's t_len.
        let mut ptrs: [u64; N_PLANES] = [0u64; N_PLANES];
        {
            let mut i = 0usize;
            while i < N_PLANES {
                ptrs[i] = caller[i].as_mut_ptr() as u64;
                i += 1;
            }
        }
        let f0 = Instant::now();
        let fillout = match scratch.fill_pinned(&ptrs, &caps) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("feed_duel: pass {}: fill_pinned: {:?}", pass_idx, e);
                std::process::exit(1);
            }
        };
        let fill_ns = f0.elapsed().as_nanos() as u64;
        if (fillout.rows as usize) != staged_total || (fillout.games_consumed as usize) != games_total
        {
            eprintln!(
                "feed_duel: pass {}: partial commit rows={}/{} games={}/{}",
                pass_idx, fillout.rows, staged_total, fillout.games_consumed, games_total
            );
            std::process::exit(1);
        }
        if scratch.rows() != 0 || scratch.games() != 0 {
            eprintln!("feed_duel: pass {}: scratch not drained", pass_idx);
            std::process::exit(1);
        }
        let wall_ns = wall_t0.elapsed().as_nanos() as u64;
        passes.push(PassOut {
            wall_ns,
            zstd_ns,
            frame_ns,
            gate_ns,
            walk_ns,
            commit_ns,
            fill_ns,
            staged_rows: staged_total,
            staged_games: games_total,
            rejects: rejects_total,
            max_hist,
            t_len,
        });
        pass_idx += 1;
    }
    for (k, p) in passes.iter().enumerate() {
        if p.staged_rows != cfg.expect_rows || p.staged_games != cfg.expect_games || p.rejects != 0 {
            eprintln!(
                "feed_duel: pass {}: counts staged_rows={}/{} staged_games={}/{} rejects={}/0",
                k, p.staged_rows, cfg.expect_rows, p.staged_games, cfg.expect_games, p.rejects
            );
            std::process::exit(1);
        }
    }
    print_report(&cfg, &cmdline, &inputs, &passes, &q_counts);
}
