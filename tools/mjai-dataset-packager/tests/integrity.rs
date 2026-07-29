//! WP-00B negative fixtures and authority contracts: corrupt valid-magic
//! objects never skip, resume yields identical authoritative hashes, repeats
//! are idempotent, and the transport manifest can never reference unverified
//! bytes.

mod common;

use common::{
    completed_skipped, convert, decode_zstd, fixture_bytes, manifest_rows, plan_field,
    sha256_file, walk_files, zstd_frame,
};
use std::fs;
use tempfile::tempdir;

const A_SOURCE: &[u8] = include_bytes!("fixtures/a.mjai.json");

fn row_for<'a>(rows: &'a [serde_json::Value], compressed_path: &str) -> &'a serde_json::Value {
    rows.iter()
        .find(|row| row["compressed_path"] == compressed_path)
        .unwrap_or_else(|| panic!("no row for {compressed_path} in {rows:?}"))
}

/// Checklist: valid zstd magic + truncated body. Legacy (no-flag) mode also
/// refuses to skip: the output is fully decoded before reuse.
#[test]
fn truncated_body_with_valid_magic_is_never_skipped() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(
        &["c.mjai.json.zst"],
        &work.path().join("input/c.mjai.json.zst"),
    );
    fs::create_dir_all(work.path().join("out")).expect("out dir");
    let frame = fixture_bytes(&["c.mjai.json.zst"]);
    let truncated_at = frame.len() / 2;
    fs::write(
        work.path().join("out/c.mjai.json.zst"),
        &frame[..truncated_at],
    )
    .expect("truncate");

    let output = convert(work.path(), "input", "out", &[]);
    assert!(output.status.success(), "stderr: {}", common::stderr_of(&output));
    assert_eq!(completed_skipped(&common::stderr_of(&output)), (1, 0));
    assert_eq!(
        fs::read(work.path().join("out/c.mjai.json.zst")).expect("rebuilt"),
        frame,
        "corrupt valid-magic object must be rebuilt byte-exactly"
    );
}

/// A valid zstd stream whose decoded payload differs from the authoritative
/// row forces a rebuild; the republished manifest matches the verified bytes.
#[test]
fn valid_stream_with_wrong_decoded_hash_forces_rebuild() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(&["a.mjai.json"], &work.path().join("input/a.mjai.json"));
    let manifest = work.path().join("manifest.jsonl");

    let first = convert(
        work.path(),
        "input",
        "out",
        &["--manifest", manifest.to_str().unwrap()],
    );
    assert!(first.status.success(), "stderr: {}", common::stderr_of(&first));

    // Sabotage the OUTPUT with a different, perfectly decodable stream.
    let output_path = work.path().join("out/a.mjai.json.zst");
    fs::write(&output_path, zstd_frame(b"{\"forged\":true}\n")).expect("forge");
    let forged_hash = sha256_file(&output_path);

    let second = convert(
        work.path(),
        "input",
        "out",
        &["--manifest", manifest.to_str().unwrap()],
    );
    assert!(second.status.success(), "stderr: {}", common::stderr_of(&second));
    assert_eq!(
        completed_skipped(&common::stderr_of(&second)),
        (1, 0),
        "wrong decoded hash must never be skipped"
    );

    let restored = fs::read(&output_path).expect("rebuilt output");
    assert_eq!(decode_zstd(&restored), A_SOURCE);
    let rows = manifest_rows(&manifest);
    assert_eq!(rows.len(), 1);
    let row = row_for(&rows, "a.mjai.json.zst");
    assert_eq!(row["compressed_bytes_sha256"], sha256_file(&output_path));
    assert_ne!(
        row["compressed_bytes_sha256"].as_str().unwrap(),
        forged_hash,
        "manifest must not keep referencing the forged bytes"
    );
    assert_eq!(
        row["decoded_bytes_sha256"],
        common::sha256_file(work.path().join("input/a.mjai.json").as_path())
    );
}

/// Valid output but no manifest row: the object has no transport authority,
/// so it is rebuilt and recorded.
#[test]
fn output_without_manifest_row_is_rebuilt() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(&["a.mjai.json"], &work.path().join("input/a.mjai.json"));
    let manifest = work.path().join("manifest.jsonl");

    let first = convert(
        work.path(),
        "input",
        "out",
        &["--manifest", manifest.to_str().unwrap()],
    );
    assert!(first.status.success(), "stderr: {}", common::stderr_of(&first));
    fs::remove_file(&manifest).expect("drop manifest");

    let second = convert(
        work.path(),
        "input",
        "out",
        &["--manifest", manifest.to_str().unwrap()],
    );
    assert!(second.status.success(), "stderr: {}", common::stderr_of(&second));
    assert_eq!(
        completed_skipped(&common::stderr_of(&second)),
        (1, 0),
        "output without an authoritative row must rebuild"
    );
    assert_eq!(manifest_rows(&manifest).len(), 1);
}

/// Manifest row without an output: rebuild and refresh the row.
#[test]
fn manifest_row_without_output_is_rebuilt() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(&["a.mjai.json"], &work.path().join("input/a.mjai.json"));
    let manifest = work.path().join("manifest.jsonl");

    let first = convert(
        work.path(),
        "input",
        "out",
        &["--manifest", manifest.to_str().unwrap()],
    );
    assert!(first.status.success(), "stderr: {}", common::stderr_of(&first));
    let old_rows = manifest_rows(&manifest);
    fs::remove_file(work.path().join("out/a.mjai.json.zst")).expect("drop output");

    let second = convert(
        work.path(),
        "input",
        "out",
        &["--manifest", manifest.to_str().unwrap()],
    );
    assert!(second.status.success(), "stderr: {}", common::stderr_of(&second));
    assert_eq!(
        completed_skipped(&common::stderr_of(&second)),
        (1, 0),
        "row without output must force rebuild"
    );
    let new_rows = manifest_rows(&manifest);
    assert_eq!(new_rows.len(), 1);
    for field in [
        "source_bytes_sha256",
        "compressed_bytes_sha256",
        "decoded_bytes_sha256",
        "decoded_bytes_length",
        "record_count",
    ] {
        assert_eq!(new_rows[0][field], old_rows[0][field], "{field}");
    }
}

/// Interrupted run debris: a stale same-directory temporary is swept, the
/// resumed build produces IDENTICAL authoritative hashes, and no temp is ever
/// mistaken for an output.
#[test]
fn interrupted_temp_output_resumes_with_identical_hashes() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(&["a.mjai.json"], &work.path().join("input/a.mjai.json"));
    let manifest = work.path().join("manifest.jsonl");

    let first = convert(
        work.path(),
        "input",
        "out",
        &["--manifest", manifest.to_str().unwrap()],
    );
    assert!(first.status.success(), "stderr: {}", common::stderr_of(&first));
    let baseline_rows = manifest_rows(&manifest);
    let baseline_output =
        fs::read(work.path().join("out/a.mjai.json.zst")).expect("first-run output");

    // Simulate a crash between write and publish: output gone, temp left.
    fs::remove_file(work.path().join("out/a.mjai.json.zst")).expect("interrupt");
    fs::write(
        work.path().join("out/.a.mjai.json.zst.tmp.999999.7"),
        b"half-written garbage",
    )
    .expect("plant stale temp");

    let second = convert(
        work.path(),
        "input",
        "out",
        &["--manifest", manifest.to_str().unwrap()],
    );
    assert!(second.status.success(), "stderr: {}", common::stderr_of(&second));
    assert_eq!(completed_skipped(&common::stderr_of(&second)), (1, 0));

    let resumed = fs::read(work.path().join("out/a.mjai.json.zst")).expect("resumed output");
    assert_eq!(resumed, baseline_output, "resume must reproduce exact bytes");
    let files = walk_files(&work.path().join("out"));
    assert_eq!(
        files,
        vec![work.path().join("out/a.mjai.json.zst")],
        "interrupted temporary must be swept, never published"
    );
    let resumed_rows = manifest_rows(&manifest);
    let new_row = row_for(&resumed_rows, "a.mjai.json.zst");
    let old_row = row_for(&baseline_rows, "a.mjai.json.zst");
    for field in [
        "source_bytes_sha256",
        "source_bytes_length",
        "compressed_bytes_sha256",
        "compressed_bytes_length",
        "decoded_bytes_sha256",
        "decoded_bytes_length",
    ] {
        assert_eq!(new_row[field], old_row[field], "authoritative {field} differs");
    }
}

/// Duplicate source identity with different bytes: the stale row is
/// superseded by exactly one fresh authoritative row.
#[test]
fn duplicate_source_identity_different_bytes_supersedes() {
    let work = tempdir().expect("workdir");
    let input = work.path().join("input/a.mjai.json");
    fs::create_dir_all(work.path().join("input")).expect("input dir");
    fs::write(&input, b"{\"type\":\"start_game\",\"names\":[\"x\",\"y\"]}\n{\"type\":\"end_game\"}\n")
        .expect("seed input v1");
    let manifest = work.path().join("manifest.jsonl");

    let first = convert(
        work.path(),
        "input",
        "out",
        &["--manifest", manifest.to_str().unwrap()],
    );
    assert!(first.status.success(), "stderr: {}", common::stderr_of(&first));

    // Same source identity (same path/kind), different bytes.
    let replacement =
        b"{\"type\":\"start_game\",\"names\":[\"z\",\"w\"]}\n{\"type\":\"end_game\"}\n";
    fs::write(&input, replacement).expect("mutate source");

    let second = convert(
        work.path(),
        "input",
        "out",
        &["--manifest", manifest.to_str().unwrap()],
    );
    assert!(second.status.success(), "stderr: {}", common::stderr_of(&second));
    assert_eq!(completed_skipped(&common::stderr_of(&second)), (1, 0));

    let rows = manifest_rows(&manifest);
    assert_eq!(rows.len(), 1, "exactly one authoritative row per path");
    assert_eq!(
        rows[0]["source_bytes_sha256"],
        sha256_file(&input),
        "row must track current source bytes"
    );
    assert_eq!(
        decode_zstd(&fs::read(work.path().join("out/a.mjai.json.zst")).expect("out")),
        replacement
    );
}

/// Archive members whose normalized names collide abort with zero writes.
#[test]
fn archive_member_path_collision_aborts() {
    let work = tempdir().expect("workdir");
    let archive_dir = work.path().join("input");
    fs::create_dir_all(&archive_dir).expect("input dir");
    let archive_path = archive_dir.join("collide.tar.zst");
    {
        use tar::Header;
        let file = fs::File::create(&archive_path).expect("archive file");
        let encoder = zstd::stream::write::Encoder::new(file, 1).expect("encoder");
        let mut tar = tar::Builder::new(encoder);
        for member in ["g.mjai.json", "g.mjson"] {
            let payload = b"{\"type\":\"start_game\",\"names\":[\"a\",\"b\"]}\n";
            let mut header = Header::new_gnu();
            header.set_size(payload.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            tar.append_data(&mut header, member, &payload[..])
                .expect("append member");
        }
        tar.into_inner().expect("tar finish").finish().expect("zstd finish");
    }

    let output = convert(work.path(), "input", "out", &[]);
    assert!(!output.status.success(), "collision must fail");
    let stderr = common::stderr_of(&output);
    assert!(stderr.contains("duplicate output collision"), "stderr: {stderr}");
    assert!(stderr.contains("g.mjai.json.zst"), "stderr: {stderr}");
    assert!(!work.path().join("out").exists(), "zero writes expected");
}

/// Repeated successful runs are idempotent: nothing is rewritten, and the
/// manifest bytes stay identical because authoritative rows are reused
/// verbatim (timestamps included).
#[test]
fn repeated_manifest_run_is_idempotent() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(&["a.mjai.json"], &work.path().join("input/a.mjai.json"));
    let manifest = work.path().join("manifest.jsonl");
    let extra = ["--manifest", manifest.to_str().unwrap()];

    let first = convert(work.path(), "input", "out", &extra);
    assert!(first.status.success(), "stderr: {}", common::stderr_of(&first));
    let manifest_before = fs::read(&manifest).expect("manifest v1");
    let output_path = work.path().join("out/a.mjai.json.zst");
    let mtime_before = fs::metadata(&output_path)
        .expect("stat")
        .modified()
        .expect("mtime");

    let second = convert(work.path(), "input", "out", &extra);
    assert!(second.status.success(), "stderr: {}", common::stderr_of(&second));
    assert_eq!(
        completed_skipped(&common::stderr_of(&second)),
        (0, 1),
        "authoritative reuse counts as skip"
    );
    assert_eq!(
        plan_field(&common::stderr_of(&second), "conservative_bytes"),
        0
    );
    assert_eq!(fs::read(&output_path).expect("output"), {
        let mut probe = Vec::new();
        probe.extend_from_slice(&fs::read(&output_path).expect("reread"));
        probe
    });
    assert_eq!(
        fs::metadata(&output_path).expect("stat").modified().expect("mtime"),
        mtime_before,
        "reuse must not rewrite the output"
    );
    assert_eq!(
        fs::read(&manifest).expect("manifest v2"),
        manifest_before,
        "idempotent runs must not churn manifest bytes"
    );
}

/// Every transport row carries the full SPEC 12.1 payload with correct
/// provenance for all three source kinds.
#[test]
fn transport_rows_carry_full_spec_payload() {
    let work = tempdir().expect("workdir");
    let input = work.path().join("input");
    fs::create_dir_all(&input).expect("input dir");
    common::copy_fixture(&["a.mjai.json"], &input.join("a.mjai.json"));
    common::copy_fixture(&["c.mjai.json.zst"], &input.join("c.mjai.json.zst"));
    common::copy_fixture(&["d.tar.zst"], &input.join("d.tar.zst"));
    let manifest = work.path().join("manifest.jsonl");

    let output = convert(
        work.path(),
        "input",
        "out",
        &["--manifest", manifest.to_str().unwrap()],
    );
    assert!(output.status.success(), "stderr: {}", common::stderr_of(&output));

    let rows = manifest_rows(&manifest);
    assert_eq!(rows.len(), 4, "raw + precompressed + two archive members");

    let raw = row_for(&rows, "a.mjai.json.zst");
    assert_eq!(raw["source_kind"], "raw");
    assert_eq!(raw["source_container_sha256"], serde_json::Value::Null);
    assert_eq!(raw["source_member_path"], serde_json::Value::Null);
    assert_eq!(raw["source_bytes_sha256"], sha256_file(&input.join("a.mjai.json")));
    assert_eq!(raw["record_count"], 3);
    assert_eq!(raw["canonical_jsonl"], false);

    let precomp = row_for(&rows, "c.mjai.json.zst");
    assert_eq!(precomp["source_kind"], "precompressed");
    assert_eq!(
        precomp["source_bytes_sha256"],
        sha256_file(&input.join("c.mjai.json.zst"))
    );
    assert_eq!(
        precomp["compressed_bytes_sha256"],
        precomp["source_bytes_sha256"],
        "precompressed copy is verbatim"
    );

    let container_hash = sha256_file(&input.join("d.tar.zst"));
    for path in ["d/games/g1.mjai.json.zst", "d/d2.mjai.json.zst"] {
        let member_row = row_for(&rows, path);
        assert_eq!(member_row["source_kind"], "archive_member");
        assert_eq!(member_row["source_container_sha256"], container_hash);
        let member_path = match path {
            "d/games/g1.mjai.json.zst" => "games/g1.mjai.json",
            _ => "d2.mjson",
        };
        assert_eq!(member_row["source_member_path"], member_path);
    }

    const REQUIRED_FIELDS: [&str; 16] = [
        "packaged_object_id",
        "source_kind",
        "source_container_sha256",
        "source_member_path",
        "source_bytes_sha256",
        "source_bytes_length",
        "compressed_path",
        "compressed_bytes_sha256",
        "compressed_bytes_length",
        "decoded_bytes_sha256",
        "decoded_bytes_length",
        "record_count",
        "canonical_jsonl",
        "packager_identity",
        "packager_config_hash",
        "created_at_utc",
    ];
    for row in &rows {
        for field in REQUIRED_FIELDS {
            assert!(row.get(field).is_some(), "missing {field} in {row}");
        }
        let id = row["packaged_object_id"].as_str().expect("hex id");
        assert_eq!(id.len(), 64, "sha256 hex id");
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
        let created = row["created_at_utc"].as_str().expect("timestamp");
        assert_eq!(created.len(), 20);
        assert!(created.ends_with('Z'));
        assert_eq!(&created[4..5], "-");
        assert_eq!(&created[10..11], "T");
        assert_eq!(row["packager_identity"].as_str().unwrap().len(), 64);
        assert_eq!(row["packager_config_hash"].as_str().unwrap().len(), 64);
    }

    let ids: std::collections::HashSet<_> = rows
        .iter()
        .map(|row| row["packaged_object_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids.len(), rows.len(), "distinct objects have distinct ids");
}

/// The published manifest can never reference unverified bytes: on-disk
/// corruption after a successful run is detected and rebuilt before any
/// manifest republication, so every retained row describes exactly the bytes
/// currently at its `compressed_path`.
#[test]
fn manifest_never_references_unverified_bytes() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(&["b.mjson"], &work.path().join("input/b.mjson"));
    let manifest = work.path().join("manifest.jsonl");
    let extra = ["--manifest", manifest.to_str().unwrap()];

    let first = convert(work.path(), "input", "out", &extra);
    assert!(first.status.success(), "stderr: {}", common::stderr_of(&first));
    let output_path = work.path().join("out/b.mjai.json.zst");
    let original = fs::read(&output_path).expect("original output");

    // Corrupt one interior byte in place (magic stays valid).
    let mut corrupted = original.clone();
    let flip = corrupted.len() - 1;
    corrupted[flip] ^= 0x55;
    fs::write(&output_path, &corrupted).expect("corrupt");

    let second = convert(work.path(), "input", "out", &extra);
    assert!(second.status.success(), "stderr: {}", common::stderr_of(&second));
    assert_eq!(
        completed_skipped(&common::stderr_of(&second)),
        (1, 0),
        "corrupted output must be rebuilt, not trusted"
    );

    let actual = fs::read(&output_path).expect("post-run output");
    let rows = manifest_rows(&manifest);
    assert_eq!(rows.len(), 1);
    let row = row_for(&rows, "b.mjai.json.zst");
    assert_eq!(
        row["compressed_bytes_sha256"],
        sha256_file(&output_path),
        "manifest hash must equal the bytes actually present"
    );
    assert_eq!(decode_zstd(&actual), fixture_bytes(&["b.mjson"]));
}

/// Corrupt output WITH a matching sealed row is still rejected: the freshly
/// measured decoded hash disagrees with the authoritative row.
#[test]
fn corrupt_object_with_valid_row_is_rebuilt_not_skipped() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(&["b.mjson"], &work.path().join("input/b.mjson"));
    let manifest = work.path().join("manifest.jsonl");
    let extra = ["--manifest", manifest.to_str().unwrap()];

    let first = convert(work.path(), "input", "out", &extra);
    assert!(first.status.success(), "stderr: {}", common::stderr_of(&first));
    let output_path = work.path().join("out/b.mjai.json.zst");
    let original = fs::read(&output_path).expect("output");
    let mut truncated = original.clone();
    truncated.truncate(original.len() / 2);
    fs::write(&output_path, &truncated).expect("truncate body, keep magic");

    let second = convert(work.path(), "input", "out", &extra);
    assert!(second.status.success(), "stderr: {}", common::stderr_of(&second));
    assert_eq!(
        completed_skipped(&common::stderr_of(&second)),
        (1, 0),
        "sealed row cannot rescue a corrupt object"
    );
    assert_eq!(fs::read(&output_path).expect("rebuilt"), original);
}
