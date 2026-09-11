# Rust Dependency Migration Guide (hydra2-replay-rs)

Status: research-complete, rollout in progress. Source: DepMigrationGuide deep passes, 2026-09-10.
Toolchain: rustc 1.98.1, Python 3.12.13, linux x86_64. Versions stable-only.
Note: Context7 quota was exhausted during research; facts via ddg search (10 queries) + fast_fetch
(9 fetches: pyo3 migration guide, pyo3 0.29 changelog, pyo3 docs.rs pages, zstd docs.rs + 0.14.0
manifest, zstd-rs releases, sha2 + digest docs.rs pages, sha2 0.11.0 manifest). RustCrypto/hashes
publishes NO GitHub releases, so sha2/digest facts come from docs.rs manifests.
`[INFERENCE]` marks unproven claims. No file edits were made during research.

## Census (cargo tree -d1 + lock)

- feed: memchr 2.8.3, serde 1.0.229, serde_json 1.0.151, zstd 0.13.3, flate2 1.1.10,
  bytemuck 1.25.2 (REMOVED 2026-09-10, zero use sites, commit 8737b76), rayon 1.12.0
- shard: serde, serde_json, sha2 0.10.9, memmap2 0.9.11, parquet =59.3.0,
  dev clap 4.6.6 / zstd / flate2
- bridge: pyo3 0.28.3 + auto-initialize, default extension-module
- smallvec ABSENT everywhere (not in lock/tree). crossbeam (deque 0.8.8 / epoch 0.9.21 /
  utils 0.8.23) indirect via rayon only, no direct use.

## No-ops (already latest, verified on docs.rs)

- serde 1.0.229 — derive sites mjai_event.rs:11,75-84; parity.rs:10,16-17,146-147,171-172. Risk NONE.
- serde_json 1.0.151 — hot BASELINE ingest.rs:65,609-616,634-652,694-697,726-728;
  walk pins 1518-1994 incl. last-wins 1519-1523/1993-1995 and huge-u64-to-0 scores map
  1809-1811; canonical parity.rs:253-330 with key sort 277-280; decisions.rs:145-247. Risk NONE.
  Gate: canonical corpus test parity.rs:728.
- memchr 2.8.3 — only memchr_iter at ingest.rs:333,409; memchr2/memmem confirmed UNUSED.
- flate2 1.1.10 — GzDecoder ingest.rs:540 + dump:85; GzEncoder test-only ingest.rs:1775.
- rayon 1.12.0 — sole pool fill.rs:736-757 build/install, fan-out 766-780
  par_iter+install+channel+scope:774; no scope/channel/par_bridge APIs; feed_duel.rs:472.
- memmap2 0.9.11 — reader.rs:19,177,179-180,196.
- clap 4.6.6 dev-only (dump:20-21).
- parquet =59.3.0 — read-only parquet_join.rs:18-19,35-36,80-84,128-176; zero direct arrow:: use.
  RECOMMEND keep the = pin, schedule dated 59.4/60.x re-eval, never silently float.

## Bump 1: sha2 0.10.9 -> 0.11.0 (LOW, migrate 2nd)

- Usage: digest.rs:9-16,41-77 / parity.rs:318-330 / stream.rs:124-127 / reader.rs:196-201.
  Nowhere names GenericArray / block-buffer / Output / cfgs (verified).
- Upstream: 0.11.0 edition 2024, rust-version 1.85; deps digest 0.11 / cfg-if / cpufeatures;
  default features alloc, oid. digest 0.11.3 deps crypto-common 0.2 + block-buffer 0.12 optional,
  Array<u8,U32> from hybrid-array (generic-array -> hybrid-array confirmed). Method names
  unchanged per digest 0.11 usage docs; Output still Deref<[u8]> + iterable + as_slice.
- Breaking touching us: NONE.
- Edit sketch (NOT pre-applied): shard Cargo.toml sha2 0.10 -> 0.11; contingency
  `let bytes: &[u8] = &sum;` at digest.rs:16 / parity.rs:322 if compiler complains
  (expected unnecessary).
- MSRV 1.85 <= 1.98.1 OK.
- Perf: 0.11 backend cfgs (x86-sha etc.) [INFERENCE-UNPROVEN] until measured; cold-path only.
- Gate: cargo test -p hydra-shard incl. s5_provenance pins + parity_84 env-gated + s7_walled.

## Bump 2: zstd 0.13.3 -> 0.14.0 (LOW, migrate 3rd)

- Usage: Decoder::new + read_to_end ingest.rs:535-538 + feed_duel.rs:528, decode_all dump:81,
  encode_all(3) + magic assert test-only ingest.rs:1765-1766. NO dict / Encoder / finish /
  with_context / bulk / seekable (verified).
- Upstream v0.14.0 (2026-09-04): breaking = with_prepared_dictionary borrows dict (unused by us),
  BSD-3-Clause relicense (hygiene only), requires zstd-safe 8.0.0; Decoder::finish consumes
  rest-of-frame + new finish_frame (we never call finish); empty-read Ok(0); zstd-safe 8
  ref_cdict/borrow + AdvancedSeekable DerefMut loss (dict-only); zstd-sys 2.1.0 bindgen-off
  default (already in lock as 2.1.0+zstd.1.5.7, C layer proven); NO decode / framing /
  level-semantics changes stated.
- Breaking touching us: NONE.
- Edit sketch: zstd 0.13 -> 0.14 in feed manifest + shard dev-deps; no code edits.
- MSRV 1.64 OK; manifest edition 2018, single dep zstd-safe 8, defaults legacy, arrays,
  zdict_builder.
- Perf: soundness release, expect bit-identical rows/s; re-proof is the gate.
- Gate: feed 76 + parity 22 + F1 vectors + feed_duel rows/s 432/96/0 + shard cold suites.

## Bump 3: pyo3 0.28.3 -> 0.29.2 (LOW-MED, migrate LAST)

- Usage (bridge ONLY): lib.rs:13,16-23 (pymodule + add_class x4); stream.rs:31-32,103-115
  (stream_py_err), 144-160, 519-598 (pyclass + get on PyFill / PyStats / PyQuar), 623-630
  (Mutex<Option> + TailStats), 633-668 (open + detach:664), 692-697 (detach whole next_into),
  741-758 (close + detach:756). Features auto-initialize + default extension-module.
  NO subclassing / new-tuples / capsules / closures / guards / Utf8Error-into-PyResult
  (only from_utf8 -> LineageError sink.rs:290), NO pyo3-build-config dep, Linux-only.
- Upstream 0.29.0 packaging: abi3t features, DROP py3.7 + 3.13t, ADD 3.15.0b1, pyo3-ffi no_std,
  pymodule PyModExport / PySlot init (internal), generate-import-lib deprecated (Windows-only
  no-op). Removed: From<Utf8Error / FromUtf16 / DecodeUtf16> for PyErr (unused),
  0.27-deprecations (unused), private FFI (untouched). Soundness splits (guards / capsule /
  closure / mutex2 — unused). 0.29.1 / 0.29.2 pure fixes (argues FOR .2 over .0).
- MSRV 1.83 OK; auto-initialize + extension-module both alive in 0.29.2, defaults unchanged;
  detach already modern (rename was 0.26); env 3.12 GIL unaffected; no abi3 flags added
  (deliberate).
- Breaking touching us: NONE; watch multi-phase init ORDER (open-once design already assumes it).
- Edit sketch: bridge Cargo.toml pyo3 0.28.3 -> 0.29.2, keep features; no .rs edits.
- Perf: call-path refcount elision [INFERENCE-UNPROVEN], expected zero (no per-row callbacks).
- Gate (strictest, LAST): cargo test -p hydra-bridge (fill_bridge, s6_sink) + feed 76 + FULL WP14
  Python set (test_rust_stream_wp14:279-297 PYO3_PYTHON recipe, replay_backend, parallel_expand,
  log_replay, stream_train, shard_build, bridge) on 3.12 + rows/s re-proof.

## Rollout order

1. dry-run no-op confirm + baseline capture (done 2026-09-10: only bitflags patch outstanding,
   applied commit 8737b76).
2. sha2.
3. zstd.
4. bytemuck removal any time (DONE commit 8737b76).
5. pyo3 LAST + WP14 smoke.
6. Keep parquet = pin.
