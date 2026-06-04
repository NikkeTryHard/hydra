# Python test fixtures

Fixtures in this directory must stay tiny, deterministic, and checked in only when they replace local-only data or slow generation in normal tests.

Rules:

- No `training/`, `local/`, `target/`, machine-specific absolute paths, or broad dataset slices.
- Prefer in-test generated tensors/records for binary data; checked-in files should be small JSON/JSONL/text fixtures.
- Keep slow full-corpus parity as explicit manual gate instead of normal test dependency.
- Do not add licensed replay data unless provenance is clear and fixture is minimal.
