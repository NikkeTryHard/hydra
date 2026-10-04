# Repository Complexity Reduction Plan

Status: proposal for review, not authorization to change existing contracts. The live RL cutover has been reported complete and its new implementation is now included in this review. No production cleanup code was changed during preparation. Targets below distinguish source-confirmed defects, exercised behavior, and investigation leads; none is blanket permission to weaken integrity or learning contracts.

## 1. Objective

Reduce the number of concepts, requirements, validation layers, artifact dependencies, compatibility branches, and states a maintainer must understand to train and evaluate Hydra2 correctly. This is not primarily a speed project. A cheap check can still encode a pointless requirement; deleting it can be worthwhile even with zero measurable runtime improvement.

The desired result is a research stack, not a software system that continually proves its own paperwork. Ordinary local work should need a valid configuration, appropriate data, correct execution, useful recovery, and honest results. It should not automatically require a registry of completion receipts, hashed command logs, recursively qualified dependencies, and canonical copies of every intermediate object.

Do not equate fewer checks with lower complexity if their removal requires caches, flags, bypass modes, exception rules, or a second execution architecture. Delete an obsolete requirement and its machinery together. Prefer one normal path over a configurable strict/loose validation framework.

### Scope

- Artifact identity and integrity machinery.
- Configuration and experiment-profile restrictions.
- Local checkpoint/recovery requirements.
- Data ingestion, manifests, readers, and resume validation.
- Native/Python implementation duplication and stale-extension fallbacks.
- Research qualification and work-package completion machinery.
- Tests and documentation that preserve implementation details rather than behavior.
- Repository-wide dead paths and obsolete scaffolding, after caller verification.

### Non-goals

- No Mahjong rule, scoring, observation, action, PPO, sampler, or search-algorithm changes disguised as cleanup.
- No dependencies, attention-kernel changes, compile-arm changes, or GPU scheduling redesign.
- No mass deletion based on file size, number of hashes, or grep results.
- No weakening confidentiality, authorization, train/eval separation, or scientific evidence for published claims.
- No reset/revert of concurrent or unrelated work.
- No promise that cleanup improves throughput or meets the RL handoff latency target.

## 2. What must remain

These protect the result or the process, rather than its paperwork:

1. **Game correctness:** legal actions, tile/state consistency, exact scoring, terminal outcomes, and authoritative engine conformance.
2. **Information boundaries:** actor-visible observations and prohibition of privileged hidden information in policy inputs or unauthorized teacher paths.
3. **Learning correctness:** finite values, correct masks and shapes, behavior-policy/source matching, complete row membership, intended advantage/return derivation, and correct optimizer boundaries.
4. **Ownership and memory safety:** native buffer geometry, nonoverlapping spans, valid handles, event/lifetime ordering, no use-after-close, and bounded allocations.
5. **Sampling and evaluation integrity:** named deterministic streams, stable cohort/order where required, no sample duplication or silent dropping, held-out data separation, and comparable evaluation conditions.
6. **Useful recovery:** atomic publication; compatible model/optimizer/scheduler state; correct learner/actor pairing; correct cursor/RNG restoration. A crash must not advertise incomplete state as durable.
7. **External data boundaries:** reject malformed/incompatible input rather than guessing, padding, or silently switching algorithms.
8. **Honest evidence:** actual measurements for superiority/performance claims. A simplified result record is not permission to make unsupported claims.

Do not remove buffer checks because upstream configuration was validated: exported FFI methods are separate memory-safety boundaries. Do not remove read-time corruption checks solely because a file was valid when originally written: disk bytes can change. Do not infer immutability from a frozen Python dataclass whose payload is a mutable dictionary.

## 3. Classification rules

For each candidate, record the protected failure and apply one of these dispositions:

| Disposition | When it applies | Required action |
| --- | --- | --- |
| Delete requirement | The condition is no longer wanted, even if implemented correctly | Update authority first, remove producer/consumer fields, validators, tests, and docs together |
| Consolidate | Two layers enforce the same property with no intervening mutation or trust boundary | Keep one owner; retain genuinely independent boundary checks |
| Move to boundary | Validation is useful for external bytes but repeated for internal owned state | Validate on import/construction; typed internal consumers use established invariants |
| Keep | It defends an actual correctness, safety, recovery, or scientific failure | Explain the failure concisely; do not add ceremony |
| Investigate | Caller/lifetime/contract evidence is insufficient | Trace and exercise before scheduling deletion |

Every proposed hash deletion must distinguish three jobs: byte corruption detection, content-addressed identity, and deterministic random-stream derivation. These are not interchangeable. Replacing SHA-256 with an incrementing ID is not a valid cleanup of a semantic seed or a content-addressed lookup.

## 4. Verified cleanup targets

### C01. Remove stale-native-extension execution fallbacks

**Observed:**

- `python/hydra2/training/stream_build.py:288-361` contains a native scheduler-factor call and a Python closed-form fallback with its own scheduler-key validation.
- `python/hydra2/training/stream_build.py:486-499` selects a native filename formatter or a Python formatter according to symbol availability.
- `python/hydra2/training/shard_reader.py:110-127` selects native suffix validation or a second Python suffix table according to symbol availability.
- `python/hydra2/search/local_abstraction.py:504-547` selects native subgame expansion or a Python graph-expansion loop for missing/stale extensions.
- `python/hydra2/search/ismcts_core.py:132-145` retains optional native import/default constant selection; this is a candidate for tracing, not proof that the whole module is obsolete.

**Problem:** the supported Pixi/native environment already requires a working bridge. Stale binaries should produce a clear build/version error, not choose a second implementation. Maintaining both paths requires duplicated formulas, validation, parity coverage, and explanations of which implementation ran.

**Proposed change:** require the native implementation for production logic that already belongs in Rust. Remove fallback branches and duplicated production formulas/tables. Keep a genuinely independent oracle only where it tests uncertain mathematics or consumer-visible behavior; place it in test/support code rather than selecting it dynamically in production. Do not replace every fallback with another registry of capabilities.

**Before deletion:** trace exact symbols through Serena and lexical exports/dynamic callers. `_scheduler_lr_factor` currently yielded only an export reference in the inspected Python reference result; that is not enough to call it dead. Check tests, CLI imports, star imports, and string registries. If unused, delete the entire helper rather than preserving an unnecessary bridge wrapper.

**Acceptance:** supported installed bridge yields the same scheduler values, suffix decisions, and graph results; missing/wrong bridge fails clearly; no stale-extension branch remains. Retain numerical parity coverage for algorithms, not tests of forwarding or filename duplication.

### C02. Delete fabricated qualification fallbacks; keep actual qualification

**Observed:** `python/hydra2/analysis/qual_gates.py:189-289` catches arbitrary exceptions from candidate-specific factories and constructs a generic `CandidateSpec` with default identities/budgets. `build_gate_record` at lines 292-366 can synthesize a tiny observation and then fall back to `_ObsStub` after any exception. `python/hydra2/distillation/_teacher_gate.py:384-409` calls this factory resolver when obtaining the real teacher candidate specification.

**Problem:** a lot of defensive plumbing exists to make qualification construction continue after the real candidate path fails. A content hash of a fallback object does not establish that the intended candidate was exercised. More hashing cannot fix this semantic substitution.

**Proposed change:** qualification requires the actual candidate specification and an explicit declared corpus/fixture. Missing implementation or failed factory is a visible blocked/failed qualification, not a generic replacement. Delete blanket `except Exception: pass`, generated substitute specs, and production synthetic-observation fallback. Where a candidate intentionally has no factory, define its real construction once or explicitly exclude it from eligible candidates; do not silently borrow another candidate's defaults.

**Keep:** privileged-information checks, deterministic replay requirements where applicable, budget/accounting comparisons, and evidence tied to the candidate actually run. Do not delete teacher eligibility protection simply because its surrounding implementation is overcomplicated.

**Acceptance:** a broken candidate factory cannot yield a successful gate using a different spec; missing real input cannot create a qualification record from a stub; valid real candidates still qualify through the intended behavior path. Tests must exercise these failures and valid candidate results, not pin fallback defaults.

### C03. Retire work-package completion paperwork as a runtime authority

**Observed:** `docs/BUILD_EXECUTION_PLAN.md:34-74` requires immutable completion envelopes, hashed inputs/outputs/command logs, an environment-manifest hash, a mutable index, dependency verification, and an exact package command matrix. `python/hydra2/artifacts/package_verify.py:1-37` exposes the verifier. `crates/bridge/src/rl/package_verify.rs:148-167` hardcodes command token expectations; lines 470-556 validate receipt shape and digest/timestamp fields.

**Problem:** checking that a command was spelled a particular way and that its log has a digest is not the same as checking the software behavior. The delivery process has become an application subsystem with its own schemas, FFI, indexing, recursive dependency rules, tests, and failure states.

**Proposed change:** ordinary development uses the scoped executable test gates and records their outcome directly. Work-package IDs may remain useful labels, but should not require runtime completion-receipt validation. Published research results retain a compact record of configuration, revision, environment, corpus, metrics, and supporting measurements. Keep old completion artifacts as historical evidence; do not rewrite them or keep operational compatibility code solely to perpetuate the old process.

**Contract change required:** this directly changes normative completion requirements. Amend the execution plan and relevant project rules first; until that amendment is approved, existing gates remain binding. Do not delete the verifier while docs/CLI still require it.

**Implementation:** trace CLI entry points, report producers, tests, and consumers. Remove completion-record producers, verifier invocation, FFI registration, command-token checks, and completion-only index handling after separating any real algorithm-result validation they currently invoke. Real report validators stay with their data-owning subsystem if still needed.

**Acceptance:** a change cannot be called correct merely because paperwork validates; local development does not require receipt publication; benchmark claims still carry reproducible measured evidence; failing checks remain visible. No replacement universal receipt framework.

### C04. Separate experiment defaults from general configuration validity

**Observed:** `crates/bridge/src/rl/lifecycle_intake.rs:173-190` rejects any pipeline cohort other than the frozen generations/games recipe. Lines 252-280 enforce exact epochs, capacity/disk/budget guards, lookahead, warmup, and objective constants. These enforce the registered pilot contract, not universal facts about Riichi or PPO.

**Problem:** research choices, machine/resource choices, and correctness invariants are mixed into one validation vocabulary. Changing a disk reserve should not conceptually require inventing a new learning algorithm. Conversely, changing PPO math must not slip through as a resource adjustment.

**Proposed change:** retain the current frozen comparator unchanged during active RL work. After it lands, classify each restriction:

- Recipe values belong in an explicitly selected experiment configuration and its recorded identity.
- Resource limits remain positive/bounded and consistent with actual allocation/geometry, but need not universally equal one machine's numbers.
- True algorithm/ownership invariants remain enforced independently of profile defaults.
- Number of generations/games can become normal validated experiment parameters only after every completion/resume/wall-ledger assumption is migrated.

Do not add dozens of profile flags or a permissive legacy mode. Prefer one explicit new contract with honest recorded settings.

**Contract change required:** registered schema/version/digest changes and all affected plan sections must be migrated before code. Exactly eight generations, the current cohort, one epoch, and one future are NOT authorized to change in the concurrent live RL implementation.

**Acceptance:** changing a permitted resource budget does not require patching several hardcoded constants; invalid capacity products and unsafe budgets still reject; experiment settings are recorded accurately; new cohort sizes work end to end including terminal recovery. Keep bounded policy lag unless separately approved as an algorithm change.

### C05. Simplify local checkpoints without pretending integrity checks are useless

**Observed:** `python/hydra2/runtime/checkpoint.py:137-188` builds six per-section state-tree hashes plus recipe/source identities. `load_checkpoint` at lines 308-374 reconstructs section hashes before returning state. `save_checkpoint` at lines 277-304 atomically saves a single container. Callers include supervised checkpoint publication, replay checkpoint publication, and the RL checkpoint wrapper.

**Problem:** local crash recovery and immutable publication are different products. Requiring a canonical rollout identity and individually hashed state-tree sections for every local recovery container couples useful restart to archival machinery.

**Proposed change:** let the active live RL cutover establish its paired local restart container first. Do not simultaneously rewrite shared supervised checkpoints. Then assess a common local recovery contract: one atomic container, explicit compatible configuration/model/source/cursor metadata, strict state loading, and correct RNG ordering. Immutable exported checkpoints keep whatever integrity/content identity guarantee their consumers actually require.

**Tradeoff:** removing section hashes removes that layer's detection of payload changes. `torch.load` and shape checks do not prove that same-shaped tensor values are the intended ones. This is an explicit guarantee reduction, not a redundant-check claim. If external artifacts need integrity, retain one well-defined byte integrity boundary or the existing export contract; do not blindly stack transport hashes and section hashes.

**Acceptance:** interrupted publication retains the previous durable state; incompatible optimizer/model/source/cursor rejects before mutation; learner/actor pairing remains correct; round-trip resume continues the intended execution. Tests for same-shape corruption belong only to formats that still promise that detection. Remove old fields, builders, readers, and tests together if the guarantee is retired.

### C06. Collapse unnecessary identity layers, not all hashing

**Observed:** `python/hydra2/artifacts/registry.py:249-282` checks stored-byte digest, decodes the envelope, computes canonical envelope identity, checks that identity, and checks type/version. `loads_canonical` in `python/hydra2/artifacts/canonical.py:259-280` accepts ordinary JSON whitespace and key ordering while rejecting duplicate keys/nonfinite constants/BOM; its name does not mean it verifies canonical byte encoding.

**Important verification:** a small Pixi runtime probe showed two differently spaced JSON byte strings parse to the same object, have different transport hashes, and have the same canonical hash. Therefore the second registry identity check is NOT automatically redundant with the first: it currently helps enforce content-addressed canonical-envelope identity. Do not delete it based on a superficial 'hashed twice' argument.

**Proposed change:** inventory objects that genuinely need immutable content identity. Local status, progress, ephemeral handles, and bookkeeping do not automatically need an artifact envelope, registry row, and canonical digest. For objects that do need content addressing, define one authoritative byte representation and verify it at import/publication. Internal typed values should not repeatedly reconstruct the same identity for ordinary operations.

**Acceptance:** surviving content-addressed stores still reject wrong bytes/type/version and malformed external JSON; local bookkeeping no longer needs registry participation unless a real consumer relies on it. Semantic seed hashing remains unchanged. Do not introduce a trusted-object cache without a proven lifetime/invalidation model.

### C07. Reduce analysis-report copies and nested identities

**Scout lead independently checked by MAIN:** `python/hydra2/analysis/qual_gates.py:405-417` hashes a comparison into a gate digest; lines 473-498 store the full comparison and gate digest inside a separately hashed report. Lines 502 and 506 assign the run ID twice. Lines 509-520 canonicalize and publish the same report at a run path, a latest path, and a content-addressed path. The inspected `analysis_gate_for` reader at lines 525-543 reads the latest path.

**Proposed change:** choose one authoritative immutable qualification result and, if needed, one small latest pointer. Do not preserve three full copies merely to satisfy historical layout assertions. Remove the overwritten run-ID calculation. Retain nested digests only if an actual consumer independently addresses/verifies an individual gate or comparison; otherwise the report's integrity boundary is enough. A hash of nested metadata is not an extra independent execution of the candidate.

**Before deletion:** trace all report-return-path consumers, CLI/test fixtures, teacher loaders, and external supported artifact contracts. The scout's claim of no content-copy readers is a lead, not an exhaustive proof. Do not remove the run result needed by a real consumer or change existing immutable records.

**Acceptance:** teacher lookup resolves the exact qualified candidate from the authoritative result; altered externally loaded reports still reject under the surviving integrity contract; reruns do not overwrite immutable results; no producer or test requires removed full-copy paths or nested fields. Schema/layout changes require new artifact contracts and migrated callers. Keep actual qualification checks described in C02.

### C08. Retire unused shard observation-identity sidecars

**Scout lead independently checked:** `python/hydra2/data/shard_build.py:490-494` writes both decision IDs and observation hashes; lines 613-634 verify both sidecars and derive dataset identity from decision IDs. `python/hydra2/training/shard_reader.py:355-362` loads observation hashes into `_observation_hashes`, while only decision IDs participate in the inspected dataset-identity check. MAIN's search across Python/Rust/tests found the observation sidecar producer, loader, filename constant, and fixtures, but no read of `_observation_hashes` after assignment.

**Proposed change:** remove the mandatory observation-hash sidecar, its manifest digest field, native filename constant, loading, and format-only tests if supported offline tooling has no consumer. Keep decision IDs needed for order/reseeding and keep actual feature-plane integrity checks. Do not introduce an optional debug flag merely to preserve unused scaffolding. Historical shard directories need not be physically modified or deleted.

**Contract change required:** update the shard format/version and supported readers explicitly. Check docs and downstream/offline consumers before concluding the sidecar is globally useless. Absence of a training consumer is strong evidence, not permission to erase required external artifacts silently.

**Acceptance:** newly built shards without the sidecar support verified reading, deterministic reseeding, and the same model inputs; corrupt required planes and inconsistent decision IDs still reject. No production assignment/constant/test requires the removed sidecar. Use isolated new shard fixtures rather than deleting user data.

### C09. Consolidate shard validation rather than maintaining a second test-only contract

**Checked:** live references to `python/hydra2/data/shard_build.py::validate_manifest` returned only its export in the defining file. Its body at lines 535-660 validates planes, sidecars, order, build statistics, and attestation. The real reader separately validates its input in `ShardReader.__init__`, `_open_planes`, and `_resolve_action_width`.

**Proposed change:** establish the supported external shard boundary once. Remove the standalone production validator if caller/export/tooling checks confirm it is only exercised by tests; migrate useful corruption scenarios to the real reader. If an offline validator has a real consumer, reuse the boundary parser rather than retaining a divergent schema. Do not move a redundant function into a new tool merely to avoid deleting it.

**Important limit:** the implementations do not enforce identical properties today. Source authorization/attestation remains mandatory at ingest/build; it must not disappear because the reader treats metadata differently. Build elapsed time and throughput are observations, not universal validity criteria. Retire required build-statistic fields only after consumers and artifact contracts are migrated.

**Acceptance:** the actual reader rejects malformed schemas, wrong action width, missing/corrupt required planes, and inconsistent required identities. Builds without authorization still reject. Test-only duplicate validators and their public exports are gone; any retained offline checker uses the same format authority.

### C10. Parse checkpoint sidecars once per resolution

**Checked:** `_iter_structural_candidates` in `python/hydra2/training/_rc_resume.py:189-208` reads sidecars to select structurally valid candidates; `resolve_resume_plan` at lines 275-299 reads each selected sidecar again before compatibility checks.

**Proposed change:** private candidate enumeration carries the already parsed sidecar through one resolution call. Keep the public resolver/plan shape and all compatibility gates. Do not add a persistent cache or assume checkpoint files can never change between separate calls. The scan result's parsed bytes are that resolution's coherent observation; atomic publication still owns writer consistency.

**Acceptance:** torn/missing sidecars and stem mismatches are skipped; newest incompatible candidates cannot bypass compatibility; correct sampler/cursor/worker/source checks remain; subsequent resolution calls observe newly published state. Exercise real selection behavior, not a permanent mock call-count test.

## 5. Additional audits before scheduling deletions

### A01. Data validation ownership

Trace raw ingest -> validation/quarantine -> canonical or columnar storage -> reader -> learner. Record which boundary owns schema, legal-mask/action-width, dtype, source authorization, split assignment, and file-integrity guarantees. Delete duplicated checks only when there is no intervening mutable storage or independent API boundary.

Do not remove shard read-time hashes simply because shard creation previously computed hashes. Do not replace exact resume-prefix checks with a blanket 'manifest is the same': determine whether the resume guarantee includes source content/order and sampler state. Treat caches as machinery to simplify or retire, not an automatic answer.

### A02. FFI and Python result revalidation

`python/hydra2/training/_rc_parse.py:69-83` calls native data parsing and then validates the shape of returned roots again. This is a concrete duplication candidate, not yet a proven safe deletion. Read the native return construction, all callsites, and supported bridge ABI assumptions. If the native producer guarantees the exact result shape and there is no external/mutable interception boundary, Python should perform typed translation rather than a second semantic validator. Keep validation of user input in the native owner.

### A03. Dead exports, aliases, and migration scaffolding

Use live references plus dynamic registry/CLI/export checks. A zero static-reference result alone is not dead-code proof. Retire obsolete aliases/re-exports and migration paths only after identifying supported formats and consumers. Historical artifacts can remain on disk without requiring every old reader to remain a production path.

### A04. Test and lint scaffolding

Remove tests that merely pin exact wording, implementation layout, duplicated forwarding, or an incidental default. Preserve observable behavior, error precedence where contractual, ownership transitions, and boundary failure coverage. Remove broad obsolete lint/type suppressions only in touched modules after the supported path is simplified; do not trigger unrelated repository-wide restyling.

### A05. Oversized subsystem fragmentation

A large file is a signal to inspect responsibilities, not a reason to split it into more files automatically. Consolidate shallow translation modules only when it removes concepts/cycles. Do not create a generic validation framework to replace a few direct functions. Keep Rust-owned logic and thin runtime-required Python boundaries.

### A06. Landed live RL preparation ownership

The completed cutover adds a move-only pool-to-tape handoff, asynchronous Rust replay preparation, and paired local restart containers. Include this code in the complexity review; it is no longer reserved for a concurrent migration.

**Correctness before cleanup:** MAIN checked `crates/bridge/src/rl/live.rs:426-534` and `crates/bridge/src/rl/trajectory.rs:228-258,438-449`:

- Both poll and wait return early when worker join reports a panic, before releasing the tape preparation claim. Poll additionally retains a job with neither worker nor result. The tape drop guard consequently retains the tape. This is a source-confirmed failure path, not a dynamically injected panic result.
- Poll takes the worker and releases the preparation mutex before storing the joined result. Concurrent wait can remove that entry in this interval; a successful replay result then has no recipient. Wait also removes the entry before joining, so begin can admit another preparation while the first worker remains active. These interleavings are source-confirmed; they were not reproduced with a concurrent runtime harness in this review.
- Begin checks job-ID exhaustion after claiming the tape and spawning the worker (`live.rs:399-418`). Exhaustion can return without publishing a job or releasing its claim. Reserve a valid ID before taking ownership. The subsequent zero-ID check is unreachable under the current checked monotonic counter; this is not a practical performance bottleneck.

**Proposed simplification:** one occupied preparation slot with a unique consumption token and one owner of join/result/release. Keep it occupied through consumption so another begin cannot overlap a pending join. Preserve stale/double-consume rejection; replacing the map with a slot does not authorize reusing a constant handle. Panic and ordinary worker error must both release the claim and clear the job without losing a successful replay. Do not replace the race with additional limbo/vanished recovery branches.

**Keep:** destructive-drain preflight, live binding/row membership checks at replay registration, exactly-once terminal freezing, source/span checks on slab fill, and buffer-export lifetime/overlap checks. They defend independent ownership or public FFI boundaries, not duplicate paperwork. Tape removal after a live replay is registered currently fails later at fill rather than enforcing drop order; determine the intended exported lifecycle before changing that contract.

**Acceptance:** exercise running poll, successful poll/wait consumption, repeated consumption, worker failure/panic, tape drop while claimed, concurrent poll/wait, and begin during wait. The claim and slot must be released on failure; successful results must have exactly one owner. These tests must assert resource/lifecycle behavior, not state-map layout or diagnostic wording.

### A07. Landed paired restart and Python owner

**Exercised:** an isolated Pixi CPU smoke performed an optimizer step, captured the paired container, saved it through the real atomic publisher, and loaded it through the real reader; recursive tensor/state comparison passed. Actual retained generation-1 and generation-2 containers from the uninterrupted and interrupted/resumed evidence roots matched bitwise for learner payload, actor state, and descriptor. This establishes those two boundaries, not all eight generations or every interruption point.

**Input-policy findings:** the same isolated smoke demonstrated that `load_restart` accepts an extra top-level container key and accepts `False` as the training epoch. `python/hydra2/training/rl_restart.py:269-315` checks selected container fields and uses equality for numeric payload cursors, whereas the native descriptor rejects boolean integers. Decide the intended closed-shape/non-boolean payload contract explicitly; if strict compatibility is required, validate it once at load before restoring state. These are observed acceptance differences, not evidence of a wrong training result.

**Bounded consolidation candidates:** reuse the existing descriptor-validation helper instead of duplicating its native call/JSON framing in restart load; establish one owner for live FFI framing rather than a second collection of async wrappers with semantic validation on both sides. The production loop calls async preparation directly while `rl_bridge` exposes synchronous tooling methods. Do not delete those synchronous methods merely because the loop does not use them. Raw exception mapping alone is not a scientific correctness defect.

**Keep distinct policies:** the paired restart writer retains one active and one newest deferred snapshot; ordinary checkpoint publication uses backpressure and landing hooks. They are not interchangeable queues. Python training/sampler comparisons are also not duplicates of native descriptor validation: the native validator does not inspect the Torch payload.

**Coverage priorities:** prior pointer preservation when a later save fails, payload/descriptor cursor mismatch, malformed or non-finite state, CUDA RNG restored after construction, and actual owner shutdown with preparation outstanding. Normal shutdown waits for preparation before dropping replay/tape; exception-path cleanup and new-pool ownership if activation fails need targeted investigation. Do not infer pool use-after-free from the old Collector remaining in Python: drain transfers ownership to the tape registry.

**Measured boundary:** retained full-run spans show 12.825–17.958 ms from both training/preparation readiness to first learner dispatch, but the next cohort completes 14.301–15.349 s after the prior training ends. The latter is a producer tail, not measured all-zero GPU time: inference still runs while collecting. The retained GPU CSV has 1,752 samples with median reported utilization 98%; utilization is not proof of useful kernel occupancy or a zero-idle guarantee. Eight restart containers are present; the retained completion log reports 344.3 s. No new full GPU run was launched for this review.

## 6. Execution sequence and concurrency

1. The live RL cutover is reported complete; its saved runtime evidence and actual APIs have been inspected. Include all new live preparation/restart/owner code in scope. Address A06 ownership failures before optional representation cleanup; preserve the frozen learning recipe and independent archive parity.
2. Approve requirement changes explicitly: completion paperwork, local checkpoint integrity promises, and experiment-profile generalization are separate choices. Consolidation/removal of stale implementation branches is a different class of change.
3. Start with fail-closed native requirements and qualification fallbacks. These have concrete duplicate/substitution machinery and relatively clear desired behavior.
4. Retire completion-only runtime machinery after authority is updated and real report validation is separated.
5. Simplify local recovery/export separation using the landed RL behavior as evidence, not as a license to widen scope blindly.
6. Generalize frozen resource/experiment restrictions only after profiling their contract assumptions and updating schemas/callers.
7. Audit/consolidate data and FFI boundary duplication.
8. Remove now-obsolete exports/tests/docs and migrate every affected caller. No shims, 'deprecated' aliases, or old fallback flags by default.

Parallel work is appropriate for independent read-only inventories and disjoint mechanical migrations after MAIN chooses the contract. One integration owner controls shared schema/API/doc changes. Subagent outputs narrow context; MAIN reopens the actual implementations and verifies their claims. No parallel full builds or GPU runs. External research is optional and should answer a specific unresolved API question; local implementation is the primary evidence.

## 7. Verification and delivery

For each slice:

- Record exact before/after requirement and observable behavior.
- Trace public symbol references before changing an exported API.
- Reproduce the unwanted branch/dependency through an appropriate isolated scenario when practical.
- Batch edits, run one portable bridge build if Rust changes require it, then exercise the real changed path.
- Run focused tests, owning file, and scoped package gates. Run `pixi run lint`, `pixi run format-check`, and `pixi run typecheck`.
- Respect CPU/GPU lane isolation. Full-suite gates are required if shared runtime/contracts/data changes justify them; do not substitute a narrow passing test for a required exit gate.
- Keep permanent tests only for plausible consumer-visible failures. Use disposable smoke probes for wiring/forwarding questions.
- Update all affected authority/docs/schema identities first where contracts change, then implementation/test documentation. Preserve immutable historical artifacts; publish a new version rather than changing their bytes.
- Report exactly which guarantees were removed, which remain, which scenarios ran, and what remains unverified. No performance claim without measured evidence.

### Completion criteria

- Every removed check has a named retired requirement or a proven surviving owner.
- No production execution silently selects a second implementation because the installed bridge is stale.
- Qualification cannot turn a broken real candidate into a passing synthetic substitute.
- Ordinary local workflows do not depend on completion receipts if their retirement was approved.
- Local recovery and immutable export have clearly different promises without duplicated production loops.
- No unsafe buffer, hidden-information, illegal-action, data-leakage, or sampling shortcuts were introduced.
- No leftover producers/readers/CLI flags/tests preserve deleted contracts.
- Documentation describes the implemented behavior, not planned scaffolding.

## 8. Evidence status

MAIN directly inspected source implementations for C01-C10, traced selected consumers with live symbols, and independently checked the strongest leads from both read-only scouts. The data scout initially returned an unusable placeholder; its recovered findings were treated as leads and checked, not accepted as proof. The JSON identity probe ran successfully through Pixi; it is not a benchmark or whole-registry test. The proposed per-batch legal-mask deletion was not promoted to a concrete target: materialization transforms planes between open-time validation and batch delivery, so a full transformation proof is needed first. Repository-wide unused-code proof, external supported-format inventory, and acceptance execution for proposed removals have not been performed. This is a planning deliverable, not a claim that cleanup is implemented.

Post-cutover review additionally checked the native preparation transitions and Python restart load gates directly, and ran `pixi run config-check` plus an isolated real save/load and retained-checkpoint comparison smoke. `pixi run test-unit tests/unit/test_rl_restart.py` actually selected the whole CPU unit directory because the task supplies that directory itself: 828 passed, one skipped for an absent corpus mount. This is not a focused-file-only result, a serial/CUDA gate, or a native panic/concurrency reproduction. Production code was unchanged; lint/type/build were not rerun for this documentation-only review. Read-only scouts supplied leads; source-confirmed and runtime-exercised findings are separated above.

Publication review found and corrected a remaining error-precedence gap: the final restart-writer close now uses the same cleanup handler as earlier shutdown stages. An isolated Pixi smoke executed the actual cleanup handler with an active and deferred real writer job both failing; the originating exception remained primary, both save failures were attached as notes, and the executor closed. This is a shutdown-failure smoke, not an additional full training run.

Whole-tree publication verification: final CPU lane 1,298 passed and three corpus-mount skips; serial lane 143 passed, 125 deselected and one MahJax GPU-soak XPASS with GPU jaxlib absent (not GPU MahJax qualification). The initial CPU run exposed a stale `stream_train.py` benchmark entry; its source hash and canonical manifest digest were updated without changing performance thresholds, the focused manifest/restart check passed six tests, and the complete CPU lane then passed. Native workspace nextest passed all 1,123 tests with the pinned libpython library path. `lake build` completed 3,114 jobs with linter warnings. Lint, format-check, typecheck and config-check passed; the actual runtime probe passed, including 85 integration tests and two missing-corpus skips. No new eight-generation training measurement was made for publication.

## 9. Landed cutover simplification (execution record)

Implemented and verified across two slices; normative sections above are
unchanged. Blocked contract changes (C02/C03/C04/C05-reduction/C07-cutover/
C08-cutover) were not touched.
### 9.1 Implemented

| Item | Change | Machinery removed |
|---|---|---|
| A06 prep ownership (`crates/bridge/src/rl/live.rs`) | Jobs map replaced by one occupied slot + monotonic token + single join-owner (`take_prep_worker`/`finish_prep_consumption`); token reserved before claim/spawn; panic and worker error release the claim and clear the slot; stale/double-consume reject; slot occupied through join so no second begin overlaps it; `limbo`/`vanished` branches deleted (unreachable by construction, not replaced) | `PrepEntry`/`jobs` map, three failure branches, claim-before-reserve ordering |
| A06 tests (same file) | Six ownership tests asserting result/claim/slot behavior, never layout or wording: failed-worker release + single consumption, panic release + cleared job, running poll + successful poll consumption with handle passthrough, 30-round poll/wait race (single winner, intact handle), begin-during-join rejection with waiter result intact, drop-while-claimed refusal then success | — |
| C01 scoped (scheduler query) | Deleted unused `_scheduler_lr_factor` (Python def + native projection + registration + math helpers + helper-only unit tests) after export/dynamic/Serena closure returned zero callers; torch `_build_scheduler` untouched | `stream_build.py` export/def/fallback, `training_leaves.rs` `linear_ramp`/`cosine_decay`/`lr_factor_impl`/`scheduler_lr_factor`/registration/tests, `PI` import, `Sequence` import |
| Prefix re-export | Removed redundant `stream_train.py` import of `_prefix_hashes_name` (no via-importer; canonical def and `stream_checkpoint.py` calls unchanged) | One import line |
| C10 parse-once | `_iter_structural_candidates_with_sidecars` carries parsed sidecars; `resolve_resume_plan` latest branch consumes them (second read deleted); explicit path unchanged; no cache; public shapes unchanged | One redundant sidecar parse per candidate per resolution |
| A02 roots | `_parse_data` keeps fetch + tuple translation, drops the byte-identical shape re-check owned by native `rc_parse_data`; siblings untouched | One duplicate validator |
| A07 bounded | `load_restart` reuses `_validate_descriptor` instead of duplicating its native-call/JSON framing (identical mapping to `schema incompatible`) | Ten duplicated framing lines |
| A07 E1 shutdown (`rl_loop.py`) | Exception exit runs the same owned shutdown as normal exit: loop error captured as primary, then writer-drain, gpu-quiesce, pool-close, prep-join (before replay/tape drops), drops, spans-write, writer-close, signal restore all still run; cleanup errors attach as notes without masking the origin | Skipped-stages-on-exception path (behavioral, not deleted machinery) |
| A07 E2 activation (`rl_loop.py`, `rl_collect.py`, `rl_gpu.py`) | Activation commits only after admission: Collector closes its own orphan pool on mid-construction failure; `replace_pool` registers-then-swaps so a registration failure keeps the old binding and never abandons the new pool here; loop closes the orphaned new pool and restores `gpu.pool` on admission failure | Orphan-on-partial-activation path; half-rebind path (swap-before-register) |
| A07 checkpoint_paths (`rl_loop.py`) | Paths recorded from the `save_restart` return value at save time, so superseded submits never appear as landed | Intended-not-landed reporting |
| A07 train_games (`rl_loop.py`) | Corrected final multiplier to `current_generation * games_per_generation` (was `generations`) | Wrong-count reporting |

Correctness guarantees preserved: move-only pool-to-tape ownership;
8/128/1-epoch/at-most-1-future recipe; sampler streams/order, observations,
terminal advantages, PPO; behavior-actor immutability and learner/actor
pairing; atomic checkpoint/pointer publication; RNG restore-after-construction;
live/archive parity; drain preflight, replay binding/row membership,
exactly-once freezing, fill-source/span gates, buffer geometry/overlap/export
lifetime checks (all kept per A06/A07 keep-lists; fill/drop exercised by
existing `test_rl_artifact.py` live-fill/drop paths).

### 9.2 Verification (commands and observed results)

- `CARGO_TEST_FORCE=1 LD_LIBRARY_PATH=$PWD/.pixi/envs/default/lib cargo test
  -p hydra-bridge --lib rl::live::tests::prep_` — before fix: panic test
  failed on leaked claim (`remove_tape` false), begin-during-join test failed
  (begin succeeded mid-join); after fix: 7/7 pass (six ownership tests plus
  `prep_slot_reusable_after_failed_consumption` for same-process re-entry).
  The 30-round poll/wait race passed both before and after (stress guard, not
  a failing-first repro: the loss window is a microsecond join race
  unreachable deterministically without scheduler control; the shared root
  cause was proved by the two deterministic repros).
- Same harness `--lib rl::` — 102 passed, 0 failed (no regressions in
  lifecycle/collector/pool/artifact/replay neighbors).
- `pixi run build-ext` (one portable build per coherent batch) — exit 0 each time.
- Throwaway Pixi smoke (`begin`/`poll`/`wait` on unknown tape/job reject as
  `ValueError`; `scheduler_lr_factor` absent, `prefix_hashes_name` and
  `plane_suffix` present) — pass; script removed after use.
- `pixi run test-training` — 338 passed (selection verified via
  `--collect-only`: covers `test_rl_restart/artifact/pool/config`,
  `test_run_config` binding gates, `test_stream_train*`, `test_loop_parity*`
  torch scheduler goldens, `test_supervised_*`). Re-ran after the A07
  lifecycle fixes: 338 passed again.
- `pixi run test-training --package WP-11` — 117 passed (run three times:
  landed batch, loop rework, ownership fixes).
- `pixi run test-serial --package WP-11` — 5 passed, 264 deselected
  (exactly `tests/unit/test_rl_gpu.py`, the serial WP-11 slice: strict CDF,
  singleton, tiny-mass, replay-preserves-updates, actor-publish).
- `pixi run test-serial` (full lane) — 143 passed, 125 deselected, 1 xpassed
  (MahJax GPU-soak xpass: GPU jaxlib absent, CPU-only evidence) in ~117 s.
- `pixi run test-unit tests/unit/test_run_config.py
  tests/unit/test_shard_reader.py tests/unit/test_shard_build.py` — selected
  the whole unit directory (task supplies the directory): 828 passed,
  1 skipped (absent Tenhou corpus mount). Covers `TestResume`,
  `TestDataRoots`, shard reader/builder suites for the C10/roots edits.
- `pixi run lint`, `pixi run format-check`, `pixi run typecheck` — all pass
  (ruff-format fixes applied to the new projection helper and the loop
  shutdown rework; `train_games` multiplier corrected to
  `current_generation * games_per_generation`).
- `cargo clippy -p hydra-bridge --lib --tests` — clean under workspace denies.
- `pixi run config-check` — pass.
- `cargo fmt --check` is not a repo gate: stable and nightly rustfmt both
  report pre-existing drift across untouched files (repo formats under
  nightly-only options); edited regions follow surrounding file style and were
  left unreformatted per instruction.

### 9.3 Investigation outputs (no code changes)

**C06 bounded inventory (table only, zero deletions).** (1) Content-addressed:
analysis reports (producer `qual_gates.py`, consumer `analysis_gate_for` +
teacher gate; latest copy mutable, reader does not verify digest),
shard manifests/dataset identity, generic `ArtifactRegistry`
(**zero production callers**: `publish`/`lookup` live refs empty, only
`tests/contracts/test_registry.py`), checkpoints/local recovery, runtime
bookkeeping (run digests, CAS blobs, tips), work-package receipts (CLI +
native verifier; C03 deletion still blocked). (2) Ordinary-file corruption
detection: plane sha, section hashes, sidecar pins, payload digests —
consumers fail closed with `CorruptArtifactError`, never confer identity.
(3) Semantic seeds: preserved, no changes (counter streams, CTR draws,
17-field keys; `counter_stream` has no in-repo def beyond docs example).
(4) Ephemeral bookkeeping: run markers, latest pointers, index rows, logs,
mirrors, caches — no digest-verified identity consumer. Unresolved externals
marked explicitly: offline shard tooling, CI/auditor readers of
`work_packages/*` and `reports/WP-12/*`, future artifact publishers,
ops/harness readers of run/checkpoint/pointer files, mount-path consumers.

**C09/A01 shard boundary map (map only, code deferred per Q1).**
Authorization/attestation: real gate is `require_attestation` at build/ingest;
the manifest holds a 2-key receipt; the reader only passes it through; the
offline validator checks dict shape only. Attestation shape is not
authorization. Plane/schema/width: producer pins (`schema_digest`,
`action_width`, plane sha), reader enforces on open (schema only when the
caller passes an expectation), validator re-checks unconditionally.
Decision-ID membership/ordering/identity: producer sorts/perms and hashes
sorted ids order-independently; reader re-derives order from sidecar and
verifies membership hash; validator checks row counts and canonical sortedness
(perm content unchecked). Three orderings stay distinct: document
representation (JSON bytes) vs decision-ID order (manifest `order`) vs
physical feature-row alignment (chunk sums + trailing shapes). Sidecar bytes:
producer pins raw-byte digests; reader checks length/membership only;
validator checks raw bytes fully. Build statistics: producer writes,
reader ignores, validator requires six presence fields plus row agreement.
Version analysis: any change to required keys, hash/order/width/shape/sha
semantics needs an explicit bump; additive informational keys do not. No compat
modes or optional-validation flags. Minimal C08/C09 decision (pending):
freeze the version-1 required set as inventoried and rule which, if any,
required semantic (byte-digest/canonical-order/attestation-minimum) moves
into the reader before the validator can retire.

**C01 broader scope (separate from the mechanical items above).** Keeps
confirmed with caller evidence: `_gate_hash_for_kind` (teacher identity),
report kind/version literals (test + Rust parity pins), deadline/cap kernels
(live in two gates). Candidates needing per-leaf equivalence plus
minimum-bridge fail-closed proof before any removal: `_plane_suffix`,
`_prefix_hashes_name`, three qual leaves, obs/utility/randomness/search/
tracking/eval/data fallbacks. Scheduler residue: zero outside this plan
document. C10/A02/prefix-export work is not counted as C01 progress.

**A05 audit (findings only; no merge authorized).** Keeps: `_qual_gate_leaves`,
`_rc_parse_aux`, qual 3-way, `_rc_*` records/parsers/root/resume split,
loop/stream/distillation/bridge splits (torch-vs-pure, IO-vs-pure,
review-size with distinct owners; full per-file anchors in the investigation
transcript). Merge candidates identified but left unmerged (each names the
removed concept, cycle status, and risk): `_rc_parse`+`_aux`,
`_rc_require`+`_rc_digest`, `loop_batch`+`adapters`, `stream_dataset`+`buffer`,
`rl_checkpoint`+`_native`, teacher `records`+`student`,
`lifecycle_ledger`+`lifecycle_common`, `metrics`+`metrics_ffi`,
`_ring_events`+`pinned_ring`.

**A07 shutdown/activation (E1–E3 fixed in this slice; input policy unchanged).**
E1: exception exit now runs the same owned shutdown — drain, quiesce,
pool-close, prep-join before drops, drops, spans-write, writer-close, signal
restore — with the loop error primary and cleanup errors attached as notes.
E2: `Collector.__init__` closes its own orphan pool on mid-construction
failure; `replace_pool` is register-then-swap (registration failure keeps the
old binding); `activate_future` commits `collector`/`future` only after
admission and closes the orphaned new pool otherwise. E3: every stage runs
inside `_cleanup`; writer errors still surface first but no longer skip later
stages; the originating failure is never masked. `GpuService.close` still
never runs on the loop path (high-water only, documented). New native test
`prep_slot_reusable_after_failed_consumption` proves same-process slot
re-entry after a consumed failure. `checkpoint_paths` and `train_games` fixed
as above.
**A07 input policy (reported, unchanged).** `load_restart` accepts extra
top-level container keys and boolean-numeric cursor equality (`False == 0`);
the native descriptor rejects boolean integers. Intended contract and any
authority change are undecided — acceptance policy untouched in this slice.

**14–15 s tail restated:** the cohort-completion gap after training ends is a
producer (collection) tail with inference still running, not measured all-zero
GPU time; 98% median CSV utilization is not useful-occupancy proof. No new
full GPU run was launched for this slice.

### 9.4 Remaining concrete decisions (for MAIN)

1. C09/C08 minimal contract (see 9.3): exact required set + reader additions
   before validator retirement; version-bump trigger.
2. C06 follow-up: `ArtifactRegistry` has no production callers — retire the
   generic registry, keep it, or narrow its supported types? Unresolved
   externals listed above apply.
3. C07/C08 versions and external-consumer migration (unchanged from §4).
4. C02 real runner/corpus, C03 retirement, C04 generalization, C05 shared
   guarantee changes: still blocked, no action taken.
5. Non-RL fallback/duplication/dead-code candidates from the new scout sweeps
   (see 9.3): each needs its stated missing proof or a MAIN removal approval —
   none authorized yet. `_encode`/`_decode` trios stay split by trust domain
   (cold/live/artifact + loop hot-path outlier); `rl_checkpoint.py`'s four
   export-only helpers stay pending out-of-repo consumer check.
