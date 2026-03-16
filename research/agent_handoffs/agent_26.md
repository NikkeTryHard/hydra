The public repo still exposes the same governing order and status that your artifact bundle describes: `HYDRA_FINAL.md`, then `HYDRA_RECONCILIATION.md`, then `docs/GAME_ENGINE.md`, and the README still says replay/sample ExIt is live, live self-play RL `delta_q` is live, while replay/offline `delta_q` and train-bin activation remain blocked pending a separate provenance/validation tranche. ([GitHub][1])

## 1. Decision

Implement replay/offline `delta_q` as a **search-derived replay sidecar** that reuses the **existing shared root-search producer** and joins into replay samples with the **same replay identity / legal-mask / source-version discipline** as ExIt.

Do **not**:

* build `delta_q` inside `mjai_loader.rs` from replay/public data,
* lift the runtime Group-C `[34]` `delta_q` feature plane into a teacher,
* create a second search semantics for replay,
* add heads,
* broaden AFBS,
* unblock `train.rs` in the same patch.

### Final verdict

* **Replay/offline provenance closure:** **repo-backed enough to implement now**.
* **Full BC/train-bin activation closure:** **still blocked**, but the blocker is no longer `delta_q` semantics or replay provenance.
* **Smallest decisive missing artifact:** a narrow **BC/train advanced-head activation hook spec** that wires `HeadActivationController` into BC/train and defines warmup-time trunk detachment for sparse heads like `DeltaQ`.

That blocker is real: public `bc.rs` still uses `HydraLoss` directly and has no visible `HeadActivationController`, `approved_loss_config`, or `extract_target_presence` integration, so BC-side `delta_q` enablement would currently bypass Hydra’s own sparse-head activation discipline. ([GitHub][2])

---

## 2. Authority boundary: what is required vs what is merely possible

### Direct artifact support — high confidence

1. **Replay/offline `delta_q` is intentionally absent today.**
   Support: README status, REC staging rule, loader hardcoded `None`, replay absence regression test, train-bin rejection tests.
   Validation path: `README L0062`, `REC L0435-L0449`, `LOADER L0451-L0454`, `LOADTEST L0783-L0792`, `POLICY L0018-L0023`, `TPOLTEST L1083-L1102`.

2. **The surviving teacher object is already semantically narrow.**
   It is masked action-space `[46]`, discard-compatible only, `Q(child)-Q(root)`, emitted by the shared root-search producer.
   Validation path: `DQOBJ L0263-L0303`, `LIVE L0281-L0366`, `ARENA L0559-L0600`, `LIVETEST L0761-L0781`.

3. **No new heads / no AFBS broadening / no runtime-bridge lift are allowed in this tranche.**
   Validation path: `REC L0493-L0498`, `REC L0504-L0508`, `REC L0512-L0515`, `REC L0537-L0538`.

4. **Loss surface and carriers already exist.**
   `HydraTargets`, `MjaiSample`, `MjaiBatch`, model head, masked MSE, RL batch collation all already carry `delta_q`.
   Validation path: `LOSS L0021-L0032`, `SAMPLE L0157-L0164`, `SAMPLE L0399-L0414`, `LOSSSEM L0321-L0330`, `RLBATCH L0477-L0485`, `ITEST L0229-L0250`.

### Code only makes this possible — not closed

1. **Typed carriers are not proof of replay/offline semantic closure.**
   `HydraTargets.delta_q_target` existing does not prove a teacher object exists for replay.
   Validation path: contrast `LOSS L0027-L0028` with `LOADER L0453-L0454`.

2. **The live RL lane proves object correctness and RL transport, not replay provenance.**
   Validation path: `DQV L0001-L0260`, `RL L0026-L0079`, `RLBATCH L0437-L0490`.

3. **The runtime Group-C `delta_q` feature plane is not a replay teacher.**
   It is a live encoder/runtime feature family, not an offline provenance contract.
   Validation path: `FINAL L0078-L0088`, `BRIDGE L0301-L0347`, `REC L0504-L0508`.

### Inference — medium confidence

1. **Replay/offline closure should follow the ExIt sidecar pattern, but not identically.**
   Keep the replay identity / provenance / version checks. Narrow the schema to signed regression semantics.
   Why only medium: authority docs require provenance-explicit closure, but they do not spell out the exact `delta_q` record shape.
   Validation path: implement roundtrip tests against ExIt-style join discipline.

---

## 3. Exact surviving v1 `delta_q` teacher object

### 3.1 Object semantics — direct support, high confidence

Let `s` be a replay decision state after replay reconstruction, and let `root` be the AFBS root built by the existing shared search producer.

Define the supported action set

[
\mathcal{A}_{\Delta q}(s)=\left{a \mid
\begin{array}{l}
a \le \texttt{DISCARD_END},\
a \notin {\texttt{AKA_5M},\texttt{AKA_5P},\texttt{AKA_5S}},\
\text{legal}(a)=1,\
\text{child}(a)\ \text{exists at the seeded root},\
N(\text{child}(a)) > 0,\
Q(root),Q(\text{child}(a))\in \mathbb{R}
\end{array}
\right}
]

Then the target is

[
m_a = \mathbf{1}[a\in \mathcal{A}_{\Delta q}(s)]
]

[
\Delta q^*_a = m_a \cdot \left(Q(\text{child}(a)) - Q(root)\right)
]

All unmasked actions must satisfy:

[
m_a = 0 \implies \Delta q^*_a = 0
]

This is not a distribution. It is not normalized. It is signed regression over a masked subset of the `[46]` action surface.
Support: `DQOBJ L0263-L0303`, `ARENA L0559-L0600`, `LOSSSEM L0321-L0330`.
Confidence is high because builder code, invariants, and loss semantics all agree.

### 3.2 Emission envelope — direct support, high confidence

The object above is only valid when emitted through the current shared root-search lane:

* compatible discard-only state,
* at least 2 legal discards,
* base policy computed from raw logits,
* hard-state gate,
* all legal discard children seeded at the root,
* learner-only value-head child evaluation,
* root-only AFBS search,
* `None` on any failed gate.

Support: `LIVE L0294-L0366`.
Confidence high because the producer code already enforces this exact envelope.

### 3.3 Worked example

If legal discard actions are `{1,2,5}`, root search yields:

* `Q(root)=0.40`
* `Q(child_1)=0.90`
* `Q(child_2)=0.10`
* child `5` exists but has `visit_count=0`

then

* `mask[1]=1`, `target[1]=+0.50`
* `mask[2]=1`, `target[2]=-0.30`
* `mask[5]=0`, `target[5]=0.0`
* all non-discard / aka / illegal actions: `mask=0`, `target=0`

Coverage is

[
\text{coverage} = \frac{2}{3}
]

This matches the builder semantics in `DQOBJ L0284-L0302`.

---

## 4. What the existing live RL lane proves, and what it does not prove

### Proved now — direct support, high confidence

1. **The teacher object is exact enough to name.**
   `Q(child)-Q(root)`, masked discard-only, finite, legal-only.
   Support: `DQOBJ`, `ARENA`.

2. **One search can emit both ExIt and `delta_q`.**
   Support: `LIVE L0349-L0366`.

3. **RL transport works.**
   Sidecar-free self-play already collates `delta_q_target`/`delta_q_mask` into `HydraTargets`.
   Support: `RLBATCH L0437-L0485`, `ITEST L0229-L0250`.

4. **Masked regression already exists.**
   Support: `LOSSSEM L0321-L0330`.

5. **There is already a structural `delta_q` validator.**
   Support: `DQV L0014-L0260`.

### Not proved now — direct support, high confidence

1. **Replay/offline provenance is not closed.**
   Loader still assigns `None`.
   Support: `LOADER L0451-L0454`, `LOADTEST L0783-L0792`.

2. **BC/train-bin activation is not safe just because the tensors exist.**
   `train.rs` still rejects `advanced_loss.delta_q`.
   Support: `POLICY L0018-L0023`, `TPOLTEST L1083-L1102`.

3. **Typed surfaces do not prove semantic readiness.**
   Support: `LOSS`, `SAMPLE`, contrasted with loader absence and policy rejection.

4. **Runtime `[34]` `delta_q` feature planes are not replay teachers.**
   Support: `FINAL L0078-L0088`, `REC L0504-L0508`.

---

## 5. Replay/offline provenance object: exact v1 contract

## 5.1 Chosen v1 schema

```rust
pub const REPLAY_DELTA_Q_SEMANTICS_V1: &str =
    "delta_q_root_child_minus_root_q_v1";
pub const REPLAY_DELTA_Q_PROVENANCE: &str = "search-derived";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplayDeltaQLookupKey {
    pub replay: ReplayDecisionKey, // keep ExIt replay identity exactly
    pub action: u8,                // sample discriminator only
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayDeltaQRecordV1 {
    pub version: u32,              // schema version; must be 1
    pub semantics: String,         // must equal REPLAY_DELTA_Q_SEMANTICS_V1
    pub provenance: String,        // must equal REPLAY_DELTA_Q_PROVENANCE

    pub key: ReplayDecisionKey,    // {source_hash, event_index, actor, obs_hash}
    pub action: u8,                // replay action id; not teacher-defining
    pub legal_mask_digest: u64,    // digest of replay sample legal mask
    pub source_net_hash: u64,      // checkpoint identity used for search/value eval
    pub source_version: u32,       // producer/model version

    pub search_budget: u32,        // exact known quantity from budget_from_legal_count()
    pub legal_discard_count: u8,
    pub supported_actions: u8,
    pub coverage: f32,             // supported_actions / legal_discard_count

    pub target: Vec<f32>,          // len == HYDRA_ACTION_SPACE
    pub mask: Vec<f32>,            // len == HYDRA_ACTION_SPACE
}
```

### Why this exact object survives

**Keep from ExIt** — direct/inference mix, confidence high-to-medium:

* `ReplayDecisionKey`
* action discriminator
* `legal_mask_digest`
* `source_net_hash`
* `source_version`
* explicit `semantics`
* explicit `provenance`
* vector `target` + vector `mask`
* loader-time mismatch rejection

**Narrow from ExIt** — medium confidence, justified by semantics:

* `search_budget` instead of blindly reusing ExIt’s audit field shape as “root visits”; budget is the exact quantity exposed by the current replay-sidecar producer code path.
* signed regression target, not a normalized policy distribution.
* no KL field.
* no top-1 agreement field in the record.

**Remove from ExIt** — high confidence:

* `kl_to_base`
* any visit-distribution interpretation
* any q-softmax alias
* any action-probability semantics

### Why `action` stays in v1

This is **inference**, confidence **medium**:

* semantically, `delta_q` is state-rooted, not action-conditioned;
* operationally, keeping `action` in the lookup key preserves loader symmetry with ExIt and avoids a wider replay-join refactor.

If later deduplication matters, `v2` can drop `action` and key only by `ReplayDecisionKey`.
For `v1`, keeping it is the lower-risk patch.

### Simpler local alternative

Key only by `ReplayDecisionKey` and make `action` audit-only.
Not chosen in `v1` because it changes the current loader/join symmetry for little immediate gain.

---

## 6. Exact join contract

## 6.1 Join-time lookup

```rust
pub fn lookup_label(
    &self,
    key: &ReplayDecisionKey,
    action: u8,
    legal_mask: &[f32; HYDRA_ACTION_SPACE],
    source_net_hash: u64,
    source_version: u32,
) -> Option<([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE])>
```

## 6.2 Required checks

Reject the record unless **all** checks pass:

```rust
record.version == 1
record.semantics == REPLAY_DELTA_Q_SEMANTICS_V1
record.provenance == REPLAY_DELTA_Q_PROVENANCE
record.legal_mask_digest == legal_mask_digest_from_f32(legal_mask)
record.source_net_hash == source_net_hash
record.source_version == source_version
record.target.len() == HYDRA_ACTION_SPACE
record.mask.len() == HYDRA_ACTION_SPACE
```

Then perform **delta_q-specific structural validation** before accepting the label:

```rust
fn delta_q_contract_ok(
    target: &[f32; HYDRA_ACTION_SPACE],
    mask: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[f32; HYDRA_ACTION_SPACE],
) -> bool {
    let mut saw_masked = false;

    for a in 0..HYDRA_ACTION_SPACE {
        let m = mask[a];
        let t = target[a];

        if !(m == 0.0 || (m - 1.0).abs() < 1e-6) {
            return false;
        }
        if !t.is_finite() {
            return false;
        }

        if m > 0.5 {
            saw_masked = true;

            if legal_mask[a] <= 0.0 {
                return false;
            }
            if a > DISCARD_END as usize {
                return false;
            }
            if matches!(a as u8, AKA_5M | AKA_5P | AKA_5S) {
                return false;
            }
        } else if t.abs() > 1e-5 {
            return false;
        }
    }

    saw_masked
}
```

Also verify the derived metadata:

```rust
let supported_actual = mask.iter().filter(|&&m| m > 0.5).count() as u8;
if supported_actual != record.supported_actions {
    return None;
}
if record.legal_discard_count == 0 {
    return None;
}
let expected_coverage = supported_actual as f32 / record.legal_discard_count as f32;
if (record.coverage - expected_coverage).abs() > 1e-4 {
    return None;
}
if record.search_budget == 0 {
    return None;
}
```

### Confidence

High.
Why: every structural check above is already implied by the current arena invariants or builder semantics; this adds no new teacher meaning, only explicit provenance rejection.
Validation path: `ARENA L0559-L0600`, `DQOBJ L0263-L0303`.

---

## 7. File-level implementation

## 7.1 `crates/hydra-train/src/training/replay_delta_q.rs` — new file

### Required content

1. Reuse from `replay_exit.rs`:

   * `ReplayDecisionKey`
   * `source_hash_from_identity`
   * `source_net_hash_from_checkpoint_identity`
   * `legal_mask_digest_from_f32`

2. Define:

   * `REPLAY_DELTA_Q_SEMANTICS_V1`
   * `REPLAY_DELTA_Q_PROVENANCE`
   * `ReplayDeltaQLookupKey`
   * `ReplayDeltaQRecordV1`
   * `DeltaQSidecarIndex`
   * `lookup_label`
   * `from_jsonl_reader`
   * `from_jsonl_path`

3. Add producer:

```rust
pub fn generate_replay_delta_q_records<B: Backend>(
    source_hash: u64,
    events: &[MjaiEvent],
    model: &HydraModel<B>,
    device: &B::Device,
    exit_cfg: &ExitConfig,      // keep shared producer config exactly
    source_net_hash: u64,
    source_version: u32,
) -> io::Result<(Vec<ReplayDeltaQRecordV1>, DeltaQValidationReport)>
```

### Producer algorithm

* reconstruct replay state exactly like `replay_exit.rs`
* for each sampled event:

  * build `obs`
  * map replay action to Hydra action
  * build `legal_mask`
  * encode observation
  * create `RootDecisionContext`
  * create `ReplayDecisionKey`
  * reconstruct validator counters:

    * `compatible_discard_state`
    * legal discard count
    * hard-state gate from base logits
  * call **shared** producer:

```rust
let labels = try_search_labels_from_context(
    &state,
    &obs,
    &ctx,
    &safety[actor],
    exit_cfg,
    &mut |obs_encoded| model.policy_value_cpu(obs_encoded, device),
    &mut adapter,
);
```

* take `labels.and_then(|l| l.delta_q)`
* if `None`, update rejection counters and continue
* if `Some(delta_q)`:

  * compute `supported_actions`
  * compute `coverage`
  * update `DeltaQValidationReport`
  * emit `ReplayDeltaQRecordV1`

### Keep / narrow / remove at producer

**Keep**

* shared search call
* learner-only value head
* same hard-state gate
* same all-legal discard seeding
* `None` on failed gate

**Narrow**

* record only `delta_q`, not ExIt target
* audit field `search_budget`, not `kl_to_base`

**Remove**

* no second search
* no replay-derived fallback label
* no bridge-plane fallback

### Simpler and stronger local alternatives

* **Simpler**: implement `replay_delta_q.rs` as a sibling of `replay_exit.rs` and tolerate duplicate search when ExIt and `delta_q` sidecars are generated separately.
  Confidence: high. Low churn.

* **Stronger**: factor a shared internal replay-search helper returning `TrajectorySearchLabels`, and let ExIt / `delta_q` writers consume one search result.
  Confidence: medium. Better compute hygiene, bigger patch.

Choose the **simpler** one for the first patch unless replay-sidecar generation is routinely dual-lane.

---

## 7.2 `crates/hydra-train/src/data/mjai_loader.rs`

### New loader entrypoint

Add a new narrow API, keep the existing ExIt-only and plain-loader APIs intact:

```rust
pub struct ReplaySidecars<'a> {
    pub exit: Option<&'a ExitSidecarIndex>,
    pub delta_q: Option<&'a DeltaQSidecarIndex>,
}
```

or, if you want less API surface churn, add:

```rust
pub fn load_game_from_events_with_sidecars(
    source_identity: &str,
    source_net_hash: u64,
    source_version: u32,
    events: Vec<MjaiEvent>,
    exit_sidecar: Option<&ExitSidecarIndex>,
    delta_q_sidecar: Option<&DeltaQSidecarIndex>,
) -> io::Result<MjaiGame>
```

### Join logic

Inside the sampled replay path:

```rust
let replay_key = source_hash.map(|source_hash| ReplayDecisionKey {
    source_hash,
    event_index: idx as u32,
    actor: actor as u8,
    obs_hash: crate::training::live_exit::obs_hash(&obs_encoded),
});

let joined_delta_q = replay_key.and_then(|key| {
    delta_q_sidecar.and_then(|sidecar| {
        source_net_hash.zip(source_version).and_then(|(source_net_hash, source_version)| {
            sidecar.lookup_label(
                &key,
                hydra_action.id(),
                &legal_mask,
                source_net_hash,
                source_version,
            )
        })
    })
});
```

Populate **both** fields from the **same** `Option`:

```rust
delta_q_target: joined_delta_q.map(|(target, _)| target),
delta_q_mask: joined_delta_q.map(|(_, mask)| mask),
```

### Loader invariants

* no sidecar => `None, None`
* any contract mismatch => `None, None`
* plain replay loader remains unchanged and keeps `delta_q` absent
* do not compute `delta_q` in loader code
* do not silently fabricate a zero mask with nonzero target

### Confidence

High.
Why: this is exactly the already-proven ExIt replay join pattern, narrowed to the `delta_q` object.
Validation path: mirror `REXITTEST` with `delta_q`-specific contract tests.

---

## 7.3 `crates/hydra-train/src/data/sample.rs`

This file needs the most important **local correction**.

### Problem in current code — direct support, high confidence

Current `sample.rs` stores `delta_q_target` and `delta_q_mask` in **parallel flat buffers** with a single `any_delta_q` flag:

* `SAMPLE L0185-L0187`
* `SAMPLE L0314-L0323`
* `SAMPLE L0399-L0414`

That means a per-sample mismatch such as:

```rust
sample.delta_q_target = Some([...]);
sample.delta_q_mask   = None;
```

can become:

* batch-level `Some(delta_q_target)`
* batch-level `Some(delta_q_mask)` (because some other row may have a mask, or because `any_delta_q` is true)
* zero mask row for the broken sample

`validate_optional_target_pairs` in `rl.rs` only checks **batch-level** `(Some,Some)` vs `(None,None)`, not per-row mismatch. `masked_action_mse` then returns zero on a zero-mask row even if the target row contains garbage.
Support: `RL L0060-L0067`, `LOSSSEM L0321-L0330`.
Confidence high because this follows directly from the current write/collate logic.

### Worked failure example

If a row has:

```text
target = [1.2, -0.7, ...]
mask   = [0.0, 0.0, ...]
```

then

[
L_{\Delta q} =
\frac{\sum \frac12 (\hat{\Delta q}-\Delta q)^2 \cdot m}{\max(1,\sum m)} = 0
]

even though the target row is invalid.

### Required fix

Replace the parallel-flat-buffer path with the already-existing canonical option-pair path used by ExIt.

#### Change `CollateBuffers`

```rust
struct CollateBuffers {
    // ...
    delta_q_samples: Vec<Option<(Vec<f32>, Vec<f32>)>>,
}
```

#### Change `write_sample`

```rust
self.delta_q_samples[index] = match (delta_q_target, delta_q_mask) {
    (Some(target), Some(mask)) => Some((target.to_vec(), mask.to_vec())),
    (None, None) => None,
    _ => panic!("delta_q target/mask mismatch at sample {index}"),
};
```

#### Change `into_batch`

```rust
let (delta_q_target, delta_q_mask) =
    collate_delta_q_targets::<B>(&self.delta_q_samples, device);

MjaiBatch {
    // ...
    delta_q_target,
    delta_q_mask,
    // ...
}
```

### Why this is the chosen local fix

* `collate_delta_q_targets` already exists (`DQOBJ L0335-L0357`)
* it matches the ExIt absent-row semantics
* it makes per-sample mismatch impossible to hide
* it is narrower than inventing a new validator layer

### Confidence

High.
Validation path:

* add `delta_q_sample_pair_mismatch_panics`
* add `delta_q_collation_roundtrips_absent_rows_as_zero_mask_rows`
* add `delta_q_target_without_mask_cannot_reach_MjaiBatch`

---

## 7.4 `crates/hydra-train/src/training/delta_q_validation.rs`

Keep the existing report and threshold types. Do **not** invent new global thresholds in this tranche.

### Reuse unchanged

```rust
DeltaQValidationReport
DeltaQValidationThresholds::default()
evaluate_report(...)
```

Current default thresholds already give the conservative structural gate:

* `sample_size >= 1000`
* `emission_rate >= 0.01`
* `mean_coverage >= 0.70`
* `mean_supported_actions >= 3.0`

Support: `DQV L0174-L0259`.

### Add replay/offline runner

```rust
pub fn run_replay_delta_q_validation<B: Backend>(
    replays: &[(String, Vec<MjaiEvent>)],
    model: &HydraModel<B>,
    device: &B::Device,
    exit_cfg: &ExitConfig,
    source_net_hash: u64,
    source_version: u32,
) -> DeltaQValidationReport
```

This runner should:

* reconstruct replay states,
* call the same shared producer,
* aggregate the same report fields,
* evaluate with the same `DeltaQValidationThresholds`.

### Add replay roundtrip validator

This is the **new provenance-specific validator**:

```rust
pub struct ReplayDeltaQRoundtripReport {
    pub total_records: u64,
    pub matched_records: u64,
    pub missing_records: u64,
    pub contract_rejections: u64,
}
```

Pass condition:

```text
matched_records == total_records
missing_records == 0
contract_rejections == 0
```

Run it on a self-generated sidecar over the exact same replay corpus and source identity/version.

### Why this survives

The existing `DeltaQValidationReport` is structural and search-lane-oriented.
The new roundtrip report is the smallest replay/offline addition that actually tests provenance closure, not just teacher sparsity.

### Confidence

* Reusing existing thresholds: **medium**.
  Why not high: current artifacts define them for the RL lane, not explicitly for replay/offline BC.
  Why still acceptable: they are already conservative and lane-local.
* Adding exact roundtrip match: **high**.
  Why: it directly tests the sidecar contract you are introducing.

### Stronger local alternative

Compare replay sidecar labels against a deeper learner-only AFBS reference on sampled replay states:

* masked sign agreement,
* top-discard agreement,
* masked MAE.

This is useful, but it is **archive-supported only**, not current authority doctrine. Keep it out of the v1 blocking gate.

---

## 8. Tests that must land with the patch

## 8.1 `training/replay_delta_q.rs`

Add:

* `delta_q_sidecar_lookup_requires_matching_contract`
* `delta_q_sidecar_lookup_rejects_non_discard_mask`
* `delta_q_sidecar_lookup_rejects_aka_mask`
* `delta_q_sidecar_lookup_rejects_nonzero_target_outside_mask`
* `delta_q_sidecar_lookup_rejects_nonfinite_target`
* `replay_delta_q_records_are_tagged_search_derived`
* `loader_with_delta_q_sidecar_populates_delta_q_fields`
* `self_generated_delta_q_sidecar_roundtrips_exactly`

## 8.2 `data/sample.rs`

Add:

* `delta_q_sample_pair_mismatch_panics`
* `delta_q_collation_uses_zero_rows_for_absent_samples`
* `delta_q_collation_preserves_present_row_values`

## 8.3 `training/delta_q_validation.rs`

Add:

* `replay_delta_q_validation_thresholds_pass_on_passing_report`
* `replay_delta_q_roundtrip_report_requires_exact_match`

## 8.4 Keep these existing tests unchanged

* replay plain-loader absence test
* train-bin rejection tests

This preserves the deliberate guardrail until activation hook lands.

---

## 9. Staged activation closure

## Stage A — implement now

Land all of this now:

1. `replay_delta_q.rs`
2. loader join path
3. `sample.rs` pair-safety fix
4. replay/offline `delta_q` validation runner
5. replay roundtrip validator
6. tests above

Keep `train.rs` rejection unchanged.

### Confidence

High.
Why: no new semantics, no new heads, no AFBS broadening, all changes are direct extensions of already-closed ExIt and RL `delta_q` lanes.

---

## Stage B — shadow validation gate

Require all of the following before any BC/train enablement discussion:

1. `DeltaQValidationResult.passed == true`
2. `ReplayDeltaQRoundtripReport` exact match pass
3. at least one self-generated replay corpus shows nonzero joined `delta_q_target` coverage
4. plain replay loader still keeps `delta_q` absent without sidecar
5. no per-sample target/mask mismatch can reach `MjaiBatch`

### Confidence

High.
Why: this is exactly the missing provenance/validation tranche the authority docs call for; it is not broader than the current lane.

---

## Stage C — still blocked

Do **not** unblock `train.rs` yet.

### Exact blocker

A narrow **BC/train advanced-head activation hook** is still missing.

The hook must define:

1. how BC/train records per-batch target presence,
2. how BC/train applies `approved_loss_config`,
3. how BC/train handles warmup-time trunk detachment for `warmup_heads()`.

The current controller docs explicitly require the caller/orchestrator to do that, and public `bc.rs` still does not. Support: `GATEA L0031-L0042`, `GATEB L0531-L0572`, plus public `bc.rs` check. ([GitHub][2])

### Smallest decisive missing contract artifact

Implement or write down this exact narrow spec:

```rust
pub struct BcAdvancedHeadActivationHook {
    pub controller: HeadActivationController,
}

pub fn bc_apply_head_gating<B: Backend>(
    controller: &mut HeadActivationController,
    base_loss: &HydraLossConfig,
    targets: &HydraTargets<B>,
) -> HydraLossConfig {
    let presence = extract_target_presence(targets);
    controller.record_batch(&presence);
    controller.approved_loss_config(base_loss)
}
```

and add the missing model-side warmup API:

```rust
pub fn forward_with_warmup_detach<B: Backend>(
    &self,
    obs: Tensor<B, 3>,
    loss_cfg: &HydraLossConfig,
    warmup_heads: &[AdvancedHead],
) -> HydraOutput<B>;
```

`DeltaQ` warmup behavior in that API must be:

* **no new head**
* **no trunk change**
* feed the existing `delta_q` head from the current shared representation,
* but if `AdvancedHead::DeltaQ` is in `warmup_heads`, detach that shared representation before the `delta_q` head so the head trains while trunk stays frozen.

Without that hook, flipping `advanced_loss.delta_q` on in BC/train would bypass Hydra’s own sparse-head activation doctrine.

### Confidence

High.
Why:

* the controller contract already says the caller must do this,
* the roadmap already flags orchestration/warmup integration as open,
* public `bc.rs` shows no controller hookup.

---

## 10. Keep / narrow / remove summary

| Piece                                       | Verdict              | Why                                       |
| ------------------------------------------- | -------------------- | ----------------------------------------- |
| `ReplayDecisionKey`                         | **Keep**             | already canonical replay identity         |
| `source_net_hash` / `source_version`        | **Keep**             | already canonical producer provenance     |
| `legal_mask_digest`                         | **Keep**             | already canonical join guard              |
| action discriminator                        | **Keep, but narrow** | join-only, not teacher-defining           |
| `search-derived` provenance tag             | **Keep**             | matches ExIt and lane semantics           |
| shared root-search producer                 | **Keep**             | already emits ExIt + `delta_q` together   |
| sidecar JSONL / index pattern               | **Keep**             | proven by ExIt                            |
| `kl_to_base`                                | **Remove**           | not meaningful for signed regression      |
| visit-distribution teacher                  | **Remove**           | wrong teacher object                      |
| q-softmax teacher                           | **Remove**           | wrong teacher object                      |
| runtime `[34]` bridge plane as label source | **Remove**           | feature lane, not teacher lane            |
| replay-derived loader builder               | **Remove**           | violates provenance-explicit rule         |
| current `sample.rs` flat `delta_q` buffers  | **Remove**           | weaker than existing option-pair collator |

---

## 11. Confidence ledger

| Conclusion                                                      | Support type       |  Confidence | Why                                                                 | Falsify by                                               |
| --------------------------------------------------------------- | ------------------ | ----------: | ------------------------------------------------------------------- | -------------------------------------------------------- |
| v1 teacher is masked discard-only `Q(child)-Q(root)`            | direct             |        high | builder, live producer, invariants, tests all agree                 | break `DQOBJ`/arena tests                                |
| replay/offline must be search-derived sidecar, not loader-built | direct             |        high | README/REC/loader absence/test/train rejection all align            | produce authority text blessing replay-built `delta_q`   |
| ExIt sidecar is the right provenance template                   | inference + direct | medium-high | already working in same repo, same replay identity problem          | implement and fail exact roundtrip                       |
| `sample.rs` must switch to option-pair collation                | direct reasoning   |        high | current code can hide per-row mismatch behind zero mask             | add mismatch test and observe current silent pass        |
| full BC/train activation is still blocked on activation hook    | direct + web check |        high | controller docs require caller integration; public `bc.rs` lacks it | show existing BC controller hookup and warmup detach API |

---

## 12. Final decision

Implement **Stage A** now:

* new `replay_delta_q.rs`,
* loader join,
* `sample.rs` pair-safety fix,
* replay validation runner,
* exact roundtrip validator,
* tests.

That closes the **replay/offline provenance question** without semantic guessing.

Do **not** claim the lane is fully activation-closed yet.
It is not.
The remaining blocker is narrow and explicit:

> **Missing artifact:** BC/train advanced-head activation hook spec (`HeadActivationController` integration + warmup detach API).

Once that generic hook exists, `delta_q` no longer needs new semantics, new provenance, or new search doctrine. It only needs the already-built controller to be honored in BC/train.

[1]: https://github.com/NikkeTryHard/hydra "https://github.com/NikkeTryHard/hydra"
[2]: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/crates/hydra-train/src/training/bc.rs "raw.githubusercontent.com"
