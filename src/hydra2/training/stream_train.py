"""Streaming-first supervised training driver: stream → expand → encode → loop.

Consumes ``.mjai.json.zst`` straight from ``data.root`` (no parquet shards)
yet carries the identical contracts as the materialized path: strict decode,
validate-then-quarantine, actor/privileged split, wall-disjoint manifests plus
the leakage check, counter-based RNG, and ``(5,)`` dora.

Pipeline per call to :func:`run_stream_training`:

1. Resolve the run layout (idempotent) and the stream manifest over
   ``data.root`` (provenance pin verified before the first game is read).
2. Scan the corpus once (ordered, ``split=None``) to enforce wall-disjoint
   train/val assignment and to collect the evaluation-wall ledger plus the
   quarantine counters. Any wall shared across the two splits fails closed
   before any runtime object is built.
3. Pull train-split games through :class:`GameStream` (or the bit-identical
   :class:`PrefetchGameStream` when ``data.num_workers > 0``), expand each
   game to actor :class:`DecisionRow` rows through the ``"python"`` oracle
   (wall-bound games bind the real S7 wall digest, wall-less games the SIM
   mark; both fail closed per game),
   quarantining-and-counting expansion failures per game — encode
   microbatches through the real actor-visible encoder, join privileged
   ranks by opaque decision id (train-split rows only; the loop gates the
   join on config weights), and drive :meth:`SupervisedLoop.train` with the
   config weights.
4. Write ``ckpt-<update>.pt`` + ResumeExactPlan sidecars every
   ``loop.checkpoint_frequency_updates`` updates, append ``metrics.jsonl``
   rows plus ``train.log`` lines, record per-microbatch feed telemetry to
   ``logs/feed-telemetry.jsonl`` (observer-only walls; resume never reads it),
   prune to ``keep_last_checkpoints``, and
   leave best-checkpoint promotion to the manual gate
   (:meth:`SupervisedLoop.evaluate_selection` /
   :meth:`SupervisedLoop.maybe_promote_best` are never called here).

Resume is row-exact: checkpoints record the pulled-game frontier cursor plus
the consumed-microbatch count of the live epoch. Resume re-verifies sidecar
identity (run digest, seed, worker plan, payload hash, RNG anchors) BEFORE
mutating any runtime object, drains the identical microbatch prefix (the
stream re-emits the same sequence from the same seed, so the drain is a
catch-up slice, never a refill), verifies the drained frontier against the
recorded cursor, restores model/optimizer/scheduler/loop/RNG state, and
continues. Same seed plus same cursor is the same sequence, so resumed
histories match fresh histories.
"""

from __future__ import annotations

import contextlib
import gc
import io
import json
import os
import threading
import time
from typing import TYPE_CHECKING, Any, cast

import torch

from hydra2.contracts.common import ContractError
from hydra2.data.stream_decode import PrefetchGameStream
from hydra2.data.stream_iter import GameStream
from hydra2.data.stream_manifest import build_manifest, manifest_digest
from hydra2.data.stream_read import StreamCursor as DataStreamCursor
from hydra2.training._rc_digest import create_run_layout, run_config_digest
from hydra2.training.loop_batch import (
    summarize_telemetry,
    summarize_update_telemetry,
)
from hydra2.training.loop_state import (
    TrainingLoopConfig,
)
from hydra2.training.loop_train import (
    SupervisedLoop,
)
from hydra2.training.stream_build import _NO_DECAY_NAME_TAGS as _NO_DECAY_NAME_TAGS
from hydra2.training.stream_build import _POLICY_HEAD_PREFIX as _POLICY_HEAD_PREFIX
from hydra2.training.stream_build import _build_model as _build_model
from hydra2.training.stream_build import _build_optimizer as _build_optimizer
from hydra2.training.stream_build import _build_scheduler as _build_scheduler
from hydra2.training.stream_build import _load_prefix_hashes as _load_prefix_hashes
from hydra2.training.stream_build import _manifest_hashes_for_loop as _manifest_hashes_for_loop
from hydra2.training.stream_build import _optimizer_fused_kwargs as _optimizer_fused_kwargs
from hydra2.training.stream_build import _optimizer_param_groups as _optimizer_param_groups
from hydra2.training.stream_build import _prefix_hashes_name as _prefix_hashes_name
from hydra2.training.stream_build import _rng_anchors as _rng_anchors
from hydra2.training.stream_build import _sidecar_cursor as _sidecar_cursor
from hydra2.training.stream_build import _verify_rng_anchors as _verify_rng_anchors
from hydra2.training.stream_checkpoint import _append_new_history as _append_new_history
from hydra2.training.stream_checkpoint import _ckpt_names as _ckpt_names
from hydra2.training.stream_checkpoint import _epoch_seed as _epoch_seed
from hydra2.training.stream_checkpoint import _prune_checkpoints as _prune_checkpoints
from hydra2.training.stream_checkpoint import _run_holdout_eval as _run_holdout_eval
from hydra2.training.stream_checkpoint import (
    _write_streaming_checkpoint as _write_streaming_checkpoint,
)
from hydra2.training.stream_dataset import _StreamDatasetCore as _StreamDatasetCore
from hydra2.training.stream_dataset_buffer import (
    _parse_dataset_buffer_sidecar as _parse_dataset_buffer_sidecar,
)
from hydra2.training.stream_dataset_buffer import _StreamDataset as _StreamDataset
from hydra2.training.stream_dataset_buffer import (
    _StreamDatasetBufferMixin as _StreamDatasetBufferMixin,
)
from hydra2.training.stream_dataset_buffer import _verify_fast_snapshots as _verify_fast_snapshots
from hydra2.training.stream_expand import _ACTION_KINDS_BY_ID as _ACTION_KINDS_BY_ID
from hydra2.training.stream_expand import _BUFFER_COMPACT_ROWS as _BUFFER_COMPACT_ROWS
from hydra2.training.stream_expand import _FEATURE_FOLD_DIM as _FEATURE_FOLD_DIM
from hydra2.training.stream_expand import _MODEL_PARAMETERS as _MODEL_PARAMETERS
from hydra2.training.stream_expand import (
    _PARALLEL_EXPAND_MAX_WORKERS as _PARALLEL_EXPAND_MAX_WORKERS,
)
from hydra2.training.stream_expand import _QUARANTINE_REASON_CLASSES as _QUARANTINE_REASON_CLASSES
from hydra2.training.stream_expand import _REPLAY_BACKENDS as _REPLAY_BACKENDS
from hydra2.training.stream_expand import _SINGLETON_RANK as _SINGLETON_RANK
from hydra2.training.stream_expand import _SINGLETON_WORLD as _SINGLETON_WORLD
from hydra2.training.stream_expand import SPLIT_RATIOS as SPLIT_RATIOS
from hydra2.training.stream_expand import _action_kind_for_id as _action_kind_for_id
from hydra2.training.stream_expand import _expand_game_planes as _expand_game_planes
from hydra2.training.stream_expand import _expand_game_rows as _expand_game_rows
from hydra2.training.stream_expand import _expand_streamed_chunk as _expand_streamed_chunk
from hydra2.training.stream_expand import _history_bucket_of as _history_bucket_of
from hydra2.training.stream_expand import _needs_privileged_labels as _needs_privileged_labels
from hydra2.training.stream_expand import _pool_worker_init as _pool_worker_init
from hydra2.training.stream_expand import _quarantine_class as _quarantine_class
from hydra2.training.stream_expand import _require_replay_backend as _require_replay_backend
from hydra2.training.stream_expand import _row_to_dict as _row_to_dict
from hydra2.training.stream_expand import _sidecar_window_hash as _sidecar_window_hash
from hydra2.training.stream_expand import _slim_row_dicts as _slim_row_dicts
from hydra2.training.stream_expand import _split_ratios as _split_ratios
from hydra2.training.stream_resume import _apply_resume_payload as _apply_resume_payload
from hydra2.training.stream_resume import _backward_pass_autocast_for as _backward_pass_autocast_for
from hydra2.training.stream_resume import _capture_prime_snapshot as _capture_prime_snapshot
from hydra2.training.stream_resume import _GatedOverlapFeed as _GatedOverlapFeed
from hydra2.training.stream_resume import _load_prime_snapshot as _load_prime_snapshot
from hydra2.training.stream_resume import _load_resume_envelope as _load_resume_envelope
from hydra2.training.stream_resume import _open_gated_feed as _open_gated_feed
from hydra2.training.stream_resume import _prime_cache_path as _prime_cache_path
from hydra2.training.stream_resume import _profile_single_update as _profile_single_update
from hydra2.training.stream_resume import _ResumeEnvelope as _ResumeEnvelope
from hydra2.training.stream_resume import _save_prime_snapshot as _save_prime_snapshot
from hydra2.training.stream_resume import _train_segment as _train_segment
from hydra2.training.stream_scan import _scan_corpus as _scan_corpus
from hydra2.training.stream_scan import _scan_corpus_cached as _scan_corpus_cached
from hydra2.training.stream_scan import _scan_corpus_parallel as _scan_corpus_parallel
from hydra2.training.stream_scan import _scan_file_games as _scan_file_games
from hydra2.training.stream_scan import _scan_report as _scan_report
from hydra2.training.stream_scan import _ScanReport as _ScanReport
from hydra2.training.stream_scan import _shared_scan_cache_path as _shared_scan_cache_path

if TYPE_CHECKING:
    from pathlib import Path

    from hydra2.training._rc_sections import ResumePlan, RunConfig

__all__ = ["SPLIT_RATIOS", "run_stream_training"]


def run_stream_training(config: RunConfig, resume: ResumePlan | None = None) -> dict[str, Any]:
    """Train a supervised run from the stream and return the run summary.

    ``resume`` is the read-only plan from :func:`resolve_resume_plan`
    (``None`` for a fresh start). Raises :class:`ContractError` fail-closed
    on wall overlap, empty train splits, config drift, or any checkpoint
    identity mismatch — always before the mismatched state is applied.
    Selection is never called: best-checkpoint promotion stays a manual,
    post-hoc gate.
    """
    if config.run.kind != "supervised":
        raise ContractError(f"stream training supports kind 'supervised', got {config.run.kind!r}")
    if config.data.world_size != _SINGLETON_WORLD:
        raise ContractError(
            f"stream training is single-process v1 (world_size 1), got {config.data.world_size}"
        )
    # Deterministic matmul pins (idempotent process locks): the bf16 path
    # must never inherit a tf32 accident-of-default. These touch
    # float32-matmul only (zero bf16-path effect): "highest" disables tf32
    # for fp32 GEMMs, the cudnn flags mirror it, and benchmark off removes
    # timing-dependent algorithm choice. Set directly (never suppressed:
    # silent numerics misconfiguration is worse than a version error).
    # TORCHINDUCTOR_CACHE_DIR setdefault mirrors tests/conftest.py
    # (explicit env wins; version-keyed so torch/triton upgrades cannot
    # poison the cache).
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    _inductor_ver = "".join(
        c if (c.isalnum() or c in "._-") else "_" for c in str(torch.__version__)
    )
    _inductor_base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        os.path.join(_inductor_base, "hydra2", f"inductor-torch{_inductor_ver}"),
    )
    run_digest = run_config_digest(config)
    if resume is not None:
        # Verify-before-mutate: every identity gate BEFORE layout/stream/loop.
        if resume.run_digest != run_digest:
            raise ContractError(
                f"resume run_digest {resume.run_digest} != run spec {run_digest} (config drift)"
            )
        if resume.cursor.seed != config.seeds.data_seed:
            raise ContractError(
                f"resume cursor seed={resume.cursor.seed} != "
                f"config seeds.data_seed={config.seeds.data_seed}"
            )
        if resume.worker_plan.num_workers != config.data.num_workers:
            raise ContractError("resume worker_plan.num_workers != config data.num_workers")
        if resume.worker_plan.world_size != config.data.world_size:
            raise ContractError("resume worker_plan.world_size != config data.world_size")
        if not resume.checkpoint.is_file():
            raise ContractError(f"resume checkpoint missing: {resume.checkpoint}")

    run_dir = create_run_layout(config)
    _t_launch = time.perf_counter()
    manifest = build_manifest(config.data.root)
    _dt = time.perf_counter() - _t_launch
    print(f"phase: manifest files={len(manifest)} t={_dt:.1f}s", flush=True)
    if len(manifest) == 0:
        raise ContractError(f"stream manifest empty under {config.data.root}")
    stream_digest = manifest_digest(manifest)
    pinned = config.data.dataset_manifest_hash
    if pinned is not None and pinned != stream_digest:
        raise ContractError(
            f"stream manifest {stream_digest} != config dataset_manifest_hash pin {pinned}"
        )
    ratios = _split_ratios(config)

    # Pre-train scan: wall-disjointness + eval ledger + quarantine counts,
    # all before any runtime object exists (content-hash cache; miss → scan).
    _t_scan = time.perf_counter()
    scan = _scan_corpus_cached(
        manifest, config=config, ratios=ratios, stream_digest=stream_digest, run_dir=run_dir
    )
    print(
        f"phase: scan train_games={scan.train_games} elapsed={time.perf_counter() - _t_scan:.1f}s",
        flush=True,
    )
    if scan.train_games == 0:
        raise ContractError(f"train split empty: no games in {config.data.train_split!r}")

    start_update = resume.global_update if resume is not None else 0
    if start_update > config.loop.max_updates:
        raise ContractError(
            f"resume update {start_update} beyond loop.max_updates {config.loop.max_updates}"
        )
    remaining = config.loop.max_updates - start_update
    microbatch = config.loop.microbatch_size

    # Deterministic roots: counter-based seeds only, never wall-clock.
    _ = torch.manual_seed(config.seeds.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seeds.train_seed)
    with contextlib.suppress(Exception):
        import numpy as np

        np.random.seed(config.seeds.train_seed % (2**32))

    envelope: _ResumeEnvelope | None = None
    if resume is not None:
        envelope = _load_resume_envelope(
            resume=resume, run_digest=run_digest, microbatch=microbatch
        )

    need_privileged = _needs_privileged_labels(config)
    # Single seek path (single-pass, epoch pinned 0): fresh runs start at the
    # origin; resumes seek to the recorded frontier with verbatim shuffle
    # buffer + RNG + dedup-prefix restore. No prefix replay, no epoch wrap.
    seek_start: DataStreamCursor | None = None
    seek_entries: list[dict[str, Any]] | None = None
    seek_rng: dict[str, Any] | None = None
    seek_prefix: list[str] | None = None
    seek_blob: Path | str | None = None
    if envelope is not None and resume is not None:
        if not envelope.has_fast_path:
            raise ContractError(
                f"checkpoint lacks seek state (dataset_buffer): {resume.checkpoint} "
                "(pre-simplification sidecars cannot resume; re-run from a fresh id)"
            )
        if envelope.start_epoch != 0:
            raise ContractError(
                f"checkpoint stream epoch {envelope.start_epoch} != 0: {resume.checkpoint} "
                "(single-pass pins epoch 0)"
            )
        if resume.cursor.epoch != 0:
            raise ContractError(
                f"resume cursor epoch {resume.cursor.epoch} != 0: {resume.checkpoint}"
            )
        seek_start = DataStreamCursor(
            file_index=resume.cursor.file_index,
            byte_offset=resume.cursor.byte_offset,
            games_seen=resume.cursor.games_seen,
            seed=resume.cursor.seed,
            epoch=0,
            shuffle_pos=int(envelope.sidecar.get("stream_shuffle_pos", 0)),
        )
        shuffle_raw = envelope.sidecar.get("shuffle")
        if not isinstance(shuffle_raw, dict):
            raise ContractError(f"checkpoint shuffle missing: {resume.checkpoint}")
        prefix_rec = shuffle_raw.get("prefix_hashes")
        if not isinstance(prefix_rec, dict):
            raise ContractError(f"checkpoint shuffle prefix_hashes missing: {resume.checkpoint}")
        seek_prefix = _load_prefix_hashes(
            resume.checkpoint.parent, prefix_rec, ckpt=resume.checkpoint
        )
        if config.data.shuffle_buffer_size > 0:
            entries = shuffle_raw.get("buffer_entries", [])
            rng_state = shuffle_raw.get("buffer_rng_state", {})
            if not isinstance(entries, list) or not isinstance(rng_state, dict):
                raise ContractError(f"checkpoint shuffle restore malformed: {resume.checkpoint}")
            # Keys must mirror entries (tamper → raise, never silent refill).
            keys = shuffle_raw.get("buffer_keys", [])
            if [str(e.get("key")) for e in entries if isinstance(e, dict)] != list(keys):
                raise ContractError(
                    f"checkpoint shuffle keys/entries mismatch: {resume.checkpoint}"
                )
            seek_entries = entries
            seek_rng = rng_state

    # Reservoir snapshot (fresh runs only, env-gated both directions):
    # HYDRA2_RESERVOIR_SNAPSHOT=1 enables load (hit) AND capture (miss);
    # unset (default) skips both — a capture nothing reads back is pure I/O.
    # Hit path stays gated: an un-gated hit restores divergent data instead
    # of the recorded prefix, so resume fails closed on the prefix-hash check.
    _prime_index: Path | None = None
    _prime_hit: dict[str, Any] | None = None
    _prime_hit_enabled = os.environ.get("HYDRA2_RESERVOIR_SNAPSHOT", "").strip() == "1"
    if resume is None and config.data.shuffle_buffer_size > 0 and _prime_hit_enabled:
        _prime_index = _prime_cache_path(
            manifest=manifest,
            stream_digest=stream_digest,
            seed=config.seeds.data_seed,
            ratios=ratios,
            train_split=config.data.train_split,
            val_split=config.data.val_split,
            buffer_size=config.data.shuffle_buffer_size,
        )
        _prime_hit = _load_prime_snapshot(path=_prime_index, stream_digest=stream_digest)
        if _prime_hit is not None:
            seek_entries = _prime_hit["buffer_entries"]
            seek_rng = _prime_hit["buffer_rng_state"]
            seek_blob = _prime_hit["blob_path"]
            seek_start = _prime_hit["stream_cursor"]
            seek_prefix = _prime_hit["prefix_hashes"]

    def _train_stream_factory() -> GameStream:
        common: dict[str, Any] = {
            "seed": config.seeds.data_seed,
            "ratios": ratios,
            "epoch": 0,
            "split": config.data.train_split,
            "shuffle_buffer": config.data.shuffle_buffer_size,
        }
        if config.data.num_workers > 0:
            return PrefetchGameStream(
                manifest,
                **common,
                start=seek_start,
                shuffle_restore_entries=seek_entries,  # type: ignore[arg-type]
                shuffle_restore_rng=seek_rng,  # type: ignore[arg-type]
                shuffle_restore_prefix_hashes=seek_prefix,  # type: ignore[arg-type]
                snapshot_blob=seek_blob,
            )
        return GameStream(
            manifest,
            **common,
            start=seek_start,
            shuffle_restore_entries=seek_entries,  # type: ignore[arg-type]
            shuffle_restore_rng=seek_rng,  # type: ignore[arg-type]
            shuffle_restore_prefix_hashes=seek_prefix,  # type: ignore[arg-type]
            snapshot_blob=seek_blob,
        )

    # Game-pull expansion (both backends): the dataset owns one GameStream and
    # expands per game — Python oracle rows (``replay_backend="python"``) or
    # Rust plane blobs (``replay_backend="rust"``, same order/shuffle/sidecar,
    # tensor-native batch assembly, no per-row Python). Resume, shuffle,
    # privileged labels, and dedup behave identically on both paths.
    dataset = _StreamDataset(
        stream_factory=_train_stream_factory,
        num_actions=config.model.action_count,
        feature_dim=_FEATURE_FOLD_DIM,
        seed=config.seeds.data_seed,
        drop_last=config.data.drop_last,
        need_privileged=need_privileged,
        replay_backend=config.data.replay_backend,
        # Parallel game expansion rides the existing prefetch/parallel path:
        # num_workers > 0 selects PrefetchGameStream above (decode parallelism)
        # and sizes the bounded (16-max) expansion pool.
        expand_workers=config.data.num_workers,
        expand_batch_games=config.data.expand_batch_games,
        homogeneous_buckets=config.data.homogeneous_buckets,
    )
    payload: Any = None
    if resume is not None and envelope is not None:
        try:
            payload = torch.load(io.BytesIO(envelope.blob), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ContractError(f"checkpoint payload unloadable: {resume.checkpoint}") from exc
        _verify_rng_anchors(payload.get("rng_state"), resume.rng_state, ckpt=resume.checkpoint)
        assert seek_start is not None
        # Verbatim tail rebuild (O(buffer) re-expansions), never the epoch.
        _verify_fast_snapshots(sidecar=envelope.sidecar, payload=payload, ckpt=resume.checkpoint)
        parsed_snapshot = _parse_dataset_buffer_sidecar(
            envelope.sidecar.get("dataset_buffer"), ckpt=resume.checkpoint
        )
        if int(parsed_snapshot["epoch"]) != 0:
            raise ContractError(
                f"checkpoint dataset epoch {parsed_snapshot['epoch']} != 0: {resume.checkpoint}"
            )
        if int(parsed_snapshot["microbatches_in_epoch"]) != envelope.drain_microbatches:
            raise ContractError(f"checkpoint microbatch count mismatch: {resume.checkpoint}")
        shuffle_raw = envelope.sidecar.get("shuffle")
        assert isinstance(shuffle_raw, dict)
        if int(shuffle_raw.get("epoch_seed", -1)) != _epoch_seed(
            data_seed=config.seeds.data_seed, epoch=dataset.epoch
        ):
            raise ContractError(f"checkpoint shuffle epoch_seed mismatch: {resume.checkpoint}")
        if int(shuffle_raw.get("buffer_size", -1)) != config.data.shuffle_buffer_size:
            raise ContractError(f"checkpoint shuffle buffer_size mismatch: {resume.checkpoint}")
        dataset.restore_buffer(parsed_snapshot)
        # Seek the live stream to the recorded frontier (no prefix drain).
        dataset._stream = dataset._factory()  # type: ignore[attr-defined]
        dataset._iter = iter(dataset._stream)  # type: ignore[attr-defined]
        drained = dataset.stream_cursor()
        if drained != seek_start:
            raise ContractError(
                f"resume seek landed on {drained!r} but checkpoint records "
                f"{seek_start!r}: {resume.checkpoint}"
            )
    # Eager first fill: warm the shuffle reservoir + first microbatch on a
    # daemon thread while the main thread builds model/optimizer/scheduler/
    # handle/feed/sampler below (fill overlaps build). Single puller (this
    # thread) preserves stream order exactly; join before loop build
    # re-raises fill failures. Deterministic: same rows, same batches.
    _eager_fill_error: list[BaseException] = []
    _eager_need = int(microbatch)

    def _eager_fill() -> None:
        try:
            dataset._fill(_eager_need)
        except BaseException as exc:  # re-raised on join, never swallowed
            _eager_fill_error.append(exc)

    _t_eager = time.perf_counter()
    _eager_thread = threading.Thread(target=_eager_fill, name="hydra2-eager-fill", daemon=True)
    _eager_thread.start()
    _eager_joined = False

    def _join_eager_fill() -> None:
        """Wait for the eager first fill; re-raise its failure (idempotent)."""
        nonlocal _eager_joined
        if _eager_joined:
            return
        _eager_joined = True
        _eager_thread.join()
        if _eager_fill_error:
            raise _eager_fill_error[0]
        print(f"phase: eager-fill joined t={time.perf_counter() - _t_eager:.1f}s", flush=True)

    model = _build_model(config)
    optimizer = _build_optimizer(config, model)
    scheduler = _build_scheduler(config, optimizer)

    from hydra2.runtime.plain import PlainPytorchAdapter
    from hydra2.runtime.protocol import RuntimeSpec, build_runtime, require_device_available

    require_device_available(config.runtime.device)
    if config.runtime.adapter_id == "plain_pytorch":
        adapter: Any = PlainPytorchAdapter()
    else:
        raise ContractError(f"unknown runtime adapter_id {config.runtime.adapter_id!r}")
    # Compiled non-fp32 needs the functorch backward shim (see
    # _backward_pass_autocast_for); eager and fp32 paths keep None, so
    # existing runtime identities are unchanged.
    spec = RuntimeSpec(
        adapter_id=config.runtime.adapter_id,  # type: ignore[arg-type]
        device=config.runtime.device,
        precision=config.loop.precision,  # type: ignore[arg-type]
        compile_mode=config.runtime.compile_mode,  # type: ignore[arg-type]
        backward_pass_autocast=_backward_pass_autocast_for(
            precision=config.loop.precision, compile_mode=config.runtime.compile_mode
        ),
    )
    handle = build_runtime(adapter=adapter, model=model, optimizer=optimizer, spec=spec)
    # bf16 routing: plain_pytorch + bf16_mixed is the launchable bf16 path
    # (loop-owned autocast, CUDA-only, fail-closed otherwise). fabric_2.6.5 +
    # the real actor model stays blocked: Fabric AMP convert_input cannot
    # traverse the frozen ActorTensorBatch (ValueError on the first forward).

    loop_config = TrainingLoopConfig(
        w_policy=config.weights.w_policy,
        w_placement=config.weights.w_placement,
        w_value=config.weights.w_value,
        w_event=dict(config.weights.w_event) if config.weights.w_event is not None else None,
        w_belief=dict(config.weights.w_belief) if config.weights.w_belief is not None else None,
        microbatch_size=microbatch,
        accumulation_steps=config.loop.accumulation_steps,
        gradient_clip_norm=config.loop.gradient_clip_norm,
        max_updates=config.loop.max_updates,
        checkpoint_frequency_updates=config.loop.checkpoint_frequency_updates,
        seed=config.seeds.train_seed,
        precision=config.loop.precision,  # type: ignore[arg-type]
        label_smoothing=config.weights.label_smoothing,
        stratified_sampling=config.loop.stratified_sampling,
        sampling_ratios=dict(config.loop.sampling_ratios)
        if config.loop.sampling_ratios is not None
        else None,
        log_per_type_metrics=config.loop.log_per_type_metrics,
        fit_temperature=config.loop.fit_temperature,
        fetch_prefetch_depth=config.loop.fetch_prefetch_depth,
    )
    from hydra2.tracking.clearml_mirror import make_mirror

    loop_manifest_hashes = _manifest_hashes_for_loop(config, model=handle.model, manifest=manifest)

    mirror = make_mirror(
        enabled=config.mirror.enabled,
        project=config.mirror.project,
        task_name=config.mirror.task_name,
        offline_dir=config.mirror.offline_dir,
        manifest_hashes=loop_manifest_hashes,
        loop_config={
            "microbatch_size": microbatch,
            "accumulation_steps": config.loop.accumulation_steps,
            "max_updates": config.loop.max_updates,
            "checkpoint_frequency_updates": config.loop.checkpoint_frequency_updates,
            "seed": config.seeds.train_seed,
            "w_policy": config.weights.w_policy,
            "w_placement": config.weights.w_placement,
            "w_value": config.weights.w_value,
            "precision": config.loop.precision,
        },
    )
    from hydra2.tracking.mlflow_mirror import make_mirror as make_mlflow_mirror

    # MLflow quiet mirror (default-on; Null under HYDRA2_MLFLOW_DISABLED in
    # tests). Store is anchored to this run's artifact root (runs/<id>/..);
    # the id file lets resume append to the same MLflow run.
    mlflow_store_dir = run_dir.parent.parent / "mirror" / "mlflow"
    mlflow_id_file = run_dir / "mirror" / "mlflow_run_id"
    stored_mlflow_run: str | None = None
    if resume is not None:
        with contextlib.suppress(Exception):
            stored_mlflow_run = mlflow_id_file.read_text(encoding="utf-8").strip() or None
    mlflow_mirror = make_mlflow_mirror(
        tracking_dir=mlflow_store_dir,
        enabled=config.telemetry.mlflow_enabled,
        run_name=config.run.run_id,
        manifest_hashes=loop_manifest_hashes,
        loop_config={
            "microbatch_size": microbatch,
            "accumulation_steps": config.loop.accumulation_steps,
            "max_updates": config.loop.max_updates,
            "checkpoint_frequency_updates": config.loop.checkpoint_frequency_updates,
            "seed": config.seeds.train_seed,
            "w_policy": config.weights.w_policy,
            "w_placement": config.weights.w_placement,
            "w_value": config.weights.w_value,
            "precision": config.loop.precision,
        },
    )
    started_mlflow_run = mlflow_mirror.start_run(run_id=stored_mlflow_run)
    if started_mlflow_run:
        with contextlib.suppress(Exception):
            mlflow_id_file.parent.mkdir(parents=True, exist_ok=True)
            mlflow_id_file.write_text(started_mlflow_run, encoding="utf-8")
    # Gated overlap feed (caller-owned): one depth-3 CUDA PinnedRing over the
    # fixed max-bucket-T schema layout (B=microbatch, A=action_count). Depth 3
    # covers fetch+compute+H2D overlap with one spare slot (~19MB per slot at
    # B2048/T256, so +~20MB vs depth 2). The probe workload buckets every
    # full microbatch at T=256, so the ring shape-matches every batch exactly
    # (byte-identical); off-bucket batches fall back to the sync move inside
    # the feed. None on CPU or when CUDA/pinned is unavailable (sync
    # fallback, CPU-safe); the loop never opens or closes the handle.
    feed = _open_gated_feed(
        microbatch=microbatch, action_count=config.model.action_count, device=handle.device
    )
    # The ring stages the H2D copy from its own pinned slots: encode-side
    # page-locking would only duplicate that work. Sync fallback (feed None)
    # keeps pinning so non_blocking transfers still overlap.
    dataset.pin_memory = feed is None
    loop = SupervisedLoop(
        model=cast("Any", handle.model),
        optimizer=cast("Any", handle.optimizer),
        scheduler=scheduler,
        dataset=dataset,
        config=loop_config,
        checkpoint_dir=run_dir / "checkpoints",
        manifest_hashes=loop_manifest_hashes,
        handle=handle,
        device=cast("Any", handle.device),
        privileged_source=dataset.privileged,
        evaluation_wall_ids=set(scan.val_walls),
        mirror=mirror,
        mlflow_mirror=mlflow_mirror,
        runtime_spec=spec,
        feed=feed,
    )
    if resume is not None:
        # Payload identity fully verified above; now mutate live objects.
        # Join the eager fill first: _apply_resume_payload snapshots dataset
        # state (get_sampler_state) which the fill thread mutates.
        _join_eager_fill()
        _apply_resume_payload(loop=loop, dataset=dataset, payload=payload, ckpt=resume.checkpoint)

    ckpt_every = config.loop.checkpoint_frequency_updates
    eval_every = config.eval.frequency_updates
    done = 0
    evals: list[dict[str, Any]] = []
    # Observer-only feed telemetry sink (run-level; the loop's own file sink
    # stays off because it truncates per train() call, which would drop every
    # checkpoint segment but the last). Fresh runs start empty; resume appends
    # the post-resume segments. Resume and RNG never read this file.
    telemetry_path = run_dir / "logs" / "feed-telemetry.jsonl"
    telemetry_all: list[Any] = []
    telemetry_updates: list[Any] = []
    if resume is None:
        telemetry_path.write_text("", encoding="utf-8")
    # Verbose GPU/CPU sampler (opt-in; default off). Run-scoped rows under
    # logs/; resume appends. Never touches train state or RNG.
    from hydra2.tracking.verbose_sampler import make_verbose_sampler

    _sampled_loop = loop
    sampler = make_verbose_sampler(
        enabled=config.telemetry.verbose_enabled,
        sink_path=run_dir / "logs" / "verbose-telemetry.jsonl",
        interval_ms=config.telemetry.verbose_interval_ms,
        run_id=config.run.run_id,
        run_digest=run_digest,
        counters_fn=lambda: (int(_sampled_loop.state.global_update), 0),
    )
    _ = sampler.start()
    print(f"phase: runtime-ready elapsed={time.perf_counter() - _t_launch:.1f}s", flush=True)
    # Profiler captures: first K checkpoint boundaries past warmup (compile
    # noise), bounded by the run window. Each captures exactly one update.
    profile_updates: set[int] = set()
    for _slot in range(2, 2 + config.telemetry.profiler_captures):
        _boundary = ckpt_every * _slot
        if start_update < _boundary <= start_update + remaining:
            profile_updates.add(_boundary)
    _device = getattr(loop, "device", "cpu")
    _profile_cuda = str(getattr(_device, "type", _device)) == "cuda"
    # GC freeze: move everything allocated so far (model, optimizer, dataset,
    # pools, imports) to the permanent generation so per-update nursery scans
    # skip it. Evidence: ``gc_collections_*`` of ``MicrobatchTelemetry``
    # (loop.py): 64-game decode waves were firing 100+ gen0 collections
    # inside single updates (stop-the-world GIL holds that starve the main
    # thread mid-backward and crater GPU util).
    # Frozen heap is never scanned or freed; only per-update garbage (batches,
    # grads, decoded games) stays collectable. Zero training-math effect.
    # Join the eager fill first: freezing mid-fill would pin in-flight batches.
    _join_eager_fill()
    # Reservoir capture (fresh runs, gated-miss only): prime buffer is fully
    # resident post-join; persist raw bytes + RNG for the next gated run.
    # Best-effort (run continues on write failure); never on resume (the
    # checkpoint machinery owns buffer state there), never after a hit
    # (identical bytes would just rewrite themselves), and never when the
    # snapshot gate is off (nothing would ever read it back: the load above
    # is gated on the same var, so an ungated write is pure I/O cost).
    if resume is None and _prime_hit_enabled and _prime_index is not None and _prime_hit is None:
        _t_capture = time.perf_counter()
        _capture_prime_snapshot(
            dataset=dataset,
            index_path=_prime_index,
            stream_digest=stream_digest,
            buffer_size=config.data.shuffle_buffer_size,
        )
        print(f"phase: reservoir-capture t={time.perf_counter() - _t_capture:.1f}s", flush=True)
    with contextlib.suppress(Exception):  # why-broad: freeze must never fail training
        gc.collect()
        gc.freeze()
    try:
        _t_first = time.perf_counter()
        while done < remaining:
            step = min(ckpt_every, remaining - done)
            done = _train_segment(
                loop=loop,
                done=done,
                step=step,
                captures=profile_updates,
                profiler_dir=run_dir / "profiler",
                train_log=run_dir / "logs" / "train.log",
                cuda_ok=_profile_cuda,
            )
            if done <= ckpt_every:
                _dt = time.perf_counter() - _t_first
                print(f"phase: seg updates={done} t={_dt:.1f}s", flush=True)
            telemetry_all.extend(loop.telemetry_records)
            telemetry_updates.extend(loop.update_records)
            with open(telemetry_path, "a", encoding="utf-8") as sink:
                for record in loop.telemetry_records:
                    _ = sink.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            # intentionally discarded: checkpoint path unneeded, manifest tracks
            _ = _write_streaming_checkpoint(
                run_dir=run_dir,
                update=loop.state.global_update,
                run_digest=run_digest,
                stream_digest=stream_digest,
                config=config,
                loop=loop,
                dataset=dataset,
                batch_size=microbatch,
            )
            # intentionally discarded: appended count unneeded
            _ = _append_new_history(run_dir, loop.loss_history)
            _prune_checkpoints(run_dir, keep=config.loop.keep_last_checkpoints)
            update = loop.state.global_update
            if eval_every > 0 and update % eval_every == 0:
                report = _run_holdout_eval(
                    manifest=manifest,
                    ratios=ratios,
                    config=config,
                    loop=loop,
                    run_dir=run_dir,
                    update=update,
                )
                if report is not None:
                    evals.append(report)
    finally:
        # Training pulled its last row: stop the sampler first (so it never
        # races the summary write), close the quiet mirror, then release
        # expansion workers (no-op serial) and the caller-owned ring.
        with contextlib.suppress(Exception):
            sampler.stop()
        with contextlib.suppress(Exception):
            mlflow_mirror.close()
        dataset.close()
        if feed is not None:
            with contextlib.suppress(Exception):
                feed.close()
    telemetry_summary: dict[str, Any] = {
        "kind": "summary",
        "microbatches": len(telemetry_all),
        **summarize_telemetry(telemetry_all),
        **summarize_update_telemetry(telemetry_updates),
    }
    with open(telemetry_path, "a", encoding="utf-8") as sink:
        _ = sink.write(json.dumps(telemetry_summary, sort_keys=True) + "\n")
    if remaining == 0:
        # intentionally discarded: appended count unneeded
        _ = _append_new_history(run_dir, loop.loss_history)
    checkpoints = _ckpt_names(run_dir)

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_digest": run_digest,
        "stream_manifest_hash": stream_digest,
        "start_update": start_update,
        "end_update": loop.state.global_update,
        "updates_run": loop.state.global_update - start_update,
        "train_games": scan.train_games,
        "val_games": scan.val_games,
        "train_sim_games": scan.train_sim_games,
        "val_sim_games": scan.val_sim_games,
        "quarantined": scan.quarantined,
        "duplicates": scan.duplicates,
        "replayed": dataset.replayed,
        "sim_replayed": dataset.sim_replayed,
        "expand_quarantined": dataset.expand_quarantined,
        "expand_quarantined_by_reason": dict(dataset.expand_quarantine_reasons),
        "privileged_labels": "joined" if need_privileged else "skipped-no-auxiliary-weights",
        "overlap_feed": "pinned-ring-cuda" if feed is not None else "sync-fallback",
        "checkpoints": checkpoints,
        "evals": evals,
        "loss_history": [dict(entry) for entry in loop.loss_history],
        "metrics_path": str(run_dir / "logs" / "metrics.jsonl"),
    }
    return summary
