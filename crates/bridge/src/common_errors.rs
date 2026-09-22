//! common_errors: SPEC 3 typed failure hierarchy (`contracts/common.py:29-107`).
//!
//! DAG: pyo3 ONLY (no feed/shard/search edge — exception *classes* carry no
//! values; message texts stay Python-side and translators keep raising the
//! `hydra2.contracts.common` names). FORBIDS: message texts, validators,
//! aliases, constructors (all stay Python in `common.py`), and any new
//! submodule (registers on the shared `hydra2._native.contracts` submodule).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + `let py =
//! sub.py()` + `sub.add(...)` on the shared contracts submodule mirror
//! `action_artifact::register` (`crates/bridge/src/action_artifact.rs:42-49`);
//! `m.add("MyError", m.py().get_type::<MyError>())` mirrors the pyo3 0.29.2
//! `create_exception!` docs example (vendored `pyo3-0.29.2/src/exceptions.rs`);
//! `Python::initialize()` + `Python::attach` in tests mirror
//! (`crates/bridge/src/canon_rng.rs:688-689`); `cast` over `downcast` per
//! wave-11.
//!
//! Module-path note: `create_exception!(hydra2.contracts.common, ...)` passes
//! a field-access expression as `$module: expr`; the macro only stringifies
//! it (`PyTypeInfo::MODULE` + the `PyErr::new_type` dotted name), so
//! `__module__` lands on the importable `hydra2.contracts.common` path —
//! never the non-package `hydra2._native.contracts` (submodules are
//! attribute-only: `import hydra2._native.contracts` fails with
//! `ModuleNotFoundError`, only `from hydra2._native import contracts` works,
//! and submodules are absent from `sys.modules`). Pickle `find_class`
//! therefore routes through the Python re-export module and round-trips
//! (xdist workers pickle exceptions).

use pyo3::prelude::*;
use pyo3::types::PyModule;

pyo3::create_exception!(
    hydra2.contracts.common,
    Hydra2Error,
    pyo3::exceptions::PyException,
    "Base class of every expected Hydra2 failure."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    ContractError,
    Hydra2Error,
    "A value violated a canonical contract (range, enum, schema shape)."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    IncompatibleSchemaError,
    ContractError,
    "Unknown or unsupported major schema version."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    CanonicalizationError,
    ContractError,
    "A value cannot be represented canonically (NaN/Inf/nondeterministic map)."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    DigestMismatchError,
    ContractError,
    "A recomputed digest does not match a recorded digest."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    RulesMismatchError,
    ContractError,
    "Rules identity differs between artifacts or runtime expectations."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    InvalidTileError,
    ContractError,
    "A physical/logical tile id is out of range or otherwise invalid."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    InvalidActionError,
    ContractError,
    "An action id is outside the canonical vocabulary or illegal here."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    VisibilityViolationError,
    ContractError,
    "Actor-visible data leaked hidden-world information."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    IllegalActionError,
    ContractError,
    "An action was taken that the legal mask excludes."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    CorruptArtifactError,
    Hydra2Error,
    "Stored bytes do not match their recorded identity."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    LineageError,
    Hydra2Error,
    "Provenance/lineage chain is missing or inconsistent."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    QuarantinedError,
    Hydra2Error,
    "Tried to consume quarantined data."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    UnsupportedRuleError,
    Hydra2Error,
    "Rule configuration is not supported by this build."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    DeterminismError,
    Hydra2Error,
    "A determinism invariant was violated."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    StaleBeliefError,
    Hydra2Error,
    "Belief state does not match the current epoch."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    PacketPartitionError,
    Hydra2Error,
    "Packet partition bookkeeping is inconsistent."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    ProposalSupportError,
    Hydra2Error,
    "Proposal distribution lacks required support."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    DeadlineExceededError,
    Hydra2Error,
    "Search deadline expired (expected control flow at runner boundary only)."
);
pyo3::create_exception!(
    hydra2.contracts.common,
    QualificationRequiredError,
    Hydra2Error,
    "Path requires a qualification token that is absent."
);

/// Register the SPEC 3 hierarchy on the shared `contracts` submodule
/// (mirrors `action_artifact::register`; single cdylib, no new entry point).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add("Hydra2Error", py.get_type::<Hydra2Error>())?;
    sub.add("ContractError", py.get_type::<ContractError>())?;
    sub.add(
        "IncompatibleSchemaError",
        py.get_type::<IncompatibleSchemaError>(),
    )?;
    sub.add(
        "CanonicalizationError",
        py.get_type::<CanonicalizationError>(),
    )?;
    sub.add("DigestMismatchError", py.get_type::<DigestMismatchError>())?;
    sub.add("RulesMismatchError", py.get_type::<RulesMismatchError>())?;
    sub.add("InvalidTileError", py.get_type::<InvalidTileError>())?;
    sub.add("InvalidActionError", py.get_type::<InvalidActionError>())?;
    sub.add(
        "VisibilityViolationError",
        py.get_type::<VisibilityViolationError>(),
    )?;
    sub.add("IllegalActionError", py.get_type::<IllegalActionError>())?;
    sub.add(
        "CorruptArtifactError",
        py.get_type::<CorruptArtifactError>(),
    )?;
    sub.add("LineageError", py.get_type::<LineageError>())?;
    sub.add("QuarantinedError", py.get_type::<QuarantinedError>())?;
    sub.add(
        "UnsupportedRuleError",
        py.get_type::<UnsupportedRuleError>(),
    )?;
    sub.add("DeterminismError", py.get_type::<DeterminismError>())?;
    sub.add("StaleBeliefError", py.get_type::<StaleBeliefError>())?;
    sub.add(
        "PacketPartitionError",
        py.get_type::<PacketPartitionError>(),
    )?;
    sub.add(
        "ProposalSupportError",
        py.get_type::<ProposalSupportError>(),
    )?;
    sub.add(
        "DeadlineExceededError",
        py.get_type::<DeadlineExceededError>(),
    )?;
    sub.add(
        "QualificationRequiredError",
        py.get_type::<QualificationRequiredError>(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyTuple, PyType};

    /// Importable re-export path the Python side aliases onto
    /// (`python/hydra2/contracts/common.py`); pickle `find_class` resolves
    /// exception instances through exactly this module.
    const MODULE_PATH: &str = "hydra2.contracts.common";

    #[test]
    fn module_paths_match_oracle() {
        Python::initialize();
        Python::attach(|py| {
            // (type, __name__, __doc__) — docs verbatim from `common.py:29-107`.
            let cases: Vec<(Bound<'_, PyType>, &str, &str)> = vec![
                (
                    py.get_type::<Hydra2Error>(),
                    "Hydra2Error",
                    "Base class of every expected Hydra2 failure.",
                ),
                (
                    py.get_type::<ContractError>(),
                    "ContractError",
                    "A value violated a canonical contract (range, enum, schema shape).",
                ),
                (
                    py.get_type::<IncompatibleSchemaError>(),
                    "IncompatibleSchemaError",
                    "Unknown or unsupported major schema version.",
                ),
                (
                    py.get_type::<CanonicalizationError>(),
                    "CanonicalizationError",
                    "A value cannot be represented canonically (NaN/Inf/nondeterministic map).",
                ),
                (
                    py.get_type::<DigestMismatchError>(),
                    "DigestMismatchError",
                    "A recomputed digest does not match a recorded digest.",
                ),
                (
                    py.get_type::<RulesMismatchError>(),
                    "RulesMismatchError",
                    "Rules identity differs between artifacts or runtime expectations.",
                ),
                (
                    py.get_type::<InvalidTileError>(),
                    "InvalidTileError",
                    "A physical/logical tile id is out of range or otherwise invalid.",
                ),
                (
                    py.get_type::<InvalidActionError>(),
                    "InvalidActionError",
                    "An action id is outside the canonical vocabulary or illegal here.",
                ),
                (
                    py.get_type::<VisibilityViolationError>(),
                    "VisibilityViolationError",
                    "Actor-visible data leaked hidden-world information.",
                ),
                (
                    py.get_type::<IllegalActionError>(),
                    "IllegalActionError",
                    "An action was taken that the legal mask excludes.",
                ),
                (
                    py.get_type::<CorruptArtifactError>(),
                    "CorruptArtifactError",
                    "Stored bytes do not match their recorded identity.",
                ),
                (
                    py.get_type::<LineageError>(),
                    "LineageError",
                    "Provenance/lineage chain is missing or inconsistent.",
                ),
                (
                    py.get_type::<QuarantinedError>(),
                    "QuarantinedError",
                    "Tried to consume quarantined data.",
                ),
                (
                    py.get_type::<UnsupportedRuleError>(),
                    "UnsupportedRuleError",
                    "Rule configuration is not supported by this build.",
                ),
                (
                    py.get_type::<DeterminismError>(),
                    "DeterminismError",
                    "A determinism invariant was violated.",
                ),
                (
                    py.get_type::<StaleBeliefError>(),
                    "StaleBeliefError",
                    "Belief state does not match the current epoch.",
                ),
                (
                    py.get_type::<PacketPartitionError>(),
                    "PacketPartitionError",
                    "Packet partition bookkeeping is inconsistent.",
                ),
                (
                    py.get_type::<ProposalSupportError>(),
                    "ProposalSupportError",
                    "Proposal distribution lacks required support.",
                ),
                (
                    py.get_type::<DeadlineExceededError>(),
                    "DeadlineExceededError",
                    "Search deadline expired (expected control flow at runner boundary only).",
                ),
                (
                    py.get_type::<QualificationRequiredError>(),
                    "QualificationRequiredError",
                    "Path requires a qualification token that is absent.",
                ),
            ];
            assert_eq!(cases.len(), 20);
            for (ty, name, doc) in cases {
                let module = ty.module().unwrap().to_str().unwrap().to_owned();
                assert_eq!(module, MODULE_PATH);
                let type_name = ty.name().unwrap().to_str().unwrap().to_owned();
                assert_eq!(type_name, name);
                let type_doc: String = ty.into_any().getattr("__doc__").unwrap().extract().unwrap();
                assert_eq!(type_doc, doc);
            }
        });
    }

    #[test]
    fn inheritance_matches_oracle() {
        Python::initialize();
        Python::attach(|py| {
            // The eight ContractError leaves stay under ContractError (HEAD:
            // `class DigestMismatchError(ContractError)` in `common.py:45`).
            for leaf in [
                py.get_type::<IncompatibleSchemaError>(),
                py.get_type::<CanonicalizationError>(),
                py.get_type::<DigestMismatchError>(),
                py.get_type::<RulesMismatchError>(),
                py.get_type::<InvalidTileError>(),
                py.get_type::<InvalidActionError>(),
                py.get_type::<VisibilityViolationError>(),
                py.get_type::<IllegalActionError>(),
            ] {
                assert!(leaf.is_subclass_of::<ContractError>().unwrap());
                assert!(leaf.is_subclass_of::<Hydra2Error>().unwrap());
            }
            // The ten direct Hydra2Error leaves stay out from under
            // ContractError (HEAD: `class CorruptArtifactError(Hydra2Error)`).
            for leaf in [
                py.get_type::<CorruptArtifactError>(),
                py.get_type::<LineageError>(),
                py.get_type::<QuarantinedError>(),
                py.get_type::<UnsupportedRuleError>(),
                py.get_type::<DeterminismError>(),
                py.get_type::<StaleBeliefError>(),
                py.get_type::<PacketPartitionError>(),
                py.get_type::<ProposalSupportError>(),
                py.get_type::<DeadlineExceededError>(),
                py.get_type::<QualificationRequiredError>(),
            ] {
                assert!(!leaf.is_subclass_of::<ContractError>().unwrap());
                assert!(leaf.is_subclass_of::<Hydra2Error>().unwrap());
            }
            assert!(
                py.get_type::<ContractError>()
                    .is_subclass_of::<Hydra2Error>()
                    .unwrap()
            );
            assert!(
                py.get_type::<Hydra2Error>()
                    .is_subclass_of::<pyo3::exceptions::PyException>()
                    .unwrap()
            );
            // Sibling exclusion: a DigestMismatchError is NOT a
            // CanonicalizationError (HEAD probe: `issubclass` False).
            assert!(
                !py.get_type::<DigestMismatchError>()
                    .is_subclass_of::<CanonicalizationError>()
                    .unwrap()
            );
            // MRO byte-shape of a representative leaf (HEAD probe:
            // DigestMismatchError -> ContractError -> Hydra2Error ->
            // Exception -> BaseException -> object).
            let mro = py.get_type::<DigestMismatchError>().mro();
            let names: Vec<String> = mro
                .iter()
                .map(|entry| {
                    entry
                        .getattr("__name__")
                        .unwrap()
                        .extract::<String>()
                        .unwrap()
                })
                .collect();
            assert_eq!(
                names,
                [
                    "DigestMismatchError",
                    "ContractError",
                    "Hydra2Error",
                    "Exception",
                    "BaseException",
                    "object"
                ]
            );
        });
    }

    #[test]
    fn str_and_reduce_match_oracle() {
        Python::initialize();
        Python::attach(|py| {
            // HEAD: `str(DigestMismatchError("hello x")) == "hello x"`.
            let inst = py
                .get_type::<DigestMismatchError>()
                .into_any()
                .call1(("hello x",))
                .unwrap();
            let text: String = inst.str().unwrap().extract().unwrap();
            assert_eq!(text, "hello x");
            // `__reduce__` is the pickle-critical contract: `(cls, args)`
            // with `cls` identical to the type and `args` round-tripping the
            // message, so `find_class("hydra2.contracts.common", ...)` on
            // loads resolves to the same object the Python side aliases.
            let reduced_any = inst.call_method0("__reduce__").unwrap();
            let reduced = reduced_any.cast::<PyTuple>().unwrap();
            let cls = reduced.get_item(0).unwrap();
            let cls_module: String = cls.getattr("__module__").unwrap().extract().unwrap();
            assert_eq!(cls_module, MODULE_PATH);
            let cls_name: String = cls.getattr("__name__").unwrap().extract().unwrap();
            assert_eq!(cls_name, "DigestMismatchError");
            assert!(std::ptr::eq(
                cls.as_ptr(),
                py.get_type::<DigestMismatchError>().as_ptr()
            ));
            let args_any = reduced.get_item(1).unwrap();
            let args = args_any.cast::<PyTuple>().unwrap();
            assert_eq!(args.len(), 1);
            let arg0: String = args.get_item(0).unwrap().extract().unwrap();
            assert_eq!(arg0, "hello x");
            // Multi-arg construction keeps the full args tuple (HEAD:
            // `DigestMismatchError("a", "b").args == ("a", "b")`).
            let multi = py
                .get_type::<DigestMismatchError>()
                .into_any()
                .call1(("a", "b"))
                .unwrap();
            let multi_reduced_any = multi.call_method0("__reduce__").unwrap();
            let multi_reduced = multi_reduced_any.cast::<PyTuple>().unwrap();
            let multi_args_any = multi_reduced.get_item(1).unwrap();
            let multi_args_tuple = multi_args_any.cast::<PyTuple>().unwrap();
            let multi_args: Vec<String> = multi_args_tuple
                .iter()
                .map(|entry| entry.extract::<String>().unwrap())
                .collect();
            assert_eq!(multi_args, ["a", "b"]);
        });
    }

    #[test]
    fn except_matching_matches_oracle() {
        Python::initialize();
        Python::attach(|py| {
            // `except ContractError` catches a DigestMismatchError; `except`
            // on the sibling CanonicalizationError does not (HEAD probe).
            let leaf = DigestMismatchError::new_err("boom");
            assert!(leaf.matches(py, py.get_type::<ContractError>()).unwrap());
            assert!(leaf.matches(py, py.get_type::<Hydra2Error>()).unwrap());
            assert!(
                !leaf
                    .matches(py, py.get_type::<CanonicalizationError>())
                    .unwrap()
            );
            // A direct Hydra2Error leaf is NOT caught by
            // `except ContractError` but IS caught by `except Hydra2Error`.
            let direct = StaleBeliefError::new_err("stale!");
            assert!(!direct.matches(py, py.get_type::<ContractError>()).unwrap());
            assert!(direct.matches(py, py.get_type::<Hydra2Error>()).unwrap());
        });
    }
}
