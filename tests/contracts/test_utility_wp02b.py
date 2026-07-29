"""WP-02B utility gate: identity round trip, seat permutation, malformed rejection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hydra2.artifacts.canonical import canonical_bytes, loads_canonical
from hydra2.contracts.common import (
    ContractError,
    DigestMismatchError,
    RulesMismatchError,
)
from hydra2.contracts.rules import RULES_ID, SOURCE_EVIDENCE_KEY, rules_manifest_from_payload
from hydra2.contracts.utility import (
    UTILITY_OBJECTIVE,
    UTILITY_TIE_POLICY,
    RawOutcome,
    SettlementFact,
    UtilityManifest,
    UtilityVector,
    make_utility_manifest,
    root_scalar,
    utility,
    utility_manifest_digest_document,
    utility_manifest_from_payload,
    utility_manifest_to_payload,
)

pytestmark = pytest.mark.contract_package("WP-02B")

_REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = _REPO_ROOT / "configs" / "rules" / "tenhou_4p_hanchan_v1.json"
#: The published manifest envelope digest is the binding rules hash.
GOLDEN_ENVELOPE_DIGEST = "sha256:3042a493280224f533d831f371275b1c96585cf1db5a2e5fb86ec259f403286b"

RANK_VALUES = (20.0, 10.0, -10.0, -20.0)


@pytest.fixture(scope="module")
def rules_hash() -> str:
    envelope_doc = loads_canonical(CONFIG_PATH.read_bytes())
    assert envelope_doc["payload"]["rules_id"] == RULES_ID
    return GOLDEN_ENVELOPE_DIGEST


@pytest.fixture(scope="module")
def umanifest(rules_hash) -> UtilityManifest:
    return make_utility_manifest(
        utility_id="expected_final_placement_tenhou_4p_hanchan_v1",
        schema_version="1.0.0",
        rules_id=RULES_ID,
        rules_hash=rules_hash,
        objective=UTILITY_OBJECTIVE,
        rank_values=RANK_VALUES,
        tie_policy=UTILITY_TIE_POLICY,
        value_min=-100.0,
        value_max=100.0,
        zero_sum=True,
    )


def make_outcome(**overrides) -> RawOutcome:
    fields = {
        "final_scores": (35200, 32100, 27400, 25300),
        "ranks": (1, 2, 3, 4),
        "point_deltas": (5200, 2100, -2600, -4700),
        "settlements": (),
        "rules_id": RULES_ID,
        "rules_hash": GOLDEN_ENVELOPE_DIGEST,
    }
    fields.update(overrides)
    return RawOutcome(**fields)


class TestRawUtilityIdentityRoundTrip:
    def test_utility_maps_rank_values_through_resolved_ranks(self, umanifest):
        outcome = make_outcome()
        vector = utility(outcome, umanifest)
        expected = tuple(RANK_VALUES[rank - 1] for rank in outcome.ranks)
        assert vector.values == expected
        assert vector.utility_id == umanifest.utility_id
        assert vector.utility_manifest_hash == umanifest.digest
        assert vector.rules_hash == outcome.rules_hash

    def test_root_scalar_selects_the_acting_seat_index(self, umanifest):
        vector = utility(make_outcome(), umanifest)
        for seat in range(4):
            assert root_scalar(vector, seat) == RANK_VALUES[make_outcome().ranks[seat] - 1]

    def test_every_top_rank_holder_gets_the_best_value(self, umanifest):
        for top_seat in range(4):
            others = [seat for seat in range(4) if seat != top_seat]
            ranks = [0] * 4
            ranks[top_seat] = 1
            for position, seat in enumerate(others, start=2):
                ranks[seat] = position
            vector = utility(make_outcome(ranks=tuple(ranks)), umanifest)
            assert root_scalar(vector, top_seat) == RANK_VALUES[0]

    def test_manifest_payload_round_trip_preserves_identity(self, umanifest):
        payload = utility_manifest_to_payload(umanifest)
        rebuilt = utility_manifest_from_payload(payload)
        assert rebuilt == umanifest
        assert rebuilt.digest == umanifest.digest
        # JSON text round trip keeps the digest stable (ES6 number formatting).
        text = json.dumps(json.loads(canonical_bytes(payload)))
        assert utility_manifest_from_payload(json.loads(text)).digest == umanifest.digest

    def test_contract_canonical_writer_matches_artifacts_authority(self, umanifest):
        from hydra2.artifacts.canonical import canonical_bytes as authority_bytes
        from hydra2.contracts.rules import canonical_contract_json_bytes

        document = utility_manifest_digest_document(umanifest)
        assert canonical_contract_json_bytes(document) == authority_bytes(document)


class TestSeatPermutationInvariance:
    def test_renamed_seats_permute_the_utility_vector(self, umanifest):
        perm = (2, 0, 3, 1)  # new seat i takes old seat perm[i]

        def remap(quad):
            return tuple(quad[perm[i]] for i in range(4))

        base_settlement = SettlementFact(
            kind="ron",
            from_seat=1,
            to_seats=(0,),
            point_deltas=(7700, -7700, 0, 0),
            detail={"han": 3},
        )
        base = make_outcome(
            settlements=(base_settlement,),
            final_scores=(32700, 27300, 35000, 25000),
            ranks=(2, 3, 1, 4),
            point_deltas=(2700, -2700, 5000, -5000),
        )
        vector = utility(base, umanifest)

        permuted_settlement = SettlementFact(
            kind=base_settlement.kind,
            from_seat=(
                None if base_settlement.from_seat is None else perm[base_settlement.from_seat]
            ),
            to_seats=tuple(perm[seat] for seat in base_settlement.to_seats),
            point_deltas=remap(base_settlement.point_deltas),
            detail=dict(base_settlement.detail),
        )
        permuted = make_outcome(
            final_scores=remap(base.final_scores),
            ranks=remap(base.ranks),
            point_deltas=remap(base.point_deltas),
            settlements=(permuted_settlement,),
        )
        permuted_vector = utility(permuted, umanifest)

        for new_seat in range(4):
            old_seat = perm[new_seat]
            assert permuted_vector.values[new_seat] == vector.values[old_seat]
            assert root_scalar(permuted_vector, new_seat) == root_scalar(vector, old_seat)
        # Identity metadata is invariant under relabeling.
        assert permuted_vector.utility_manifest_hash == vector.utility_manifest_hash
        assert permuted_vector.rules_hash == vector.rules_hash


class TestMalformedSettlementRejection:
    def test_non_permutation_ranks_rejected_at_construction(self):
        for bad in ((1, 2, 2, 4), (4, 3, 2, 2)):
            with pytest.raises(ContractError, match="permutation"):
                make_outcome(ranks=bad)
        for out_of_range in ((0, 1, 2, 3), (1, 2, 3, 5)):
            with pytest.raises(ContractError, match="ranks"):
                make_outcome(ranks=out_of_range)

    def test_tied_ranks_have_no_utility_even_via_bypass(self, umanifest):
        outcome = RawOutcome.__new__(RawOutcome)
        object.__setattr__(outcome, "final_scores", (30000,) * 4)
        object.__setattr__(outcome, "ranks", (1, 1, 3, 4))  # unresolved tie
        object.__setattr__(outcome, "point_deltas", (0, 0, 0, 0))
        object.__setattr__(outcome, "settlements", ())
        object.__setattr__(outcome, "rules_id", RULES_ID)
        object.__setattr__(outcome, "rules_hash", GOLDEN_ENVELOPE_DIGEST)
        with pytest.raises(ContractError, match=r"tied|permutation"):
            utility(outcome, umanifest)

    @pytest.mark.parametrize("field", ["rules_id", "rules_hash"])
    def test_rules_identity_mismatch_raises_rules_mismatch(self, umanifest, field):
        if field == "rules_id":
            outcome = make_outcome(rules_id="some_other_rules")
        else:
            outcome = make_outcome(rules_hash="sha256:" + "cd" * 32)
        with pytest.raises(RulesMismatchError):
            utility(outcome, umanifest)

    def test_non_finite_and_bound_violating_manifests_rejected(self, rules_hash):
        base = {
            "utility_id": "x",
            "schema_version": "1.0.0",
            "rules_id": RULES_ID,
            "rules_hash": rules_hash,
            "objective": UTILITY_OBJECTIVE,
            "tie_policy": UTILITY_TIE_POLICY,
            "zero_sum": False,
            "value_min": -100.0,
            "value_max": 100.0,
        }
        with pytest.raises(ContractError, match="finite"):
            make_utility_manifest(rank_values=(float("nan"), 0.0, 0.0, 0.0), **base)
        with pytest.raises(ContractError, match="bounds"):
            make_utility_manifest(rank_values=(120.0, -40.0, -40.0, -40.0), **base)
        nonzero_sum_base = {k: v for k, v in base.items() if k != "zero_sum"}
        with pytest.raises(ContractError, match="zero"):
            make_utility_manifest(
                rank_values=(31.0, 0.0, -10.0, -20.0), zero_sum=True, **nonzero_sum_base
            )

    def test_wrong_objective_or_tie_policy_literal_rejected(self, rules_hash):
        kwargs = {
            "utility_id": "x",
            "schema_version": "1.0.0",
            "rules_id": RULES_ID,
            "rules_hash": rules_hash,
            "rank_values": RANK_VALUES,
            "value_min": -100.0,
            "value_max": 100.0,
            "zero_sum": True,
        }
        with pytest.raises(ContractError, match="objective"):
            make_utility_manifest(
                objective="expected_profit", tie_policy=UTILITY_TIE_POLICY, **kwargs
            )
        with pytest.raises(ContractError, match="tie_policy"):
            make_utility_manifest(
                objective=UTILITY_OBJECTIVE, tie_policy="split_ties_evenly", **kwargs
            )

    def test_tampered_recorded_digest_rejected_on_load(self, umanifest):
        payload = utility_manifest_to_payload(umanifest)
        payload["rank_values"] = [21.0, 10.0, -11.0, -20.0]
        with pytest.raises(DigestMismatchError):
            utility_manifest_from_payload(payload)

    def test_extra_manifest_payload_key_rejected(self, umanifest):
        payload = utility_manifest_to_payload(umanifest)
        payload["engine_hint"] = "never trust engine defaults"
        with pytest.raises(ContractError, match="extra"):
            utility_manifest_from_payload(payload)

    def test_settlement_fact_structural_rejections(self):
        deltas = (8000, -8000, 0, 0)
        with pytest.raises(ContractError, match="must not repeat"):
            SettlementFact(kind="ron", from_seat=1, to_seats=(0, 0), point_deltas=deltas, detail={})
        with pytest.raises(ContractError, match="must not appear among to_seats"):
            SettlementFact(kind="ron", from_seat=0, to_seats=(0,), point_deltas=deltas, detail={})
        with pytest.raises(ContractError, match=r"to_seats\[0\]"):
            SettlementFact(
                kind="ron", from_seat=None, to_seats=(4,), point_deltas=deltas, detail={}
            )
        with pytest.raises(ContractError, match="bool"):
            SettlementFact(
                kind="ron", from_seat=1, to_seats=(0,), point_deltas=(True, -8000, 0, 0), detail={}
            )
        with pytest.raises(ContractError, match="non-finite"):
            SettlementFact(
                kind="ron",
                from_seat=1,
                to_seats=(0,),
                point_deltas=deltas,
                detail={"riichi_stick": float("inf")},
            )
        with pytest.raises(ContractError, match="mapping"):
            SettlementFact(
                kind="ron", from_seat=1, to_seats=(0,), point_deltas=deltas, detail="not-a-map"
            )

    def test_utility_vector_and_root_scalar_input_guards(self, umanifest):
        with pytest.raises(ContractError, match="finite"):
            UtilityVector(
                values=(float("nan"), 0.0, 0.0, 0.0),
                utility_id="x",
                utility_manifest_hash=umanifest.digest,
                rules_hash=GOLDEN_ENVELOPE_DIGEST,
            )
        vector = utility(make_outcome(), umanifest)
        with pytest.raises(ContractError, match="seat"):
            root_scalar(vector, 4)
        with pytest.raises(ContractError, match="int"):
            root_scalar(vector, True)  # bool MUST NOT pass integer validation

    def test_rules_manifest_config_feeds_the_utility_binding(self, umanifest):
        """The config on disk is the same rules identity the utility binds to."""
        envelope_doc = loads_canonical(CONFIG_PATH.read_bytes())
        manifest = rules_manifest_from_payload(envelope_doc["payload"])
        assert manifest.rules_id == umanifest.rules_id
        assert envelope_doc["payload"][SOURCE_EVIDENCE_KEY]["fields"]
