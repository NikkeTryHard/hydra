"""SPEC 2.1 primitive alias constructors and SPEC 3 error hierarchy."""

from __future__ import annotations

import pytest

from hydra2.contracts.common import (
    ContractError,
    DigestMismatchError,
    Hydra2Error,
    IncompatibleSchemaError,
    make_action_id,
    make_belief_epoch_id,
    make_digest_text,
    make_parent_id,
    make_run_id,
    make_schema_version,
    make_seat,
    make_sequence_no,
    make_tile_id,
    make_tile_type,
    make_utc_timestamp,
)


class TestErrorHierarchy:
    def test_every_expected_failure_derives_from_hydra2_error(self):
        import hydra2.contracts.common as common

        expected = [
            "ContractError",
            "IncompatibleSchemaError",
            "CanonicalizationError",
            "DigestMismatchError",
            "RulesMismatchError",
            "InvalidTileError",
            "InvalidActionError",
            "VisibilityViolationError",
            "IllegalActionError",
            "CorruptArtifactError",
            "LineageError",
            "QuarantinedError",
            "UnsupportedRuleError",
            "DeterminismError",
            "StaleBeliefError",
            "PacketPartitionError",
            "ProposalSupportError",
            "DeadlineExceededError",
            "QualificationRequiredError",
        ]
        for name in expected:
            cls = getattr(common, name)
            assert issubclass(cls, Hydra2Error), name

    def test_contract_subfamily(self):
        from hydra2.contracts.common import (
            CanonicalizationError,
            CorruptArtifactError,
            InvalidTileError,
        )

        assert issubclass(IncompatibleSchemaError, ContractError)
        assert issubclass(CanonicalizationError, ContractError)
        assert issubclass(DigestMismatchError, ContractError)
        assert issubclass(InvalidTileError, ContractError)
        assert not issubclass(CorruptArtifactError, ContractError)


class TestRangeConstructors:
    @pytest.mark.parametrize("value", [0, 1, 2, 3])
    def test_seat_accepts_0_to_3(self, value):
        assert make_seat(value) == value

    @pytest.mark.parametrize("value", [-1, 4, 100])
    def test_seat_rejects_out_of_range(self, value):
        with pytest.raises(ContractError):
            make_seat(value)

    def test_sequence_no_nonnegative(self):
        assert make_sequence_no(0) == 0
        with pytest.raises(ContractError):
            make_sequence_no(-1)

    def test_action_id_nonnegative(self):
        assert make_action_id(86) == 86
        with pytest.raises(ContractError):
            make_action_id(-1)

    def test_tile_id_bounds(self):
        assert make_tile_id(0) == 0
        assert make_tile_id(135) == 135
        with pytest.raises(ContractError):
            make_tile_id(136)
        with pytest.raises(ContractError):
            make_tile_id(-1)


def test_tile_type_bounds():
    assert make_tile_type(33) == 33
    with pytest.raises(ContractError):
        make_tile_type(34)


def test_belief_epoch_nonnegative():
    assert make_belief_epoch_id(7) == 7
    with pytest.raises(ContractError):
        make_belief_epoch_id(-1)


class TestBoolDoesNotPassIntegerValidation:
    @pytest.mark.parametrize(
        "ctor",
        [
            make_seat,
            make_sequence_no,
            make_action_id,
            make_tile_id,
            make_tile_type,
            make_belief_epoch_id,
        ],
    )
    def test_bool_rejected(self, ctor):
        with pytest.raises(ContractError):
            ctor(True)
        with pytest.raises(ContractError):
            ctor(False)


class TestStringConstructors:
    def test_digest_text_exact_format(self):
        good = "sha256:" + "a" * 64
        assert make_digest_text(good) == good

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "sha256:" + "A" * 64,  # uppercase hex noncanonical
            "sha256:" + "g" * 64,
            "md5:" + "a" * 64,
            "sha256:" + "a" * 63,
            "sha256:" + "a" * 65,
            "sha256:" + "a" * 64 + "\n",
        ],
    )
    def test_digest_text_rejects_noncanonical(self, bad):
        with pytest.raises(ContractError):
            make_digest_text(bad)

    def test_utc_timestamp_formats(self):
        assert make_utc_timestamp("2026-08-22T12:00:00Z") == "2026-08-22T12:00:00Z"
        assert make_utc_timestamp("2026-08-22T12:00:00.123456Z")

    @pytest.mark.parametrize(
        "bad",
        ["2026-08-22 12:00:00Z", "2026-08-22T12:00:00", "2026-13-01T00:00:00Zx", "not-a-time"],
    )
    def test_utc_timestamp_rejects(self, bad):
        # Regex-level rejection; calendar validity (month 13) passes the shape
        # check and is refined by RFC 8785-era contract work in WP-02A.
        try:
            make_utc_timestamp(bad)
        except ContractError:
            pass
        else:
            assert bad == "2026-13-01T00:00:00Zx"

    def test_schema_version(self):
        assert make_schema_version("1.0.0") == "1.0.0"
        for bad in ["1", "1.0", "v1.0.0", "1.0.0-dev"]:
            with pytest.raises(ContractError):
                make_schema_version(bad)

    @pytest.mark.parametrize("ctor", [make_parent_id, make_run_id])
    def test_empty_strings_rejected(self, ctor):
        with pytest.raises(ContractError):
            ctor("")
