"""WP-02C Action Contract gate: SPEC 6 kinds, invariants, table, codec.

Fixture exercised (IMPLEMENTATION_SPEC section 22):
- ACT-ROUNDTRIP-001: every action kind round trips with red identity preserved.
Golden anchors: the committed ``configs/contracts/action_table_v1.json`` bytes
are engine-independent identity for all downstream adapters and models.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.action import (
    ACTION_KIND_ORDINALS,
    ACTION_KINDS,
    ACTION_PHASES,
    ACTION_TABLE_ARTIFACT_TYPE,
    ACTION_TABLE_SCHEMA_VERSION,
    PHASES,
    ActionContext,
    CanonicalAction,
    CanonicalActionTemplate,
    VisibleMeld,
    action_table_envelope,
    build_action_table,
    canonical_json_bytes,
    generate_action_templates,
    load_action_table,
    template_sort_key,
    visible_meld_id,
)
from hydra2.contracts.action import (
    canonical_action_codec as codec,
)
from hydra2.contracts.common import (
    ContractError,
    DigestMismatchError,
    IncompatibleSchemaError,
    InvalidActionError,
    make_seat,
    make_tile_id,
)

pytestmark = pytest.mark.contract_package("WP-02C")

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_TABLE_PATH = REPO_ROOT / "configs" / "contracts" / "action_table_v1.json"

#: Engine-independent golden identity of action_table_v1 (regenerate-stable).
GOLDEN_TABLE_DIGEST = "sha256:7b55693428384713f6a6ab7292f57657259c2ca9f05139944d4c8c6197ae76e8"
GOLDEN_ARTIFACT_SHA256 = "4a66338f6e5d9bfd8432dbde75f7941acfcaa308358740c55e9ec23649e42214"

#: Analytic census per kind; generation order is ordinal-sorted.
EXPECTED_COUNTS = {
    "pass": 4,
    "discard": 136,
    "tsumogiri": 136,
    "riichi_discard": 136,
    "chi": 4032,
    "pon": 1224,
    "daiminkan": 408,
    "ankan": 34,
    "kakan": 136,
    "ron": 408,
    "tsumo": 136,
    "abort_nine_terminals": 1,
    "accept_abortive_draw": 1,
}

S0, S1, S2, S3 = make_seat(0), make_seat(1), make_seat(2), make_seat(3)


def mk(raw: int):
    return make_tile_id(raw)


@pytest.fixture(scope="module")
def table():
    return load_action_table(ACTION_TABLE_PATH)


@pytest.fixture(scope="module")
def regenerated():
    return build_action_table(generate_action_templates())


def action(
    kind, actor=S0, *, tile=None, called=None, consumed=(), source=None, declares=False, metadata=()
):
    return CanonicalAction(
        kind=kind,
        actor=actor,
        tile=tile,
        called_tile=called,
        consumed_tiles=tuple(consumed),
        source_seat=source,
        declares_riichi=declares,
        metadata=tuple(metadata),
    )


def context(
    table_obj, phase, *, actor=S0, offered=None, offered_by=None, hand=(), melds=(), table_hash=None
):
    return ActionContext(
        actor=actor,
        action_table_hash=table_hash or table_obj.digest,
        phase=phase,
        offered_tile=offered,
        offered_by=offered_by,
        own_concealed_tiles=tuple(sorted(hand)),
        visible_melds=tuple(melds),
    )


def round_trip(action_obj, table_obj, ctx) -> int:
    index = codec.encode(action_obj, table=table_obj, context=ctx)
    decoded = codec.decode(index, table=table_obj, context=ctx)
    assert decoded == action_obj
    assert codec.encode(decoded, table=table_obj, context=ctx) == index
    return index


# ---------------------------------------------------------------------------
# Kind ordinals (SPEC 6.1).
# ---------------------------------------------------------------------------


class TestKindOrdinalsFrozen:
    def test_ordinals_match_spec_exactly(self):
        assert ACTION_KIND_ORDINALS == {
            "pass": 0,
            "discard": 1,
            "tsumogiri": 2,
            "riichi_discard": 3,
            "chi": 4,
            "pon": 5,
            "daiminkan": 6,
            "ankan": 7,
            "kakan": 8,
            "ron": 9,
            "tsumo": 10,
            "abort_nine_terminals": 11,
            "accept_abortive_draw": 12,
        }

    def test_kind_tuple_follows_ordinal_order(self):
        ordered = tuple(sorted(ACTION_KIND_ORDINALS, key=ACTION_KIND_ORDINALS.get))
        assert ordered == ACTION_KINDS
        assert len(ACTION_KINDS) == 13

    def test_every_kind_declares_phase_gate(self):
        for kind in ACTION_KINDS:
            gate = ACTION_PHASES[kind]
            assert gate, f"missing phase gate for {kind}"
            assert gate <= set(PHASES), kind


# ---------------------------------------------------------------------------
# SPEC 6.2 invariants on CanonicalAction.
# ---------------------------------------------------------------------------


class TestCanonicalActionInvariants:
    def test_each_kind_constructs_when_invariants_hold(self):
        meld = VisibleMeld(None, "pon", S0, S3, mk(110), (mk(108), mk(109), mk(110)))
        ok = [
            action("pass"),
            action("pass", source=S2),
            action("ron", tile=mk(52), source=S2),
            action("discard", tile=mk(16)),
            action("tsumogiri", tile=mk(17)),
            action("riichi_discard", tile=mk(36), declares=True),
            action("chi", called=mk(24), consumed=(mk(16), mk(20)), source=S3),
            action("pon", called=mk(18), consumed=(mk(16), mk(17)), source=S2),
            action("daiminkan", called=mk(111), consumed=(mk(108), mk(109), mk(110)), source=S1),
            action("ankan", consumed=(mk(16), mk(17), mk(18), mk(19))),
            action(
                "kakan",
                tile=mk(111),
                metadata=(("prior_pon_meld_id", visible_meld_id(meld)),),
            ),
            action("tsumo", tile=mk(36)),
            action("abort_nine_terminals"),
            action("accept_abortive_draw"),
        ]
        kinds = {item.kind for item in ok}
        assert kinds == set(ACTION_KIND_ORDINALS)

    def test_declares_riichi_only_on_riichi_discard(self):
        with pytest.raises(ContractError):
            action("discard", tile=mk(16), declares=True)
        with pytest.raises(ContractError):
            action("riichi_discard", tile=mk(16), declares=False)

    def test_chi_requires_previous_seat_and_run(self):
        action("chi", called=mk(24), consumed=(mk(16), mk(20)), source=S3)  # 5m-6m-7m
        with pytest.raises(ContractError):  # source must be the previous seat
            action("chi", called=mk(24), consumed=(mk(16), mk(20)), source=S2)
        with pytest.raises(ContractError):  # honor called tile forbidden
            action("chi", called=mk(108), consumed=(mk(76), mk(80)), source=S3)
        with pytest.raises(ContractError):  # gap: 5m+8m around 7m is not a run
            action("chi", called=mk(28), consumed=(mk(16), mk(24)), source=S3)
        with pytest.raises(ContractError):  # duplicate logical type in consumed pair
            action("chi", called=mk(20), consumed=(mk(16), mk(17)), source=S3)

    def test_pon_daiminkan_same_type_distinct_source(self):
        action("pon", called=mk(18), consumed=(mk(16), mk(17)), source=S2)
        with pytest.raises(ContractError):  # consumes the called physical tile itself
            action("pon", called=mk(18), consumed=(mk(18), mk(17)), source=S2)
        with pytest.raises(ContractError):  # mixed logical types
            action("pon", called=mk(18), consumed=(mk(16), mk(52)), source=S2)
        with pytest.raises(ContractError):  # source equals actor
            action("pon", called=mk(18), consumed=(mk(16), mk(17)), source=S0)
        with pytest.raises(ContractError):  # daiminkan short one tile
            action("daiminkan", called=mk(111), consumed=(mk(108), mk(109)), source=S1)

    def test_ankan_needs_all_four_copies_of_one_type(self):
        action("ankan", consumed=(mk(16), mk(17), mk(18), mk(19)))
        with pytest.raises(ContractError):
            action("ankan", consumed=(mk(16), mk(17), mk(18)))
        with pytest.raises(ContractError):  # 3x5m + 6m
            action("ankan", consumed=(mk(16), mk(17), mk(18), mk(20)))
        with pytest.raises(ContractError):  # unsorted physical ids rejected
            action("ankan", consumed=(mk(17), mk(16), mk(18), mk(19)))

    def test_kakan_metadata_schema(self):
        meld = VisibleMeld(None, "pon", S0, S3, mk(110), (mk(108), mk(109), mk(110)))
        good = (("prior_pon_meld_id", visible_meld_id(meld)),)
        action("kakan", tile=mk(111), metadata=good)
        with pytest.raises(ContractError):  # missing reference
            action("kakan", tile=mk(111))
        with pytest.raises(ContractError):  # undeclared key
            action("kakan", tile=mk(111), metadata=(("meld", "x"),))
        with pytest.raises(ContractError):  # extension metadata on other kinds
            action("discard", tile=mk(16), metadata=(("note", 1),))

    def test_unique_physical_tiles_and_sorted_metadata(self):
        with pytest.raises(ContractError):  # duplicate physical tile inside chi
            action("chi", called=mk(20), consumed=(mk(16), mk(16)), source=S3)
        dup_meta = (("prior_pon_meld_id", "b"), ("prior_pon_meld_id", "a"))
        with pytest.raises(ContractError):  # duplicate keys
            action("kakan", tile=mk(111), metadata=dup_meta)

    def test_parameterless_and_exact_shape_rules(self):
        with pytest.raises(InvalidActionError):
            action("pass", tile=mk(16))
        with pytest.raises(InvalidActionError):
            action("ron", tile=None, source=S2)
        with pytest.raises(InvalidActionError):
            action("tsumo", tile=mk(16), source=S1)
        with pytest.raises(InvalidActionError):
            action("abort_nine_terminals", tile=mk(16))
        with pytest.raises(InvalidActionError):
            action("accept_abortive_draw", consumed=(mk(16),))


# ---------------------------------------------------------------------------
# Red-five physical identity (ACT-ROUNDTRIP-001).
# ---------------------------------------------------------------------------


class TestRedFiveIdentityPreserved:
    HAND = (mk(16), mk(17), mk(18), mk(19), mk(36), mk(37))

    def test_discard_ids_distinguish_red_from_normal_fives(self, table):
        ctx = context(table, "draw_decision", hand=self.HAND)
        red = round_trip(action("discard", tile=mk(16)), table, ctx)
        normal_a = round_trip(action("discard", tile=mk(17)), table, ctx)
        normal_b = round_trip(action("discard", tile=mk(18)), table, ctx)
        assert len({red, normal_a, normal_b}) == 3

    def test_tsumogiri_preserves_red_physical_id(self, table):
        ctx = context(table, "draw_decision", hand=[*self.HAND, mk(88)])
        index = round_trip(action("tsumogiri", tile=mk(88)), table, ctx)
        assert codec.decode(index, table=table, context=ctx).tile == mk(88)

    def test_chi_red_consumption_changes_identity(self, table):
        offered = mk(24)  # 7m copy0; run 5m-6m-7m
        red_ctx = context(
            table, "discard_response", offered=offered, offered_by=S3, hand=[mk(16), mk(20)]
        )
        norm_ctx = context(
            table, "discard_response", offered=offered, offered_by=S3, hand=[mk(17), mk(20)]
        )
        red = round_trip(
            action("chi", called=offered, consumed=(mk(16), mk(20)), source=S3), table, red_ctx
        )
        norm = round_trip(
            action("chi", called=offered, consumed=(mk(17), mk(20)), source=S3), table, norm_ctx
        )
        assert red != norm

    def test_pon_called_red_vs_normal_distinct(self, table):
        offered = mk(18)
        red_ctx = context(
            table, "discard_response", offered=offered, offered_by=S2, hand=[mk(16), mk(19)]
        )
        norm_ctx = context(
            table, "discard_response", offered=offered, offered_by=S2, hand=[mk(17), mk(19)]
        )
        red = round_trip(
            action("pon", called=offered, consumed=(mk(16), mk(19)), source=S2), table, red_ctx
        )
        norm = round_trip(
            action("pon", called=offered, consumed=(mk(17), mk(19)), source=S2), table, norm_ctx
        )
        assert red != norm

    def test_ankan_of_five_man_includes_red_copy(self, table):
        ctx = context(
            table,
            "draw_decision",
            hand=[*self.HAND, mk(38), mk(39), mk(100), mk(101), mk(102), mk(103)],
        )
        five_man = round_trip(
            action("ankan", consumed=(mk(16), mk(17), mk(18), mk(19))), table, ctx
        )
        other = round_trip(
            action("ankan", consumed=(mk(100), mk(101), mk(102), mk(103))), table, ctx
        )
        assert five_man != other
        decoded = codec.decode(five_man, table=table, context=ctx)
        assert decoded.consumed_tiles[0] == mk(16)

    def test_tsumo_winning_red_tile_survives_round_trip(self, table):
        ctx = context(table, "draw_decision", hand=[*self.HAND, mk(52)])
        index = round_trip(action("tsumo", tile=mk(52)), table, ctx)
        assert codec.decode(index, table=table, context=ctx).tile == mk(52)


# ---------------------------------------------------------------------------
# Table generation order and ID stability (golden bytes).
# ---------------------------------------------------------------------------


class TestGenerationOrderStability:
    def test_regeneration_is_identical(self, regenerated, table):
        again = build_action_table(generate_action_templates())
        assert again.actions == regenerated.actions
        assert again.digest == regenerated.digest == GOLDEN_TABLE_DIGEST
        assert regenerated.actions == table.actions
        assert regenerated.digest == table.digest

    def test_strict_order_without_duplicates(self, regenerated):
        keys = [template_sort_key(t) for t in regenerated.actions]
        assert keys == sorted(keys)
        assert len(set(keys)) == len(keys) == 6792

    def test_census_per_kind(self, regenerated):
        counts = dict.fromkeys(ACTION_KINDS, 0)
        for tpl in regenerated.actions:
            counts[tpl.kind] += 1
        assert counts == EXPECTED_COUNTS

    def test_kind_base_indices_are_analytic(self, regenerated):
        bases = {}
        cursor = 0
        for kind in ACTION_KINDS:
            bases[kind] = cursor
            cursor += EXPECTED_COUNTS[kind]
        seen = {}
        for index, tpl in enumerate(regenerated.actions):
            seen.setdefault(tpl.kind, index)
        assert seen == bases
        assert bases["pass"] == 0
        assert bases["discard"] == 4
        assert bases["chi"] == 412
        assert bases["accept_abortive_draw"] == 6791

    def test_none_first_offset_ordering_inside_pass_block(self, regenerated):
        offsets = [tpl.source_offset for tpl in regenerated.actions[:4]]
        assert offsets == [None, -1, 1, 2]

    def test_committed_bytes_equal_regenerated_canonical_bytes(self, regenerated, table):
        doc = action_table_envelope(regenerated)
        assert canonical_json_bytes(doc) == ACTION_TABLE_PATH.read_bytes()
        assert table.digest == GOLDEN_TABLE_DIGEST
        raw_sha = hashlib.sha256(ACTION_TABLE_PATH.read_bytes()).hexdigest()
        assert raw_sha == GOLDEN_ARTIFACT_SHA256


# ---------------------------------------------------------------------------
# Golden artifact loading and corruption rejection.
# ---------------------------------------------------------------------------


class TestGoldenArtifactEngineIndependent:
    def test_envelope_header(self, table):
        doc = json.loads(ACTION_TABLE_PATH.read_bytes())
        assert doc["artifact_type"] == ACTION_TABLE_ARTIFACT_TYPE
        assert doc["schema_version"] == ACTION_TABLE_SCHEMA_VERSION == "1.0.0"
        assert doc["compatibility"] == "exact"
        assert doc["payload"]["digest"] == GOLDEN_TABLE_DIGEST
        assert len(doc["payload"]["actions"]) == 6792

    def test_local_jcs_matches_wp02a_authority(self, regenerated):
        doc = action_table_envelope(regenerated)
        assert canonical_json_bytes(doc) == canonical_bytes(doc)

    def test_digest_mismatch_rejected(self, tmp_path):
        doc = json.loads(ACTION_TABLE_PATH.read_bytes())
        # Tamper with a structurally valid discard template (index 4) so the
        # loader fails on digest recomputation, not on template parsing.
        assert doc["payload"]["actions"][4]["kind"] == "discard"
        doc["payload"]["actions"][4]["tile"] = 135
        tampered = tmp_path / "tampered.json"
        tampered.write_bytes(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode())
        with pytest.raises(DigestMismatchError):
            load_action_table(tampered)

    def test_truncated_bytes_rejected(self, tmp_path):
        raw = ACTION_TABLE_PATH.read_bytes()
        truncated = tmp_path / "cut.json"
        truncated.write_bytes(raw[: len(raw) // 2])
        with pytest.raises(ContractError):
            load_action_table(truncated)

    def test_unknown_major_version_rejected(self, tmp_path):
        doc = json.loads(ACTION_TABLE_PATH.read_bytes())
        doc["schema_version"] = "2.0.0"
        doc["payload"]["schema_version"] = "2.0.0"
        path = tmp_path / "v2.json"
        path.write_bytes(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode())
        with pytest.raises(IncompatibleSchemaError):
            load_action_table(path)

    def test_duplicate_json_key_rejected(self, tmp_path):
        raw_text = ACTION_TABLE_PATH.read_text()
        # JCS orders object keys; injecting a second "kind" ahead of the real one
        # yields a document stdlib json accepts but the I-JSON loader must reject.
        poisoned = raw_text.replace('{"called_tile"', '{"called_tile":{"x":1},', 1)
        path = tmp_path / "dup.json"
        path.write_text(poisoned, encoding="utf-8")
        with pytest.raises((ContractError, json.JSONDecodeError)):
            load_action_table(path)


# ---------------------------------------------------------------------------
# Codec bijection over sampled legal contexts (mask/action alignment).
# ---------------------------------------------------------------------------

#: Fresh copy of the phase gates so the mask predicate does not lean on the
#: codec's own constant table beyond shared contract data.
_PHASE_GATE_INDEPENDENT = {
    "pass": {"discard_response", "kan_response"},
    "discard": {"draw_decision"},
    "tsumogiri": {"draw_decision"},
    "riichi_discard": {"draw_decision"},
    "chi": {"discard_response"},
    "pon": {"discard_response"},
    "daiminkan": {"discard_response"},
    "ankan": {"draw_decision"},
    "kakan": {"draw_decision"},
    "ron": {"discard_response", "kan_response"},
    "tsumo": {"draw_decision"},
    "abort_nine_terminals": {"draw_decision"},
    "accept_abortive_draw": {"discard_response", "kan_response"},
}
_OFFSET_DELTA_INDEPENDENT = {-1: 3, 0: 0, 1: 1, 2: 2}


def _independent_legal(index: int, table_obj, ctx: ActionContext) -> bool:
    """Fresh mask predicate written from SPEC text; guards codec circularity."""
    tpl = table_obj.actions[index]
    if ctx.phase not in _PHASE_GATE_INDEPENDENT[tpl.kind]:
        return False
    src = None
    if tpl.source_offset is not None:
        src = make_seat((ctx.actor + _OFFSET_DELTA_INDEPENDENT[tpl.source_offset]) % 4)
    hand = set(ctx.own_concealed_tiles)
    kind = tpl.kind
    if kind in ("chi", "pon", "daiminkan"):
        if src is None or ctx.offered_tile != tpl.called_tile or ctx.offered_by != src:
            return False
    elif kind == "ron":
        if src is None or ctx.offered_tile != tpl.tile or ctx.offered_by != src:
            return False
    elif kind == "pass" and src is not None and (ctx.offered_by != src or ctx.offered_tile is None):
        return False
    owned: list[object] = []
    if kind in ("discard", "tsumogiri", "riichi_discard", "tsumo", "kakan"):
        owned = [tpl.tile]
    elif kind in ("chi", "pon", "daiminkan", "ankan"):
        owned = list(tpl.consumed_tiles)
    if any(item not in hand for item in owned):
        return False
    if kind == "kakan":
        base = [
            m
            for m in ctx.visible_melds
            if m.kind == "pon" and m.owner == ctx.actor and m.tiles[0] // 4 == tpl.tile // 4
        ]
        if len(base) != 1:
            return False
        missing = set(range(4 * (tpl.tile // 4), 4 * (tpl.tile // 4) + 4)) - {
            int(x) for x in base[0].tiles
        }
        if missing != {int(tpl.tile)}:
            return False
    return True


def _sample_contexts(table_obj) -> list[ActionContext]:
    rng = random.Random(20260822)
    contexts: list[ActionContext] = []
    for _trial in range(24):
        actor = make_seat(rng.randrange(4))
        others = [s for s in (S0, S1, S2, S3) if s != actor]
        size = rng.choice([1, 4, 7, 13])
        hand = set(rng.sample(range(136), size))
        melds: list[VisibleMeld] = []
        if rng.random() < 0.35:
            ctype = rng.randrange(34)
            copies = list(range(4 * ctype, 4 * ctype + 4))
            rng.shuffle(copies)
            meld_tiles, free = sorted(copies[:3]), copies[3]
            melds.append(
                VisibleMeld(
                    None,
                    "pon",
                    actor,
                    others[rng.randrange(3)],
                    make_tile_id(meld_tiles[-1]),
                    tuple(make_tile_id(x) for x in meld_tiles),
                )
            )
            hand.add(free)
        offered = None
        offered_by = None
        if rng.random() < 0.65:
            offered = make_tile_id(rng.randrange(136))
            offered_by = others[rng.randrange(3)]
        phase = rng.choice(["draw_decision", "discard_response", "kan_response"])
        contexts.append(
            context(
                table_obj,
                phase,  # type: ignore[arg-type]
                actor=actor,
                offered=offered,
                offered_by=offered_by,
                hand=[make_tile_id(x) for x in sorted(hand)],
                melds=melds,
            )
        )
    return contexts


class TestCodecMaskBijectionProperty:
    def test_decode_iff_mask_bit_encode_stable_over_samples(self, table):
        contexts = _sample_contexts(table)
        # targeted specials: red fives in hand, honor-triplet offer, kakan window
        contexts.append(
            context(table, "draw_decision", hand=[mk(16), mk(52), mk(88), mk(17), mk(53)])
        )
        contexts.append(
            context(
                table,
                "discard_response",
                offered=mk(111),
                offered_by=S1,
                hand=[mk(108), mk(109), mk(110)],
            )
        )
        meld = VisibleMeld(None, "pon", S0, S3, mk(110), (mk(108), mk(109), mk(110)))
        contexts.append(context(table, "draw_decision", hand=[mk(111)], melds=[meld]))
        checked_support = 0
        for ctx in contexts:
            support = 0
            for index in range(len(table.actions)):
                legal = _independent_legal(index, table, ctx)
                if legal:
                    action_obj = codec.decode(index, table=table, context=ctx)
                    assert codec.encode(action_obj, table=table, context=ctx) == index
                    support += 1
                else:
                    with pytest.raises(InvalidActionError):
                        codec.decode(index, table=table, context=ctx)
            checked_support += support
        assert checked_support > 100

    def test_decoded_actions_are_pairwise_distinct(self, table):
        ctx = context(table, "draw_decision", hand=[mk(16), mk(17), mk(18), mk(19), mk(36)])
        seen = {}
        for index in range(len(table.actions)):
            try:
                action_obj = codec.decode(index, table=table, context=ctx)
            except InvalidActionError:
                continue
            marker = (
                action_obj.kind,
                action_obj.tile,
                action_obj.called_tile,
                action_obj.consumed_tiles,
                action_obj.source_seat,
                action_obj.declares_riichi,
                action_obj.metadata,
            )
            assert marker not in seen, f"indices {seen.get(marker)} and {index} collide"
            seen[marker] = index


# ---------------------------------------------------------------------------
# Malformed calls and context misuse.
# ---------------------------------------------------------------------------


class TestMalformedCallsRejected:
    def test_wrong_table_hash_blocks_encode_and_decode(self, table):
        ctx = context(table, "draw_decision", hand=[mk(16)], table_hash="sha256:" + "0" * 64)
        with pytest.raises(InvalidActionError):
            codec.encode(action("discard", tile=mk(16)), table=table, context=ctx)
        with pytest.raises(InvalidActionError):
            codec.decode(4, table=table, context=ctx)

    def test_foreign_owned_tiles_rejected(self, table):
        ctx = context(table, "draw_decision", hand=[mk(36)])
        with pytest.raises(InvalidActionError):
            codec.encode(action("discard", tile=mk(16)), table=table, context=ctx)
        with pytest.raises(InvalidActionError):
            codec.encode(action("tsumo", tile=mk(16)), table=table, context=ctx)

    def test_missing_prior_meld_blocks_kakan(self, table):
        ctx = context(table, "draw_decision", hand=[mk(111)])
        meld = VisibleMeld(None, "pon", S0, S3, mk(110), (mk(108), mk(109), mk(110)))
        good_ref = ("prior_pon_meld_id", visible_meld_id(meld))
        with pytest.raises(InvalidActionError):  # no meld visible at all
            codec.encode(
                action("kakan", tile=mk(111), metadata=(good_ref,)), table=table, context=ctx
            )
        meld_other = VisibleMeld(None, "pon", S1, S3, mk(110), (mk(108), mk(109), mk(110)))
        ctx_other = context(table, "draw_decision", hand=[mk(111)], melds=[meld_other])
        with pytest.raises(InvalidActionError):  # meld owned by someone else
            codec.encode(
                action("kakan", tile=mk(111), metadata=(good_ref,)), table=table, context=ctx_other
            )
        wrong_ref = ("prior_pon_meld_id", "pon:1.2.3")
        ctx_right = context(table, "draw_decision", hand=[mk(111)], melds=[meld])
        with pytest.raises(InvalidActionError):  # stale/foreign meld id
            codec.encode(
                action("kakan", tile=mk(111), metadata=(wrong_ref,)), table=table, context=ctx_right
            )
        with pytest.raises(InvalidActionError):  # added tile is not the free copy
            codec.encode(
                action("kakan", tile=mk(107), metadata=(good_ref,)), table=table, context=ctx_right
            )

    def test_offer_mismatch_rejected(self, table):
        ctx = context(
            table, "discard_response", offered=mk(83), offered_by=S2, hand=[mk(80), mk(81)]
        )
        with pytest.raises(InvalidActionError):  # claimed seat differs from offerer
            codec.encode(
                action("pon", called=mk(83), consumed=(mk(80), mk(81)), source=S1),
                table=table,
                context=ctx,
            )
        with pytest.raises(InvalidActionError):  # called tile differs from offer
            codec.encode(
                action("pon", called=mk(82), consumed=(mk(80), mk(81)), source=S2),
                table=table,
                context=ctx,
            )
        chi_ctx = context(
            table, "discard_response", offered=mk(84), offered_by=S3, hand=[mk(76), mk(80)]
        )
        with pytest.raises(InvalidActionError):  # chi claimed from the wrong seat
            codec.encode(
                action("chi", called=mk(84), consumed=(mk(76), mk(80)), source=S2),
                table=table,
                context=chi_ctx,
            )

    def test_ron_requires_matching_offer(self, table):
        ctx = context(table, "discard_response", offered=mk(52), offered_by=S2)
        round_trip(action("ron", tile=mk(52), source=S2), table, ctx)
        bad = context(table, "discard_response", offered=mk(88), offered_by=S2)
        with pytest.raises(InvalidActionError):
            codec.encode(action("ron", tile=mk(52), source=S2), table=table, context=bad)
        with pytest.raises(InvalidActionError):  # ron without an offerer is void
            codec.encode(action("ron", tile=mk(52), source=None), table=table, context=ctx)

    def test_phase_gates_block_structurally_valid_actions(self, table):
        draw_ctx = context(table, "draw_decision", hand=[mk(80), mk(81), mk(83)])
        with pytest.raises(InvalidActionError):  # claims live only at discard_response
            codec.encode(
                action("pon", called=mk(83), consumed=(mk(80), mk(81)), source=S1),
                table=table,
                context=draw_ctx,
            )
        resp_ctx = context(table, "discard_response", hand=[mk(16)])
        with pytest.raises(InvalidActionError):  # discards live only at draw_decision
            codec.encode(action("discard", tile=mk(16)), table=table, context=resp_ctx)
        with pytest.raises(InvalidActionError):  # abort acceptance gated to responses
            codec.encode(action("accept_abortive_draw"), table=table, context=draw_ctx)
        start_ctx = context(table, "round_start", hand=[mk(16)])
        with pytest.raises(InvalidActionError):
            codec.encode(action("abort_nine_terminals"), table=table, context=start_ctx)

    def test_out_of_range_and_non_int_ids_rejected(self, table):
        ctx = context(table, "draw_decision", hand=[mk(16)])
        with pytest.raises(InvalidActionError):
            codec.decode(len(table.actions), table=table, context=ctx)
        with pytest.raises(ContractError):
            codec.decode(True, table=table, context=ctx)
        with pytest.raises(InvalidActionError):
            codec.decode(-1, table=table, context=ctx)

    def test_actor_mismatch_rejected(self, table):
        ctx = context(table, "draw_decision", actor=S1, hand=[mk(16)])
        with pytest.raises(InvalidActionError):
            codec.encode(action("discard", actor=S0, tile=mk(16)), table=table, context=ctx)

    def test_template_construction_rejects_bad_offsets(self):
        with pytest.raises(ContractError):  # chi claims only from previous seat
            CanonicalActionTemplate(
                kind="chi",
                tile=None,
                called_tile=mk(24),
                consumed_tiles=(mk(16), mk(20)),
                source_offset=1,
                declares_riichi=False,
                meld_ref_required=False,
            )
        with pytest.raises(ContractError):  # kakan has no source offset at all
            CanonicalActionTemplate(
                kind="kakan",
                tile=mk(111),
                called_tile=None,
                consumed_tiles=(),
                source_offset=-1,
                declares_riichi=False,
                meld_ref_required=True,
            )
