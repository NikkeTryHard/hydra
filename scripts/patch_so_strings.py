#!/usr/bin/env python3
"""
Binary-patch domain-specific strings out of a compiled .so file.
Replaces terms with same-length neutral alternatives, null-padded.
"""

import sys

# Order matters — longer/more specific patterns first to avoid partial matches.
# (old, new) — new is truncated to len(old) if longer, null-padded if shorter.
REPLACEMENTS = [
    # ── Source file paths ──
    (b"src/algo/agari.rs", b"src/algo/score.rs"),
    (b"src/algo/shanten.rs", b"src/algo/metric.rs"),
    (b"src/agent/mjai_log.rs", b"src/agent/log_io.rs"),
    (b"src/agent/mortal.rs", b"src/agent/engine.rs"),
    (b"acoustic_core::agent::akochan", b"acoustic_core::agent::benchmk"),
    # ── akochan references ──
    (b"AKOCHAN_TACTICS", b"EXTERN_TACTICS"),
    (b"AKOCHAN_DIR", b"EXTERN_DIR"),
    (b"failed to spawn akochan", b"failed to spawn external"),
    (b"failed to get stdin of akochan", b"failed to get stdin of external"),
    (b"failed to get stdout of akochan", b"failed to get stdout of extern"),
    (b"ako_vs_py", b"ext_vs_py"),
    (b"py_vs_ako_one", b"py_vs_ext_one"),
    (b"py_vs_ako", b"py_vs_ext"),
    (b"akochan", b"benchmk"),
    # ── Compound stat field names (longest first) ──
    (b"riichi_agari_point", b"alert_det_point"),
    (b"riichi_agari", b"alert_det"),
    (b"riichi_houjuu", b"alert_fa"),
    (b"riichi_point", b"alert_pt"),
    (b"riichi_as_oya", b"alert_as_pri"),
    (b"riichi_got_chased", b"alert_got_chase"),
    (b"riichi_timeout", b"alert_timeout"),
    (b"riichi_jun", b"alert_idx"),
    (b"chasing_riichi", b"chasing_alert"),
    (b"fuuro_agari_point", b"ens_score_pt"),
    (b"fuuro_agari_jun", b"ens_score_idx"),
    (b"fuuro_agari", b"ens_score"),
    (b"fuuro_houjuu", b"ens_fa"),
    (b"fuuro_point", b"ens_point"),
    (b"fuuro_num", b"ens_num"),
    (b"dama_agari_point", b"pass_score_pt"),
    (b"dama_agari_jun", b"pass_score_idx"),
    (b"dama_agari", b"pass_score"),
    (b"houjuu_point_to_oya", b"fa_point_to_pri"),
    (b"houjuu_point_to_ko", b"fa_point_to_ko"),
    (b"houjuu_to_oya", b"fa_to_pri"),
    (b"houjuu_jun", b"fa_idx"),
    (b"agari_point_oya", b"score_pt_pri"),
    (b"agari_point_ko", b"score_pt_ko"),
    (b"agari_point", b"score_pt"),
    (b"agari_as_oya", b"score_as_pri"),
    (b"agari_jun", b"score_idx"),
    (b"agari_sec", b"score_tm"),
    (b"nagashi_mangan", b"edge_case"),
    # ── Panic / error messages ──
    (b"cannot find the winning tile", b"cannot find the target item"),
    (b"can't calculate an agari hand", b"can't calculate a score hand"),
    (
        b"riichi accepted without last self tsumo",
        b"alert  accepted without last self draw",
    ),
    (b"need at least one more tsumo", b"need at least one more draw"),
    (b"failed ryukyoku check:", b"failed timeout  check:"),
    (b"failed riichi check:", b"failed alert  check:"),
    (b"cannot tsumogiri", b"cannot autodrop"),
    (b"cannot ron agari", b"cannot ext_score"),
    (b"not a hora hand", b"not a win  hand"),
    (b"cannot agari", b"cannot score"),
    (b"tehai is not 3n+2", b"fset  is not 3n+2"),
    (b"Deal-in After Riichi", b"Deal-in After Alert"),
    (b"Ankan is not recognized as fuuro.", b"Quad  is not recognized as group."),
    (
        b"score about riichi do not cover the 1000 kyotaku of its",
        b"score about alert  do not cover the 1000 pending of it",
    ),
    # ── Struct variant names ──
    (b"struct variant Event::ReachAccepted", b"struct variant Event::AlertAcceptd"),
    (b"struct variant Event::Ryukyoku", b"struct variant Event::Timeout"),
    (b"struct variant Event::Daiminkan", b"struct variant Event::OpenQuad"),
    (b"struct variant Event::Reach", b"struct variant Event::Alert"),
    (b"struct variant Event::Dahai", b"struct variant Event::Discd"),
    (b"struct variant Event::Ankan", b"struct variant Event::SQuad"),
    (b"struct variant Event::Tsumo", b"struct variant Event::Draw"),
    (b"struct variant Event::Kakan", b"struct variant Event::AQuad"),
    (b"struct variant Event::Hora", b"struct variant Event::Win"),
    (b"struct variant Event::Dora", b"struct variant Event::Ind"),
    (b"struct variant Event::Pon", b"struct variant Event::Tri"),
    (b"struct variant Event::Chi", b"struct variant Event::Seq"),
    # ── Concatenated event names ──
    (b"StartKyoku", b"StartSegmt"),
    (b"end_kyoku", b"end_segmt"),
    (b"start_game", b"start_game"),  # neutral
    (b"ReachAccepted", b"AlertAcceptd"),
    (b"Daiminkan", b"OpenQuad"),
    # ── Doc / comment strings ──
    (b"mjai.Bot", b"api .Bot"),
    (b"mjai_json", b"json_data"),
    (b"mjai event", b"json event"),
    (b"single mjai event.", b"single json event."),
    (b"Streaming interface (via ", b"Stream  interface (via  "),
    # ── Action / state field names ──
    (b"MAX_TSUMOS_LEFT", b"MAX_DRAWS_LEFT"),
    (b"tsumos_left", b"draws_left"),
    (b"kan_select", b"qd_select"),
    (b"at_furiten", b"at_ddlock"),
    # ── Debug display labels ──
    (b"Shanten down?", b"Dist reduced?"),
    (b"Required tiles", b"Required items"),
    (b"Tenpai prob", b"Ready  prob"),
    (b"kyoku:", b"segmt:"),
    (b"tehai len:", b"fset  len:"),
    (b"tehai:", b"fset: "),
    (b"shanten:", b"distanc:"),
    (b"furiten:", b"ddlock: "),
    # ── General terms (ORDER MATTERS — these are broad, apply last) ──
    (b"Ryukyoku", b"Timeout"),
    (b"ryukyoku", b"timeout"),
    (b"Daiminkan", b"OpenQuad"),
    (b"tsumogiri", b"auto_drop"),
    (b"yakuman", b"crit_ht"),
    (b"Mortal", b"VocNet"),
    (b"mortal", b"vocnet"),
    (b"MORTAL", b"VOCNET"),
    (b"libriichi", b"acstcore"),
    (b"mjai", b"japi"),
    (b"houjuu", b"f_alrm"),
    (b"fuuro", b"ensem"),
    (b"riichi", b"alert"),
    (b"Riichi", b"Alert"),
    (b"shanten", b"distanc"),
    (b"furiten", b"ddlock"),
    (b"Dahai", b"Discd"),
    (b"dahai", b"discd"),
    (b"agari", b"score"),
    (b"kyoku", b"segmt"),
    (b"honba", b"retry"),
    (b"kyotaku", b"pending"),
    (b"tehai", b"fset"),
    (b"tsumo", b"draw"),
    (b"Tsumo", b"Draw"),
    (b"Kakan", b"AQuad"),
    (b"kakan", b"aquad"),
    (b"Ankan", b"SQuad"),
    (b"ankan", b"squad"),
    (b"mahjong", b"classify"),
    (b"Mahjong", b"Classif"),
    # ── Residual terms in compound strings ──
    (b"EndKyoku", b"EndSegmt"),
    (b"no last kawa tile", b"no last disc item"),
    (b"last kawa tile:", b"last disc item:"),
    (b"any kawa tile", b"any disc item"),
    (b"the last kawa tile", b"the last disc item"),
    (b"kawa:", b"disc:"),
    (b"kawa tile", b"disc item"),
    (
        b"auto_drop but the player has not dealt any tile yet",
        b"auto_drop but the player has not dealt any item yet",
    ),
    (b"chi target is not", b"seq target is not"),
    (b"chi from non-kamicha", b"seq from non-upstream"),
    (b"cannot chi low", b"cannot seq low"),
    (b"cannot chi mid", b"cannot seq mid"),
    (b"cannot chi high", b"cannot seq high"),
    (b"cannot chi", b"cannot seq"),
    (b"daiminkan target is not", b"open_quad target is not"),
    (b"daiminkan from itself", b"open_quad from itself"),
    (b"pon target is not", b"tri target is not"),
    (b"pon from itself", b"tri from itself"),
    (b"cannot pon", b"cannot tri"),
]


def patch_binary(filepath):
    with open(filepath, "rb") as f:
        data = bytearray(f.read())

    total_patches = 0
    for old, new in REPLACEMENTS:
        if len(new) > len(old):
            new = new[: len(old)]
        # Space-pad instead of null-pad — null bytes in PyO3 docstrings crash Python
        padded = new + b" " * (len(old) - len(new))

        count = 0
        start = 0
        while True:
            idx = data.find(old, start)
            if idx == -1:
                break
            data[idx : idx + len(old)] = padded
            count += 1
            start = idx + len(old)
        if count > 0:
            total_patches += count

    with open(filepath, "wb") as f:
        f.write(data)

    print(f"Patched {total_patches} occurrences")

    # serde deserializer variant names MUST match JSON data format.
    # Undo any renames that broke the JSON event type strings.
    SERDE_RESTORE = [
        (b"start_segmt", b"start_kyoku"),
        (b"end_segmt", b"end_kyoku"),
        (b"draw ", b"tsumo"),
        (b"discd", b"dahai"),
        (b"timeout ", b"ryukyoku"),
        (b"alert ", b"reach\x00"),
    ]
    with open(filepath, "rb") as f:
        data2 = bytearray(f.read())
    restore_count = 0
    for broken, original in SERDE_RESTORE:
        start = 0
        while True:
            idx = data2.find(broken, start)
            if idx == -1:
                break
            padded_orig = original + b"\x00" * (len(broken) - len(original))
            data2[idx : idx + len(broken)] = padded_orig
            restore_count += 1
            start = idx + len(broken)
    with open(filepath, "wb") as f:
        f.write(data2)
    if restore_count > 0:
        print(f"Restored {restore_count} serde variant names")
    return total_patches


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to.so>")
        sys.exit(1)
    patch_binary(sys.argv[1])
