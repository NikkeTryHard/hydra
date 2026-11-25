"""Engine parity against a trace recorded from the pinned distribution.

The fixture in ``tests/fixtures/engine_trace.json`` was recorded by running this
script with ``HYDRA3_ENGINE=installed`` and ``--record``, so it encodes the
behaviour of the pinned MahJax release. The default mode replays the same seeds
and actions through whichever engine is selected and compares whole-state
digests, which makes any rule, scoring, wall, or call difference fail.

Re-record only when engine behaviour is expected to change::

    HYDRA3_ENGINE=installed pixi run python tests/engine_parity.py --record
"""

import hashlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from hydra3 import engine

engine.select()
import mahjax
from mahjax.common.hand import Hand

FIXTURE = Path(__file__).with_name("fixtures") / "engine_trace.json"
LANES = 512
TURNS = 24
CHECK_EVERY = 4


def fold(state) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(state):
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def trace() -> dict:
    env = mahjax.make("red_4p", round_mode="half", order_points=[0, 0, 0, 0])
    init, step = jax.vmap(env.init), jax.vmap(env.step)
    state = init(jax.random.split(jax.random.key(20261123), LANES))
    checkpoints = {}
    for turn in range(TURNS):
        legal = state.legal_action_mask
        keys = jax.random.fold_in(jax.random.key(7), turn)
        scores = jax.random.uniform(keys, legal.shape)
        action = jnp.argmax(jnp.where(legal, scores, -1.0), axis=1)
        state = step(state, action, jax.random.split(jax.random.fold_in(keys, 1), LANES))
        if turn % CHECK_EVERY == 0:
            checkpoints[str(turn)] = fold(state)
        if bool((state.terminated | state.truncated).all()):
            break
    checkpoints["final"] = fold(state)
    return {"engine": mahjax.__file__, "turns": turn, "checkpoints": checkpoints}


def check_waits() -> None:
    """Exercise hands rarely reached in the opening state trace, including dead waits."""
    orphans = [0, 8, 9, 17, 18, 26, *range(27, 34)]
    cases = (
        ([9, 10, 11, 12, 13, 14, 18, 19, 20, 27, 27, 3, 5], True),
        ([0, 0, 0, 0, 9, 10, 11, 12, 13, 14, 18, 19, 20], False),
        ([30] * 4 + [27] * 3 + [28] * 3 + [29] * 3, False),
        ([4, 34, 0, 0, 8, 8, 9, 9, 17, 17, 18, 18, 33], True),
        (orphans, True),
        (orphans[:-1] + [0], True),
    )
    counts = np.stack([np.bincount(tiles, minlength=37) for tiles, _ in cases])
    hands = jnp.asarray(counts, jnp.int8)
    expected = np.asarray([ready for _, ready in cases])
    # Red and ordinary fives share a face for completion but not a discard id.
    # Compare both wire representations so flattening cannot create a fifth copy.
    for rows in (hands, jax.vmap(Hand.to_34)(hands)):
        np.testing.assert_array_equal(np.asarray(jax.vmap(Hand.is_tenpai)(rows)), expected)


def main() -> int:
    check_waits()
    current = trace()
    if "--record" in sys.argv:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"recorded {len(current['checkpoints'])} checkpoints from {current['engine']}")
        return 0
    if not FIXTURE.is_file():
        print(f"FAIL: missing fixture {FIXTURE}")
        return 1
    expected = json.loads(FIXTURE.read_text())
    if current["checkpoints"] != expected["checkpoints"]:
        for turn, digest in expected["checkpoints"].items():
            if current["checkpoints"].get(turn) != digest:
                print(f"MISMATCH at turn {turn}: {current['checkpoints'].get(turn)} != {digest}")
                break
        return 1
    print(
        f"OK: {len(current['checkpoints'])} checkpoints reproduce the pinned trace "
        f"({current['turns']} turns)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
