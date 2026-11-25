#!/bin/sh
# Pin the canonical action source. A moving revision or local action edits
# could change the meaning of IDs stored in Hydra3 corpora.
set -eu
source_dir=vendor/hydra2
revision=0dc5ed89b68b2b95ae61abd0d9d4c662ff41393e
if [ ! -e "$source_dir" ]; then
    git clone --quiet --depth=1 --revision="$revision" https://github.com/NikkeTryHard/hydra.git "$source_dir"
fi
actual=$(git -C "$source_dir" rev-parse HEAD)
changes=$(git -C "$source_dir" status --porcelain --untracked-files=all --ignored -- crates/feed/src/census.rs crates/bridge/src/rl/actions.rs)
if [ "$actual" != "$revision" ] || [ -n "$changes" ]; then
    printf 'Expected unmodified reference sources at %s in %s.\n' "$revision" "$source_dir" >&2
    exit 1
fi
