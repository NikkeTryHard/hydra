# Combined archive corpus

This directory is the raw historical archive corpus for Hydra handoff work.

Historical wording disclaimer:
- these files are preserved snapshots of what agents produced at the time
- they may still contain older authority-flow language that no longer matches the live repo docs
- that mismatch is intentional historical preservation, not a signal that the current routing model failed to update

What it is for:
- preserve full `answer_*_combined.md` artifacts and prompt exemplars
- keep broad prior exploration available as evidence
- provide source material for the canonical archive SSOT in `ARCHIVE_CANONICAL_CLAIMS.jsonl`

What it is not:
- not the canonical archive SSOT
- not promoted doctrine
- not runtime reality
- not expected to be DRY in the same way as the live docs

Important path remaps after doc cleanup:
- historical references to `research/BUILD_AGENT_PROMPT.md` should be interpreted through the current routing chain instead: `README.md` -> `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` -> `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md` -> `research/design/HYDRA_RECONCILIATION.md` / `research/design/HYDRA_FINAL.md` as needed -> `research/design/IMPLEMENTATION_ROADMAP.md`
- historical references to `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.md` now map to `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` (source) and `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md` (generated render)

If a combined answer conflicts with the canonical archive SSOT, promoted doctrine summaries, or current code, treat it as historical raw evidence and validate before reusing it.
