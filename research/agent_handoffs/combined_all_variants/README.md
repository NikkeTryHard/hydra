# Combined archive corpus

This directory is the raw historical archive corpus for Hydra handoff work.

What it is for:
- preserve full `answer_*_combined.md` artifacts and prompt exemplars
- keep broad prior exploration available as evidence
- provide source material for `ARCHIVE_CANONICAL_CLAIMS.jsonl`

What it is not:
- not SSOT
- not active doctrine
- not expected to be DRY in the same way as the live docs

Important path remaps after doc cleanup:
- historical references to `research/BUILD_AGENT_PROMPT.md` should be interpreted through the current routing chain instead: `README.md` -> root/subtree `AGENTS.md` -> `research/design/HYDRA_RECONCILIATION.md` -> `research/design/IMPLEMENTATION_ROADMAP.md`
- historical references to `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.md` now map to `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` (source) and `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md` (generated render)

If a combined answer conflicts with the live repo docs or current code, treat it as historical evidence and validate before reusing it.
