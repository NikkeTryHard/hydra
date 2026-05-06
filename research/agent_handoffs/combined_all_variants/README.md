# Combined archive corpus

Dir = raw historical archive corpus for Hydra handoff work.

Historical wording disclaimer:
- files = preserved snapshots of agent output at time
- may contain older authority-flow language not matching live repo docs
- mismatch intentional historical preservation, not signal current routing model failed update

What it is for:
- preserve full `answer_*_combined.md` artifacts and prompt exemplars
- keep broad prior exploration as evidence
- provide source material for canonical archive SSOT in `ARCHIVE_CANONICAL_CLAIMS.jsonl`

What it is not:
- not canonical archive SSOT
- not promoted doctrine
- not runtime reality
- not expected DRY same as live docs

Important path remaps after doc cleanup:
- historical references to `research/BUILD_AGENT_PROMPT.md` should be interpreted through current routing chain instead: `README.md` -> `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` -> `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md` -> `research/design/HYDRA_RECONCILIATION.md` / `research/design/HYDRA_FINAL.md` as needed -> `research/design/IMPLEMENTATION_ROADMAP.md`
- historical references to `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.md` now map to `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` (source) and `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md` (generated render)

If combined answer conflicts with canonical archive SSOT, promoted doctrine summaries, or current code, treat as historical raw evidence; validate before reuse.