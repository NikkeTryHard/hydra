# Research Handoffs

Archive/research intake front door. Do not browse raw historical prompt dumps first.

Authority order:
1. `ARCHIVE_CANONICAL_CLAIMS.jsonl` — canonical claim ledger. JSONL wins.
2. `ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md` — compact derived triage.
3. `research/design/HYDRA_RECONCILIATION.md` — active path.
4. `research/design/HYDRA_FINAL.md` — max architecture.
5. Current code/docs for runtime truth.

Current files:

| File | Use |
|---|---|
| `ARCHIVE_CANONICAL_CLAIMS.jsonl` | source ledger; preserve |
| `ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md` | human triage over ledger |
| `PROMPT_STYLE_GUIDE.md` | current prompt-generator workflow |
| `genie_hidden_world_external_artifact_bank.md` | hidden-world external mechanism bank |
| `genie_hidden_world_external_pdf_extractions.md` | dense paper extraction notes |
| `historical_markdown_snapshot.tar.xz` | archived raw/generated Markdown removed from active tree |

Archived out of active Markdown:
- `combined_all_variants/*.md`
- generated prompt packets (`genie_*prompt*.md`, `hydra_dcrl_*prompt.md`)
- generated ledger render (`ARCHIVE_CANONICAL_CLAIMS_RENDERED.md`)
- standalone returned answer / external-artifact shortlist now superseded by ledger/promoted docs

Rules:
- Raw archive = evidence only, not doctrine.
- If JSONL `source_ref` needs raw lines, extract snapshot locally and inspect exact source.
- Do not restore raw Markdown dumps into active tree unless refreshing ledger/provenance.
- Generated render can be rebuilt with `generate_archive_canonical_claims.py` if needed.
