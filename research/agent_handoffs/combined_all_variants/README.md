# Combined archive corpus

Raw `answer_*_combined.md` and reference prompt Markdown moved out of active docs.

Use parent archive snapshot:

```text
research/agent_handoffs/historical_markdown_snapshot.tar.xz
```

Why:
- raw answer files are historical evidence, not doctrine
- canonical claim ledger is `../ARCHIVE_CANONICAL_CLAIMS.jsonl`
- derived triage is `../ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md`
- promoted doctrine lives in `../../design/HYDRA_RECONCILIATION.md` and `../../design/HYDRA_FINAL.md`

If validating ledger `source_ref`, extract snapshot outside active docs, inspect exact raw file/line, then return to promoted docs/current code before acting.
