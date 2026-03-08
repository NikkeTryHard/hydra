from pathlib import Path
import json


def render_cell(value: object) -> str:
    text = str(value)
    text = text.replace("|", r"\|")
    text = text.replace("\r\n", "<br>").replace("\n", "<br>").replace("\r", "<br>")
    return text


def render_row(values: list[object]) -> str:
    return "| " + " | ".join(render_cell(value) for value in values) + " |"


base = Path(__file__).resolve().parent
jsonl = base / "ARCHIVE_CANONICAL_CLAIMS.jsonl"
md = base / "ARCHIVE_CANONICAL_CLAIMS.md"
records = []
with jsonl.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))
coverage = next(r["items"] for r in records if r["type"] == "coverage")
legend = next(r["items"] for r in records if r["type"] == "legend")
appendix = next(r["items"] for r in records if r["type"] == "appendix")
claims = [r for r in records if r["type"] == "claim"]
lines = []
lines.append("## Archive canonical claims ledger")
lines.append(
    "This file is a stricter canonical ledger for the archive corpus in `research/agent_handoffs/combined_all_variants` only. Archive files are the extraction corpus. Current Hydra docs/code are used only for post-extraction validation and contradiction checks."
)
lines.append("## Coverage")
lines.extend(f"- {item}" for item in coverage)
lines.append("## Legend")
lines.extend(f"- {item}" for item in legend)
header = "| canonical_claim | tag | all_source_refs | supporting_source_quotes | repo_supported | repo_support_detail | hydra_docs_present | hydra_docs_detail | in_code_now | in_code_detail | reproduced | reproduced_detail | validated_pass_1 | validated_pass_2 | trustworthy | implementation_ready | promising | strength_upside | risk | fallback_worthy | fallback_role | notes |"
sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
lines.append(header)
lines.append(sep)
for c in claims:
    row = [
        c["canonical_claim"],
        c["tag"],
        c["all_source_refs"],
        c["supporting_source_quotes"],
        c["repo_supported"],
        c["repo_support_detail"],
        c["hydra_docs_present"],
        c["hydra_docs_detail"],
        c["in_code_now"],
        c["in_code_detail"],
        c["reproduced"],
        c["reproduced_detail"],
        c["validated_pass_1"],
        c["validated_pass_2"],
        c["trustworthy"],
        c["implementation_ready"],
        c["promising"],
        c["strength_upside"],
        c["risk"],
        c["fallback_worthy"],
        c["fallback_role"],
        c["notes"],
    ]
    lines.append(render_row(row))
lines.append("## Appendix — archive file -> outcome")
lines.extend(appendix)
md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {md}")
