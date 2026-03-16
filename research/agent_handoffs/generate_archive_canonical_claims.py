from pathlib import Path
import json


SPECIAL_SECTION_TYPES = ("coverage", "legend", "appendix")
FIELD_ALIASES = {
    "not_done": "not_done_yet",
    "pass_1": "validated_pass_1",
    "pass_2_3": "validated_pass_2",
    "trust": "trustworthy",
    "promise": "promising",
}
KNOWN_COLUMNS = [
    "record_type",
    "status",
    "scope",
    "artifact",
    "summary",
    "what_we_did",
    "not_done_yet",
    "canonical_claim",
    "tag",
    "all_source_refs",
    "supporting_source_quotes",
    "repo_supported",
    "repo_support_detail",
    "hydra_docs_present",
    "hydra_docs_detail",
    "in_code_now",
    "in_code_detail",
    "reproduced",
    "reproduced_detail",
    "validated_pass_1",
    "validated_pass_2",
    "trustworthy",
    "implementation_ready",
    "promising",
    "strength_upside",
    "risk",
    "fallback_worthy",
    "fallback_role",
    "notes",
]
GENERATOR_PATH = "research/agent_handoffs/generate_archive_canonical_claims.py"


def render_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = "<br>".join(f"- {item}" for item in value)
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value)
    text = text.replace("|", r"\|")
    text = text.replace("\r\n", "<br>").replace("\n", "<br>").replace("\r", "<br>")
    return text


def render_row(columns: list[str], record: dict[str, object]) -> str:
    return (
        "| "
        + " | ".join(render_cell(record.get(column, "")) for column in columns)
        + " |"
    )


def discover_jsonl_inputs(base: Path) -> list[Path]:
    return sorted(path for path in base.glob("*.jsonl") if path.is_file())


def normalize_record(
    record: object, source: Path, line_number: int
) -> dict[str, object]:
    if not isinstance(record, dict):
        raise ValueError(
            f"Expected a JSON object in {source.name}:{line_number}, got {type(record).__name__}"
        )
    if "type" not in record:
        raise ValueError(
            f"Missing required 'type' field in {source.name}:{line_number}"
        )

    normalized = dict(record)
    record_type = normalized.pop("type")
    normalized["record_type"] = record_type

    for alias, canonical in FIELD_ALIASES.items():
        if alias not in normalized:
            continue
        if canonical not in normalized:
            normalized[canonical] = normalized.pop(alias)
            continue
        if normalized[alias] == normalized[canonical]:
            normalized.pop(alias, None)

    return normalized


def load_records(jsonl_inputs: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for jsonl_path in jsonl_inputs:
        with jsonl_path.open("r", encoding="utf-8") as file_handle:
            for line_number, line in enumerate(file_handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                records.append(
                    normalize_record(json.loads(stripped), jsonl_path, line_number)
                )
    return records


def collect_sections(
    records: list[dict[str, object]],
) -> tuple[dict[str, list[object]], list[dict[str, object]]]:
    sections = {section_type: [] for section_type in SPECIAL_SECTION_TYPES}
    table_records: list[dict[str, object]] = []

    for record in records:
        record_type = str(record["record_type"])
        if record_type in sections:
            items = record.get("items", [])
            if isinstance(items, list):
                sections[record_type].extend(items)
            elif items not in (None, ""):
                sections[record_type].append(items)
            continue
        table_records.append(record)

    return sections, table_records


def derive_columns(records: list[dict[str, object]]) -> list[str]:
    columns = list(KNOWN_COLUMNS)
    seen = set(columns)

    for record in records:
        for key in record:
            if key not in seen:
                columns.append(key)
                seen.add(key)

    return columns


def render_bullets(items: list[object]) -> list[str]:
    return [f"- {item}" for item in items]


def render_appendix_items(items: list[object]) -> list[str]:
    rendered: list[str] = []
    for item in items:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped.startswith(("#", "-", "*")):
                rendered.append(item)
            else:
                rendered.append(f"- {item}")
        else:
            rendered.append(f"- {render_cell(item)}")
    return rendered


def render_lines(
    jsonl_inputs: list[Path], records: list[dict[str, object]]
) -> list[str]:
    sections, table_records = collect_sections(records)
    columns = derive_columns(table_records)
    input_names = ", ".join(f"`{path.name}`" for path in jsonl_inputs)
    lines: list[str] = []

    lines.append("# Archive canonical claims ledger (generated)")
    lines.append("")
    lines.append("> [!WARNING]")
    lines.append(f"> This file is generated by `{GENERATOR_PATH}`.")
    lines.append(
        "> Do not edit this file directly. Regenerate it after updating the JSONL inputs in this folder."
    )
    lines.append(
        "> After regenerating, inspect the output. If it is still incorrect, update the generator."
    )
    lines.append(">")
    lines.append(f"> Active JSONL inputs: {input_names}")
    lines.append(
        "This file is the generated human-readable mirror of Hydra's canonical archive SSOT in `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`. The raw archive corpus in `research/agent_handoffs/combined_all_variants` is the intake material behind that ledger. Promoted doctrine docs and current code/runtime are used to validate, sharpen, and detect lag in the archive canon rather than to replace its source role."
    )
    lines.append("## Coverage")
    lines.extend(render_bullets(sections["coverage"]))
    lines.append("## Legend")
    lines.extend(render_bullets(sections["legend"]))
    lines.append(render_row(columns, {column: column for column in columns}))
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for record in table_records:
        lines.append(render_row(columns, record))
    lines.append("## Appendix — archive file -> outcome")
    lines.extend(render_appendix_items(sections["appendix"]))
    return lines


def main() -> None:
    base = Path(__file__).resolve().parent
    md = base / "ARCHIVE_CANONICAL_CLAIMS_RENDERED.md"
    jsonl_inputs = discover_jsonl_inputs(base)
    if not jsonl_inputs:
        raise ValueError(f"No JSONL inputs found in {base}")

    records = load_records(jsonl_inputs)
    lines = render_lines(jsonl_inputs, records)
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
