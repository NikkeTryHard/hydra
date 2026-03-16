#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, TextIO


COMBINED_SCHEMA_VERSION = 1
DEFAULT_LAYOUT = "single_markdown_file_prompt_shell_manifest_then_answer"
DEFAULT_PROMPT_SOURCE_PATH = "embedded_prompt_shell_and_manifest"
LEGACY_FALLBACK_PROMPT_SOURCE_PATH = (
    "embedded_prompt_shell_and_manifest_legacy_shell_fallback"
)
LEGACY_MODE_ERROR_SUFFIX = (
    ".mode: explicit mode is required when shell_source_path is used; "
    "template text is preserved by default"
)


class CombinedArtifactError(Exception):
    pass


def _load_prompt_generator() -> ModuleType:
    module_path = Path(__file__).resolve().with_name("generate_prompt.py")
    spec = importlib.util.spec_from_file_location("generate_prompt", module_path)
    if spec is None or spec.loader is None:
        raise CombinedArtifactError(
            f"failed to load prompt generator from {module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def _infer_agent_number(answer_path: Path) -> str:
    match = re.search(r"agent_(\d+)$", answer_path.stem)
    if match is None:
        raise CombinedArtifactError(
            "could not infer agent number from answer filename; pass --agent-number"
        )
    return match.group(1)


def _default_output_path(repo_root: Path, agent_number: str) -> Path:
    return (
        repo_root
        / "research/agent_handoffs/combined_all_variants"
        / f"answer_{agent_number}_combined.md"
    )


def _default_note(agent_number: str) -> str:
    return (
        f"Self-contained combined record for Agent {agent_number}. "
        "It preserves the compact prompt shell and artifact manifest generated from "
        "the authoritative prompt config, plus the preserved answer text."
    )


def _wrap_cdata(text: str) -> str:
    return text.replace("]]>", "]]]]><![CDATA[>")


def _uses_legacy_shell_mode_fallback(errors: list[str]) -> bool:
    return bool(errors) and all(
        error.endswith(LEGACY_MODE_ERROR_SUFFIX) for error in errors
    )


def _artifact_heading(
    prompt_generator: ModuleType, repo_root: Path, artifact: Any
) -> str:
    return (
        artifact.label
        or artifact.artifact_id
        or artifact.source_label
        or prompt_generator.artifact_path_display(repo_root, artifact)
    )


def _render_compact_artifact_manifest_entry(
    prompt_generator: ModuleType,
    repo_root: Path,
    artifact: Any,
    index: int,
) -> list[str]:
    lines = [
        f"## Artifact {index:02d} — {_artifact_heading(prompt_generator, repo_root, artifact)}"
    ]
    if artifact.artifact_id:
        lines.append(f"Artifact id: `{artifact.artifact_id}`")
    if artifact.source_label:
        lines.append(f"Source label: {artifact.source_label}")
    lines.append(f"Type: `{artifact.kind}`")
    source_text = prompt_generator.artifact_source_text(repo_root, artifact)
    if source_text:
        lines.append(f"Source: {source_text}")
    if artifact.explanation:
        lines.append(f"Why it matters: {artifact.explanation}")
    lines.append("")
    return lines


def _render_compact_prompt_text(
    prompt_generator: ModuleType,
    config: Any,
    variant_name: str,
) -> str:
    variant = prompt_generator.get_variant(config, variant_name)
    template = (
        prompt_generator.load_prompt_template(variant.shell_source_path)
        if variant.shell_source_path is not None
        else None
    )
    title = (
        variant.title
        or (template.title if template is not None else None)
        or config.defaults.title
        or variant.name
    )
    base_shell_sections = (
        template.shell_sections
        if template is not None
        else config.defaults.shell_sections
    )
    shell_sections = prompt_generator.merge_shell_sections(
        base_shell_sections, variant.shell_sections
    )
    artifact_specs = prompt_generator.resolve_variant_artifacts(config, variant)

    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")

    for section in shell_sections:
        lines.append(f"<{section.tag}>")
        lines.extend(section.lines)
        lines.append(f"</{section.tag}>")
        lines.append("")

    lines.append("<artifacts_manifest>")
    lines.append("")
    for index, artifact in enumerate(artifact_specs, start=1):
        lines.extend(
            _render_compact_artifact_manifest_entry(
                prompt_generator,
                config.repo_root,
                artifact,
                index,
            )
        )
    lines.append("</artifacts_manifest>")
    lines.append("")
    return "\n".join(lines)


def _parse_config_with_legacy_shell_mode_fallback(
    prompt_generator: ModuleType,
    config_path: Path,
    repo_root_override: Path | None = None,
):
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    root = prompt_generator._ensure_dict(raw, "<root>", errors, config_path)
    version = root.get("version")
    if version != getattr(prompt_generator, "SCHEMA_VERSION"):
        errors.append(
            prompt_generator._config_error(
                config_path,
                f"version: expected {prompt_generator.SCHEMA_VERSION}, got {version!r}",
            )
        )

    repo_root_value = root.get("repo_root", ".")
    repo_root_str = prompt_generator._ensure_string(
        repo_root_value,
        "repo_root",
        errors,
        config_path,
    )
    repo_root = (
        config_path.parent.resolve()
        if repo_root_str is None
        else (config_path.parent / repo_root_str).resolve()
    )
    if repo_root_override is not None:
        repo_root = repo_root_override.resolve()

    defaults_obj = prompt_generator._ensure_dict(
        root.get("defaults", {}), "defaults", errors, config_path
    )
    default_title = (
        prompt_generator._ensure_string(
            defaults_obj.get("title"),
            "defaults.title",
            errors,
            config_path,
        )
        if "title" in defaults_obj
        else None
    )
    default_shell_sections = prompt_generator._parse_shell_sections(
        defaults_obj.get("shell_sections", []),
        "defaults.shell_sections",
        errors,
        config_path,
    )
    default_artifact_ids_raw = prompt_generator._ensure_list(
        defaults_obj.get("artifact_ids", []),
        "defaults.artifact_ids",
        errors,
        config_path,
    )
    default_artifact_ids: list[str] = []
    for index, artifact_id in enumerate(default_artifact_ids_raw):
        value = prompt_generator._ensure_string(
            artifact_id,
            f"defaults.artifact_ids[{index}]",
            errors,
            config_path,
        )
        if value is not None:
            default_artifact_ids.append(value)
    artifact_container_tag = defaults_obj.get("artifact_container_tag", "artifacts")
    if not isinstance(artifact_container_tag, str) or not artifact_container_tag:
        errors.append(
            prompt_generator._config_error(
                config_path,
                "defaults.artifact_container_tag: expected non-empty string",
            )
        )
        artifact_container_tag = "artifacts"

    artifacts_raw = prompt_generator._ensure_list(
        root.get("artifacts", []), "artifacts", errors, config_path
    )
    artifacts: dict[str, Any] = {}
    for index, raw_artifact in enumerate(artifacts_raw):
        artifact = prompt_generator._parse_artifact(
            raw_artifact,
            f"artifacts[{index}]",
            errors,
            config_path,
            repo_root,
            artifact_id_required=True,
        )
        if artifact is None or artifact.artifact_id is None:
            continue
        if artifact.artifact_id in artifacts:
            errors.append(
                prompt_generator._config_error(
                    config_path,
                    f"artifacts[{index}].id: duplicate artifact id {artifact.artifact_id!r}",
                )
            )
            continue
        artifacts[artifact.artifact_id] = artifact

    variants_raw = prompt_generator._ensure_list(
        root.get("variants"), "variants", errors, config_path
    )
    variants: list[Any] = []
    variant_names: set[str] = set()
    for index, raw_variant in enumerate(variants_raw):
        variant_obj = prompt_generator._ensure_dict(
            raw_variant, f"variants[{index}]", errors, config_path
        )
        name = prompt_generator._ensure_string(
            variant_obj.get("name"),
            f"variants[{index}].name",
            errors,
            config_path,
        )
        if name is not None and name in variant_names:
            errors.append(
                prompt_generator._config_error(
                    config_path,
                    f"variants[{index}].name: duplicate variant name {name!r}",
                )
            )
        elif name is not None:
            variant_names.add(name)
        title = (
            prompt_generator._ensure_string(
                variant_obj.get("title"),
                f"variants[{index}].title",
                errors,
                config_path,
            )
            if "title" in variant_obj
            else None
        )
        shell_source_path = (
            prompt_generator._resolve_repo_path(
                prompt_generator._ensure_string(
                    variant_obj.get("shell_source_path"),
                    f"variants[{index}].shell_source_path",
                    errors,
                    config_path,
                ),
                f"variants[{index}].shell_source_path",
                errors,
                config_path,
                repo_root,
            )
            if "shell_source_path" in variant_obj
            else None
        )
        shell_sections = prompt_generator._parse_shell_sections(
            variant_obj.get("shell_sections", []),
            f"variants[{index}].shell_sections",
            errors,
            config_path,
        )
        artifact_ids_raw = prompt_generator._ensure_list(
            variant_obj.get("artifact_ids", []),
            f"variants[{index}].artifact_ids",
            errors,
            config_path,
        )
        artifact_ids: list[str] = []
        for artifact_index, artifact_id in enumerate(artifact_ids_raw):
            value = prompt_generator._ensure_string(
                artifact_id,
                f"variants[{index}].artifact_ids[{artifact_index}]",
                errors,
                config_path,
            )
            if value is not None:
                artifact_ids.append(value)

        extra_artifacts_raw = prompt_generator._ensure_list(
            variant_obj.get("artifacts", []),
            f"variants[{index}].artifacts",
            errors,
            config_path,
        )
        extra_artifacts: list[Any] = []
        for artifact_index, raw_artifact in enumerate(extra_artifacts_raw):
            artifact = prompt_generator._parse_artifact(
                raw_artifact,
                f"variants[{index}].artifacts[{artifact_index}]",
                errors,
                config_path,
                repo_root,
                artifact_id_required=False,
            )
            if artifact is not None:
                extra_artifacts.append(artifact)

        output_file = (
            prompt_generator._ensure_string(
                variant_obj.get("output_file"),
                f"variants[{index}].output_file",
                errors,
                config_path,
            )
            if "output_file" in variant_obj
            else None
        )

        if name is not None:
            variants.append(
                prompt_generator.VariantSpec(
                    name=name,
                    title=title,
                    shell_source_path=shell_source_path,
                    shell_sections=shell_sections,
                    artifact_ids=artifact_ids,
                    extra_artifacts=extra_artifacts,
                    output_file=output_file,
                )
            )

    config = prompt_generator.PromptConfig(
        config_path=config_path,
        repo_root=repo_root,
        defaults=prompt_generator.DefaultsSpec(
            title=default_title,
            shell_sections=default_shell_sections,
            artifact_ids=default_artifact_ids,
            artifact_container_tag=artifact_container_tag,
        ),
        artifacts=artifacts,
        variants=variants,
    )

    errors.extend(prompt_generator._validate_config(config))
    filtered_errors = [
        error for error in errors if not error.endswith(LEGACY_MODE_ERROR_SUFFIX)
    ]
    if filtered_errors:
        raise prompt_generator.ValidationError(filtered_errors)
    return config


def _render_prompt_with_fallback(
    prompt_generator: ModuleType,
    config_path: Path,
    variant_name: str | None,
    repo_root_override: Path | None = None,
) -> tuple[str, str, list[str], str]:
    validation_error_type = getattr(prompt_generator, "ValidationError")
    try:
        config = prompt_generator._parse_config(config_path, repo_root_override)
        selected_variant = prompt_generator.choose_variant(config, variant_name)
        prompt_text = _render_compact_prompt_text(
            prompt_generator,
            config,
            selected_variant,
        )
        return prompt_text, selected_variant, [], DEFAULT_PROMPT_SOURCE_PATH
    except validation_error_type as exc:
        if not _uses_legacy_shell_mode_fallback(exc.errors):
            raise

        config = _parse_config_with_legacy_shell_mode_fallback(
            prompt_generator,
            config_path,
            repo_root_override,
        )
        selected_variant = prompt_generator.choose_variant(config, variant_name)
        prompt_text = _render_compact_prompt_text(
            prompt_generator,
            config,
            selected_variant,
        )
        fallback_note = (
            "Legacy prompt-schema fallback applied while generating this combined "
            "artifact. The preserved prompt config predates the current explicit "
            "shell_section mode requirement for template-backed variants, so missing "
            "shell-section modes were interpreted as replacement to reconstruct "
            "the historical compact prompt shell and artifact manifest."
        )
        return (
            prompt_text,
            selected_variant,
            [fallback_note],
            LEGACY_FALLBACK_PROMPT_SOURCE_PATH,
        )


def build_combined_record(
    *,
    run_id: str,
    variant_id: str,
    prompt_text: str,
    answer_text: str,
    answer_source_path: str,
    notes: list[str],
    layout: str = DEFAULT_LAYOUT,
    prompt_source_path: str = DEFAULT_PROMPT_SOURCE_PATH,
) -> str:
    metadata_lines = ["  <metadata>"]
    for note in notes:
        metadata_lines.append(f"    <notes>{note}</notes>")
    metadata_lines.append(f"    <layout>{layout}</layout>")
    metadata_lines.append("  </metadata>")

    prompt_cdata = _wrap_cdata(prompt_text)
    answer_cdata = _wrap_cdata(answer_text)

    return "\n".join(
        [
            f'<combined_run_record run_id="{run_id}" variant_id="{variant_id}" schema_version="{COMBINED_SCHEMA_VERSION}">',
            *metadata_lines,
            "",
            "  <prompt_section>",
            f'  <prompt_text status="preserved" source_path="{prompt_source_path}">',
            f"  <![CDATA[{prompt_cdata}]]>",
            "  </prompt_text>",
            "  </prompt_section>",
            "",
            "  <answer_section>",
            f'  <answer_text status="preserved" source_path="{answer_source_path}">',
            f"  <![CDATA[{answer_cdata}]]>",
            "  </answer_text>",
            "  </answer_section>",
            "</combined_run_record>",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Hydra combined archive artifact from an answer markdown file "
            "and a prompt config JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--answer", required=True, help="Path to the agent answer markdown"
    )
    parser.add_argument(
        "--config", required=True, help="Path to the prompt config JSON"
    )
    parser.add_argument(
        "--variant", help="Variant name to render via generate_prompt.py"
    )
    parser.add_argument(
        "--agent-number",
        help="Agent number for output naming; inferred from answer filename when omitted",
    )
    parser.add_argument(
        "--output", help="Explicit output path for the combined artifact"
    )
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Additional metadata <notes> entry to include; can be passed multiple times",
    )
    parser.add_argument(
        "--prompt-source-path",
        default=DEFAULT_PROMPT_SOURCE_PATH,
        help="source_path attribute for <prompt_text>",
    )
    parser.add_argument(
        "--layout",
        default=DEFAULT_LAYOUT,
        help="layout metadata value",
    )
    return parser


def main(
    argv: list[str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    parser = build_parser()
    out: TextIO = sys.stdout if stdout is None else stdout
    err: TextIO = sys.stderr if stderr is None else stderr

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        return 1

    answer_path = Path(args.answer).resolve()
    config_path = Path(args.config).resolve()
    prompt_generator = _load_prompt_generator()
    prompt_error_type = getattr(prompt_generator, "PromptGeneratorError")

    try:
        if not answer_path.is_file():
            raise CombinedArtifactError(f"answer file not found: {answer_path}")
        if not config_path.is_file():
            raise CombinedArtifactError(f"config file not found: {config_path}")

        answer_text = answer_path.read_text(encoding="utf-8")
        if not answer_text.strip():
            raise CombinedArtifactError(f"answer file is empty: {answer_path}")

        prompt_text, selected_variant, fallback_notes, prompt_source_path = (
            _render_prompt_with_fallback(prompt_generator, config_path, args.variant)
        )

        agent_number = args.agent_number or _infer_agent_number(answer_path)
        run_id = f"answer_{agent_number}"
        output_path = (
            Path(args.output).resolve()
            if args.output
            else _default_output_path(Path(__file__).resolve().parents[1], agent_number)
        )
        notes = [_default_note(agent_number), *fallback_notes, *args.note]

        combined = build_combined_record(
            run_id=run_id,
            variant_id=selected_variant,
            prompt_text=prompt_text,
            answer_text=answer_text,
            answer_source_path=answer_path.name,
            notes=notes,
            layout=args.layout,
            prompt_source_path=(
                args.prompt_source_path
                if args.prompt_source_path != DEFAULT_PROMPT_SOURCE_PATH
                else prompt_source_path
            ),
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(combined, encoding="utf-8")
        out.write(f"generated combined artifact at {output_path}\n")
        return 0
    except getattr(prompt_generator, "ValidationError") as exc:
        for error in exc.errors:
            err.write(f"{error}\n")
        return 1
    except KeyError as exc:
        err.write(f"{exc}\n")
        return 1
    except CombinedArtifactError as exc:
        err.write(f"{exc}\n")
        return 1
    except Exception as exc:
        if isinstance(exc, prompt_error_type):
            err.write(f"{exc}\n")
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
