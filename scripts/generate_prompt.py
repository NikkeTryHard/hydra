#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


SCHEMA_VERSION = 1


class PromptGeneratorError(Exception):
    pass


class ValidationError(PromptGeneratorError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


class PromptTemplateParseError(PromptGeneratorError):
    pass


@dataclass(frozen=True)
class ShellSection:
    tag: str
    lines: list[str]


@dataclass(frozen=True)
class PromptTemplate:
    title: str | None
    shell_sections: list[ShellSection]
    artifact_container_tag: str


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str | None
    kind: str
    label: str | None
    explanation: str | None
    path: Path | None
    start_line: int | None
    end_line: int | None
    content: str | None
    fence_language: str
    source_label: str | None
    show_line_numbers: bool


@dataclass(frozen=True)
class DefaultsSpec:
    title: str | None
    shell_sections: list[ShellSection]
    artifact_ids: list[str]
    artifact_container_tag: str


@dataclass(frozen=True)
class VariantSpec:
    name: str
    title: str | None
    shell_source_path: Path | None
    shell_sections: list[ShellSection]
    artifact_ids: list[str]
    extra_artifacts: list[ArtifactSpec]
    output_file: str | None


@dataclass(frozen=True)
class PromptConfig:
    config_path: Path
    repo_root: Path
    defaults: DefaultsSpec
    artifacts: dict[str, ArtifactSpec]
    variants: list[VariantSpec]


def _config_error(config_path: Path, message: str) -> str:
    return f"{config_path}: {message}"


def _ensure_dict(
    value: Any, location: str, errors: list[str], config_path: Path
) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    errors.append(_config_error(config_path, f"{location}: expected object"))
    return {}


def _ensure_list(
    value: Any, location: str, errors: list[str], config_path: Path
) -> list[Any]:
    if isinstance(value, list):
        return value
    errors.append(_config_error(config_path, f"{location}: expected array"))
    return []


def _ensure_string(
    value: Any,
    location: str,
    errors: list[str],
    config_path: Path,
    *,
    allow_empty: bool = False,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        errors.append(_config_error(config_path, f"{location}: expected string"))
        return None
    if not allow_empty and value == "":
        errors.append(
            _config_error(config_path, f"{location}: string must not be empty")
        )
        return None
    return value


def _ensure_int(
    value: Any, location: str, errors: list[str], config_path: Path
) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    errors.append(_config_error(config_path, f"{location}: expected integer"))
    return None


def _resolve_repo_path(
    path_str: str | None,
    location: str,
    errors: list[str],
    config_path: Path,
    repo_root: Path,
) -> Path | None:
    if path_str is None:
        return None
    return (repo_root / path_str).resolve()


def _parse_shell_sections(
    raw_sections: Any,
    location: str,
    errors: list[str],
    config_path: Path,
) -> list[ShellSection]:
    sections: list[ShellSection] = []
    for index, raw_section in enumerate(
        _ensure_list(raw_sections, location, errors, config_path)
    ):
        section_obj = _ensure_dict(
            raw_section,
            f"{location}[{index}]",
            errors,
            config_path,
        )
        tag = _ensure_string(
            section_obj.get("tag"),
            f"{location}[{index}].tag",
            errors,
            config_path,
        )
        raw_lines = _ensure_list(
            section_obj.get("lines"),
            f"{location}[{index}].lines",
            errors,
            config_path,
        )
        lines: list[str] = []
        for line_index, raw_line in enumerate(raw_lines):
            line = _ensure_string(
                raw_line,
                f"{location}[{index}].lines[{line_index}]",
                errors,
                config_path,
                allow_empty=True,
            )
            if line is not None:
                lines.append(line)
        if tag is not None:
            sections.append(ShellSection(tag=tag, lines=lines))
    return sections


def _parse_artifact(
    raw_artifact: Any,
    location: str,
    errors: list[str],
    config_path: Path,
    repo_root: Path,
    *,
    artifact_id_required: bool,
) -> ArtifactSpec | None:
    artifact_obj = _ensure_dict(raw_artifact, location, errors, config_path)
    artifact_id = _ensure_string(
        artifact_obj.get("id"),
        f"{location}.id",
        errors,
        config_path,
    )
    if not artifact_id_required:
        artifact_id = (
            artifact_obj.get("id") if isinstance(artifact_obj.get("id"), str) else None
        )
    kind = _ensure_string(
        artifact_obj.get("type"),
        f"{location}.type",
        errors,
        config_path,
    )
    label = (
        _ensure_string(
            artifact_obj.get("label"),
            f"{location}.label",
            errors,
            config_path,
            allow_empty=False,
        )
        if "label" in artifact_obj
        else None
    )
    explanation = (
        _ensure_string(
            artifact_obj.get("explanation"),
            f"{location}.explanation",
            errors,
            config_path,
            allow_empty=False,
        )
        if "explanation" in artifact_obj
        else None
    )
    source_label = (
        _ensure_string(
            artifact_obj.get("source_label"),
            f"{location}.source_label",
            errors,
            config_path,
            allow_empty=False,
        )
        if "source_label" in artifact_obj
        else None
    )

    fence_language = artifact_obj.get("fence_language", "text")
    if not isinstance(fence_language, str):
        errors.append(
            _config_error(config_path, f"{location}.fence_language: expected string")
        )
        fence_language = "text"

    show_line_numbers = artifact_obj.get("show_line_numbers", True)
    if not isinstance(show_line_numbers, bool):
        errors.append(
            _config_error(
                config_path, f"{location}.show_line_numbers: expected boolean"
            )
        )
        show_line_numbers = True

    path: Path | None = None
    start_line: int | None = None
    end_line: int | None = None
    content: str | None = None

    if kind == "file_range":
        path_str = _ensure_string(
            artifact_obj.get("path"),
            f"{location}.path",
            errors,
            config_path,
        )
        start_line = _ensure_int(
            artifact_obj.get("start_line"),
            f"{location}.start_line",
            errors,
            config_path,
        )
        end_line = _ensure_int(
            artifact_obj.get("end_line"),
            f"{location}.end_line",
            errors,
            config_path,
        )
        path = _resolve_repo_path(
            path_str, f"{location}.path", errors, config_path, repo_root
        )
    elif kind == "file_full":
        path_str = _ensure_string(
            artifact_obj.get("path"),
            f"{location}.path",
            errors,
            config_path,
        )
        path = _resolve_repo_path(
            path_str, f"{location}.path", errors, config_path, repo_root
        )
    elif kind == "literal":
        if "content_lines" in artifact_obj:
            raw_content_lines = _ensure_list(
                artifact_obj.get("content_lines"),
                f"{location}.content_lines",
                errors,
                config_path,
            )
            content_lines: list[str] = []
            for line_index, raw_line in enumerate(raw_content_lines):
                line = _ensure_string(
                    raw_line,
                    f"{location}.content_lines[{line_index}]",
                    errors,
                    config_path,
                    allow_empty=True,
                )
                if line is not None:
                    content_lines.append(line)
            content = "\n".join(content_lines)
        else:
            content = _ensure_string(
                artifact_obj.get("content"),
                f"{location}.content",
                errors,
                config_path,
                allow_empty=True,
            )
    else:
        if kind is not None:
            errors.append(
                _config_error(
                    config_path,
                    f"{location}.type: unsupported artifact type {kind!r} (expected file_range, file_full, or literal)",
                )
            )

    return ArtifactSpec(
        artifact_id=artifact_id,
        kind=kind or "",
        label=label,
        explanation=explanation,
        path=path,
        start_line=start_line,
        end_line=end_line,
        content=content,
        fence_language=fence_language,
        source_label=source_label,
        show_line_numbers=show_line_numbers,
    )


def _parse_config(
    config_path: Path, repo_root_override: Path | None = None
) -> PromptConfig:
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(
            [_config_error(config_path, f"invalid JSON: {exc}")]
        ) from exc

    errors: list[str] = []
    root = _ensure_dict(raw, "<root>", errors, config_path)

    version = root.get("version")
    if version != SCHEMA_VERSION:
        errors.append(
            _config_error(
                config_path,
                f"version: expected {SCHEMA_VERSION}, got {version!r}",
            )
        )

    repo_root_value = root.get("repo_root", ".")
    repo_root_str = _ensure_string(
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

    defaults_obj = _ensure_dict(
        root.get("defaults", {}), "defaults", errors, config_path
    )
    default_title = (
        _ensure_string(
            defaults_obj.get("title"),
            "defaults.title",
            errors,
            config_path,
        )
        if "title" in defaults_obj
        else None
    )
    default_shell_sections = _parse_shell_sections(
        defaults_obj.get("shell_sections", []),
        "defaults.shell_sections",
        errors,
        config_path,
    )
    default_artifact_ids_raw = _ensure_list(
        defaults_obj.get("artifact_ids", []),
        "defaults.artifact_ids",
        errors,
        config_path,
    )
    default_artifact_ids: list[str] = []
    for index, artifact_id in enumerate(default_artifact_ids_raw):
        value = _ensure_string(
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
            _config_error(
                config_path,
                "defaults.artifact_container_tag: expected non-empty string",
            )
        )
        artifact_container_tag = "artifacts"

    artifacts_raw = _ensure_list(
        root.get("artifacts", []), "artifacts", errors, config_path
    )
    artifacts: dict[str, ArtifactSpec] = {}
    for index, raw_artifact in enumerate(artifacts_raw):
        artifact = _parse_artifact(
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
                _config_error(
                    config_path,
                    f"artifacts[{index}].id: duplicate artifact id {artifact.artifact_id!r}",
                )
            )
            continue
        artifacts[artifact.artifact_id] = artifact

    variants_raw = _ensure_list(root.get("variants"), "variants", errors, config_path)
    variants: list[VariantSpec] = []
    variant_names: set[str] = set()
    for index, raw_variant in enumerate(variants_raw):
        variant_obj = _ensure_dict(
            raw_variant, f"variants[{index}]", errors, config_path
        )
        name = _ensure_string(
            variant_obj.get("name"),
            f"variants[{index}].name",
            errors,
            config_path,
        )
        if name is not None and name in variant_names:
            errors.append(
                _config_error(
                    config_path,
                    f"variants[{index}].name: duplicate variant name {name!r}",
                )
            )
        elif name is not None:
            variant_names.add(name)
        title = (
            _ensure_string(
                variant_obj.get("title"),
                f"variants[{index}].title",
                errors,
                config_path,
            )
            if "title" in variant_obj
            else None
        )
        shell_source_path = (
            _resolve_repo_path(
                _ensure_string(
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
        shell_sections = _parse_shell_sections(
            variant_obj.get("shell_sections", []),
            f"variants[{index}].shell_sections",
            errors,
            config_path,
        )
        artifact_ids_raw = _ensure_list(
            variant_obj.get("artifact_ids", []),
            f"variants[{index}].artifact_ids",
            errors,
            config_path,
        )
        artifact_ids: list[str] = []
        for artifact_index, artifact_id in enumerate(artifact_ids_raw):
            value = _ensure_string(
                artifact_id,
                f"variants[{index}].artifact_ids[{artifact_index}]",
                errors,
                config_path,
            )
            if value is not None:
                artifact_ids.append(value)

        extra_artifacts_raw = _ensure_list(
            variant_obj.get("artifacts", []),
            f"variants[{index}].artifacts",
            errors,
            config_path,
        )
        extra_artifacts: list[ArtifactSpec] = []
        for artifact_index, raw_artifact in enumerate(extra_artifacts_raw):
            artifact = _parse_artifact(
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
            _ensure_string(
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
                VariantSpec(
                    name=name,
                    title=title,
                    shell_source_path=shell_source_path,
                    shell_sections=shell_sections,
                    artifact_ids=artifact_ids,
                    extra_artifacts=extra_artifacts,
                    output_file=output_file,
                )
            )

    config = PromptConfig(
        config_path=config_path,
        repo_root=repo_root,
        defaults=DefaultsSpec(
            title=default_title,
            shell_sections=default_shell_sections,
            artifact_ids=default_artifact_ids,
            artifact_container_tag=artifact_container_tag,
        ),
        artifacts=artifacts,
        variants=variants,
    )
    errors.extend(_validate_config(config))
    if errors:
        raise ValidationError(errors)
    return config


def _validate_config(config: PromptConfig) -> list[str]:
    errors: list[str] = []
    if not config.variants:
        errors.append(
            _config_error(config.config_path, "variants: expected at least one variant")
        )

    repo_root = config.repo_root.resolve()
    default_ids = config.defaults.artifact_ids
    for index, artifact_id in enumerate(default_ids):
        if artifact_id not in config.artifacts:
            errors.append(
                _config_error(
                    config.config_path,
                    f"defaults.artifact_ids[{index}]: unknown artifact id {artifact_id!r}",
                )
            )

    for artifact_id, artifact in config.artifacts.items():
        errors.extend(
            _validate_artifact(
                config.config_path, repo_root, f"artifacts[{artifact_id!r}]", artifact
            )
        )

    for variant_index, variant in enumerate(config.variants):
        for artifact_index, artifact_id in enumerate(variant.artifact_ids):
            if artifact_id not in config.artifacts:
                errors.append(
                    _config_error(
                        config.config_path,
                        f"variants[{variant_index}].artifact_ids[{artifact_index}]: unknown artifact id {artifact_id!r}",
                    )
                )
        for artifact_index, artifact in enumerate(variant.extra_artifacts):
            errors.extend(
                _validate_artifact(
                    config.config_path,
                    repo_root,
                    f"variants[{variant_index}].artifacts[{artifact_index}]",
                    artifact,
                )
            )
        if variant.shell_source_path is not None:
            errors.extend(
                _validate_shell_source(
                    config.config_path,
                    repo_root,
                    f"variants[{variant_index}].shell_source_path",
                    variant.shell_source_path,
                )
            )
    return errors


def _validate_artifact(
    config_path: Path,
    repo_root: Path,
    location: str,
    artifact: ArtifactSpec,
) -> list[str]:
    errors: list[str] = []
    if artifact.kind in {"file_range", "file_full"}:
        if artifact.path is None:
            errors.append(
                _config_error(config_path, f"{location}: file artifact missing path")
            )
            return errors
        try:
            artifact.path.relative_to(repo_root)
        except ValueError:
            errors.append(
                _config_error(
                    config_path,
                    f"{location}.path: resolved path {artifact.path} escapes repo root {repo_root}",
                )
            )
            return errors
        if not artifact.path.exists():
            errors.append(
                _config_error(
                    config_path,
                    f"{location}.path: file not found at {artifact.path}",
                )
            )
            return errors
        if not artifact.path.is_file():
            errors.append(
                _config_error(
                    config_path,
                    f"{location}.path: expected file, got {artifact.path}",
                )
            )
            return errors
        lines = artifact.path.read_text(encoding="utf-8").splitlines()
        if artifact.kind == "file_range":
            if artifact.start_line is None or artifact.end_line is None:
                errors.append(
                    _config_error(
                        config_path,
                        f"{location}: file_range requires start_line and end_line",
                    )
                )
                return errors
            if artifact.start_line < 1:
                errors.append(
                    _config_error(config_path, f"{location}.start_line: must be >= 1")
                )
            if artifact.end_line < artifact.start_line:
                errors.append(
                    _config_error(
                        config_path,
                        f"{location}.end_line: must be >= start_line ({artifact.start_line})",
                    )
                )
            if lines and artifact.end_line > len(lines):
                errors.append(
                    _config_error(
                        config_path,
                        f"{location}.end_line: {artifact.end_line} exceeds file length {len(lines)}",
                    )
                )
            if not lines:
                errors.append(
                    _config_error(
                        config_path,
                        f"{location}: file_range cannot target an empty file",
                    )
                )
    elif artifact.kind == "literal":
        if artifact.content is None:
            errors.append(
                _config_error(
                    config_path,
                    f"{location}: literal artifact requires content or content_lines",
                )
            )
    else:
        errors.append(
            _config_error(
                config_path, f"{location}: unsupported artifact kind {artifact.kind!r}"
            )
        )
    return errors


def _validate_shell_source(
    config_path: Path,
    repo_root: Path,
    location: str,
    shell_source_path: Path,
) -> list[str]:
    errors: list[str] = []
    try:
        shell_source_path.relative_to(repo_root)
    except ValueError:
        errors.append(
            _config_error(
                config_path,
                f"{location}: resolved path {shell_source_path} escapes repo root {repo_root}",
            )
        )
        return errors
    if not shell_source_path.exists():
        errors.append(
            _config_error(
                config_path,
                f"{location}: file not found at {shell_source_path}",
            )
        )
        return errors
    if not shell_source_path.is_file():
        errors.append(
            _config_error(
                config_path,
                f"{location}: expected file, got {shell_source_path}",
            )
        )
        return errors
    try:
        load_prompt_template(shell_source_path)
    except PromptTemplateParseError as exc:
        errors.append(_config_error(config_path, f"{location}: {exc}"))
    return errors


def extract_prompt_body(text: str) -> str:
    match = re.search(
        r"<prompt_text\b[^>]*>\s*<!\[CDATA\[(.*?)\]\]>\s*</prompt_text>",
        text,
        re.DOTALL,
    )
    if match:
        return match.group(1)
    return text


def parse_prompt_template(prompt_text: str) -> PromptTemplate:
    lines = prompt_text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)

    title: str | None = None
    if lines and lines[0].startswith("# "):
        title = lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)

    shell_sections: list[ShellSection] = []
    artifact_container_tag: str | None = None
    index = 0
    tag_pattern = re.compile(r"<([a-zA-Z0-9_]+)>")

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        match = tag_pattern.fullmatch(line)
        if match is None:
            raise PromptTemplateParseError(
                f"expected section tag while parsing prompt shell, got {line!r}"
            )
        tag = match.group(1)
        closing_tag = f"</{tag}>"
        index += 1
        block_lines: list[str] = []
        while index < len(lines) and lines[index].strip() != closing_tag:
            block_lines.append(lines[index])
            index += 1
        if index >= len(lines):
            raise PromptTemplateParseError(
                f"missing closing tag {closing_tag!r} while parsing prompt shell"
            )
        if artifact_container_tag is not None:
            raise PromptTemplateParseError(
                f"found section <{tag}> after artifact container <{artifact_container_tag}>"
            )
        if tag == "artifacts":
            artifact_container_tag = tag
        else:
            shell_sections.append(ShellSection(tag=tag, lines=block_lines))
        index += 1

    if artifact_container_tag is None:
        raise PromptTemplateParseError(
            "prompt shell is missing an <artifacts> container"
        )

    return PromptTemplate(
        title=title,
        shell_sections=shell_sections,
        artifact_container_tag=artifact_container_tag,
    )


def load_prompt_template(path: Path) -> PromptTemplate:
    return parse_prompt_template(extract_prompt_body(path.read_text(encoding="utf-8")))


def merge_shell_sections(
    default_sections: list[ShellSection], override_sections: list[ShellSection]
) -> list[ShellSection]:
    merged: list[ShellSection] = [
        ShellSection(tag=section.tag, lines=list(section.lines))
        for section in default_sections
    ]
    index_by_tag = {section.tag: index for index, section in enumerate(merged)}
    for section in override_sections:
        replacement = ShellSection(tag=section.tag, lines=list(section.lines))
        if section.tag in index_by_tag:
            merged[index_by_tag[section.tag]] = replacement
        else:
            index_by_tag[section.tag] = len(merged)
            merged.append(replacement)
    return merged


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "prompt"


def render_prompt(config: PromptConfig, variant_name: str) -> str:
    variant = get_variant(config, variant_name)
    template = (
        load_prompt_template(variant.shell_source_path)
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
    shell_sections = merge_shell_sections(base_shell_sections, variant.shell_sections)
    artifact_specs = resolve_variant_artifacts(config, variant)

    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")

    for section in shell_sections:
        lines.append(f"<{section.tag}>")
        lines.extend(section.lines)
        lines.append(f"</{section.tag}>")
        lines.append("")

    container_tag = (
        template.artifact_container_tag
        if template is not None
        else config.defaults.artifact_container_tag
    )
    lines.append(f"<{container_tag}>")
    lines.append("")
    for index, artifact in enumerate(artifact_specs, start=1):
        lines.extend(render_artifact_block(config.repo_root, artifact, index))
    lines.append(f"</{container_tag}>")
    lines.append("")
    return "\n".join(lines)


def get_variant(config: PromptConfig, variant_name: str) -> VariantSpec:
    for variant in config.variants:
        if variant.name == variant_name:
            return variant
    raise KeyError(f"unknown variant {variant_name!r}")


def resolve_variant_artifacts(
    config: PromptConfig, variant: VariantSpec
) -> list[ArtifactSpec]:
    artifacts: list[ArtifactSpec] = []
    for artifact_id in config.defaults.artifact_ids:
        artifacts.append(config.artifacts[artifact_id])
    for artifact_id in variant.artifact_ids:
        artifacts.append(config.artifacts[artifact_id])
    artifacts.extend(variant.extra_artifacts)
    return artifacts


def render_artifact_block(
    repo_root: Path, artifact: ArtifactSpec, index: int
) -> list[str]:
    heading = (
        artifact.label
        or artifact.artifact_id
        or artifact.source_label
        or artifact_path_display(repo_root, artifact)
    )
    lines = [f"## Artifact {index:02d} — {heading}"]
    if artifact.artifact_id:
        lines.append(f"Artifact id: `{artifact.artifact_id}`")
    if artifact.source_label:
        lines.append(f"Source label: {artifact.source_label}")
    lines.append(f"Type: `{artifact.kind}`")
    source_text = artifact_source_text(repo_root, artifact)
    if source_text:
        lines.append(f"Source: {source_text}")
    if artifact.explanation:
        lines.append(f"Why it matters: {artifact.explanation}")
    lines.append("")
    lines.append(f"```{artifact.fence_language}")
    lines.extend(render_artifact_content(repo_root, artifact))
    lines.append("```")
    lines.append("")
    return lines


def artifact_path_display(repo_root: Path, artifact: ArtifactSpec) -> str:
    if artifact.path is None:
        return "literal"
    return str(artifact.path.relative_to(repo_root))


def artifact_source_text(repo_root: Path, artifact: ArtifactSpec) -> str | None:
    if artifact.kind == "literal":
        return None
    if artifact.path is None:
        return None
    relative_path = artifact.path.relative_to(repo_root)
    if artifact.kind == "file_range":
        return f"`{relative_path}:{artifact.start_line}-{artifact.end_line}`"
    return f"`{relative_path}`"


def render_artifact_content(repo_root: Path, artifact: ArtifactSpec) -> list[str]:
    if artifact.kind == "literal":
        return (artifact.content or "").splitlines()
    if artifact.path is None:
        return []
    file_lines = artifact.path.read_text(encoding="utf-8").splitlines()
    if artifact.kind == "file_range":
        assert artifact.start_line is not None
        assert artifact.end_line is not None
        selected = file_lines[artifact.start_line - 1 : artifact.end_line]
        start_number = artifact.start_line
    else:
        selected = file_lines
        start_number = 1
    if not artifact.show_line_numbers:
        return selected
    prefix_label = (
        artifact.source_label or artifact.artifact_id or artifact.path.stem.upper()
    )
    return [
        f"[{prefix_label} L{line_number:04d}] {line}"
        for line_number, line in enumerate(selected, start=start_number)
    ]


def list_variant_names(config: PromptConfig) -> list[str]:
    return [variant.name for variant in config.variants]


def choose_variant(config: PromptConfig, variant_name: str | None) -> str:
    if variant_name:
        return variant_name
    if len(config.variants) == 1:
        return config.variants[0].name
    available = ", ".join(list_variant_names(config))
    raise ValidationError(
        [
            _config_error(
                config.config_path,
                f"multiple variants available; pass --variant NAME (choices: {available})",
            )
        ]
    )


def generate_all_variants(config: PromptConfig) -> dict[str, str]:
    return {
        variant.name: render_prompt(config, variant.name) for variant in config.variants
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Hydra-style artifact-first prompts from JSON configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to prompt config JSON")
    parser.add_argument("--variant", help="Variant name to generate")
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Generate every variant in the config",
    )
    parser.add_argument(
        "--output", help="Output file path for a single generated variant"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory when generating all variants",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="List variant names and exit",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config and exit without generating prompt text",
    )
    parser.add_argument(
        "--repo-root",
        help="Override repo_root from config",
    )
    return parser


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
        if code is None:
            return 0
        return 1

    if args.all_variants and args.output:
        err.write("--output cannot be combined with --all-variants\n")
        return 1
    if args.all_variants and not args.output_dir and not args.validate_only:
        err.write("--all-variants requires --output-dir unless using --validate-only\n")
        return 1
    if args.output_dir and not args.all_variants and not args.validate_only:
        err.write("--output-dir is only valid with --all-variants\n")
        return 1

    config_path = Path(args.config).resolve()
    repo_root_override = Path(args.repo_root).resolve() if args.repo_root else None

    try:
        config = _parse_config(config_path, repo_root_override)
        if args.list_variants:
            for name in list_variant_names(config):
                out.write(f"{name}\n")
            return 0
        if args.validate_only:
            out.write("config is valid\n")
            return 0

        if args.all_variants:
            output_dir = Path(args.output_dir).resolve()
            rendered = generate_all_variants(config)
            for variant in config.variants:
                output_name = variant.output_file or f"{slugify(variant.name)}.md"
                _write_text(output_dir / output_name, rendered[variant.name])
            out.write(f"generated {len(rendered)} prompt(s) in {output_dir}\n")
            return 0

        selected_variant = choose_variant(config, args.variant)
        rendered = render_prompt(config, selected_variant)
        if args.output:
            output_path = Path(args.output).resolve()
            _write_text(output_path, rendered)
            out.write(f"generated prompt at {output_path}\n")
        else:
            out.write(rendered)
            if not rendered.endswith("\n"):
                out.write("\n")
        return 0
    except ValidationError as exc:
        for error in exc.errors:
            err.write(f"{error}\n")
        return 1
    except KeyError as exc:
        err.write(f"{exc}\n")
        return 1
    except PromptGeneratorError as exc:
        err.write(f"{exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
