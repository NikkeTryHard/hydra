from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "generate_prompt.py"
SPEC = importlib.util.spec_from_file_location("generate_prompt", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"failed to load module from {MODULE_PATH}")
generate_prompt = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generate_prompt
SPEC.loader.exec_module(generate_prompt)


class PromptGeneratorTests(unittest.TestCase):
    def make_repo(self) -> Path:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        repo_root = Path(tempdir.name)
        (repo_root / "docs").mkdir(parents=True)
        (repo_root / "docs/example.md").write_text(
            "alpha\nbeta\ngamma\ndelta\n",
            encoding="utf-8",
        )
        return repo_root

    def write_config(self, repo_root: Path, payload: dict) -> Path:
        config_dir = repo_root / "configs"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "prompt.json"
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return config_path

    def load_config(self, config_path: Path):
        return generate_prompt._parse_config(config_path)

    def test_extract_prompt_body_from_wrapped_reference_example(self) -> None:
        wrapped = """<combined_run_record>
  <prompt_section>
  <prompt_text status=\"preserved\"><![CDATA[# Title

<role>
Body
</role>

<artifacts>
...
</artifacts>]]></prompt_text>
  </prompt_section>
</combined_run_record>
"""

        extracted = generate_prompt.extract_prompt_body(wrapped)

        self.assertTrue(extracted.startswith("# Title\n"))
        self.assertIn("<role>", extracted)
        self.assertIn("<artifacts>", extracted)

    def test_parse_prompt_template_reads_title_order_and_artifact_tag(self) -> None:
        template = generate_prompt.parse_prompt_template(
            """# Template title

<role>
Role body
</role>

<direction>
Direction body
</direction>

<style>
Style body
</style>

<artifact_note>
Artifact note
</artifact_note>

<artifacts>
...
</artifacts>
"""
        )

        self.assertEqual(template.title, "# Template title")
        self.assertEqual(
            [section.tag for section in template.shell_sections],
            ["role", "direction", "style", "artifact_note"],
        )
        self.assertEqual(template.artifact_container_tag, "artifacts")

    def test_render_prompt_uses_shell_source_order(self) -> None:
        repo_root = self.make_repo()
        canonical = repo_root / "docs/reference.md"
        canonical.write_text(
            """# Canonical title

<role>
Canonical role
</role>

<direction>
Canonical direction
</direction>

<style>
Canonical style
</style>

<artifact_note>
Canonical note
</artifact_note>

<artifacts>
...
</artifacts>
""",
            encoding="utf-8",
        )
        config_path = self.write_config(
            repo_root,
            {
                "version": 1,
                "repo_root": "..",
                "defaults": {
                    "title": "Fallback title",
                    "shell_sections": [
                        {"tag": "role", "lines": ["Default role"]},
                        {"tag": "style", "lines": ["Default style"]},
                    ],
                    "artifact_ids": [],
                },
                "artifacts": [],
                "variants": [
                    {
                        "name": "main",
                        "title": "Rendered title",
                        "shell_source_path": "docs/reference.md",
                        "shell_sections": [
                            {"tag": "direction", "lines": ["Variant direction"]},
                            {"tag": "style", "lines": ["Variant style"]},
                        ],
                        "artifact_ids": [],
                    }
                ],
            },
        )

        config = self.load_config(config_path)
        rendered = generate_prompt.render_prompt(config, "main")

        role_index = rendered.index("<role>")
        direction_index = rendered.index("<direction>")
        style_index = rendered.index("<style>")
        artifact_note_index = rendered.index("<artifact_note>")
        artifacts_index = rendered.index("<artifacts>")

        self.assertTrue(
            role_index
            < direction_index
            < style_index
            < artifact_note_index
            < artifacts_index
        )
        self.assertIn("Variant direction", rendered)
        self.assertIn("Variant style", rendered)
        self.assertIn("Canonical note", rendered)
        self.assertTrue(rendered.startswith("# Rendered title\n"))

    def test_validation_rejects_unparseable_shell_source(self) -> None:
        repo_root = self.make_repo()
        (repo_root / "docs/bad.md").write_text("not a prompt shell\n", encoding="utf-8")
        config_path = self.write_config(
            repo_root,
            {
                "version": 1,
                "repo_root": "..",
                "defaults": {
                    "shell_sections": [],
                    "artifact_ids": [],
                },
                "artifacts": [],
                "variants": [
                    {
                        "name": "main",
                        "shell_source_path": "docs/bad.md",
                        "shell_sections": [],
                        "artifact_ids": [],
                    }
                ],
            },
        )

        with self.assertRaises(generate_prompt.ValidationError) as ctx:
            self.load_config(config_path)

        self.assertTrue(
            any("shell_source_path" in error for error in ctx.exception.errors)
        )

    def test_render_file_range_includes_expected_lines(self) -> None:
        repo_root = self.make_repo()
        config_path = self.write_config(
            repo_root,
            {
                "version": 1,
                "repo_root": "..",
                "defaults": {
                    "shell_sections": [
                        {"tag": "role", "lines": ["Produce a blueprint."]},
                        {"tag": "direction", "lines": ["Use the artifacts below."]},
                    ],
                    "artifact_ids": ["slice"],
                },
                "artifacts": [
                    {
                        "id": "slice",
                        "type": "file_range",
                        "path": "docs/example.md",
                        "start_line": 2,
                        "end_line": 3,
                        "label": "Example slice",
                        "explanation": "Pull the middle lines.",
                        "source_label": "EX",
                    }
                ],
                "variants": [
                    {"name": "main", "shell_sections": [], "artifact_ids": []}
                ],
            },
        )

        config = self.load_config(config_path)
        rendered = generate_prompt.render_prompt(config, "main")

        self.assertIn("## Artifact 01 — Example slice", rendered)
        self.assertIn("Why it matters: Pull the middle lines.", rendered)
        self.assertIn("[EX L0002] beta", rendered)
        self.assertIn("[EX L0003] gamma", rendered)
        self.assertNotIn("[EX L0001] alpha", rendered)

    def test_render_literal_artifact(self) -> None:
        repo_root = self.make_repo()
        config_path = self.write_config(
            repo_root,
            {
                "version": 1,
                "repo_root": "..",
                "defaults": {
                    "shell_sections": [
                        {"tag": "role", "lines": ["Produce a blueprint."]},
                    ],
                    "artifact_ids": [],
                },
                "artifacts": [],
                "variants": [
                    {
                        "name": "main",
                        "shell_sections": [
                            {"tag": "direction", "lines": ["Focus on a narrow task."]}
                        ],
                        "artifact_ids": [],
                        "artifacts": [
                            {
                                "type": "literal",
                                "label": "Literal note",
                                "explanation": "Freeform author note.",
                                "content_lines": ["line one", "line two"],
                            }
                        ],
                    }
                ],
            },
        )

        config = self.load_config(config_path)
        rendered = generate_prompt.render_prompt(config, "main")

        self.assertIn("## Artifact 01 — Literal note", rendered)
        self.assertIn("line one", rendered)
        self.assertIn("line two", rendered)
        self.assertIn("Type: `literal`", rendered)

    def test_variant_shell_section_overrides_default_by_tag(self) -> None:
        repo_root = self.make_repo()
        config_path = self.write_config(
            repo_root,
            {
                "version": 1,
                "repo_root": "..",
                "defaults": {
                    "shell_sections": [
                        {"tag": "role", "lines": ["Produce a blueprint."]},
                        {"tag": "direction", "lines": ["Default direction."]},
                    ],
                    "artifact_ids": [],
                },
                "artifacts": [],
                "variants": [
                    {
                        "name": "main",
                        "shell_sections": [
                            {"tag": "direction", "lines": ["Variant direction."]},
                            {"tag": "style", "lines": ["- no vague answer"]},
                        ],
                        "artifact_ids": [],
                    }
                ],
            },
        )

        config = self.load_config(config_path)
        rendered = generate_prompt.render_prompt(config, "main")

        self.assertIn("Variant direction.", rendered)
        self.assertNotIn("Default direction.", rendered)
        self.assertIn("<style>", rendered)

    def test_validation_reports_missing_artifact_reference(self) -> None:
        repo_root = self.make_repo()
        config_path = self.write_config(
            repo_root,
            {
                "version": 1,
                "repo_root": "..",
                "defaults": {
                    "shell_sections": [],
                    "artifact_ids": ["missing"],
                },
                "artifacts": [],
                "variants": [
                    {"name": "main", "shell_sections": [], "artifact_ids": []}
                ],
            },
        )

        with self.assertRaises(generate_prompt.ValidationError) as ctx:
            self.load_config(config_path)

        self.assertTrue(
            any(
                "unknown artifact id 'missing'" in error
                for error in ctx.exception.errors
            )
        )

    def test_validation_reports_out_of_bounds_range(self) -> None:
        repo_root = self.make_repo()
        config_path = self.write_config(
            repo_root,
            {
                "version": 1,
                "repo_root": "..",
                "defaults": {
                    "shell_sections": [],
                    "artifact_ids": ["slice"],
                },
                "artifacts": [
                    {
                        "id": "slice",
                        "type": "file_range",
                        "path": "docs/example.md",
                        "start_line": 1,
                        "end_line": 99,
                    }
                ],
                "variants": [
                    {"name": "main", "shell_sections": [], "artifact_ids": []}
                ],
            },
        )

        with self.assertRaises(generate_prompt.ValidationError) as ctx:
            self.load_config(config_path)

        self.assertTrue(
            any("exceeds file length" in error for error in ctx.exception.errors)
        )

    def test_cli_list_variants(self) -> None:
        repo_root = self.make_repo()
        config_path = self.write_config(
            repo_root,
            {
                "version": 1,
                "repo_root": "..",
                "defaults": {
                    "shell_sections": [],
                    "artifact_ids": [],
                },
                "artifacts": [],
                "variants": [
                    {"name": "a", "shell_sections": [], "artifact_ids": []},
                    {"name": "b", "shell_sections": [], "artifact_ids": []},
                ],
            },
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = generate_prompt.main(
            ["--config", str(config_path), "--list-variants"],
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(stdout.getvalue().splitlines(), ["a", "b"])

    def test_cli_validate_only_success(self) -> None:
        repo_root = self.make_repo()
        config_path = self.write_config(
            repo_root,
            {
                "version": 1,
                "repo_root": "..",
                "defaults": {
                    "shell_sections": [],
                    "artifact_ids": [],
                },
                "artifacts": [],
                "variants": [
                    {"name": "main", "shell_sections": [], "artifact_ids": []}
                ],
            },
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = generate_prompt.main(
            ["--config", str(config_path), "--validate-only"],
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(code, 0)
        self.assertIn("config is valid", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_cli_requires_variant_when_multiple_exist(self) -> None:
        repo_root = self.make_repo()
        config_path = self.write_config(
            repo_root,
            {
                "version": 1,
                "repo_root": "..",
                "defaults": {
                    "shell_sections": [],
                    "artifact_ids": [],
                },
                "artifacts": [],
                "variants": [
                    {"name": "a", "shell_sections": [], "artifact_ids": []},
                    {"name": "b", "shell_sections": [], "artifact_ids": []},
                ],
            },
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = generate_prompt.main(
            ["--config", str(config_path)], stdout=stdout, stderr=stderr
        )

        self.assertEqual(code, 1)
        self.assertIn("multiple variants available", stderr.getvalue())

    def test_cli_generate_all_variants_requires_output_dir(self) -> None:
        repo_root = self.make_repo()
        config_path = self.write_config(
            repo_root,
            {
                "version": 1,
                "repo_root": "..",
                "defaults": {
                    "shell_sections": [],
                    "artifact_ids": [],
                },
                "artifacts": [],
                "variants": [
                    {"name": "main", "shell_sections": [], "artifact_ids": []}
                ],
            },
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = generate_prompt.main(
            ["--config", str(config_path), "--all-variants"],
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(code, 1)
        self.assertIn("requires --output-dir", stderr.getvalue())

    def test_cli_generates_all_variants_to_directory(self) -> None:
        repo_root = self.make_repo()
        config_path = self.write_config(
            repo_root,
            {
                "version": 1,
                "repo_root": "..",
                "defaults": {
                    "shell_sections": [
                        {"tag": "role", "lines": ["Produce a blueprint."]}
                    ],
                    "artifact_ids": [],
                },
                "artifacts": [],
                "variants": [
                    {
                        "name": "first variant",
                        "output_file": "first.md",
                        "shell_sections": [{"tag": "direction", "lines": ["First."]}],
                        "artifact_ids": [],
                    },
                    {
                        "name": "second variant",
                        "shell_sections": [{"tag": "direction", "lines": ["Second."]}],
                        "artifact_ids": [],
                    },
                ],
            },
        )
        output_dir = repo_root / "generated"
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = generate_prompt.main(
            [
                "--config",
                str(config_path),
                "--all-variants",
                "--output-dir",
                str(output_dir),
            ],
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertTrue((output_dir / "first.md").exists())
        self.assertTrue((output_dir / "second-variant.md").exists())

    def test_example_config_declares_all_reference_variants(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        config = generate_prompt._parse_config(
            repo_root / "scripts/examples/prompt_config.example.json"
        )

        self.assertEqual(
            [variant.name for variant in config.variants],
            ["narrow-focused", "broad-fuse-loop", "balanced-narrow"],
        )
        self.assertTrue(
            all(variant.shell_source_path is not None for variant in config.variants)
        )


if __name__ == "__main__":
    unittest.main()
