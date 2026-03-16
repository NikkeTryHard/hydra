from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "generate_combined_archive_artifact.py"
)
SPEC = importlib.util.spec_from_file_location(
    "generate_combined_archive_artifact", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"failed to load module from {MODULE_PATH}")
generate_combined_archive_artifact = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generate_combined_archive_artifact
SPEC.loader.exec_module(generate_combined_archive_artifact)


class CombinedArchiveArtifactGeneratorTests(unittest.TestCase):
    def make_repo(self) -> Path:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        repo_root = Path(tempdir.name)
        (repo_root / "docs").mkdir(parents=True)
        (repo_root / "docs/reference.md").write_text(
            """# Template title

<role>
Canonical role
</role>

<artifacts>
...
</artifacts>
""",
            encoding="utf-8",
        )
        return repo_root

    def write_config(self, repo_root: Path, payload: dict) -> Path:
        config_dir = repo_root / "configs"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "prompt.json"
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return config_path

    def write_answer(self, repo_root: Path, name: str, content: str) -> Path:
        answer_path = repo_root / name
        answer_path.write_text(content, encoding="utf-8")
        return answer_path

    def test_main_generates_combined_artifact(self) -> None:
        repo_root = self.make_repo()
        config_path = self.write_config(
            repo_root,
            {
                "version": 1,
                "repo_root": "..",
                "defaults": {
                    "title": "Hydra prompt — example",
                    "artifact_container_tag": "artifacts",
                    "shell_sections": [
                        {"tag": "role", "lines": ["Produce the thing."]},
                    ],
                    "artifact_ids": ["task"],
                },
                "artifacts": [
                    {
                        "id": "task",
                        "type": "literal",
                        "label": "Task",
                        "fence_language": "text",
                        "content_lines": ["hello from artifact"],
                    }
                ],
                "variants": [{"name": "main", "artifact_ids": []}],
            },
        )
        answer_path = self.write_answer(
            repo_root, "agent_7.md", "# answer\nreal stuff\n"
        )
        output_path = repo_root / "out/answer_7_combined.md"
        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = generate_combined_archive_artifact.main(
            [
                "--answer",
                str(answer_path),
                "--config",
                str(config_path),
                "--output",
                str(output_path),
            ],
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        rendered = output_path.read_text(encoding="utf-8")
        self.assertIn(
            '<combined_run_record run_id="answer_7" variant_id="main"', rendered
        )
        self.assertIn(
            "<layout>single_markdown_file_prompt_shell_manifest_then_answer</layout>",
            rendered,
        )
        self.assertIn(
            '<prompt_text status="preserved" source_path="embedded_prompt_shell_and_manifest">',
            rendered,
        )
        self.assertIn(
            '<answer_text status="preserved" source_path="agent_7.md">', rendered
        )
        self.assertIn("# Hydra prompt — example", rendered)
        self.assertIn("<artifacts_manifest>", rendered)
        self.assertIn("## Artifact 01 — Task", rendered)
        self.assertIn("Artifact id: `task`", rendered)
        self.assertIn("Type: `literal`", rendered)
        self.assertNotIn("```text", rendered)
        self.assertNotIn("<config_section>", rendered)
        self.assertIn("# answer\nreal stuff", rendered)
        self.assertIn("generated combined artifact at", stdout.getvalue())

    def test_main_rejects_empty_answer(self) -> None:
        repo_root = self.make_repo()
        config_path = self.write_config(
            repo_root,
            {
                "version": 1,
                "repo_root": "..",
                "defaults": {
                    "title": "Hydra prompt — example",
                    "artifact_container_tag": "artifacts",
                    "shell_sections": [],
                    "artifact_ids": [],
                },
                "artifacts": [],
                "variants": [{"name": "main", "artifact_ids": []}],
            },
        )
        answer_path = self.write_answer(repo_root, "agent_8.md", "\n\n")
        stderr = io.StringIO()

        exit_code = generate_combined_archive_artifact.main(
            [
                "--answer",
                str(answer_path),
                "--config",
                str(config_path),
                "--output",
                str(repo_root / "out/answer_8_combined.md"),
            ],
            stdout=io.StringIO(),
            stderr=stderr,
        )

        self.assertEqual(exit_code, 1)
        self.assertIn("answer file is empty", stderr.getvalue())

    def test_main_requires_variant_when_generator_config_has_multiple(self) -> None:
        repo_root = self.make_repo()
        config_path = self.write_config(
            repo_root,
            {
                "version": 1,
                "repo_root": "..",
                "defaults": {
                    "title": "Hydra prompt — example",
                    "artifact_container_tag": "artifacts",
                    "shell_sections": [],
                    "artifact_ids": [],
                },
                "artifacts": [],
                "variants": [
                    {"name": "first", "artifact_ids": []},
                    {"name": "second", "artifact_ids": []},
                ],
            },
        )
        answer_path = self.write_answer(repo_root, "agent_9.md", "answer\n")
        stderr = io.StringIO()

        exit_code = generate_combined_archive_artifact.main(
            [
                "--answer",
                str(answer_path),
                "--config",
                str(config_path),
                "--output",
                str(repo_root / "out/answer_9_combined.md"),
            ],
            stdout=io.StringIO(),
            stderr=stderr,
        )

        self.assertEqual(exit_code, 1)
        self.assertIn("multiple variants available", stderr.getvalue())

    def test_main_uses_legacy_shell_mode_fallback_for_template_variants(self) -> None:
        repo_root = self.make_repo()
        config_path = self.write_config(
            repo_root,
            {
                "version": 1,
                "repo_root": "..",
                "defaults": {
                    "title": "Hydra prompt — legacy",
                    "artifact_container_tag": "artifacts",
                    "shell_sections": [],
                    "artifact_ids": ["task"],
                },
                "artifacts": [
                    {
                        "id": "task",
                        "type": "literal",
                        "label": "Task",
                        "fence_language": "text",
                        "content_lines": ["legacy artifact"],
                    }
                ],
                "variants": [
                    {
                        "name": "legacy",
                        "title": "Hydra prompt — legacy",
                        "shell_source_path": "docs/reference.md",
                        "artifact_ids": [],
                        "shell_sections": [
                            {
                                "tag": "role",
                                "lines": ["Legacy replacement role"],
                            }
                        ],
                    }
                ],
            },
        )
        answer_path = self.write_answer(repo_root, "agent_10.md", "legacy answer\n")
        output_path = repo_root / "out/answer_10_combined.md"

        exit_code = generate_combined_archive_artifact.main(
            [
                "--answer",
                str(answer_path),
                "--config",
                str(config_path),
                "--output",
                str(output_path),
            ],
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )

        self.assertEqual(exit_code, 0)
        rendered = output_path.read_text(encoding="utf-8")
        self.assertIn("Legacy replacement role", rendered)
        self.assertNotIn("Canonical role", rendered)
        self.assertIn(
            '<prompt_text status="preserved" source_path="embedded_prompt_shell_and_manifest_legacy_shell_fallback">',
            rendered,
        )
        self.assertIn(
            "Legacy prompt-schema fallback applied while generating this combined artifact.",
            rendered,
        )

    def test_main_still_rejects_nonlegacy_validation_errors(self) -> None:
        repo_root = self.make_repo()
        config_path = self.write_config(
            repo_root,
            {
                "version": 2,
                "repo_root": "..",
                "defaults": {
                    "title": "Hydra prompt — invalid",
                    "artifact_container_tag": "artifacts",
                    "shell_sections": [],
                    "artifact_ids": [],
                },
                "artifacts": [],
                "variants": [{"name": "main", "artifact_ids": []}],
            },
        )
        answer_path = self.write_answer(repo_root, "agent_11.md", "answer\n")
        stderr = io.StringIO()

        exit_code = generate_combined_archive_artifact.main(
            [
                "--answer",
                str(answer_path),
                "--config",
                str(config_path),
                "--output",
                str(repo_root / "out/answer_11_combined.md"),
            ],
            stdout=io.StringIO(),
            stderr=stderr,
        )

        self.assertEqual(exit_code, 1)
        self.assertIn("version: expected 1", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
