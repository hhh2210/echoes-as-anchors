from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PackageContractTests(unittest.TestCase):
    def test_distribution_has_stable_package_and_cli_names(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(project["project"]["name"], "echoes-as-anchors")
        self.assertEqual(
            project["project"]["scripts"]["echoes-lm-eval"],
            "echoes_as_anchors.evaluation.two_stage_eval:main",
        )
        self.assertEqual(project["project"]["optional-dependencies"]["harness"], ["lm-eval==0.4.12"])
        self.assertEqual(
            project["tool"]["setuptools"]["packages"]["find"]["include"],
            ["echoes_as_anchors", "echoes_as_anchors.*"],
        )

    def test_base_package_import_does_not_require_gpu_dependencies(self) -> None:
        sys.path.insert(0, str(ROOT))
        try:
            from src.evaluation import two_stage_common
        finally:
            sys.path.pop(0)
        self.assertEqual(two_stage_common.DEFAULT_PROTOCOL_ID, "repo-standalone-v1-not-figure4")

    def test_release_builds_wheel_from_a_clean_source_archive(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("uv build --sdist --clear --out-dir dist", makefile)
        self.assertIn("uv build --wheel dist/*.tar.gz --out-dir dist", makefile)
        self.assertNotIn("uv build --wheel --out-dir dist", makefile)


if __name__ == "__main__":
    unittest.main()
