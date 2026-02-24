from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.pipeline.config import load_effective_config, validate_required_paths, PathValidationError


class TestPipelineConfig(unittest.TestCase):
    def test_override_precedence_and_path_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg_path = root / "config.yaml"
            cfg_path.write_text(
                yaml.safe_dump(
                    {
                        "paths": {"repo_root": "."},
                        "train": {"model_name_or_path": "checkpoints/custom-model"},
                    }
                ),
                encoding="utf-8",
            )

            cfg = load_effective_config(cfg_path, ["train.model_name_or_path=checkpoints/override-model"])
            self.assertTrue(Path(cfg["train"]["model_name_or_path"]).is_absolute())
            self.assertTrue(str(cfg["train"]["model_name_or_path"]).endswith("checkpoints/override-model"))

    def test_required_path_validation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg_path = root / "config.yaml"
            cfg_path.write_text(yaml.safe_dump({"paths": {"repo_root": "."}}), encoding="utf-8")
            cfg = load_effective_config(cfg_path, ["train.model_name_or_path=/tmp/not-exist"])

            with self.assertRaises(PathValidationError):
                validate_required_paths(cfg, "train")


if __name__ == "__main__":
    unittest.main()
