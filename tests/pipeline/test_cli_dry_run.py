from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


class TestPipelineCliDryRun(unittest.TestCase):
    def test_all_subcommands_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "checkpoints" / "GEM-7B").mkdir(parents=True)
            (root / "data").mkdir(parents=True)
            (root / "eval_out").mkdir(parents=True)
            (root / "scripts").mkdir(parents=True)
            (root / "chunks").mkdir(parents=True)
            (root / "inputs").mkdir(parents=True)
            (root / "scores").mkdir(parents=True)

            (root / "data" / "mixed_train.json").write_text("[]", encoding="utf-8")
            (root / "scripts" / "zero2.json").write_text("{}", encoding="utf-8")
            (root / "question.json").write_text("[]", encoding="utf-8")
            (root / "chunks" / "chunk_0.json").write_text("[]", encoding="utf-8")
            (root / "pred.jsonl").write_text('{"question_id":"x-1","text":"ok"}\n', encoding="utf-8")
            (root / "gold.json").write_text(
                json.dumps([{"id": "x-1", "conversations": [{"value": "q"}, {"value": "a"}]}]),
                encoding="utf-8",
            )
            (root / "merge_input.json").write_text("[]", encoding="utf-8")
            (root / "template.txt").write_text("{{generated}} {{groundtruth}}", encoding="utf-8")
            (root / "datasets.yaml").write_text(
                yaml.safe_dump(
                    {
                        "datasets": {
                            "dummy": {
                                "task": "multichoice",
                                "gold_file": str(root / "question.json"),
                                "prediction_match": "dummy",
                                "options": ["A", "B", "C", "D"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            cfg = {
                "paths": {
                    "repo_root": str(root),
                    "ecg_tower": str(root / "ecg_tower.pt"),
                },
                "train": {
                    "model_name_or_path": str(root / "checkpoints" / "GEM-7B"),
                    "data_path": str(root / "data" / "mixed_train.json"),
                    "image_folder": str(root),
                    "ecg_folder": str(root),
                    "deepspeed_config": str(root / "scripts" / "zero2.json"),
                },
                "finetune": {
                    "model_name_or_path": str(root / "checkpoints" / "GEM-7B"),
                    "data_path": str(root / "data" / "mixed_train.json"),
                    "image_folder": str(root),
                    "ecg_folder": str(root),
                    "ecg_tower": str(root / "ecg_tower.pt"),
                    "deepspeed_config": str(root / "scripts" / "zero2.json"),
                },
                "evaluation": {
                    "datasets_config": str(root / "datasets.yaml"),
                    "ecgbench": {
                        "model_path": str(root / "checkpoints" / "GEM-7B"),
                        "question_file": str(root / "question.json"),
                        "answers_dir": str(root / "eval_out"),
                    },
                    "grounding": {
                        "model_path": str(root / "checkpoints" / "GEM-7B"),
                        "question_files": [str(root / "chunks" / "chunk_0.json")],
                        "answers_dir": str(root / "eval_out"),
                        "gpus": [0],
                    },
                    "score_ecgbench": {
                        "pred_root": str(root),
                        "output_json": str(root / "scores" / "ecgbench.json"),
                    },
                    "report_eval": {
                        "predictions_jsonl": str(root / "pred.jsonl"),
                        "gold_json": str(root / "gold.json"),
                        "output_dir": str(root / "scores" / "report"),
                    },
                },
                "grounding": {
                    "merge": {
                        "result_jsonl": str(root / "pred.jsonl"),
                        "test_file": str(root / "gold.json"),
                        "output_json": str(root / "scores" / "merged.json"),
                    },
                    "gpt_eval": {
                        "input_json": str(root / "merge_input.json"),
                        "template_file": str(root / "template.txt"),
                        "output_dir": str(root / "scores" / "gpt"),
                    },
                    "score": {
                        "input_dir": str(root / "scores"),
                        "output_json": str(root / "scores" / "summary.json"),
                        "per_id_json": str(root / "scores" / "per_id.json"),
                    },
                },
            }

            (root / "ecg_tower.pt").write_text("stub", encoding="utf-8")
            cfg_path = root / "pipeline.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

            commands = [
                "validate-config",
                "train",
                "finetune",
                "eval-generate-ecgbench",
                "eval-generate-grounding",
                "eval-score-ecgbench",
                "eval-score-report",
                "grounding-merge",
                "grounding-gpt-eval",
                "grounding-score",
            ]

            for cmd in commands:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "scripts/gem_pipeline.py",
                        cmd,
                        "--config",
                        str(cfg_path),
                        "--dry-run",
                    ],
                    cwd="/home/yushanhui/workspace/GEM",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, msg=f"{cmd} failed: {proc.stderr}")


if __name__ == "__main__":
    unittest.main()
