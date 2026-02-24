from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


class TestEvaluateEcgBench(unittest.TestCase):
    def test_multichoice_split(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pred_root = root / "pred"
            pred_root.mkdir(parents=True)

            gold_path = root / "mmmu-ecg.json"
            gold_path.write_text(
                json.dumps(
                    [
                        {"id": "q-1", "conversations": [{"value": "Q"}, {"value": "A"}]},
                        {"id": "q-2", "conversations": [{"value": "Q"}, {"value": "B"}]},
                    ]
                ),
                encoding="utf-8",
            )

            pred_path = pred_root / "mmmu-ecg-step-final.jsonl"
            pred_path.write_text(
                "\n".join(
                    [
                        json.dumps({"question_id": "q-1", "text": "A"}),
                        json.dumps({"question_id": "q-2", "text": "Answer: B"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            datasets_cfg = root / "datasets.yaml"
            datasets_cfg.write_text(
                yaml.safe_dump(
                    {
                        "datasets": {
                            "mmmu-ecg": {
                                "task": "multichoice",
                                "gold_file": str(gold_path),
                                "prediction_match": "mmmu-ecg",
                                "options": ["A", "B", "C", "D", "E", "F", "G", "H"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "evaluation/evaluate_ecgbench.py",
                    "--pred-root",
                    str(pred_root),
                    "--datasets-config",
                    str(datasets_cfg),
                    "--splits",
                    "mmmu-ecg",
                ],
                cwd="/home/yushanhui/workspace/GEM",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            payload = json.loads(proc.stdout)
            accuracy = payload["splits"]["mmmu-ecg"]["best"]["metrics"]["accuracy"]
            self.assertAlmostEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
