from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestNotebookEquivalence(unittest.TestCase):
    def test_process_gem_outputs_and_grounding_scores(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            result_jsonl = root / "result.jsonl"
            result_jsonl.write_text(
                json.dumps({"question_id": "id-1", "text": "model text", "model_id": "gem"}) + "\n",
                encoding="utf-8",
            )

            test_json = root / "test.json"
            test_json.write_text(
                json.dumps(
                    [
                        {
                            "id": "id-1",
                            "ecg": "a",
                            "image": "b",
                            "machine_measurements": {"x": 1},
                            "report": "r",
                            "conversations": [{"value": "q"}, {"value": "gold"}],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            merged_json = root / "merged.json"
            proc_merge = subprocess.run(
                [
                    sys.executable,
                    "gem_evaluation/process_gem_outputs.py",
                    "--result-jsonl",
                    str(result_jsonl),
                    "--test-file",
                    str(test_json),
                    "--output-json",
                    str(merged_json),
                ],
                cwd="/home/yushanhui/workspace/GEM",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(proc_merge.returncode, 0, msg=proc_merge.stderr)
            merged = json.loads(merged_json.read_text(encoding="utf-8"))
            self.assertEqual(merged[0]["GEM_generated"], "model text")

            score_dir = root / "score_dir"
            score_dir.mkdir(parents=True)
            (score_dir / "id-1.json").write_text(
                json.dumps(
                    {
                        "id": "id-1",
                        "results": json.dumps(
                            {
                                "DiagnosisAccuracy": [{"Score": 2, "Explanation": "ok"}],
                                "LeadAssessmentCoverage": [{"Score": 12, "Explanation": "ok"}],
                                "LeadAssessmentAccuracy": [{"Score": 24, "Explanation": "ok"}],
                                "GroundedECGUnderstanding": [{"Score": 80, "Explanation": "ok"}],
                                "EvidenceBasedReasoning": [{"Score": 70, "Explanation": "ok"}],
                                "RealisticDiagnosticProcess": [{"Score": 60, "Explanation": "ok"}],
                            }
                        ),
                    }
                ),
                encoding="utf-8",
            )

            summary_json = root / "summary.json"
            per_id_json = root / "per_id.json"
            proc_score = subprocess.run(
                [
                    sys.executable,
                    "gem_evaluation/process_grounding_scores.py",
                    "--input-dir",
                    str(score_dir),
                    "--output-json",
                    str(summary_json),
                    "--per-id-json",
                    str(per_id_json),
                ],
                cwd="/home/yushanhui/workspace/GEM",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(proc_score.returncode, 0, msg=proc_score.stderr)

            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["num_parsed"], 1)
            self.assertIn("DiagnosisAccuracy", summary["means"])


if __name__ == "__main__":
    unittest.main()
