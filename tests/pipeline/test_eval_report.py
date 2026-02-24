from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation import eval_report


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]


class _FakeCompletions:
    def create(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        return _FakeCompletion(
            '{"Diagnosis":{"Score":8,"Explanation":"ok"},"Waveform":{"Score":7,"Explanation":"ok"},"Rhythm":{"Score":9,"Explanation":"ok"}}'
        )


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeOpenAI:
    def __init__(self, api_key: str) -> None:
        _ = api_key
        self.chat = _FakeChat()


class TestEvalReport(unittest.TestCase):
    def test_eval_report_with_mock_openai(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pred = root / "pred.jsonl"
            gold = root / "gold.json"
            out = root / "out"

            pred.write_text(
                json.dumps({"question_id": "ecg-1", "text": "generated report"}) + "\n",
                encoding="utf-8",
            )
            gold.write_text(
                json.dumps([{"id": "ecg-1", "conversations": [{"value": "q"}, {"value": "gold report"}]}]),
                encoding="utf-8",
            )

            with patch.object(eval_report, "OpenAI", _FakeOpenAI):
                os.environ["TEST_OPENAI_KEY"] = "dummy"
                rc = eval_report.main(
                    [
                        "--predictions-jsonl",
                        str(pred),
                        "--gold-json",
                        str(gold),
                        "--output-dir",
                        str(out),
                        "--model",
                        "fake-model",
                        "--api-key-env",
                        "TEST_OPENAI_KEY",
                    ]
                )
                self.assertEqual(rc, 0)

            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["num_scored"], 1)
            self.assertGreater(summary["average_score"], 0)


if __name__ == "__main__":
    unittest.main()
