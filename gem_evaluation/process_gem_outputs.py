#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_prediction(row: Dict[str, Any], preferred_field: str) -> str:
    if preferred_field and isinstance(row.get(preferred_field), str):
        return row[preferred_field]
    for key in ("text", "response"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge GEM model outputs with grounding test metadata")
    parser.add_argument("--result-jsonl", required=True, help="Path to generated JSONL (e.g. model_ecg_resume output)")
    parser.add_argument("--test-file", required=True, help="Path to test JSON file")
    parser.add_argument("--output-json", required=True, help="Path to merged output JSON")
    parser.add_argument("--id-field", default="question_id", help="Primary id field in result JSONL")
    parser.add_argument("--prediction-field", default="text", help="Primary generated text field in result JSONL")
    parser.add_argument("--generated-key", default="GEM_generated", help="Output key for model generation")
    parser.add_argument("--groundtruth-key", default="GPT4o_generated", help="Output key for ground truth generation")
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing ids instead of failing")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    result_path = Path(args.result_jsonl).expanduser().resolve()
    test_path = Path(args.test_file).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()

    if not result_path.exists():
        raise FileNotFoundError(f"result-jsonl not found: {result_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"test-file not found: {test_path}")

    results = _load_jsonl(result_path)
    test_rows = _load_json(test_path)
    if not isinstance(test_rows, list):
        raise ValueError("test-file must contain a JSON list")

    result_map: Dict[str, Dict[str, Any]] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        row_id = row.get(args.id_field)
        if not isinstance(row_id, str) or not row_id.strip():
            row_id = row.get("question_id") or row.get("id")
        if isinstance(row_id, str) and row_id.strip():
            result_map[row_id.strip()] = row

    merged: List[Dict[str, Any]] = []
    missing_ids: List[str] = []

    for row in test_rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        if not isinstance(row_id, str):
            continue

        pred_row = result_map.get(row_id)
        if pred_row is None:
            missing_ids.append(row_id)
            if args.allow_missing:
                continue
            raise KeyError(f"Missing prediction for id: {row_id}")

        conversations = row.get("conversations", [])
        gold_value = ""
        if isinstance(conversations, list) and len(conversations) > 1:
            tail = conversations[1]
            if isinstance(tail, dict) and isinstance(tail.get("value"), str):
                gold_value = tail["value"]

        merged.append(
            {
                "id": row_id,
                "ecg": row.get("ecg"),
                "image": row.get("image"),
                args.generated_key: _extract_prediction(pred_row, args.prediction_field),
                args.groundtruth_key: gold_value,
                "machine_measurements": row.get("machine_measurements"),
                "report": row.get("report"),
                "model_id": pred_row.get("model_id"),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2 if args.pretty else None)
        if args.pretty:
            f.write("\n")

    print(
        json.dumps(
            {
                "result_jsonl": str(result_path),
                "test_file": str(test_path),
                "output_json": str(output_path),
                "merged_count": len(merged),
                "missing_count": len(missing_ids),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
