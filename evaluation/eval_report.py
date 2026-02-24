#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - import availability depends on runtime env
    from openai import OpenAI as _OpenAIClient
except ModuleNotFoundError:  # pragma: no cover
    _OpenAIClient = None

try:
    from .prompts import report_eval_prompt
except ImportError:  # pragma: no cover
    from prompts import report_eval_prompt

OpenAI = _OpenAIClient


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


def _extract_qid(row: Mapping[str, Any]) -> Optional[str]:
    value = row.get("question_id") or row.get("id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_generated_report(row: Mapping[str, Any]) -> str:
    value = row.get("text")
    if isinstance(value, str):
        return value
    value = row.get("response")
    if isinstance(value, str):
        return value
    return ""


def _extract_ecg_tail(qid: str) -> str:
    return qid.split("-")[-1]


def _extract_json_text(raw: str) -> Optional[Dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    block = text[start : end + 1]
    try:
        obj = json.loads(block)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None
    return None


def _build_prompt(golden_report: str, generated_report: str) -> str:
    return (
        f"{report_eval_prompt}\n"
        f"[The Start of Ground Truth Report]\n{golden_report}\n[The End of Ground Truth Report]\n"
        f"[The Start of Generated Report]\n{generated_report}\n[The End of Generated Report]"
    )


def _request_once(client: OpenAI, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content
    return content or ""


def _request_with_retry(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int,
) -> str:
    last_error: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            return _request_once(client, model, prompt)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"OpenAI request failed after {max_retries} retries: {last_error}")


def _score_report_object(report_obj: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    dimension_scores: Dict[str, float] = {}
    for key, value in report_obj.items():
        if isinstance(value, dict) and "Score" in value:
            try:
                dimension_scores[key] = float(value["Score"]) * 10.0
            except (TypeError, ValueError):
                continue
    avg = float(mean(dimension_scores.values())) if dimension_scores else 0.0
    return avg, dimension_scores


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated ECG reports with OpenAI")
    parser.add_argument("--predictions-jsonl", required=True, help="Path to generated report JSONL")
    parser.add_argument("--gold-json", required=True, help="Path to gold report JSON file")
    parser.add_argument("--output-dir", required=True, help="Directory for per-id JSON and summary")
    parser.add_argument("--model", default=os.getenv("EVAL_MODEL", "gpt-4o-2024-08-06"))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--resume", action="store_true", help="Skip ids that already have output files")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing API key: set {args.api_key_env}")
    if OpenAI is None:
        raise RuntimeError("Missing dependency 'openai'. Install it with: pip install openai")

    predictions_path = Path(args.predictions_jsonl).expanduser().resolve()
    gold_path = Path(args.gold_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions_jsonl not found: {predictions_path}")
    if not gold_path.exists():
        raise FileNotFoundError(f"gold_json not found: {gold_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pred_rows = _load_jsonl(predictions_path)
    gold_rows = _load_json(gold_path)
    if not isinstance(gold_rows, list):
        raise ValueError("gold_json must contain a JSON list")

    gold_report_map: Dict[str, str] = {}
    for row in gold_rows:
        if not isinstance(row, dict):
            continue
        qid = _extract_qid(row)
        if not qid:
            continue
        tail = _extract_ecg_tail(qid)
        conversations = row.get("conversations", [])
        if isinstance(conversations, list) and conversations:
            last = conversations[-1]
            if isinstance(last, dict) and isinstance(last.get("value"), str):
                gold_report_map[tail] = last["value"]

    tasks: List[Tuple[str, str, str]] = []
    for row in pred_rows:
        qid = _extract_qid(row)
        if not qid:
            continue
        tail = _extract_ecg_tail(qid)
        generated = _extract_generated_report(row)
        golden = gold_report_map.get(tail)
        if not golden:
            continue
        output_file = output_dir / f"{tail}.json"
        if args.resume and output_file.exists():
            continue
        tasks.append((tail, generated, golden))

    client = OpenAI(api_key=api_key)

    def _worker(ecg_id: str, generated: str, golden: str) -> str:
        prompt = _build_prompt(golden, generated)
        return _request_with_retry(client, args.model, prompt, args.max_retries)

    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as pool:
        for ecg_id, generated, golden in tasks:
            futures[pool.submit(_worker, ecg_id, generated, golden)] = ecg_id

        for future in as_completed(futures):
            ecg_id = futures[future]
            raw_text = future.result()
            payload = {"id": ecg_id, "results": raw_text}
            with (output_dir / f"{ecg_id}.json").open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    per_id_scores: Dict[str, Any] = {}
    per_dimension: Dict[str, List[float]] = {}
    for path in sorted(output_dir.glob("*.json")):
        try:
            item = _load_json(path)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(item, dict):
            continue
        raw_text = item.get("results")
        ecg_id = str(item.get("id") or path.stem)
        if not isinstance(raw_text, str):
            continue
        parsed = _extract_json_text(raw_text)
        if not parsed:
            continue

        avg_score, dims = _score_report_object(parsed)
        per_id_scores[ecg_id] = {
            "average_score": avg_score,
            "dimensions": dims,
        }
        for key, value in dims.items():
            per_dimension.setdefault(key, []).append(value)

    summary = {
        "model": args.model,
        "predictions_jsonl": str(predictions_path),
        "gold_json": str(gold_path),
        "output_dir": str(output_dir),
        "num_predictions": len(pred_rows),
        "num_scored": len(per_id_scores),
        "average_score": float(
            mean([item["average_score"] for item in per_id_scores.values()])
        )
        if per_id_scores
        else 0.0,
        "dimension_means": {
            key: float(mean(values))
            for key, values in sorted(per_dimension.items())
            if values
        },
        "per_id_scores": per_id_scores,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2 if args.pretty else None)
        if args.pretty:
            f.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
