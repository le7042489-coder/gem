#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple


CANONICAL_KEYS = [
    "DiagnosisAccuracy",
    "AnalysisCompleteness",
    "AnalysisRelevance",
    "LeadAssessmentCoverage",
    "LeadAssessmentAccuracy",
    "GroundedECGUnderstanding",
    "EvidenceBasedReasoning",
    "RealisticDiagnosticProcess",
]

KEY_ALIASES = {
    "ECGFeatureGrounding": "GroundedECGUnderstanding",
    "ClinicalDiagnosticFidelity": "RealisticDiagnosticProcess",
}


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def _extract_json_dict(text: str) -> Optional[Dict[str, Any]]:
    cleaned = _strip_code_fence(text)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    block = cleaned[start : end + 1]
    try:
        obj = json.loads(block)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None
    return None


def _normalize_key(key: str) -> str:
    return KEY_ALIASES.get(key, key)


def _extract_scores(value: Any) -> List[float]:
    scores: List[float] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and "Score" in item:
                try:
                    scores.append(float(item["Score"]))
                except (TypeError, ValueError):
                    continue
            elif isinstance(item, (int, float)):
                scores.append(float(item))
    elif isinstance(value, dict) and "Score" in value:
        try:
            scores.append(float(value["Score"]))
        except (TypeError, ValueError):
            pass
    elif isinstance(value, (int, float)):
        scores.append(float(value))
    return scores


def _aggregate_metric(key: str, scores: List[float]) -> float:
    if not scores:
        return 0.0

    if key == "DiagnosisAccuracy":
        return (sum(1 for s in scores if s > 0) / len(scores)) * 100.0
    if key == "LeadAssessmentCoverage":
        return min(sum(scores), 12.0) / 12.0 * 100.0
    if key == "LeadAssessmentAccuracy":
        return sum(scores) / 24.0 * 100.0

    max_score = max(scores)
    if max_score <= 1.0:
        return float(mean(scores) * 100.0)
    if max_score <= 10.0:
        return float(sum(scores))
    return float(mean(scores))


def _parse_record(raw_result: str) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    parsed = _extract_json_dict(raw_result)
    if not parsed:
        return {}, {}

    metric_scores: Dict[str, float] = {}
    detail_scores: Dict[str, List[float]] = {}

    for key, value in parsed.items():
        canonical = _normalize_key(str(key))
        if canonical not in CANONICAL_KEYS:
            continue
        scores = _extract_scores(value)
        detail_scores[canonical] = scores
        metric_scores[canonical] = _aggregate_metric(canonical, scores)

    return metric_scores, detail_scores


def _read_result_files(input_dir: Path) -> Dict[str, str]:
    raw: Dict[str, str] = {}
    for path in sorted(input_dir.rglob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue

        if isinstance(obj, dict) and "id" in obj and "results" in obj:
            rid = str(obj["id"])
            if isinstance(obj["results"], str):
                raw[rid] = obj["results"]
            continue

        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    raw[str(key)] = value
                elif isinstance(value, dict) and isinstance(value.get("results"), str):
                    raw[str(key)] = value["results"]
    return raw


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process grounding GPT evaluation results")
    parser.add_argument("--input-dir", required=True, help="Directory containing per-id JSON score files")
    parser.add_argument("--output-json", required=True, help="Path to summary JSON")
    parser.add_argument("--per-id-json", required=True, help="Path to per-id normalized score JSON")
    parser.add_argument("--raw-merged-json", default="", help="Optional path to write merged raw id->result map")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    per_id_json = Path(args.per_id_json).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"input-dir not found: {input_dir}")

    raw_results = _read_result_files(input_dir)

    if args.raw_merged_json:
        merged_path = Path(args.raw_merged_json).expanduser().resolve()
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        with merged_path.open("w", encoding="utf-8") as f:
            json.dump(raw_results, f, ensure_ascii=False, indent=2 if args.pretty else None)
            if args.pretty:
                f.write("\n")

    per_id_scores: Dict[str, Dict[str, float]] = {}
    per_id_details: Dict[str, Dict[str, List[float]]] = {}

    parse_failed: List[str] = []
    for rid, raw in raw_results.items():
        metrics, detail = _parse_record(raw)
        if not metrics:
            parse_failed.append(rid)
            continue
        per_id_scores[rid] = metrics
        per_id_details[rid] = detail

    means: Dict[str, float] = {}
    for key in CANONICAL_KEYS:
        values = [m[key] for m in per_id_scores.values() if key in m]
        means[key] = float(mean(values)) if values else 0.0

    summary = {
        "input_dir": str(input_dir),
        "num_raw_results": len(raw_results),
        "num_parsed": len(per_id_scores),
        "num_failed": len(parse_failed),
        "failed_ids": parse_failed,
        "means": means,
    }

    per_id_payload = {
        "scores": per_id_scores,
        "details": per_id_details,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2 if args.pretty else None)
        if args.pretty:
            f.write("\n")

    per_id_json.parent.mkdir(parents=True, exist_ok=True)
    with per_id_json.open("w", encoding="utf-8") as f:
        json.dump(per_id_payload, f, ensure_ascii=False, indent=2 if args.pretty else None)
        if args.pretty:
            f.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
