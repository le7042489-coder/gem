from __future__ import annotations

from typing import Any, Dict, List, Tuple


SECTION_KEYS = ("rhythm", "conduction", "st_t", "axis")


def _ensure_section(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {"text": "", "evidence_ids": []}
    text = str(value.get("text", "") or "")
    evidence_ids = value.get("evidence_ids", [])
    if not isinstance(evidence_ids, list):
        evidence_ids = []
    return {"text": text, "evidence_ids": [str(x) for x in evidence_ids if isinstance(x, (str, int))]}


def validate_predict_plus_output(
    structured: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    max_evidence: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    max_evidence = max(1, int(max_evidence))

    # Normalize evidence list and enforce max count.
    normalized_evidence: List[Dict[str, Any]] = []
    seen = set()
    for item in evidence:
        if not isinstance(item, dict):
            continue
        ev_id = str(item.get("id", "")).strip()
        if not ev_id:
            ev_id = f"ev_{len(normalized_evidence) + 1}"
        if ev_id in seen:
            warnings.append(f"Duplicate evidence id removed: {ev_id}")
            continue
        seen.add(ev_id)
        obj = dict(item)
        obj["id"] = ev_id
        normalized_evidence.append(obj)
        if len(normalized_evidence) >= max_evidence:
            warnings.append(f"Evidence truncated to GEM_PLUS_MAX_EVIDENCE={max_evidence}.")
            break

    valid_ids = {x["id"] for x in normalized_evidence}

    # Normalize structured report.
    structured = structured if isinstance(structured, dict) else {}
    out_structured: Dict[str, Any] = {}
    for key in SECTION_KEYS:
        sec = _ensure_section(structured.get(key))
        raw_ids = list(sec["evidence_ids"])
        sec["evidence_ids"] = [x for x in raw_ids if x in valid_ids]
        removed = [x for x in raw_ids if x not in valid_ids]
        if removed:
            warnings.append(f"{key}: removed invalid evidence_ids {removed}.")
        out_structured[key] = sec

    summary = structured.get("summary", "")
    out_structured["summary"] = str(summary or "")

    return out_structured, normalized_evidence, warnings

