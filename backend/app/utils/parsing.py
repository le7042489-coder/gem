import re
from typing import List, Optional, Dict, Any

STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_INDEX = {lead: i for i, lead in enumerate(STANDARD_LEADS)}
SEGMENT_SECONDS = 2.5
TOTAL_SECONDS = 10.0
DEFAULT_DIAG_PROMPT = "Please interpret this ECG and provide a diagnosis."
CIRCLED_NUMBERS = [
    "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩",
    "⑪", "⑫", "⑬", "⑭", "⑮", "⑯", "⑰", "⑱", "⑲", "⑳"
]


def normalize_lead_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip()
    if not s:
        return None
    s_lower = s.lower()
    if s_lower == "avr":
        return "aVR"
    if s_lower == "avl":
        return "aVL"
    if s_lower == "avf":
        return "aVF"
    if s_lower in {"i", "ii", "iii"}:
        return s.upper()
    if re.match(r"^v[1-6]$", s_lower):
        return f"V{s_lower[1]}"
    return s


def expand_lead_field(lead_field: str) -> List[str]:
    if not lead_field:
        return []
    field = lead_field.replace("and", ",").replace("/", ",")
    parts = [p.strip() for p in field.split(",") if p.strip()]
    leads: List[str] = []
    roman_order = ["I", "II", "III"]
    for part in parts:
        match = re.match(r"^(V)([1-6])\s*-\s*V?([1-6])$", part, re.IGNORECASE)
        if match:
            start = int(match.group(2))
            end = int(match.group(3))
            step = 1 if start <= end else -1
            for i in range(start, end + step, step):
                leads.append(f"V{i}")
            continue
        match = re.match(r"^(I|II|III)\s*-\s*(I|II|III)$", part, re.IGNORECASE)
        if match:
            start = roman_order.index(match.group(1).upper())
            end = roman_order.index(match.group(2).upper())
            step = 1 if start <= end else -1
            for i in range(start, end + step, step):
                leads.append(roman_order[i])
            continue
        normalized = normalize_lead_name(part)
        if normalized:
            leads.append(normalized)

    seen = set()
    ordered = []
    for lead in leads:
        if lead not in seen:
            seen.add(lead)
            ordered.append(lead)
    return ordered


def parse_time_value(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower()
    if s in {"na", "n/a", "none", "unknown", "?"}:
        return None
    s = s.replace("sec", "").replace("s", "")
    try:
        return float(s)
    except ValueError:
        return None


def clean_findings(findings: List[Dict[str, Any]], duration: float = TOTAL_SECONDS) -> List[Dict[str, Any]]:
    cleaned = []
    for f in findings:
        lead = normalize_lead_name(f.get("lead"))
        if lead not in STANDARD_LEADS:
            continue
        symptom = (f.get("symptom") or "").strip()
        if not symptom:
            symptom = "Finding"
        start = parse_time_value(f.get("start"))
        end = parse_time_value(f.get("end"))
        if start is None or end is None:
            start = None
            end = None
        else:
            if start > end:
                start, end = end, start
            start = max(0.0, min(float(start), duration))
            end = max(0.0, min(float(end), duration))
        cleaned.append({"symptom": symptom, "lead": lead, "start": start, "end": end})

    unique = []
    seen = set()
    for f in cleaned:
        key = (f["symptom"].lower(), f["lead"], round(f["start"] or -1.0, 3), round(f["end"] or -1.0, 3))
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def parse_findings_output(text: str, default_symptom: Optional[str] = None) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.upper().startswith("FINDING|"):
            continue
        parts = line.split("|", 4)
        if len(parts) < 5:
            continue
        _, symptom, lead_field, start_s, end_s = parts
        symptom = symptom.strip() or (default_symptom or "Finding")
        leads = expand_lead_field(lead_field.strip())
        start = parse_time_value(start_s)
        end = parse_time_value(end_s)
        for lead in leads:
            findings.append({"symptom": symptom, "lead": lead, "start": start, "end": end})

    if findings:
        return clean_findings(findings)

    pattern_struct = (
        r"Symptom\s*[:\-]\s*(?P<symptom>[^\n;]+?)\s*"
        r"(?:;|,)?\s*Lead\s*[:\-]\s*(?P<lead>[^\n;]+?)\s*"
        r"(?:;|,)?\s*Time\s*[:\-]\s*(?P<start>\d+\.?\d*)\s*(?:s|sec)?\s*"
        r"(?:-|to)\s*(?P<end>\d+\.?\d*)"
    )
    for match in re.finditer(pattern_struct, text, re.IGNORECASE):
        symptom = match.group("symptom").strip()
        leads = expand_lead_field(match.group("lead").strip())
        start = parse_time_value(match.group("start"))
        end = parse_time_value(match.group("end"))
        for lead in leads:
            findings.append({"symptom": symptom, "lead": lead, "start": start, "end": end})

    if findings:
        return clean_findings(findings)

    pattern_time = r"(I|II|III|aVR|aVL|aVF|V[1-6])\D+?(\d+\.?\d*)\s*(?:s|sec)?\s*(?:-|to)\s*(\d+\.?\d*)"
    matches_time = re.findall(pattern_time, text, re.IGNORECASE)
    if matches_time:
        symptom = default_symptom or "Finding"
        for lead, start, end in matches_time:
            lead = normalize_lead_name(lead)
            findings.append({"symptom": symptom, "lead": lead, "start": start, "end": end})

    return clean_findings(findings)


def extract_diagnosis_summary(text: str) -> Optional[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if line.lower().startswith("diagnosis"):
            parts = line.split(":", 1)
            return parts[1].strip() if len(parts) > 1 else None
        if line.lower().startswith("impression"):
            parts = line.split(":", 1)
            return parts[1].strip() if len(parts) > 1 else None
    for line in lines:
        if not line.startswith("FINDING|") and not line.lower().startswith("findings"):
            return line
    return None


def format_time_range(start: Optional[float], end: Optional[float], duration: float = TOTAL_SECONDS) -> str:
    if start is None or end is None:
        return f"0.00-{duration:.2f}s (full segment)"
    return f"{start:.2f}-{end:.2f}s"


def format_diagnosis_report(summary: Optional[str], findings: List[Dict[str, Any]]) -> str:
    lines = []
    if summary:
        lines.append(f"Diagnosis: {summary}")
    else:
        lines.append("Diagnosis: (model did not provide a clear summary)")

    if not findings:
        lines.append("Findings: (no structured findings parsed)")
        return "\n".join(lines)

    symptom_map: Dict[str, List[Dict[str, Any]]] = {}
    for f in findings:
        symptom_map.setdefault(f["symptom"], []).append(f)

    lines.append("Findings:")
    lead_order = {lead: i for i, lead in enumerate(STANDARD_LEADS)}
    for symptom in sorted(symptom_map.keys()):
        entries = symptom_map[symptom]
        lead_groups: Dict[str, List[Any]] = {}
        for f in entries:
            lead_groups.setdefault(f["lead"], []).append((f["start"], f["end"]))
        lead_chunks = []
        for lead in sorted(lead_groups.keys(), key=lambda x: lead_order.get(x, 999)):
            ranges = sorted(lead_groups[lead], key=lambda t: (t[0] or -1, t[1] or -1))
            range_str = ", ".join(format_time_range(s, e) for s, e in ranges)
            lead_chunks.append(f"{lead} ({range_str})")
        lines.append(f"- {symptom}: " + "; ".join(lead_chunks))

    return "\n".join(lines)


def format_findings_only(findings: List[Dict[str, Any]], default_symptom: Optional[str] = None) -> str:
    if not findings:
        return "No structured findings parsed."
    lines = []
    for f in findings:
        symptom = f.get("symptom") or default_symptom or "Finding"
        lead = f.get("lead")
        lines.append(f"- {symptom} | {lead} | {format_time_range(f.get('start'), f.get('end'))}")
    return "\n".join(lines)


def assign_findings_indices(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    indexed = []
    for i, f in enumerate(findings, 1):
        item = dict(f)
        item["index"] = i
        item["label"] = CIRCLED_NUMBERS[i - 1] if i - 1 < len(CIRCLED_NUMBERS) else str(i)
        indexed.append(item)
    return indexed
