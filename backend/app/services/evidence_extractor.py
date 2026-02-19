from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.signal

from ..utils.parsing import LEAD_INDEX, STANDARD_LEADS, normalize_lead_name
from .measurement_store import MeasurementStore


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_time_to_ms(value: Any) -> Optional[float]:
    v = _to_float(value)
    if v is None:
        return None
    # Most ECG windows in this project are 0-10s. Treat small values as seconds.
    if abs(v) <= 20:
        return v * 1000.0
    return v


def _sanitize_window(t_start_ms: Optional[float], t_end_ms: Optional[float]) -> Tuple[float, float]:
    if t_start_ms is None or t_end_ms is None:
        return 0.0, 10_000.0
    start = max(0.0, min(float(t_start_ms), 10_000.0))
    end = max(0.0, min(float(t_end_ms), 10_000.0))
    if end < start:
        start, end = end, start
    if end == start:
        end = min(10_000.0, start + 1.0)
    return start, end


def _make_evidence(
    lead: str,
    t_start_ms: Optional[float],
    t_end_ms: Optional[float],
    measurement_name: str,
    value: Any,
    unit: str,
    source: str,
    quality: str,
) -> Optional[Dict[str, Any]]:
    lead_norm = normalize_lead_name(lead)
    if lead_norm not in STANDARD_LEADS:
        return None
    val = _to_float(value)
    if val is None:
        return None
    start_ms, end_ms = _sanitize_window(t_start_ms, t_end_ms)
    return {
        "lead": lead_norm,
        "t_start_ms": round(start_ms, 2),
        "t_end_ms": round(end_ms, 2),
        "measurement_name": str(measurement_name),
        "value": round(val, 4),
        "unit": str(unit),
        "source": source,
        "quality": quality,
    }


def _candidate_values(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return None


def _extract_from_measurement_list(items: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in items:
        lead = _candidate_values(item, ("lead", "lead_name", "Lead", "leadName"))
        measurement_name = _candidate_values(
            item,
            ("measurement_name", "measurement", "name", "feature", "metric"),
        )
        value = _candidate_values(item, ("value", "measurement_value", "val"))
        unit = _candidate_values(item, ("unit", "units")) or "ms"
        t_start = _candidate_values(item, ("t_start_ms", "start_ms", "t_start", "start", "start_s"))
        t_end = _candidate_values(item, ("t_end_ms", "end_ms", "t_end", "end", "end_s"))
        evidence = _make_evidence(
            lead=lead or "II",
            t_start_ms=_normalize_time_to_ms(t_start),
            t_end_ms=_normalize_time_to_ms(t_end),
            measurement_name=measurement_name or "measurement",
            value=value,
            unit=unit,
            source=source,
            quality="high",
        )
        if evidence is not None:
            out.append(evidence)
    return out


def _extract_from_flat_measurements(payload: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
    key_map = {
        "RR_Interval": ("RR", "ms"),
        "PR_Interval": ("PR", "ms"),
        "QRS_Complex": ("QRS", "ms"),
        "QT_Interval": ("QT", "ms"),
        "QTc_Interval": ("QTc", "ms"),
        "Heart_Rate": ("HR", "bpm"),
    }
    out: List[Dict[str, Any]] = []
    for key, (name, unit) in key_map.items():
        if key not in payload:
            continue
        evidence = _make_evidence(
            lead="II",
            t_start_ms=0,
            t_end_ms=10_000,
            measurement_name=name,
            value=payload.get(key),
            unit=unit,
            source=source,
            quality="medium",
        )
        if evidence is not None:
            out.append(evidence)
    return out


def _extract_measurement_evidence(payload: Any, source: str) -> List[Dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        nested = payload.get("machine_measurements")
        if isinstance(nested, list):
            return _extract_from_measurement_list([x for x in nested if isinstance(x, dict)], source=source)
        if isinstance(nested, dict):
            return _extract_from_flat_measurements(nested, source=source)
        list_like_keys = ("measurements", "features", "items")
        for key in list_like_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return _extract_from_measurement_list([x for x in value if isinstance(x, dict)], source=source)
        as_list = _extract_from_measurement_list([payload], source=source)
        if as_list:
            return as_list
        return _extract_from_flat_measurements(payload, source=source)
    if isinstance(payload, list):
        return _extract_from_measurement_list([x for x in payload if isinstance(x, dict)], source=source)
    return []


def _detect_r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    centered = signal - np.median(signal)
    distance = max(1, int(0.25 * fs))
    prominence = max(float(np.std(centered) * 0.35), 1e-3)
    peaks_pos, _ = scipy.signal.find_peaks(centered, distance=distance, prominence=prominence)
    peaks_neg, _ = scipy.signal.find_peaks(-centered, distance=distance, prominence=prominence)
    peaks = peaks_pos if len(peaks_pos) >= len(peaks_neg) else peaks_neg
    return peaks


def _estimate_qrs_ms(signal: np.ndarray, peaks: np.ndarray, fs: int) -> Optional[float]:
    if len(peaks) == 0:
        return None
    widths = []
    for p in peaks:
        amp = abs(signal[p])
        if amp < 1e-6:
            continue
        th = amp * 0.2
        left = p
        right = p
        left_min = max(0, p - int(0.15 * fs))
        right_max = min(len(signal) - 1, p + int(0.15 * fs))
        while left > left_min and abs(signal[left]) > th:
            left -= 1
        while right < right_max and abs(signal[right]) > th:
            right += 1
        widths.append((right - left) / fs * 1000.0)
    if not widths:
        return None
    return float(np.mean(widths))


def _estimate_qt_ms(signal: np.ndarray, peaks: np.ndarray, fs: int, qrs_ms: Optional[float]) -> Optional[float]:
    if len(peaks) == 0:
        return None
    qrs_half = int(((qrs_ms or 90.0) / 1000.0) * fs / 2.0)
    qts = []
    for p in peaks:
        q_start = max(0, p - qrs_half)
        t_start = min(len(signal) - 1, p + int(0.12 * fs))
        t_end = min(len(signal), p + int(0.60 * fs))
        if t_end <= t_start:
            continue
        window = np.abs(signal[t_start:t_end])
        if window.size == 0:
            continue
        t_peak = t_start + int(np.argmax(window))
        qts.append((t_peak - q_start) / fs * 1000.0)
    if not qts:
        return None
    return float(np.mean(qts))


def _estimate_st_deviation_mv(signal: np.ndarray, peaks: np.ndarray, fs: int, qrs_ms: Optional[float]) -> Optional[float]:
    if len(peaks) == 0:
        return None
    qrs_half = int(((qrs_ms or 90.0) / 1000.0) * fs / 2.0)
    vals = []
    for p in peaks:
        base_l = max(0, p - int(0.20 * fs))
        base_r = max(base_l + 1, p - int(0.12 * fs))
        st_idx = min(len(signal) - 1, p + qrs_half + int(0.08 * fs))
        if base_r <= base_l:
            continue
        baseline = float(np.mean(signal[base_l:base_r]))
        vals.append(float(signal[st_idx] - baseline))
    if not vals:
        return None
    return float(np.mean(vals))


def _extract_algorithmic_evidence(signal_12l: np.ndarray, fs_used: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    lead_ii = signal_12l[LEAD_INDEX["II"]]
    peaks = _detect_r_peaks(lead_ii, fs=fs_used)

    if len(peaks) >= 2:
        rr_ms = np.diff(peaks) / fs_used * 1000.0
        rr_mean = float(np.mean(rr_ms))
        hr = 60_000.0 / rr_mean if rr_mean > 1e-6 else None
        out.append(
            _make_evidence(
                lead="II",
                t_start_ms=peaks[0] / fs_used * 1000.0,
                t_end_ms=peaks[-1] / fs_used * 1000.0,
                measurement_name="RR",
                value=rr_mean,
                unit="ms",
                source="algorithmic",
                quality="medium",
            )
        )
        if hr is not None:
            out.append(
                _make_evidence(
                    lead="II",
                    t_start_ms=peaks[0] / fs_used * 1000.0,
                    t_end_ms=peaks[-1] / fs_used * 1000.0,
                    measurement_name="HR",
                    value=hr,
                    unit="bpm",
                    source="algorithmic",
                    quality="medium",
                )
            )

    qrs_ms = _estimate_qrs_ms(lead_ii, peaks, fs=fs_used)
    if qrs_ms is not None:
        out.append(
            _make_evidence(
                lead="II",
                t_start_ms=0,
                t_end_ms=10_000,
                measurement_name="QRS",
                value=qrs_ms,
                unit="ms",
                source="algorithmic",
                quality="medium",
            )
        )

    qt_ms = _estimate_qt_ms(lead_ii, peaks, fs=fs_used, qrs_ms=qrs_ms)
    if qt_ms is not None:
        out.append(
            _make_evidence(
                lead="II",
                t_start_ms=0,
                t_end_ms=10_000,
                measurement_name="QT",
                value=qt_ms,
                unit="ms",
                source="algorithmic",
                quality="low",
            )
        )

    for lead in ("II", "V5"):
        sig = signal_12l[LEAD_INDEX[lead]]
        st_dev = _estimate_st_deviation_mv(sig, peaks, fs=fs_used, qrs_ms=qrs_ms)
        if st_dev is not None:
            out.append(
                _make_evidence(
                    lead=lead,
                    t_start_ms=0,
                    t_end_ms=10_000,
                    measurement_name="ST_deviation",
                    value=st_dev,
                    unit="mV",
                    source="algorithmic",
                    quality="low",
                )
            )

    return [x for x in out if x is not None]


def _assign_ids(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(evidence, 1):
        obj = dict(item)
        obj["id"] = f"ev_{i}"
        out.append(obj)
    return out


def build_evidence_candidates(
    signal_12l: np.ndarray,
    fs_used: int,
    sample_id: Optional[str],
    sample_record: Optional[Dict[str, Any]],
    max_evidence: int,
) -> List[Dict[str, Any]]:
    max_evidence = max(1, int(max_evidence))

    measurement_payload = None
    if isinstance(sample_record, dict):
        measurement_payload = sample_record.get("machine_measurements")
        if measurement_payload is None:
            metadata = sample_record.get("metadata")
            if isinstance(metadata, dict):
                measurement_payload = metadata.get("machine_measurements")

    evidence = _extract_measurement_evidence(measurement_payload, source="machine_measurements")

    if not evidence and sample_id:
        external_payload = MeasurementStore.get().get_measurements(sample_id)
        evidence = _extract_measurement_evidence(external_payload, source="machine_measurements")

    if not evidence:
        evidence = _extract_algorithmic_evidence(signal_12l, fs_used=fs_used)

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in evidence:
        key = (
            item["lead"],
            item["measurement_name"],
            round(float(item["t_start_ms"]), 2),
            round(float(item["t_end_ms"]), 2),
            round(float(item["value"]), 4),
            item["unit"],
            item["source"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return _assign_ids(deduped[:max_evidence])

