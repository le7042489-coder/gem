from __future__ import annotations

from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal
import wfdb
from fastapi import UploadFile

from ..config import TARGET_FS, TARGET_LEN, TEMP_UPLOAD_DIR, UPLOAD_DEFAULT_FS
from ..utils.parsing import STANDARD_LEADS, normalize_lead_name


def _save_upload_files(files: List[UploadFile], temp_dir: Path) -> Optional[str]:
    temp_dir.mkdir(exist_ok=True)
    base_name = None
    for file_obj in files:
        filename = Path(file_obj.filename).name
        dst_path = temp_dir / filename
        with dst_path.open("wb") as f:
            content = file_obj.file.read()
            f.write(content)
        if filename.endswith(".hea"):
            base_name = filename[:-4]
    return base_name


def _transpose_to_12_leads(signal: np.ndarray, source_name: str = "signal") -> np.ndarray:
    if signal.ndim != 2:
        raise ValueError(f"{source_name} must be a 2D array. Got shape={signal.shape}.")
    if signal.shape[0] == 12:
        return signal
    if signal.shape[1] == 12:
        return np.transpose(signal, (1, 0))
    raise ValueError(f"{source_name} must be shaped (12, L) or (L, 12). Got shape={signal.shape}.")


def _align_wfdb_leads(signal_lc: np.ndarray, sig_names: Optional[List[str]]) -> np.ndarray:
    if not sig_names:
        return signal_lc
    normalized = [normalize_lead_name(n) for n in sig_names]
    if all(lead in normalized for lead in STANDARD_LEADS):
        indices = [normalized.index(lead) for lead in STANDARD_LEADS]
        return signal_lc[:, indices]
    return signal_lc


def _sanitize(signal_12l: np.ndarray) -> np.ndarray:
    out = np.asarray(signal_12l).astype(np.float32, copy=False)
    out[np.isnan(out)] = 0
    out[np.isinf(out)] = 0
    return out


def _resample_signal(signal_12l: np.ndarray, fs_original: float, fs_target: int) -> Tuple[np.ndarray, bool]:
    fs_original = float(fs_original)
    if fs_original <= 0:
        fs_original = float(fs_target)
    if abs(fs_original - float(fs_target)) < 1e-6:
        return signal_12l, False

    ratio = Fraction(fs_target / fs_original).limit_denominator(10_000)
    up = ratio.numerator
    down = ratio.denominator
    resampled = scipy.signal.resample_poly(signal_12l, up=up, down=down, axis=1)
    return np.asarray(resampled, dtype=np.float32), True


def normalize_signal(
    signal_12l: np.ndarray,
    fs_original: float,
    target_fs: int = TARGET_FS,
    target_len: int = TARGET_LEN,
) -> Tuple[np.ndarray, Dict[str, object]]:
    signal_12l = _sanitize(signal_12l)
    len_original = int(signal_12l.shape[1])
    fs_original = float(fs_original) if fs_original is not None else float(target_fs)
    signal_12l, resampled = _resample_signal(signal_12l, fs_original=fs_original, fs_target=target_fs)

    cropped = False
    padded = False
    if signal_12l.shape[1] > target_len:
        signal_12l = signal_12l[:, :target_len]
        cropped = True
    elif signal_12l.shape[1] < target_len:
        pad = target_len - signal_12l.shape[1]
        signal_12l = np.pad(signal_12l, ((0, 0), (0, pad)))
        padded = True

    preprocess = {
        "fs_original": fs_original,
        "len_original": len_original,
        "resampled": resampled,
        "resample_method": "resample_poly" if resampled else "none",
        "cropped": cropped,
        "padded": padded,
    }
    return np.asarray(signal_12l, dtype=np.float32), preprocess


def _load_wfdb_record(record_path: str) -> Tuple[np.ndarray, float]:
    signal, meta = wfdb.rdsamp(record_path)
    signal = _align_wfdb_leads(signal, meta.get("sig_name"))
    signal_12l = _transpose_to_12_leads(np.transpose(signal, (1, 0)), source_name="wfdb signal")
    return signal_12l, float(meta["fs"])


def load_signal_from_npy_path(path: Path, fs_original: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, object]]:
    signal = np.load(str(path))
    signal_12l = _transpose_to_12_leads(np.asarray(signal), source_name=f"time_series {path}")
    fs = float(fs_original) if fs_original is not None else float(UPLOAD_DEFAULT_FS)
    return normalize_signal(signal_12l, fs_original=fs)


def load_signal_from_uploaded_npy(
    time_series_file: UploadFile,
    fs_original: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    raw = time_series_file.file.read()
    array = np.load(BytesIO(raw))
    signal_12l = _transpose_to_12_leads(np.asarray(array), source_name="uploaded time_series")
    fs = float(fs_original) if fs_original is not None else float(UPLOAD_DEFAULT_FS)
    return normalize_signal(signal_12l, fs_original=fs)


def process_uploaded_files(files: List[UploadFile]) -> Tuple[Optional[np.ndarray], Optional[float], str]:
    if not files:
        return None, None, "未上传文件"

    base_name = _save_upload_files(files, TEMP_UPLOAD_DIR)
    if not base_name:
        return None, None, "❌ 缺少 .hea 头文件，请同时上传 .dat 和 .hea"

    try:
        signal_12l, fs_original = _load_wfdb_record(str(TEMP_UPLOAD_DIR / base_name))
        signal_norm, _ = normalize_signal(signal_12l, fs_original=fs_original)
        return signal_norm, float(TARGET_FS), "✅ 文件读取成功"
    except Exception as e:
        return None, None, f"❌ 读取失败: {str(e)}"

