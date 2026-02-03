from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import wfdb
from fastapi import UploadFile

from ..config import TARGET_FS, TARGET_LEN, TEMP_UPLOAD_DIR
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


def process_uploaded_files(files: List[UploadFile]) -> Tuple[Optional[np.ndarray], Optional[float], str]:
    if not files:
        return None, None, "未上传文件"

    base_name = _save_upload_files(files, TEMP_UPLOAD_DIR)
    if not base_name:
        return None, None, "❌ 缺少 .hea 头文件，请同时上传 .dat 和 .hea"

    try:
        record = wfdb.rdsamp(str(TEMP_UPLOAD_DIR / base_name))
        signal = record[0]
        meta = record[1]
        fs = meta['fs']

        sig_names = meta.get('sig_name', None)
        if sig_names:
            normalized = [normalize_lead_name(n) for n in sig_names]
            if all(lead in normalized for lead in STANDARD_LEADS):
                indices = [normalized.index(lead) for lead in STANDARD_LEADS]
                signal = signal[:, indices]

        if fs != TARGET_FS:
            step = fs / TARGET_FS
            indices = np.arange(0, signal.shape[0], step).astype(int)
            indices = indices[indices < signal.shape[0]]
            signal = signal[indices]

        if signal.shape[0] < TARGET_LEN:
            pad_len = TARGET_LEN - signal.shape[0]
            signal = np.pad(signal, ((0, pad_len), (0, 0)))
        else:
            signal = signal[:TARGET_LEN, :]

        signal = signal.T

        return signal, TARGET_FS, "✅ 文件读取成功"

    except Exception as e:
        return None, None, f"❌ 读取失败: {str(e)}"
