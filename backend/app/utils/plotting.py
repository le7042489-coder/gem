from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .parsing import LEAD_INDEX, SEGMENT_SECONDS, TOTAL_SECONDS

LAYOUT_MAP = [
    [("I", LEAD_INDEX["I"], 0), ("aVR", LEAD_INDEX["aVR"], 1), ("V1", LEAD_INDEX["V1"], 2), ("V4", LEAD_INDEX["V4"], 3)],
    [("II", LEAD_INDEX["II"], 0), ("aVL", LEAD_INDEX["aVL"], 1), ("V2", LEAD_INDEX["V2"], 2), ("V5", LEAD_INDEX["V5"], 3)],
    [("III", LEAD_INDEX["III"], 0), ("aVF", LEAD_INDEX["aVF"], 1), ("V3", LEAD_INDEX["V3"], 2), ("V6", LEAD_INDEX["V6"], 3)],
]

LEAD_POS_MAP: Dict[str, Tuple[int, int, float]] = {}
for r, row in enumerate(LAYOUT_MAP):
    for c, (lead_name, _, time_slot) in enumerate(row):
        LEAD_POS_MAP[lead_name] = (r, c, time_slot * SEGMENT_SECONDS)


def plot_ecg_3x4_grid(signal_data: np.ndarray) -> Image.Image:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    plt.subplots_adjust(wspace=0, hspace=0)

    seg_len = signal_data.shape[1] // 4

    for r in range(3):
        for c in range(4):
            lead_name, ch_idx, time_slot = LAYOUT_MAP[r][c]
            ax = axes[r, c]

            start = time_slot * seg_len
            end = start + seg_len
            segment = signal_data[ch_idx, start:end]

            ax.plot(segment, color='black', linewidth=1.2)
            ax.text(0.02, 0.9, lead_name, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', color='red')

            ax.set_xlim([0, seg_len])
            y_min, y_max = segment.min(), segment.max()
            margin = (y_max - y_min) * 0.1
            if margin == 0:
                margin = 1
            ax.set_ylim([y_min - margin, y_max + margin])

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.grid(True, which='major', color='#f2a7a7', linestyle='-', linewidth=0.8, alpha=0.6)
            ax.set_facecolor('#fff7f2')

    buf = BytesIO()
    plt.savefig(buf, format='jpg', dpi=100, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def compute_box_coords(lead: str, start: Optional[float], end: Optional[float], width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    if lead not in LEAD_POS_MAP:
        return None

    col_w = width / 4
    row_h = height / 3
    r, c, offset = LEAD_POS_MAP[lead]
    y1 = r * row_h
    y2 = (r + 1) * row_h
    base_x = c * col_w

    if start is None or end is None:
        x1 = base_x
        x2 = base_x + col_w
    else:
        start = max(0.0, min(float(start), TOTAL_SECONDS))
        end = max(0.0, min(float(end), TOTAL_SECONDS))
        if start > end:
            start, end = end, start
        rel_start = max(0.0, min(start - offset, SEGMENT_SECONDS))
        rel_end = max(0.0, min(end - offset, SEGMENT_SECONDS))
        if rel_end <= rel_start:
            return None
        pixels_per_sec = col_w / SEGMENT_SECONDS
        x1 = base_x + rel_start * pixels_per_sec
        x2 = base_x + rel_end * pixels_per_sec

    margin = 4
    x1 = x1 + margin
    x2 = x2 - margin
    y1 = y1 + margin
    y2 = y2 - margin
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, x2, y1, y2)


def findings_to_boxes(findings: List[Dict[str, Any]], image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
    width, height = image_size
    boxes = []
    for f in findings:
        coords = compute_box_coords(f.get("lead"), f.get("start"), f.get("end"), width, height)
        if not coords:
            continue
        x1, x2, y1, y2 = coords
        boxes.append({
            "index": f.get("index"),
            "label": f.get("label"),
            "x1": x1 / width,
            "x2": x2 / width,
            "y1": y1 / height,
            "y2": y2 / height
        })
    return boxes


def image_to_base64(image: Image.Image) -> str:
    import base64
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
