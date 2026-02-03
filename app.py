import gradio as gr
import torch
import os
import re
import numpy as np
import wfdb
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from io import BytesIO
from transformers import BitsAndBytesConfig
import plotly.express as px
import plotly.graph_objects as go
import textwrap
import tempfile
import scipy.signal

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from backend.app.config import MODEL_PATH, MODEL_BASE, DEVICE_MAP
from backend.app.config import TARGET_FS, TARGET_LEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

# ==================== 1. 模型初始化 (只运行一次) ====================
print("⏳ 正在初始化 GEM 模型 (4-bit)...")
disable_torch_init()

model_name = get_model_name_from_path(MODEL_PATH)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    llm_int8_skip_modules=["ecg_tower", "vision_tower", "mm_projector"]
)

# 强制 device_map 使用 GPU 0，防止加速库将部分层切分到 CPU 导致报错
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH,
    MODEL_BASE,
    model_name,
    quantization_config=bnb_config,
    device_map=DEVICE_MAP,
)

print("✅ 模型加载完成！")


# ==================== 2. ECG 元信息与解析辅助 ====================

STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_INDEX = {lead: i for i, lead in enumerate(STANDARD_LEADS)}
SEGMENT_SECONDS = 2.5
TOTAL_SECONDS = 10.0
DEFAULT_DIAG_PROMPT = "Please interpret this ECG and provide a diagnosis."
CIRCLED_NUMBERS = [
    "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩",
    "⑪", "⑫", "⑬", "⑭", "⑮", "⑯", "⑰", "⑱", "⑲", "⑳"
]

# 3x4 布局 (Row, Col) -> (Lead, Channel Index, Time Slot)
LAYOUT_MAP = [
    [("I", LEAD_INDEX["I"], 0), ("aVR", LEAD_INDEX["aVR"], 1), ("V1", LEAD_INDEX["V1"], 2), ("V4", LEAD_INDEX["V4"], 3)],
    [("II", LEAD_INDEX["II"], 0), ("aVL", LEAD_INDEX["aVL"], 1), ("V2", LEAD_INDEX["V2"], 2), ("V5", LEAD_INDEX["V5"], 3)],
    [("III", LEAD_INDEX["III"], 0), ("aVF", LEAD_INDEX["aVF"], 1), ("V3", LEAD_INDEX["V3"], 2), ("V6", LEAD_INDEX["V6"], 3)],
]

LEAD_POS_MAP = {}
for r, row in enumerate(LAYOUT_MAP):
    for c, (lead_name, _, time_slot) in enumerate(row):
        LEAD_POS_MAP[lead_name] = (r, c, time_slot * SEGMENT_SECONDS)


def normalize_lead_name(name):
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


def expand_lead_field(lead_field):
    if not lead_field:
        return []
    field = lead_field.replace("and", ",").replace("/", ",")
    parts = [p.strip() for p in field.split(",") if p.strip()]
    leads = []
    roman_order = ["I", "II", "III"]
    for part in parts:
        m = re.match(r"^(V)([1-6])\s*-\s*V?([1-6])$", part, re.IGNORECASE)
        if m:
            start = int(m.group(2))
            end = int(m.group(3))
            step = 1 if start <= end else -1
            for i in range(start, end + step, step):
                leads.append(f"V{i}")
            continue
        m = re.match(r"^(I|II|III)\s*-\s*(I|II|III)$", part, re.IGNORECASE)
        if m:
            start = roman_order.index(m.group(1).upper())
            end = roman_order.index(m.group(2).upper())
            step = 1 if start <= end else -1
            for i in range(start, end + step, step):
                leads.append(roman_order[i])
            continue
        normalized = normalize_lead_name(part)
        if normalized:
            leads.append(normalized)
    # 去重但保留顺序
    seen = set()
    ordered = []
    for lead in leads:
        if lead not in seen:
            seen.add(lead)
            ordered.append(lead)
    return ordered


def parse_time_value(val):
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


def clean_findings(findings, duration=TOTAL_SECONDS):
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
    # 去重
    unique = []
    seen = set()
    for f in cleaned:
        key = (f["symptom"].lower(), f["lead"], round(f["start"] or -1.0, 3), round(f["end"] or -1.0, 3))
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def parse_findings_output(text, default_symptom=None):
    findings = []

    # 1) 优先解析 FINDING| 格式
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

    # 2) 解析 "Symptom/Lead/Time" 行
    pattern_struct = (
        r"Symptom\s*[:\-]\s*(?P<symptom>[^\n;]+?)\s*"
        r"(?:;|,)?\s*Lead\s*[:\-]\s*(?P<lead>[^\n;]+?)\s*"
        r"(?:;|,)?\s*Time\s*[:\-]\s*(?P<start>\d+\.?\d*)\s*(?:s|sec)?\s*"
        r"(?:-|to)\s*(?P<end>\d+\.?\d*)"
    )
    for m in re.finditer(pattern_struct, text, re.IGNORECASE):
        symptom = m.group("symptom").strip()
        leads = expand_lead_field(m.group("lead").strip())
        start = parse_time_value(m.group("start"))
        end = parse_time_value(m.group("end"))
        for lead in leads:
            findings.append({"symptom": symptom, "lead": lead, "start": start, "end": end})

    if findings:
        return clean_findings(findings)

    # 3) 兜底：只解析导联+时间，症状用 default_symptom
    pattern_time = r"(I|II|III|aVR|aVL|aVF|V[1-6])\D+?(\d+\.?\d*)\s*(?:s|sec)?\s*(?:-|to)\s*(\d+\.?\d*)"
    matches_time = re.findall(pattern_time, text, re.IGNORECASE)
    if matches_time:
        symptom = default_symptom or "Finding"
        for lead, start, end in matches_time:
            lead = normalize_lead_name(lead)
            findings.append({"symptom": symptom, "lead": lead, "start": start, "end": end})

    return clean_findings(findings)


def extract_diagnosis_summary(text):
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


def format_time_range(start, end, duration=TOTAL_SECONDS):
    if start is None or end is None:
        return f"0.00-{duration:.2f}s (full segment)"
    return f"{start:.2f}-{end:.2f}s"


def format_diagnosis_report(summary, findings):
    lines = []
    if summary:
        lines.append(f"Diagnosis: {summary}")
    else:
        lines.append("Diagnosis: (model did not provide a clear summary)")

    if not findings:
        lines.append("Findings: (no structured findings parsed)")
        return "\n".join(lines)

    # 按症状分组，按导联顺序排序
    symptom_map = {}
    for f in findings:
        symptom_map.setdefault(f["symptom"], []).append(f)

    lines.append("Findings:")
    lead_order = {lead: i for i, lead in enumerate(STANDARD_LEADS)}
    for symptom in sorted(symptom_map.keys()):
        entries = symptom_map[symptom]
        lead_groups = {}
        for f in entries:
            lead_groups.setdefault(f["lead"], []).append((f["start"], f["end"]))
        lead_chunks = []
        for lead in sorted(lead_groups.keys(), key=lambda x: lead_order.get(x, 999)):
            ranges = sorted(lead_groups[lead], key=lambda t: (t[0] or -1, t[1] or -1))
            range_str = ", ".join(format_time_range(s, e) for s, e in ranges)
            lead_chunks.append(f"{lead} ({range_str})")
        lines.append(f"- {symptom}: " + "; ".join(lead_chunks))

    return "\n".join(lines)


def format_findings_only(findings, default_symptom=None):
    if not findings:
        return "No structured findings parsed."
    lines = []
    for f in findings:
        symptom = f.get("symptom") or default_symptom or "Finding"
        lead = f.get("lead")
        lines.append(f"- {symptom} | {lead} | {format_time_range(f.get('start'), f.get('end'))}")
    return "\n".join(lines)


def assign_findings_indices(findings):
    indexed = []
    for i, f in enumerate(findings, 1):
        item = dict(f)
        item["index"] = i
        item["label"] = CIRCLED_NUMBERS[i - 1] if i - 1 < len(CIRCLED_NUMBERS) else str(i)
        indexed.append(item)
    return indexed


def build_findings_rows(findings):
    if not findings:
        return [["-", "No findings parsed", "-", "-"]]
    rows = []
    for f in findings:
        rows.append([
            f.get("label", str(f.get("index", ""))),
            f.get("symptom", "Finding"),
            f.get("lead", ""),
            format_time_range(f.get("start"), f.get("end"))
        ])
    return rows


def compute_box_coords(lead, start, end, width, height):
    lead = normalize_lead_name(lead)
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


def build_ecg_figure(ecg_image, findings, focus_index=None):
    if ecg_image is None:
        return None

    img = np.array(ecg_image)
    height, width = img.shape[0], img.shape[1]

    fig = px.imshow(img)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff"
    )
    fig.update_xaxes(visible=False, showgrid=False)
    fig.update_yaxes(visible=False, showgrid=False, autorange="reversed", scaleanchor="x", scaleratio=1)

    shapes = []
    annotations = []
    for f in findings or []:
        coords = compute_box_coords(f.get("lead"), f.get("start"), f.get("end"), width, height)
        if not coords:
            continue
        x1, x2, y1, y2 = coords
        shapes.append(dict(
            type="rect",
            x0=x1,
            x1=x2,
            y0=y1,
            y1=y2,
            line=dict(color="rgba(255,122,89,0.95)", width=2, dash="dot"),
            fillcolor="rgba(255,122,89,0.25)"
        ))
        annotations.append(dict(
            x=x1 + 10,
            y=y1 + 14,
            text=f.get("label", str(f.get("index", ""))),
            showarrow=False,
            font=dict(color="#1f2937", size=14, family="Inter, sans-serif"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(255,122,89,0.9)",
            borderwidth=1
        ))

    fig.update_layout(shapes=shapes, annotations=annotations)

    if focus_index is not None:
        target = next((f for f in findings if f.get("index") == focus_index), None)
        if target:
            coords = compute_box_coords(target.get("lead"), target.get("start"), target.get("end"), width, height)
            if coords:
                x1, x2, y1, y2 = coords
                pad = 40
                x0 = max(0, x1 - pad)
                x3 = min(width, x2 + pad)
                y0 = max(0, y1 - pad)
                y3 = min(height, y2 + pad)
                fig.update_xaxes(range=[x0, x3])
                fig.update_yaxes(range=[y3, y0])

    return fig


def export_report_pdf(report_text):
    if not report_text:
        return None
    wrapped = "\n".join(textwrap.wrap(report_text, width=100))
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.95, wrapped, va="top", fontsize=12, family="Inter")
    tmp_dir = Path("temp_upload")
    tmp_dir.mkdir(exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=str(tmp_dir))
    fig.savefig(tmp.name, format="pdf", bbox_inches="tight")
    plt.close(fig)
    tmp.close()
    return tmp.name


# ==================== 3. 信号处理与绘图核心 (核心复现逻辑) ====================

def process_uploaded_files(files, window_start_s=0.0):
    """
    处理上传的文件列表，寻找成对的 .dat 和 .hea
    返回: signal (12, 5000), sampling_rate, temp_base_path
    """
    if not files:
        return None, None, "未上传文件"

    # 创建临时目录存放文件，因为 wfdb 需要通过路径读取
    temp_dir = Path("temp_upload")
    temp_dir.mkdir(exist_ok=True)

    base_name = None

    # 将文件移动到临时目录
    for file_obj in files:
        # Gradio 传入的是 NamedString 或 tempfile 路径
        src_path = file_obj.name
        filename = os.path.basename(src_path)
        dst_path = temp_dir / filename
        shutil.copy(src_path, dst_path)

        if filename.endswith(".hea"):
            base_name = filename[:-4]  # 去掉后缀

    if not base_name:
        return None, None, "❌ 缺少 .hea 头文件，请同时上传 .dat 和 .hea"

    # 读取信号
    try:
        record = wfdb.rdsamp(str(temp_dir / base_name))
        signal = record[0]  # (Samples, Channels)
        meta = record[1]
        fs = meta['fs']
        if not fs or fs <= 0:
            return None, None, "❌ 采样率无效，请检查 .hea 文件"

        # 尝试按标准 12 导联顺序重排
        sig_names = meta.get('sig_name', None)
        if sig_names:
            normalized = [normalize_lead_name(n) for n in sig_names]
            if all(lead in normalized for lead in STANDARD_LEADS):
                indices = [normalized.index(lead) for lead in STANDARD_LEADS]
                signal = signal[:, indices]

        # 简单的重采样逻辑：GEM 需要 500Hz
        # 如果是 MIMIC-IV (通常 500Hz)，则直接用
        target_fs = TARGET_FS
        if fs != target_fs:
            num_samples = int(round(signal.shape[0] * target_fs / fs))
            num_samples = max(1, num_samples)
            signal = scipy.signal.resample(signal, num_samples, axis=0)

        # 调整长度到 10秒 (5000点)
        target_len = TARGET_LEN
        if signal.shape[0] < target_len:
            # 填充
            pad_len = target_len - signal.shape[0]
            signal = np.pad(signal, ((0, pad_len), (0, 0)))
        else:
            # 智能切片：支持用户指定时间窗口起点
            try:
                window_start_s = float(window_start_s or 0.0)
            except (TypeError, ValueError):
                window_start_s = 0.0
            start_idx = int(round(window_start_s * target_fs))
            max_start = max(0, signal.shape[0] - target_len)
            start_idx = max(0, min(start_idx, max_start))
            signal = signal[start_idx:start_idx + target_len, :]

        # 转置为 (Channels, Samples) -> (12, 5000)
        signal = signal.T

        return signal, target_fs, "✅ 文件读取成功"

    except Exception as e:
        return None, None, f"❌ 读取失败: {str(e)}"


def plot_ecg_3x4_grid(signal_data):
    """
    将 (12, 5000) 的信号绘制为 GEM 标准的 3x4 布局图片。
    布局逻辑：
    - 4 列，每列代表 2.5 秒的时间窗口 (0-2.5, 2.5-5.0, 5.0-7.5, 7.5-10.0)
    - 3 行，按标准 12 导联顺序排列
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    plt.subplots_adjust(wspace=0, hspace=0)  # 无间隙

    seg_len = signal_data.shape[1] // 4  # 每个时间段长度

    for r in range(3):
        for c in range(4):
            lead_name, ch_idx, time_slot = LAYOUT_MAP[r][c]
            ax = axes[r, c]

            # 计算切片范围
            start = time_slot * seg_len
            end = start + seg_len
            segment = signal_data[ch_idx, start:end]

            # 绘图
            ax.plot(segment, color='black', linewidth=1.2)

            # 标注
            ax.text(0.02, 0.9, lead_name, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', color='red')

            # 美化：去除坐标轴，保留网格
            ax.set_xlim([0, seg_len])
            # 简单的自动缩放 Y 轴
            y_min, y_max = segment.min(), segment.max()
            margin = (y_max - y_min) * 0.1
            if margin == 0: margin = 1
            ax.set_ylim([y_min - margin, y_max + margin])

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # 画背景网格 (模拟 ECG 纸 细格子)
            ax.grid(True, which='major', color='#f2a7a7', linestyle='-', linewidth=0.8, alpha=0.6)
            ax.set_facecolor('#fff7f2')

    # 转为 PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='jpg', dpi=100, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return img


# ==================== 4. Grounding 可视化 ====================

def draw_grounding_boxes(image, findings):
    """
    根据解析结果在图上画框
    findings: list of dicts {symptom, lead, start, end} 或 (lead, start, end)
    """
    if not findings:
        return image

    img_draw = image.convert("RGBA")
    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = image.size
    col_w = w / 4
    row_h = h / 3

    box_fill = (255, 0, 0, 60)
    box_outline = (255, 0, 0, 200)

    for item in findings:
        if isinstance(item, dict):
            lead = item.get("lead")
            start = item.get("start")
            end = item.get("end")
        else:
            lead, start, end = item

        lead = normalize_lead_name(lead)
        if lead not in LEAD_POS_MAP:
            continue

        r, c, offset = LEAD_POS_MAP[lead]

        y1 = r * row_h
        y2 = (r + 1) * row_h
        base_x = c * col_w

        # 计算 X 轴范围
        if start is None or end is None:
            # 没时间，画满整格
            x1 = base_x
            x2 = base_x + col_w
        else:
            # 相对时间
            start = max(0.0, min(float(start), TOTAL_SECONDS))
            end = max(0.0, min(float(end), TOTAL_SECONDS))
            if start > end:
                start, end = end, start
            rel_start = start - offset
            rel_end = end - offset

            # 限制在当前格子的 0-2.5s 范围内
            rel_start = max(0, min(rel_start, SEGMENT_SECONDS))
            rel_end = max(0, min(rel_end, SEGMENT_SECONDS))
            if rel_end <= rel_start:
                continue

            # 映射到像素
            pixels_per_sec = col_w / SEGMENT_SECONDS
            x1 = base_x + rel_start * pixels_per_sec
            x2 = base_x + rel_end * pixels_per_sec

        # 稍微加一点 margin
        margin = 4
        draw.rectangle([x1 + margin, y1 + margin, x2 - margin, y2 - margin],
                       fill=box_fill, outline=box_outline, width=3)

    return Image.alpha_composite(img_draw, overlay).convert("RGB")


def plot_single_lead(signal_data, lead_name, findings=None, fs=500):
    if signal_data is None:
        return None
    lead = normalize_lead_name(lead_name)
    if lead not in LEAD_INDEX:
        return None

    idx = LEAD_INDEX[lead]
    segment = signal_data[idx]
    duration = segment.shape[0] / fs
    times = np.arange(segment.shape[0]) / fs

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(times, segment, color='black', linewidth=1.0)
    ax.set_title(f"Lead {lead} (0-{duration:.1f}s)")
    ax.set_xlim([0, duration])

    # 高亮症状时间段
    if findings:
        for f in findings:
            f_lead = normalize_lead_name(f.get("lead"))
            if f_lead != lead:
                continue
            start = f.get("start")
            end = f.get("end")
            if start is None or end is None:
                start, end = 0.0, duration
            else:
                start = max(0.0, min(float(start), duration))
                end = max(0.0, min(float(end), duration))
                if start > end:
                    start, end = end, start
            if end <= start:
                continue
            ax.axvspan(start, end, color='red', alpha=0.2)

    ax.grid(True, which='major', color='#f2a7a7', linestyle='-', linewidth=0.8, alpha=0.6)
    ax.set_facecolor('#fff7f2')

    buf = BytesIO()
    plt.savefig(buf, format='jpg', dpi=100, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


# ==================== 5. 主推理逻辑 ====================

def run_inference(files, text_query, mode, window_start_s=0.0, return_state=False):
    # 1. 处理输入文件
    signal_np, fs, msg = process_uploaded_files(files, window_start_s=window_start_s)
    if signal_np is None:
        if return_state:
            return None, f"❌ {msg}", [], None, None
        return None, f"❌ {msg}"

    # 2. 生成图像 Input
    ecg_image = plot_ecg_3x4_grid(signal_np)

    # 3. 生成信号 Tensor Input
    # 打印调试信息，确认信号强度
    print(
        f"\n[Debug] Signal Stat - Mean: {signal_np.mean():.4f}, Std: {signal_np.std():.4f}, Max: {signal_np.max()}, Shape: {signal_np.shape}")

    sig_tensor = torch.from_numpy(signal_np).float()
    # 简单的 Z-Score 归一化，防止数值过大或过小
    sig_mean = sig_tensor.mean()
    sig_std = sig_tensor.std()
    if sig_std > 1e-6:
        sig_tensor = (sig_tensor - sig_mean) / sig_std
    ecgs_tensor = sig_tensor.unsqueeze(0).half().cuda()

    # 4. 图像预处理
    if hasattr(image_processor, "preprocess"):
        image_tensor = image_processor.preprocess(ecg_image, return_tensors='pt')['pixel_values'][0]
    else:
        image_tensor = image_processor(ecg_image, return_tensors='pt')['pixel_values'][0]
    image_tensor = image_tensor.unsqueeze(0).half().cuda()

    # 5. 构造 Prompt
    if mode == "grounding":
        qs = (
            f"Task: Locate {text_query} in the ECG.\n"
            "Output format (one per line):\n"
            "FINDING|<symptom>|<lead>|<start_s>|<end_s>\n"
            "Rules:\n"
            "- Use lead names exactly: I, II, III, aVR, aVL, aVF, V1-V6.\n"
            "- Use absolute time in seconds within 0-10.\n"
            "- If a finding spans the full 10s, use 0-10.\n"
            "Begin."
        )
    else:
        qs = (
            f"User request: {text_query}\n"
            "Task: Provide a concise diagnosis summary, then list each observable ECG symptom with lead and time.\n"
            "Output format:\n"
            "Diagnosis: <short summary>\n"
            "Findings:\n"
            "FINDING|<symptom>|<lead>|<start_s>|<end_s>\n"
            "Rules:\n"
            "- Use lead names exactly: I, II, III, aVR, aVL, aVF, V1-V6.\n"
            "- Use absolute time in seconds within 0-10.\n"
            "- If a finding spans the full 10s, use 0-10.\n"
            "- One finding per line.\n"
            "Begin."
        )

    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    print(f"Running Inference: {mode}...")

    # 6. 生成
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                ecgs=ecgs_tensor,
                # 【修改点】Grounding 任务改用贪婪搜索，更稳定
                do_sample=False if mode in {"grounding", "diagnosis"} else True,
                temperature=0.0 if mode in {"grounding", "diagnosis"} else 0.2,
                max_new_tokens=512,
                min_new_tokens=16 if mode == "grounding" else None,
                use_cache=True,
                stopping_criteria=None
            )
    except Exception as e:
        print(f"❌ Inference Error: {e}")
        if return_state:
            return ecg_image, f"Error: {e}", [], signal_np, fs
        return ecg_image, f"Error: {e}"

    # 7. 调试输出
    new_tokens = output_ids[0, input_ids.shape[1]:]
    print(f"[Debug] Generated Token IDs: {new_tokens.tolist()}")  # 看看到底输出了什么 ID

    # 解码（带调试对比）
    eos_token_id = tokenizer.eos_token_id
    output_text_no_skip = tokenizer.decode(new_tokens, skip_special_tokens=False)
    output_text_skip = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"[Debug] EOS Token ID: {eos_token_id}")
    print(f"[Debug] First 10 New Tokens: {new_tokens[:10].tolist()}")
    print(f"[Debug] Decode (no-skip): '{output_text_no_skip}'")
    print(f"[Debug] Decode (skip): '{output_text_skip}'")
    output_text = output_text_skip.strip()
    print(f"[Debug] Raw Output Text: '{output_text}'")  # 打印带引号的文本，检查是否为空串

    # 8. 后处理
    if not output_text:
        output_text = output_text_no_skip.strip()
        if not output_text:
            output_text = "[Error] 模型输出了空内容。请检查终端的 Token IDs。"

    if mode == "grounding":
        findings = parse_findings_output(output_text, default_symptom=text_query)
        print(f"[Debug] Parsed Findings: {findings}")
        report_text = format_findings_only(findings, default_symptom=text_query)
        if not findings:
            report_text = output_text
    else:
        findings = parse_findings_output(output_text)
        print(f"[Debug] Parsed Findings: {findings}")
        summary = extract_diagnosis_summary(output_text)
        report_text = format_diagnosis_report(summary, findings)
        if not findings:
            report_text = report_text + "\n\nRaw Output:\n" + output_text

    if return_state:
        return ecg_image, report_text, findings, signal_np, fs
    return ecg_image, report_text


def run_diagnosis(files, window_start_s=0.0):
    return run_inference(files, DEFAULT_DIAG_PROMPT, "diagnosis", window_start_s=window_start_s, return_state=True)


def update_lead_view(lead_name, signal_state, findings_state, fs_state):
    if signal_state is None:
        return None
    fs = fs_state or 500
    return plot_single_lead(signal_state, lead_name, findings_state, fs=fs)


def start_analysis(files, window_start_s):
    if not files:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            build_findings_rows([]),
            "❌ 请先上传 .dat 和 .hea 文件。",
            [],
            None,
            "❌ 请先上传 .dat 和 .hea 文件。"
        )

    ecg_image, report_text, findings, _, _ = run_diagnosis(files, window_start_s=window_start_s)
    if ecg_image is None or report_text.startswith("❌") or report_text.startswith("Error"):
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            build_findings_rows([]),
            report_text,
            [],
            None,
            report_text
        )

    indexed_findings = assign_findings_indices(findings)
    fig = build_ecg_figure(ecg_image, indexed_findings)
    rows = build_findings_rows(indexed_findings)

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        fig,
        rows,
        report_text,
        indexed_findings,
        ecg_image,
        ""
    )


def reset_to_landing():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        None,
        build_findings_rows([]),
        "",
        [],
        None,
        "",
        None,
        0.0
    )


def focus_on_finding(evt: gr.SelectData, image_state, findings_state):
    if image_state is None:
        return None
    if not findings_state:
        return build_ecg_figure(image_state, [])
    row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row_index is None or row_index >= len(findings_state):
        return build_ecg_figure(image_state, findings_state)
    focus_index = findings_state[row_index].get("index")
    return build_ecg_figure(image_state, findings_state, focus_index=focus_index)


def reset_ecg_view(image_state, findings_state):
    if image_state is None:
        return None
    return build_ecg_figure(image_state, findings_state or [])


# ==================== 6. Web 界面 ====================

CLINIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #f4f7fb;
  --card: #ffffff;
  --text: #1f2937;
  --muted: #6b7280;
  --accent: #5aa7ff;
  --accent-warm: #ff9a6a;
  --border: rgba(148, 163, 184, 0.35);
  --shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
}

body, .gradio-container {
  background: var(--bg);
  font-family: 'Inter', sans-serif;
  color: var(--text);
}

.gradio-container {
  max-width: 1400px;
  margin: 0 auto;
}

h1, h2, h3 {
  color: #1e293b;
}

#landing-card, #workbench-left, #workbench-right {
  background: var(--card);
  border-radius: 18px;
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  padding: 24px;
}

#landing-hero h1 {
  margin-bottom: 6px;
}

#landing-drop {
  border: 2px dashed #cbd5e1;
  background: #f8fbff;
  border-radius: 16px;
  padding: 28px;
}

#landing-cta button {
  background: linear-gradient(135deg, #5aa7ff, #7bd3ff);
  color: #fff;
  border: none;
  font-weight: 600;
  border-radius: 999px;
  padding: 10px 24px;
}

#landing-status {
  color: #ef4444;
  font-weight: 500;
}

#workbench-header {
  align-items: center;
  gap: 12px;
}

#findings-table table {
  border-radius: 12px;
  overflow: hidden;
}

#report-box textarea {
  border-radius: 12px;
  border: 1px solid var(--border);
  background: #f8fafc;
}

#export-btn button, #reset-view button, #workbench-reset button {
  border-radius: 999px;
  font-weight: 600;
}
"""

with gr.Blocks(title="GEM Clinical ECG Workbench", css=CLINIC_CSS) as demo:
    findings_state = gr.State([])
    image_state = gr.State(None)

    with gr.Group(elem_id="landing", visible=True) as landing_group:
        with gr.Column(elem_id="landing-card"):
            gr.Markdown(
                "<div id='landing-hero'><h1>Clinical ECG Diagnosis Workbench</h1>"
                "<p>Upload raw <strong>.dat</strong> and <strong>.hea</strong> files for AI-assisted analysis.</p></div>"
            )
            file_input = gr.File(
                label="Drag & Drop .dat/.hea files",
                file_count="multiple",
                type="file",
                file_types=[".dat", ".hea"],
                elem_id="landing-drop"
            )
            window_start_slider = gr.Slider(
                label="时间窗口起点 (秒)",
                minimum=0,
                maximum=60,
                value=0,
                step=0.5,
                info="仅当记录长于 10 秒时生效，会自动裁剪到有效范围。"
            )
            start_btn = gr.Button("Start AI Analysis", elem_id="landing-cta")
            landing_status = gr.Markdown("", elem_id="landing-status")

    with gr.Group(elem_id="workbench", visible=False) as workbench_group:
        with gr.Row(elem_id="workbench-header"):
            gr.Markdown("## Diagnosis Workbench")
            new_btn = gr.Button("New Analysis", elem_id="workbench-reset")

        with gr.Row():
            with gr.Column(scale=7):
                with gr.Column(elem_id="workbench-left"):
                    gr.Markdown("### Evidence Viewer")
                    ecg_plot = gr.Plot(label="ECG Viewer")
                    reset_btn = gr.Button("Reset View", elem_id="reset-view")

            with gr.Column(scale=3):
                with gr.Column(elem_id="workbench-right"):
                    gr.Markdown("### AI Findings")
                    gr.Markdown("<span style='color:#6b7280'>Click a row to focus the waveform.</span>")
                    findings_table = gr.Dataframe(
                        headers=["ID", "Finding", "Lead", "Time"],
                        value=build_findings_rows([]),
                        interactive=False,
                        elem_id="findings-table"
                    )
                    gr.Markdown("### Editable Report")
                    report_box = gr.Textbox(lines=12, label="Report", elem_id="report-box")
                    export_btn = gr.Button("Export PDF", elem_id="export-btn")
                    export_file = gr.File(label="Download PDF", elem_id="export-file")

    start_btn.click(
        fn=start_analysis,
        inputs=[file_input, window_start_slider],
        outputs=[landing_group, workbench_group, ecg_plot, findings_table, report_box, findings_state, image_state, landing_status]
    )

    new_btn.click(
        fn=reset_to_landing,
        inputs=[],
        outputs=[landing_group, workbench_group, ecg_plot, findings_table, report_box, findings_state, image_state, landing_status, file_input, window_start_slider]
    )

    reset_btn.click(
        fn=reset_ecg_view,
        inputs=[image_state, findings_state],
        outputs=[ecg_plot]
    )

    findings_table.select(
        fn=focus_on_finding,
        inputs=[image_state, findings_state],
        outputs=[ecg_plot]
    )

    export_btn.click(
        fn=export_report_pdf,
        inputs=[report_box],
        outputs=[export_file]
    )

if __name__ == "__main__":
    # 允许局域网访问
    demo.launch(server_name="0.0.0.0", share=False)
