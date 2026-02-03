import threading
from typing import Dict, Any

import torch
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token

from ..config import DEFAULT_DIAG_PROMPT, MAX_NEW_TOKENS, GROUNDING_MIN_NEW_TOKENS
from ..utils.parsing import (
    parse_findings_output,
    extract_diagnosis_summary,
    format_diagnosis_report,
    format_findings_only,
    assign_findings_indices,
)
from ..utils.plotting import plot_ecg_3x4_grid
from .model_manager import ModelManager


_infer_lock = threading.Lock()


def _build_prompt(text_query: str, mode: str) -> str:
    if mode == "grounding":
        return (
            f"Task: Locate {text_query} in the ECG.\n"
            "Output format (one per line):\n"
            "FINDING|<symptom>|<lead>|<start_s>|<end_s>\n"
            "Rules:\n"
            "- Use lead names exactly: I, II, III, aVR, aVL, aVF, V1-V6.\n"
            "- Use absolute time in seconds within 0-10.\n"
            "- If a finding spans the full 10s, use 0-10.\n"
            "Begin."
        )
    return (
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


def run_inference(signal_np: np.ndarray, text_query: str = DEFAULT_DIAG_PROMPT, mode: str = "diagnosis") -> Dict[str, Any]:
    manager = ModelManager.get()

    sig_mean = signal_np.mean()
    sig_std = signal_np.std()
    if sig_std > 1e-6:
        signal_np = (signal_np - sig_mean) / sig_std

    ecg_image = plot_ecg_3x4_grid(signal_np)

    sig_tensor = torch.from_numpy(signal_np).float()
    ecgs_tensor = sig_tensor.unsqueeze(0).half().cuda()

    if hasattr(manager.image_processor, "preprocess"):
        image_tensor = manager.image_processor.preprocess(ecg_image, return_tensors='pt')['pixel_values'][0]
    else:
        image_tensor = manager.image_processor(ecg_image, return_tensors='pt')['pixel_values'][0]
    image_tensor = image_tensor.unsqueeze(0).half().cuda()

    qs = _build_prompt(text_query, mode)
    if manager.model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, manager.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    with _infer_lock:
        with torch.inference_mode():
            output_ids = manager.model.generate(
                input_ids,
                images=image_tensor,
                ecgs=ecgs_tensor,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=GROUNDING_MIN_NEW_TOKENS if mode == "grounding" else None,
                use_cache=True,
                stopping_criteria=None
            )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    output_text = manager.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not output_text:
        output_text = manager.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
        if not output_text:
            output_text = "[Error] 模型输出了空内容。"

    if mode == "grounding":
        findings = parse_findings_output(output_text, default_symptom=text_query)
        report_text = format_findings_only(findings, default_symptom=text_query)
    else:
        findings = parse_findings_output(output_text)
        summary = extract_diagnosis_summary(output_text)
        report_text = format_diagnosis_report(summary, findings)
        if not findings:
            report_text = report_text + "\n\nRaw Output:\n" + output_text

    indexed_findings = assign_findings_indices(findings)

    return {
        "image": ecg_image,
        "report": report_text,
        "findings": indexed_findings,
        "raw_output": output_text,
    }
