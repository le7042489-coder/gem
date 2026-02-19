from __future__ import annotations

import json
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token

from ..config import (
    DEFAULT_DIAG_PROMPT,
    DEFAULT_PLUS_PROMPT,
    GROUNDING_MIN_NEW_TOKENS,
    MAX_NEW_TOKENS,
    TARGET_FS,
    TARGET_LEN,
    GEM_PLUS_MAX_EVIDENCE,
)
from ..utils.parsing import (
    parse_findings_output,
    extract_diagnosis_summary,
    format_diagnosis_report,
    format_findings_only,
    assign_findings_indices,
)
from ..utils.plotting import plot_ecg_3x4_grid
from .evidence_extractor import build_evidence_candidates
from .image_store import ImageStore
from .model_manager import ModelManager
from .response_validator import validate_predict_plus_output


_infer_lock = threading.Lock()


def _zscore_signal(signal_np: np.ndarray) -> np.ndarray:
    sig_mean = float(signal_np.mean())
    sig_std = float(signal_np.std())
    if sig_std > 1e-6:
        return (signal_np - sig_mean) / sig_std
    return signal_np


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


def _build_plus_prompt(
    text_query: str,
    patient_context: Dict[str, Any],
    evidence_candidates: List[Dict[str, Any]],
) -> str:
    allowed_ids = [x["id"] for x in evidence_candidates]
    prompt = {
        "task": text_query,
        "patient_context": patient_context,
        "evidence_candidates": evidence_candidates,
        "required_output_schema": {
            "rhythm": {"text": "string", "evidence_ids": ["subset of allowed evidence IDs"]},
            "conduction": {"text": "string", "evidence_ids": ["subset of allowed evidence IDs"]},
            "st_t": {"text": "string", "evidence_ids": ["subset of allowed evidence IDs"]},
            "axis": {"text": "string", "evidence_ids": ["subset of allowed evidence IDs"]},
            "summary": "string",
        },
        "constraints": [
            "Output MUST be valid JSON only (no markdown fence, no extra text).",
            "evidence_ids MUST come from allowed IDs only.",
            "If uncertain, keep text concise and evidence_ids as empty list.",
            f"Allowed evidence IDs: {allowed_ids}",
        ],
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def _generate_output_text(
    signal_np: np.ndarray,
    prompt_body: str,
    mode: str,
) -> Tuple[str, Any]:
    manager = ModelManager.get()

    ecg_image = plot_ecg_3x4_grid(signal_np)

    sig_tensor = torch.from_numpy(signal_np).float()
    ecgs_tensor = sig_tensor.unsqueeze(0).half().cuda()

    if hasattr(manager.image_processor, "preprocess"):
        image_tensor = manager.image_processor.preprocess(ecg_image, return_tensors="pt")["pixel_values"][0]
    else:
        image_tensor = manager.image_processor(ecg_image, return_tensors="pt")["pixel_values"][0]
    image_tensor = image_tensor.unsqueeze(0).half().cuda()

    qs = prompt_body
    if manager.model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, manager.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

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
                stopping_criteria=None,
            )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    output_text = manager.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not output_text:
        output_text = manager.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
        if not output_text:
            output_text = "[Error] 模型输出了空内容。"

    return output_text, ecg_image


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def _extract_json_block(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1]


def _json_repair(text: str) -> Optional[Dict[str, Any]]:
    block = _extract_json_block(text)
    if block is None:
        return None
    candidate = block.replace("\u201c", "\"").replace("\u201d", "\"").replace("\u2019", "'")
    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*]", "]", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _fallback_structured(output_text: str) -> Dict[str, Any]:
    summary = extract_diagnosis_summary(output_text) or output_text
    return {
        "rhythm": {"text": "", "evidence_ids": []},
        "conduction": {"text": "", "evidence_ids": []},
        "st_t": {"text": "", "evidence_ids": []},
        "axis": {"text": "", "evidence_ids": []},
        "summary": summary.strip(),
    }


def _parse_structured_output(output_text: str) -> Tuple[Dict[str, Any], str]:
    cleaned = _strip_code_fence(output_text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed, "strict_json"
    except json.JSONDecodeError:
        pass

    repaired = _json_repair(cleaned)
    if isinstance(repaired, dict):
        return repaired, "json_repair"

    return _fallback_structured(output_text), "legacy_finding_fallback"


def _normalize_context(patient_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    ctx = patient_context if isinstance(patient_context, dict) else {}
    return {
        "age": ctx.get("age"),
        "sex": ctx.get("sex"),
        "encounter": ctx.get("encounter"),
    }


def run_inference(signal_np: np.ndarray, text_query: str = DEFAULT_DIAG_PROMPT, mode: str = "diagnosis") -> Dict[str, Any]:
    signal_for_model = _zscore_signal(signal_np)
    prompt_body = _build_prompt(text_query, mode=mode)
    output_text, ecg_image = _generate_output_text(signal_for_model, prompt_body, mode=mode)

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


def run_inference_plus(
    signal_np: np.ndarray,
    sample_id: str,
    preprocess_meta: Dict[str, Any],
    text_query: Optional[str] = None,
    patient_context: Optional[Dict[str, Any]] = None,
    sample_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    fs_used = TARGET_FS
    context_used = _normalize_context(patient_context)

    evidence_candidates = build_evidence_candidates(
        signal_12l=signal_np,
        fs_used=fs_used,
        sample_id=sample_id,
        sample_record=sample_record,
        max_evidence=GEM_PLUS_MAX_EVIDENCE,
    )

    signal_for_model = _zscore_signal(signal_np)
    prompt_body = _build_plus_prompt(
        text_query or DEFAULT_PLUS_PROMPT,
        patient_context=context_used,
        evidence_candidates=evidence_candidates,
    )
    output_text, ecg_image = _generate_output_text(signal_for_model, prompt_body, mode="diagnosis")

    structured_raw, parser_status = _parse_structured_output(output_text)
    structured, evidence, validation_warnings = validate_predict_plus_output(
        structured=structured_raw,
        evidence=evidence_candidates,
        max_evidence=GEM_PLUS_MAX_EVIDENCE,
    )
    image_id = ImageStore.get().put_image(ecg_image)

    if parser_status != "strict_json":
        validation_warnings = [f"Output parsed via {parser_status}."] + validation_warnings

    return {
        "sample_id": sample_id,
        "structured": structured,
        "evidence": evidence,
        "raw_model_output": output_text,
        "parser_status": parser_status,
        "context_used": context_used,
        "fs_used": fs_used,
        "preprocess": preprocess_meta,
        "viewer": {
            "image_id": image_id,
            "layout": "standard_3x4",
            "total_ms": int(round(TARGET_LEN / TARGET_FS * 1000.0)),
            "segment_ms": int(round((TARGET_LEN / TARGET_FS * 1000.0) / 4.0)),
        },
        "validation_warnings": validation_warnings,
    }

