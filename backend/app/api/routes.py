from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form

from ..services.ecg_processor import process_uploaded_files
from ..services.inference import run_inference
from ..utils.parsing import format_time_range, DEFAULT_DIAG_PROMPT
from ..utils.plotting import findings_to_boxes, image_to_base64
from .schemas import PredictResponse, ProcessFileResponse, Finding, Box

router = APIRouter()


@router.post("/process_file", response_model=ProcessFileResponse)
async def process_file(files: List[UploadFile] = File(...)):
    signal, fs, msg = process_uploaded_files(files)
    if signal is None:
        return ProcessFileResponse(message=msg, fs=None, shape=None)
    return ProcessFileResponse(message=msg, fs=fs, shape=list(signal.shape))


@router.post("/predict", response_model=PredictResponse)
async def predict(
    files: List[UploadFile] = File(...),
    mode: str = Form("diagnosis"),
    query: Optional[str] = Form(None),
):
    signal, fs, msg = process_uploaded_files(files)
    if signal is None:
        return PredictResponse(
            report=msg,
            findings=[],
            boxes=[],
            image_base64="",
            image_width=0,
            image_height=0,
            raw_output=msg,
        )

    text_query = query or DEFAULT_DIAG_PROMPT
    result = run_inference(signal, text_query=text_query, mode=mode)
    image = result["image"]
    findings = result["findings"]

    image_width, image_height = image.size
    boxes = findings_to_boxes(findings, (image_width, image_height))
    image_b64 = image_to_base64(image)

    finding_models = [
        Finding(
            index=f["index"],
            label=f["label"],
            symptom=f["symptom"],
            lead=f["lead"],
            start=f.get("start"),
            end=f.get("end"),
            time_range=format_time_range(f.get("start"), f.get("end"))
        )
        for f in findings
    ]

    box_models = [
        Box(
            index=b["index"],
            label=b["label"],
            x1=b["x1"],
            x2=b["x2"],
            y1=b["y1"],
            y2=b["y2"]
        )
        for b in boxes
    ]

    return PredictResponse(
        report=result["report"],
        findings=finding_models,
        boxes=box_models,
        image_base64=image_b64,
        image_width=image_width,
        image_height=image_height,
        raw_output=result["raw_output"],
    )
