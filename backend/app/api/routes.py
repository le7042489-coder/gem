import json
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Response

from ..services.ecg_processor import (
    process_uploaded_files,
    load_signal_from_npy_path,
    load_signal_from_uploaded_npy,
)
from ..services.image_store import ImageStore
from ..services.inference import run_inference, run_inference_plus
from ..services.sample_catalog import SampleCatalog
from ..utils.parsing import format_time_range, DEFAULT_DIAG_PROMPT
from ..utils.plotting import findings_to_boxes, image_to_base64
from .schemas import (
    PredictResponse,
    ProcessFileResponse,
    Finding,
    Box,
    SampleListResponse,
    SampleItem,
    PredictPlusByIdRequest,
    PredictPlusResponse,
)

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


@router.get("/samples", response_model=SampleListResponse)
async def list_samples(
    source: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
):
    catalog = SampleCatalog.get()
    items, total = catalog.list_samples(source=source, q=q, offset=offset, limit=limit)
    return SampleListResponse(
        total=total,
        offset=offset,
        limit=limit,
        items=[
            SampleItem(
                id=item["id"],
                source=item["source"],
                image=item["image"],
                time_series=item["time_series"],
                mask_path=item.get("mask_path"),
            )
            for item in items
        ],
    )


@router.post("/predict_plus/by_id", response_model=PredictPlusResponse)
async def predict_plus_by_id(payload: PredictPlusByIdRequest):
    catalog = SampleCatalog.get()
    sample = catalog.get_sample(payload.sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"sample_id not found: {payload.sample_id}")

    signal_path = catalog.resolve_path(sample["time_series"])
    if not signal_path.exists():
        raise HTTPException(status_code=404, detail=f"time_series not found: {signal_path}")

    fs_original = None
    metadata = sample.get("metadata")
    if isinstance(metadata, dict) and metadata.get("fs_original") is not None:
        fs_original = metadata.get("fs_original")

    try:
        signal, preprocess = load_signal_from_npy_path(signal_path, fs_original=fs_original)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load sample {payload.sample_id}: {e}")

    if payload.patient_context is None:
        context_dict = {}
    elif hasattr(payload.patient_context, "model_dump"):
        context_dict = payload.patient_context.model_dump()
    else:
        context_dict = payload.patient_context.dict()
    result = run_inference_plus(
        signal_np=signal,
        sample_id=sample["id"],
        preprocess_meta=preprocess,
        text_query=payload.query,
        patient_context=context_dict,
        sample_record=sample,
    )
    return PredictPlusResponse(**result)


def _parse_patient_context_json(raw: Optional[str]) -> Dict[str, Any]:
    if raw is None or raw.strip() == "":
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid patient_context_json: {e}")
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="patient_context_json must be a JSON object.")
    return parsed


@router.post("/predict_plus/upload", response_model=PredictPlusResponse)
async def predict_plus_upload(
    time_series_file: UploadFile = File(...),
    image_file: Optional[UploadFile] = File(None),
    query: Optional[str] = Form(None),
    patient_context_json: Optional[str] = Form(None),
    fs_original: Optional[float] = Form(None),
):
    filename = (time_series_file.filename or "").lower()
    if not filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="time_series_file must be a .npy file.")

    try:
        signal, preprocess = load_signal_from_uploaded_npy(time_series_file, fs_original=fs_original)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded time_series_file: {e}")

    # Keep uploaded PNG as sidecar preview only (not used in model inference in v1).
    _ = image_file
    patient_context = _parse_patient_context_json(patient_context_json)
    sample_id = f"upload_{uuid.uuid4().hex[:8]}"

    result = run_inference_plus(
        signal_np=signal,
        sample_id=sample_id,
        preprocess_meta=preprocess,
        text_query=query,
        patient_context=patient_context,
        sample_record={"id": sample_id, "metadata": {}},
    )
    if image_file is not None:
        result["validation_warnings"] = list(result.get("validation_warnings", [])) + [
            "Uploaded image_file is stored as sidecar only and not used for model inference in this phase."
        ]
    return PredictPlusResponse(**result)


@router.get("/images/{image_id}")
async def get_image(image_id: str):
    payload = ImageStore.get().get_image(image_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"image_id not found: {image_id}")
    return Response(content=payload, media_type="image/png")
