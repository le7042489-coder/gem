from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class Finding(BaseModel):
    index: int
    label: str
    symptom: str
    lead: str
    start: Optional[float] = None
    end: Optional[float] = None
    time_range: str


class Box(BaseModel):
    index: int
    label: str
    x1: float
    x2: float
    y1: float
    y2: float


class PredictResponse(BaseModel):
    report: str
    findings: List[Finding]
    boxes: List[Box]
    image_base64: str
    image_width: int
    image_height: int
    raw_output: str


class ProcessFileResponse(BaseModel):
    message: str
    fs: Optional[float]
    shape: Optional[List[int]]


class SampleItem(BaseModel):
    id: str
    source: str
    image: str
    time_series: str
    mask_path: Optional[str] = None


class SampleListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    items: List[SampleItem]


class PatientContext(BaseModel):
    age: Optional[Any] = None
    sex: Optional[Any] = None
    encounter: Optional[Any] = None


class PredictPlusByIdRequest(BaseModel):
    sample_id: str
    query: Optional[str] = None
    patient_context: Optional[PatientContext] = None


class EvidenceItem(BaseModel):
    id: str
    lead: str
    t_start_ms: float
    t_end_ms: float
    measurement_name: str
    value: float
    unit: str
    source: str
    quality: str


class StructuredSection(BaseModel):
    text: str
    evidence_ids: List[str]


class StructuredReport(BaseModel):
    rhythm: StructuredSection
    conduction: StructuredSection
    st_t: StructuredSection
    axis: StructuredSection
    summary: str


class ViewerInfo(BaseModel):
    image_id: str
    layout: str
    total_ms: int
    segment_ms: int


class PreprocessInfo(BaseModel):
    fs_original: float
    len_original: int
    resampled: bool
    resample_method: str
    cropped: bool
    padded: bool


class PredictPlusResponse(BaseModel):
    sample_id: str
    structured: StructuredReport
    evidence: List[EvidenceItem]
    raw_model_output: str
    parser_status: str
    context_used: Dict[str, Any]
    fs_used: int
    preprocess: PreprocessInfo
    viewer: ViewerInfo
    validation_warnings: List[str]
