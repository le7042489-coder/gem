from typing import List, Optional
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
