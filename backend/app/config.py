import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _none_if_empty(value):
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _parse_device_map(value):
    if value is None:
        return "auto"
    raw = str(value).strip()
    if raw == "":
        return "auto"
    if raw.lower() == "auto":
        return "auto"
    if raw.startswith("{") or raw.startswith("["):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    if raw.isdigit():
        return {"": int(raw)}
    return {"": raw}


def _parse_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/GEM-7B")
MODEL_BASE = _none_if_empty(os.getenv("MODEL_BASE"))
DEVICE_MAP = _parse_device_map(os.getenv("DEVICE_MAP", "auto"))
TARGET_FS = 500
TARGET_LEN = 5000
TEMP_UPLOAD_DIR = Path("temp_upload")
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DIAG_PROMPT = "Please interpret this ECG and provide a diagnosis."
DEFAULT_PLUS_PROMPT = "Interpret this ECG and provide a grounded structured report."

# Generation defaults
MAX_NEW_TOKENS = 512
GROUNDING_MIN_NEW_TOKENS = 16

# Data / inference
LOCAL_INDEX_PATH = Path(os.getenv("LOCAL_INDEX_PATH", "data/local_index.json"))
LOCAL_INFER_PATH = Path(os.getenv("LOCAL_INFER_PATH", "data/local_infer.json"))
MIXED_TRAIN_PATH = Path(os.getenv("MIXED_TRAIN_PATH", "data/mixed_train.json"))
MACHINE_MEASUREMENTS_PATH = _none_if_empty(os.getenv("MACHINE_MEASUREMENTS_PATH"))
GEM_PLUS_MAX_EVIDENCE = _parse_int(os.getenv("GEM_PLUS_MAX_EVIDENCE"), 24)
IMAGE_CACHE_MAX_ITEMS = _parse_int(os.getenv("IMAGE_CACHE_MAX_ITEMS"), 256)
UPLOAD_DEFAULT_FS = _parse_float(os.getenv("UPLOAD_DEFAULT_FS"), float(TARGET_FS))

# API
API_TITLE = "GEM Clinical ECG Backend"
