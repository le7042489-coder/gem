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

MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/GEM-7B")
MODEL_BASE = _none_if_empty(os.getenv("MODEL_BASE"))
DEVICE_MAP = _parse_device_map(os.getenv("DEVICE_MAP", "auto"))
TARGET_FS = 500
TARGET_LEN = 5000
TEMP_UPLOAD_DIR = Path("temp_upload")

DEFAULT_DIAG_PROMPT = "Please interpret this ECG and provide a diagnosis."

# Generation defaults
MAX_NEW_TOKENS = 512
GROUNDING_MIN_NEW_TOKENS = 16

# API
API_TITLE = "GEM Clinical ECG Backend"
