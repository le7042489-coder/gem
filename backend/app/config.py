from pathlib import Path

MODEL_PATH = "checkpoints/GEM-7B"
MODEL_BASE = None
TARGET_FS = 500
TARGET_LEN = 5000
TEMP_UPLOAD_DIR = Path("temp_upload")

DEFAULT_DIAG_PROMPT = "Please interpret this ECG and provide a diagnosis."

# Generation defaults
MAX_NEW_TOKENS = 512
GROUNDING_MIN_NEW_TOKENS = 16

# API
API_TITLE = "GEM Clinical ECG Backend"
