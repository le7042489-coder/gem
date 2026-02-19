from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import MACHINE_MEASUREMENTS_PATH


_KEY_CANDIDATES = (
    "id",
    "sample_id",
    "question_id",
    "study_id",
    "ecg_id",
    "record_id",
    "file_name",
    "path",
)


class MeasurementStore:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._records: Dict[str, Any] = {}
        self._ready = False
        self._load()

    @classmethod
    def get(cls) -> "MeasurementStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _register_record(self, key: str, payload: Any) -> None:
        if key is None:
            return
        key = str(key).strip()
        if not key:
            return
        self._records[key] = payload
        # Fallback key forms for easier matching by sample_id suffix.
        stem = Path(key).stem
        if stem and stem not in self._records:
            self._records[stem] = payload

    def _pick_row_key(self, row: Dict[str, Any]) -> Optional[str]:
        for k in _KEY_CANDIDATES:
            if k in row and str(row[k]).strip():
                return str(row[k]).strip()
        return None

    def _load_json(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            for k, v in obj.items():
                self._register_record(k, v)
            return
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    key = self._pick_row_key(item)
                    if key is not None:
                        self._register_record(key, item)

    def _load_jsonl(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    key = self._pick_row_key(item)
                    if key is not None:
                        self._register_record(key, item)

    def _load_csv(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = self._pick_row_key(row)
                if key is not None:
                    self._register_record(key, row)

    def _load(self) -> None:
        self._ready = True
        if not MACHINE_MEASUREMENTS_PATH:
            return
        path = Path(MACHINE_MEASUREMENTS_PATH)
        if not path.exists():
            return
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                self._load_json(path)
            elif suffix == ".jsonl":
                self._load_jsonl(path)
            elif suffix == ".csv":
                self._load_csv(path)
        except Exception:
            # Keep inference service resilient; just disable external measurements on error.
            self._records = {}

    def get_measurements(self, sample_id: str) -> Optional[Any]:
        if not self._ready:
            self._load()
        if not sample_id:
            return None
        sid = str(sample_id).strip()
        if sid in self._records:
            return self._records[sid]
        stem = Path(sid).stem
        if stem in self._records:
            return self._records[stem]
        if "_" in sid:
            tail = sid.split("_")[-1]
            if tail in self._records:
                return self._records[tail]
        return None

