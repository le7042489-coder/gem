from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import LOCAL_INDEX_PATH, LOCAL_INFER_PATH, MIXED_TRAIN_PATH, REPO_ROOT


def _as_path(value: Path) -> Path:
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def _read_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if isinstance(data.get("items"), list):
            return [x for x in data["items"] if isinstance(x, dict)]
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _infer_source(record: Dict[str, Any]) -> str:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        src = metadata.get("source")
        if src:
            return str(src)
    for key in ("time_series", "image"):
        path = str(record.get(key, "")).lower()
        if "ptbxl" in path:
            return "ptbxl"
        if "ludb" in path:
            return "ludb"
        if "mimic" in path:
            return "mimic_demo"
    return "unknown"


def _normalize_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sample_id = str(raw.get("id", "")).strip()
    image = str(raw.get("image", "")).strip()
    time_series = str(raw.get("time_series", "")).strip()
    if not sample_id or not image or not time_series:
        return None
    mask_path = raw.get("mask_path")
    if isinstance(mask_path, str) and mask_path.strip().lower() == "none":
        mask_path = None
    return {
        "id": sample_id,
        "source": _infer_source(raw),
        "image": image,
        "time_series": time_series,
        "mask_path": mask_path,
        "metadata": raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
        "machine_measurements": raw.get("machine_measurements"),
    }


class SampleCatalog:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._items: List[Dict[str, Any]] = []
        self._item_map: Dict[str, Dict[str, Any]] = {}
        self.reload()

    @classmethod
    def get(cls) -> "SampleCatalog":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_from_first_existing(self) -> List[Dict[str, Any]]:
        for candidate in (LOCAL_INDEX_PATH, LOCAL_INFER_PATH, MIXED_TRAIN_PATH):
            path = _as_path(candidate)
            if path.exists():
                return _read_json_list(path)
        return []

    def reload(self) -> None:
        raw_items = self._load_from_first_existing()
        items: List[Dict[str, Any]] = []
        item_map: Dict[str, Dict[str, Any]] = {}
        for raw in raw_items:
            item = _normalize_record(raw)
            if item is None:
                continue
            items.append(item)
            item_map[item["id"]] = item
        self._items = items
        self._item_map = item_map

    def get_sample(self, sample_id: str) -> Optional[Dict[str, Any]]:
        if sample_id in self._item_map:
            return self._item_map[sample_id]
        # fallback: lazy reload for hot-swapped index files
        self.reload()
        return self._item_map.get(sample_id)

    def list_samples(
        self,
        source: Optional[str] = None,
        q: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> Tuple[List[Dict[str, Any]], int]:
        src = (source or "").strip().lower()
        query = (q or "").strip().lower()
        filtered: List[Dict[str, Any]] = []
        for item in self._items:
            if src and item.get("source", "").lower() != src:
                continue
            if query:
                hay = " ".join([
                    str(item.get("id", "")),
                    str(item.get("source", "")),
                    str(item.get("time_series", "")),
                    str(item.get("image", "")),
                ]).lower()
                if query not in hay:
                    continue
            filtered.append(item)
        total = len(filtered)
        offset = max(0, int(offset))
        limit = max(1, min(200, int(limit)))
        return filtered[offset: offset + limit], total

    def resolve_path(self, rel_or_abs: str) -> Path:
        path = Path(rel_or_abs)
        if path.is_absolute():
            return path
        return REPO_ROOT / path
