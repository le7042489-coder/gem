#!/usr/bin/env python3
import argparse
import io
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _npy_bytes(shape: Tuple[int, int] = (12, 5000)) -> bytes:
    import numpy as np

    arr = np.zeros(shape, dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _pick_sample_id(client) -> str:
    r = client.get("/samples", params={"limit": 1})
    _require(r.status_code == 200, f"GET /samples expected 200, got {r.status_code}: {r.text}")
    payload = r.json()
    items = payload.get("items") or []
    _require(items and isinstance(items, list), f"GET /samples returned empty items: {payload}")
    sample_id = items[0].get("id")
    _require(isinstance(sample_id, str) and sample_id, f"Invalid sample id in /samples payload: {items[0]}")
    return sample_id


def _smoke_common(client) -> None:
    r = client.get("/health")
    _require(r.status_code == 200, f"GET /health expected 200, got {r.status_code}: {r.text}")
    _require(r.json().get("status") == "ok", f"GET /health unexpected payload: {r.json()}")

    r = client.get("/samples", params={"limit": 2})
    _require(r.status_code == 200, f"GET /samples expected 200, got {r.status_code}: {r.text}")
    payload = r.json()
    for k in ("total", "offset", "limit", "items"):
        _require(k in payload, f"GET /samples missing key {k}: {payload}")


def _smoke_predict_plus_upload(client) -> Dict[str, Any]:
    npy = _npy_bytes((12, 5000))
    r = client.post(
        "/predict_plus/upload",
        files={"time_series_file": ("smoke.npy", npy, "application/octet-stream")},
        data={"query": "smoke"},
    )
    _require(r.status_code == 200, f"POST /predict_plus/upload expected 200, got {r.status_code}: {r.text}")
    payload = r.json()
    _require("viewer" in payload and isinstance(payload["viewer"], dict), f"upload missing viewer: {payload}")
    _require("image_id" in payload["viewer"], f"upload missing viewer.image_id: {payload}")

    r_bad = client.post(
        "/predict_plus/upload",
        files={"time_series_file": ("smoke.txt", b"abc", "text/plain")},
    )
    _require(r_bad.status_code == 400, f"POST /predict_plus/upload bad ext expected 400, got {r_bad.status_code}: {r_bad.text}")

    return payload


def _smoke_predict_plus_by_id(client, sample_id: str) -> Dict[str, Any]:
    t0 = time.time()
    r = client.post("/predict_plus/by_id", json={"sample_id": sample_id, "query": "smoke"})
    dt = time.time() - t0
    _require(r.status_code == 200, f"POST /predict_plus/by_id expected 200, got {r.status_code}: {r.text}")
    payload = r.json()
    _require(payload.get("sample_id") == sample_id, f"by_id sample_id mismatch: {payload.get('sample_id')} != {sample_id}")
    _require("structured" in payload, f"by_id missing structured: {payload}")
    _require("viewer" in payload and isinstance(payload["viewer"], dict), f"by_id missing viewer: {payload}")
    _require("image_id" in payload["viewer"], f"by_id missing viewer.image_id: {payload}")
    print(f"[smoke] /predict_plus/by_id ok in {dt:.2f}s (parser_status={payload.get('parser_status')})")

    r_404 = client.post("/predict_plus/by_id", json={"sample_id": "does_not_exist"})
    _require(r_404.status_code == 404, f"POST /predict_plus/by_id 404 expected, got {r_404.status_code}: {r_404.text}")

    return payload


def _smoke_fetch_image(client, image_id: str) -> None:
    r = client.get(f"/images/{image_id}")
    _require(r.status_code == 200, f"GET /images/{{id}} expected 200, got {r.status_code}: {r.text}")
    ct = (r.headers.get("content-type") or "").lower()
    _require(ct.startswith("image/"), f"GET /images/{{id}} unexpected content-type: {ct}")
    _require(len(r.content) > 0, "GET /images/{id} returned empty body")


def _install_stub_inference() -> Tuple[Any, Any]:
    import backend.app.api.routes as routes
    from backend.app.services.image_store import ImageStore
    from PIL import Image

    original = routes.run_inference_plus

    def fake_run_inference_plus(
        *,
        signal_np,
        sample_id: str,
        preprocess_meta: Dict[str, Any],
        text_query: Optional[str] = None,
        patient_context: Optional[Dict[str, Any]] = None,
        sample_record: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        _ = (signal_np, text_query, patient_context, sample_record)
        img = Image.new("RGB", (64, 64), "white")
        image_id = ImageStore.get().put_image(img)
        return {
            "sample_id": sample_id,
            "structured": {
                "rhythm": {"text": "ok", "evidence_ids": ["ev_1"]},
                "conduction": {"text": "ok", "evidence_ids": []},
                "st_t": {"text": "ok", "evidence_ids": []},
                "axis": {"text": "ok", "evidence_ids": []},
                "summary": "ok",
            },
            "evidence": [
                {
                    "id": "ev_1",
                    "lead": "II",
                    "t_start_ms": 100.0,
                    "t_end_ms": 200.0,
                    "measurement_name": "RR",
                    "value": 800.0,
                    "unit": "ms",
                    "source": "algorithmic",
                    "quality": "low",
                }
            ],
            "raw_model_output": "{}",
            "parser_status": "strict_json",
            "context_used": {"age": None, "sex": None, "encounter": None},
            "fs_used": 500,
            "preprocess": preprocess_meta,
            "viewer": {"image_id": image_id, "layout": "standard_3x4", "total_ms": 10000, "segment_ms": 2500},
            "validation_warnings": [],
        }

    routes.run_inference_plus = fake_run_inference_plus
    return routes, original


def _run(mode: str) -> None:
    try:
        from fastapi.testclient import TestClient
        from backend.app.main import app
    except Exception as e:
        raise RuntimeError(
            "Failed to import backend dependencies. Run with: conda run -n gem python scripts/smoke_backend_api.py"
        ) from e

    client = TestClient(app)

    if mode in ("stub", "both"):
        print("[smoke] running stub smoke...")
        routes_mod, original = _install_stub_inference()
        try:
            _smoke_common(client)
            sample_id = _pick_sample_id(client)
            payload_by_id = _smoke_predict_plus_by_id(client, sample_id)
            payload_upload = _smoke_predict_plus_upload(client)
            _smoke_fetch_image(client, payload_by_id["viewer"]["image_id"])
            _smoke_fetch_image(client, payload_upload["viewer"]["image_id"])
        finally:
            routes_mod.run_inference_plus = original
        print("[smoke] stub smoke ok")

    if mode in ("real", "both"):
        print("[smoke] running real smoke (will load model)...")
        _smoke_common(client)
        sample_id = _pick_sample_id(client)
        payload_by_id = _smoke_predict_plus_by_id(client, sample_id)
        _smoke_fetch_image(client, payload_by_id["viewer"]["image_id"])
        payload_upload = _smoke_predict_plus_upload(client)
        _smoke_fetch_image(client, payload_upload["viewer"]["image_id"])
        print("[smoke] real smoke ok")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="GEM+ backend API smoke tests (stub + real).")
    parser.add_argument("--mode", choices=("stub", "real", "both"), default="both")
    args = parser.parse_args(argv)

    _run(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
