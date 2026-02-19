# GEM FastAPI Backend

## Quick start

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

- `GET /health`
- `POST /process_file` (multipart files: .dat + .hea)
- `POST /predict` (multipart files + optional form fields: mode, query)
- `GET /samples` (local sample index for GEM+)
- `POST /predict_plus/by_id` (JSON body with `sample_id`, optional `query`, optional `patient_context`)
- `POST /predict_plus/upload` (multipart with required `.npy` + optional `.png`)
- `GET /images/{image_id}` (fetch standard-rendered ECG image used by GEM+ viewer)

Response from `/predict` includes:
- `report`
- `findings` (structured)
- `boxes` (normalized coordinates 0-1)
- `image_base64` (JPEG)
- `image_width`, `image_height`

Use the normalized box coordinates to overlay anomaly regions in the frontend.

Response from `/predict_plus/*` is lightweight and omits inline image payload:
- `structured` (rhythm / conduction / st_t / axis / summary)
- `evidence` (lead + ms time window + measurement/value/unit/source/quality)
- `raw_model_output`, `parser_status`, `validation_warnings`
- `fs_used`
- `preprocess` (`fs_original`, `len_original`, `resampled`, `resample_method`, `cropped`, `padded`)
- `viewer` (`image_id`, `layout`, `total_ms`, `segment_ms`)

Frontend should compute highlight boxes locally from `lead + t_start_ms + t_end_ms` against the fixed `standard_3x4` layout, and fetch the image via `GET /images/{image_id}`.
