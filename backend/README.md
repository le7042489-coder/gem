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

Response from `/predict` includes:
- `report`
- `findings` (structured)
- `boxes` (normalized coordinates 0-1)
- `image_base64` (JPEG)
- `image_width`, `image_height`

Use the normalized box coordinates to overlay anomaly regions in the frontend.
