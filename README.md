# GEM Engineering Platform / GEM 工程平台

## 项目定位 / Project Scope
This repository is maintained as an engineering project for ECG multimodal model development and system delivery.
本仓库定位为工程交付项目，聚焦安装、运行、调试、测试与集成，不作为科研论文主页。

## 核心能力 / Core Capabilities
- `backend/`: FastAPI service for ECG inference, structured output, and image serving.
- `frontend/`: Next.js workbench for sample browsing, prediction triggering, and visualization.
- `scripts/gem_pipeline.py`: unified CLI for config validation, training, fine-tuning, and evaluation orchestration.
- `predict_plus` flow: sample-based and upload-based inference APIs with lightweight viewer metadata.
- `tests/` + `scripts/smoke_backend_api.py`: unit checks and API smoke verification paths.

## 系统架构 / System Architecture
```text
[Browser/User]
   |
   v
[frontend (Next.js)]
   | REST
   v
[backend (FastAPI)]
   | calls model runtime + preprocessors
   v
[llava/* + ecg_coca/*]
   |
   +--> [Image store / sample catalog]
   |
   +--> [scripts/gem_pipeline.py]
           |
           v
      [checkpoints/ + eval_outputs/ + reports]
```

## 快速开始 / Quick Start
1. Clone and install dependencies.

```bash
git clone https://github.com/lanxiang1017/GEM.git
cd GEM
bash setup.sh
```

`setup.sh` supports `ENV_NAME`, `PYTHON_VERSION`, `VENV_DIR`, `INSTALL_BACKEND`, `FLASH_ATTN_MODE`.

2. Create local environment file.

```bash
cp .env.example .env
```

3. Start backend.

```bash
# If backend deps were skipped in setup:
# pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

4. Start frontend.

```bash
cd frontend
npm install
export NEXT_PUBLIC_API_BASE=http://localhost:8000
npm run dev
```

5. Verify service health.

```bash
curl http://localhost:8000/health
curl "http://localhost:8000/samples?limit=1"
```

Frontend default URL: `http://localhost:3000`

## 环境变量 / Environment Variables
Use `.env` at repo root for backend/evaluation values.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | empty | Optional; required by OpenAI-based scoring/evaluation scripts. |
| `AZURE_OPENAI_API_KEY` | empty | Optional Azure OpenAI credential. |
| `AZURE_OPENAI_ENDPOINT` | empty | Optional Azure OpenAI endpoint. |
| `EVAL_MODEL` | `gpt-4o-2024-08-06` | Default model override for report evaluation tooling. |
| `MODEL_PATH` | `checkpoints/GEM-7B` | Backend model path. |
| `MODEL_BASE` | empty | Optional base model for merged/delta style loading. |
| `DEVICE_MAP` | `auto` | Runtime placement (`auto`, JSON map, or device id). |
| `LOCAL_INDEX_PATH` | `data/local_index.json` | Sample catalog index consumed by `/samples` and `/predict_plus/by_id`. |
| `LOCAL_INFER_PATH` | `data/local_infer.json` | Optional local inference metadata source. |
| `MIXED_TRAIN_PATH` | `data/mixed_train.json` | Default mixed-train manifest path. |
| `MACHINE_MEASUREMENTS_PATH` | empty | Optional machine measurement source for enriched outputs. |
| `GEM_PLUS_MAX_EVIDENCE` | `24` | Maximum evidence items kept in structured response. |
| `IMAGE_CACHE_MAX_ITEMS` | `256` | In-memory rendered image cache size. |
| `UPLOAD_DEFAULT_FS` | `500.0` | Default sampling frequency for uploaded `.npy` if not provided. |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | Frontend backend base URL (export in shell or set in frontend env). |

## 后端接口速查 / Backend API Quick Reference
Core endpoints for engineering integration:

- `GET /health`
- `GET /samples`
- `POST /predict_plus/by_id`
- `POST /predict_plus/upload`
- `GET /images/{image_id}`

Minimal curl sequence:

```bash
# 1) Health
curl http://localhost:8000/health

# 2) List samples
curl "http://localhost:8000/samples?limit=3"

# 3) Predict by indexed sample id
curl -X POST "http://localhost:8000/predict_plus/by_id" \
  -H "Content-Type: application/json" \
  -d '{"sample_id":"<sample_id>","query":"Interpret this ECG and provide a grounded structured report."}'

# 4) Predict by uploaded .npy
curl -X POST "http://localhost:8000/predict_plus/upload" \
  -F "time_series_file=@/absolute/path/to/sample.npy" \
  -F "query=Interpret this ECG and provide a grounded structured report."

# 5) Fetch rendered image via viewer.image_id from predict_plus response
curl "http://localhost:8000/images/<image_id>" --output ecg.png
```

Detailed response fields and compatibility endpoints are documented in [backend/README.md](backend/README.md).

## Pipeline 命令入口 / Unified Pipeline Entry
Unified command shape:

```bash
python scripts/gem_pipeline.py <subcommand> --config configs/pipelines/gem_default.yaml
```

Recommended entry commands:

```bash
# Validate config schema/types
python scripts/gem_pipeline.py validate-config --config configs/pipelines/gem_default.yaml

# Inspect resolved command without launching heavy jobs
python scripts/gem_pipeline.py train \
  --config configs/pipelines/gem_default.yaml \
  --dry-run \
  --print-effective-config

# Override any config value
python scripts/gem_pipeline.py finetune \
  --config configs/pipelines/gem_default.yaml \
  --set finetune.output_dir=checkpoints/gem-medts-v1 \
  --dry-run
```

Training/evaluation full runbook: [docs/engineering/training-evaluation.md](docs/engineering/training-evaluation.md)

## 测试与验证 / Testing & Validation
Run unit + pipeline tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Run backend smoke tests:

```bash
# Fast mode: no real model execution
python scripts/smoke_backend_api.py --mode stub

# Full mode: loads model and runs real inference
python scripts/smoke_backend_api.py --mode real
```

Recommended smoke path for new environment:
1. `GET /health`
2. `GET /samples`
3. `scripts/smoke_backend_api.py --mode stub`

## 运维文档索引 / Engineering Docs Index
- Training and evaluation runbook: [docs/engineering/training-evaluation.md](docs/engineering/training-evaluation.md)
- Data and model asset guide: [docs/engineering/data-model-assets.md](docs/engineering/data-model-assets.md)
- Backend service details: [backend/README.md](backend/README.md)
- Frontend setup details: [frontend/README.md](frontend/README.md)
- Legacy script migration notes: [docs/migrations/llava_scripts_legacy.md](docs/migrations/llava_scripts_legacy.md)

## 许可证 / License
Licensed under Apache License 2.0. See [LICENSE](LICENSE).
