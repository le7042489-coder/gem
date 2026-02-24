#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

model_name=""
split=""
question_file=""
pipeline_config="${PIPELINE_CONFIG:-configs/pipelines/gem_default.yaml}"
extra_args=()

usage() {
  cat <<'USAGE'
Usage: bench_ecgbench.sh -m MODEL_PATH -d SPLIT [-q QUESTION_FILE] [-c CONFIG] [--dry-run]

Legacy compatibility:
  -m model path/name
  -d dataset split

Optional:
  -q explicit question JSON path (default: data/ecg_bench/<split>.json)
  -c pipeline config path
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m)
      model_name="${2:-}"; shift 2 ;;
    -d)
      split="${2:-}"; shift 2 ;;
    -q)
      question_file="${2:-}"; shift 2 ;;
    -c)
      pipeline_config="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      extra_args+=("$1"); shift ;;
  esac
done

if [[ -z "$model_name" || -z "$split" ]]; then
  usage
  echo "Error: -m and -d are required." >&2
  exit 1
fi

if [[ -z "$question_file" ]]; then
  question_file="data/ecg_bench/${split}.json"
fi

model_path="$model_name"
if [[ -n "${CKPT_DIR:-}" && ! "$model_name" = /* ]]; then
  model_path="${CKPT_DIR%/}/${model_name}"
fi

echo "[migration] evaluation/gem_bench/bench_ecgbench.sh delegates to scripts/gem_pipeline.py eval-generate-ecgbench"

exec python "$ROOT_DIR/scripts/gem_pipeline.py" \
  eval-generate-ecgbench \
  --config "$pipeline_config" \
  --set "evaluation.ecgbench.model_path=${model_path}" \
  --set "evaluation.ecgbench.split=${split}" \
  --set "evaluation.ecgbench.question_file=${question_file}" \
  "${extra_args[@]}"
