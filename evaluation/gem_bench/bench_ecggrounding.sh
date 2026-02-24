#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

model_name=""
question_glob="data/ecg_bench/chunks/*.json"
question_files=""
gpu_list="${GPU_LIST:-0,1,2,3,4,5,6,7}"
pipeline_config="${PIPELINE_CONFIG:-configs/pipelines/gem_default.yaml}"
extra_args=()

usage() {
  cat <<'USAGE'
Usage: bench_ecggrounding.sh -m MODEL_PATH [-g GLOB] [-f FILE1,FILE2] [-p GPU_IDS] [-c CONFIG] [--dry-run]

Legacy compatibility:
  -m model path/name

Optional:
  -g glob pattern for chunked question files (default: data/ecg_bench/chunks/*.json)
  -f comma-separated explicit question files (overrides -g)
  -p comma-separated GPU ids (default: 0,1,2,3,4,5,6,7)
  -c pipeline config path
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m)
      model_name="${2:-}"; shift 2 ;;
    -g)
      question_glob="${2:-}"; shift 2 ;;
    -f)
      question_files="${2:-}"; shift 2 ;;
    -p)
      gpu_list="${2:-}"; shift 2 ;;
    -c)
      pipeline_config="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      extra_args+=("$1"); shift ;;
  esac
done

if [[ -z "$model_name" ]]; then
  usage
  echo "Error: -m is required." >&2
  exit 1
fi

model_path="$model_name"
if [[ -n "${CKPT_DIR:-}" && ! "$model_name" = /* ]]; then
  model_path="${CKPT_DIR%/}/${model_name}"
fi

echo "[migration] evaluation/gem_bench/bench_ecggrounding.sh delegates to scripts/gem_pipeline.py eval-generate-grounding"

set_args=(
  "--set" "evaluation.grounding.model_path=${model_path}"
  "--set" "evaluation.grounding.gpus=[${gpu_list}]"
)

if [[ -n "$question_files" ]]; then
  IFS=',' read -r -a file_array <<<"$question_files"
  json_array="["
  for i in "${!file_array[@]}"; do
    file="$(echo "${file_array[$i]}" | xargs)"
    [[ -z "$file" ]] && continue
    [[ $i -gt 0 ]] && json_array+=", "
    json_array+="\"${file}\""
  done
  json_array+="]"
  set_args+=("--set" "evaluation.grounding.question_files=${json_array}")
else
  set_args+=("--set" "evaluation.grounding.question_files_glob=${question_glob}")
fi

exec python "$ROOT_DIR/scripts/gem_pipeline.py" \
  eval-generate-grounding \
  --config "$pipeline_config" \
  "${set_args[@]}" \
  "${extra_args[@]}"
