#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[migration] scripts/train_gem.sh now delegates to scripts/gem_pipeline.py train"
echo "[migration] prefer: python scripts/gem_pipeline.py train --config configs/pipelines/gem_default.yaml"

PIPELINE_CONFIG="${PIPELINE_CONFIG:-configs/pipelines/gem_default.yaml}"
SET_ARGS=()

append_set() {
  local key="$1"
  local value="$2"
  SET_ARGS+=("--set" "${key}=${value}")
}

[[ -n "${LLM_VERSION:-}" ]] && append_set "train.model_name_or_path" "${LLM_VERSION}"
[[ -n "${DATA_PATH:-}" ]] && append_set "train.data_path" "${DATA_PATH}"
[[ -n "${IMAGE_FOLDER:-}" ]] && append_set "train.image_folder" "${IMAGE_FOLDER}"
[[ -n "${ECG_FOLDER:-}" ]] && append_set "train.ecg_folder" "${ECG_FOLDER}"
[[ -n "${ECG_TOWER:-}" ]] && append_set "paths.ecg_tower" "${ECG_TOWER}"
[[ -n "${DEEPSPEED_CONFIG:-}" ]] && append_set "train.deepspeed_config" "${DEEPSPEED_CONFIG}"
[[ -n "${VERSION:-}" ]] && append_set "train.version" "${VERSION}"
[[ -n "${REPORT_TO:-}" ]] && append_set "train.report_to" "${REPORT_TO}"

[[ -n "${GPUS_PER_NODE:-}" ]] && append_set "train.gpus_per_node" "${GPUS_PER_NODE}"
[[ -n "${NNODES:-}" ]] && append_set "train.nnodes" "${NNODES}"
[[ -n "${NODE_RANK:-}" ]] && append_set "train.node_rank" "${NODE_RANK}"
[[ -n "${MASTER_ADDR:-}" ]] && append_set "train.master_addr" "${MASTER_ADDR}"
[[ -n "${MASTER_PORT:-}" ]] && append_set "train.master_port" "${MASTER_PORT}"

[[ -n "${NUM_EPOCHS:-}" ]] && append_set "train.num_train_epochs" "${NUM_EPOCHS}"
[[ -n "${GRAD_ACC_STEP:-}" ]] && append_set "train.grad_acc_step" "${GRAD_ACC_STEP}"
[[ -n "${BATCH_PER_GPU:-}" ]] && append_set "train.batch_per_gpu" "${BATCH_PER_GPU}"
[[ -n "${DATALOADER_NUM_WORKERS:-}" ]] && append_set "train.dataloader_num_workers" "${DATALOADER_NUM_WORKERS}"

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  append_set "train.output_dir" "${OUTPUT_DIR}"
elif [[ -n "${BASE_RUN_NAME:-}" ]]; then
  append_set "train.output_dir" "${ROOT_DIR}/checkpoints/${BASE_RUN_NAME}"
fi

exec python "$ROOT_DIR/scripts/gem_pipeline.py" \
  train \
  --config "$PIPELINE_CONFIG" \
  "${SET_ARGS[@]}" \
  "$@"
