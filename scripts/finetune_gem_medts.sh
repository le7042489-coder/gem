#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[migration] scripts/finetune_gem_medts.sh now delegates to scripts/gem_pipeline.py finetune"
echo "[migration] prefer: python scripts/gem_pipeline.py finetune --config configs/pipelines/gem_default.yaml"

PIPELINE_CONFIG="${PIPELINE_CONFIG:-configs/pipelines/gem_default.yaml}"
SET_ARGS=()

append_set() {
  local key="$1"
  local value="$2"
  SET_ARGS+=("--set" "${key}=${value}")
}

[[ -n "${MODEL_NAME_OR_PATH:-}" ]] && append_set "finetune.model_name_or_path" "${MODEL_NAME_OR_PATH}"
[[ -n "${DATA_PATH:-}" ]] && append_set "finetune.data_path" "${DATA_PATH}"
[[ -n "${IMAGE_FOLDER:-}" ]] && append_set "finetune.image_folder" "${IMAGE_FOLDER}"
[[ -n "${ECG_FOLDER:-}" ]] && append_set "finetune.ecg_folder" "${ECG_FOLDER}"
[[ -n "${OUTPUT_DIR:-}" ]] && append_set "finetune.output_dir" "${OUTPUT_DIR}"
[[ -n "${ECG_TOWER:-}" ]] && append_set "finetune.ecg_tower" "${ECG_TOWER}"
[[ -n "${OPEN_CLIP_CONFIG:-}" ]] && append_set "finetune.open_clip_config" "${OPEN_CLIP_CONFIG}"
[[ -n "${VISION_TOWER:-}" ]] && append_set "finetune.vision_tower" "${VISION_TOWER}"
[[ -n "${DEEPSPEED_CONFIG:-}" ]] && append_set "finetune.deepspeed_config" "${DEEPSPEED_CONFIG}"

[[ -n "${NUM_TRAIN_EPOCHS:-}" ]] && append_set "finetune.num_train_epochs" "${NUM_TRAIN_EPOCHS}"
[[ -n "${PER_DEVICE_TRAIN_BATCH_SIZE:-}" ]] && append_set "finetune.per_device_train_batch_size" "${PER_DEVICE_TRAIN_BATCH_SIZE}"
[[ -n "${GRADIENT_ACCUMULATION_STEPS:-}" ]] && append_set "finetune.gradient_accumulation_steps" "${GRADIENT_ACCUMULATION_STEPS}"
[[ -n "${LEARNING_RATE:-}" ]] && append_set "finetune.learning_rate" "${LEARNING_RATE}"

exec python "$ROOT_DIR/scripts/gem_pipeline.py" \
  finetune \
  --config "$PIPELINE_CONFIG" \
  "${SET_ARGS[@]}" \
  "$@"
