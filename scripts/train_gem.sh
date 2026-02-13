#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Distributed training configuration.
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-1234}"
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

# Optional Hugging Face cache path.
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"

LLM_VERSION="${LLM_VERSION:-$ROOT_DIR/checkpoints/GEM-7B}"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
DATA_VERSION="${DATA_VERSION:-mixed_train}"
BASE_RUN_NAME="${BASE_RUN_NAME:-GEM-${LLM_VERSION_CLEAN}-${DATA_VERSION}-finetune}"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

version="${VERSION:-llava_v1}"

data_path="${DATA_PATH:-$ROOT_DIR/data/mixed_train.json}"
image_folder="${IMAGE_FOLDER:-$ROOT_DIR}"
ecg_folder="${ECG_FOLDER:-$ROOT_DIR}"
ecg_tower="${ECG_TOWER:-$ROOT_DIR/ecg_coca/open_clip/checkpoint/cpt_wfep_epoch_20.pt}"
deepspeed_config="${DEEPSPEED_CONFIG:-$ROOT_DIR/scripts/zero2.json}"
report_to="${REPORT_TO:-none}"

num_epochs="${NUM_EPOCHS:-1}"
GRAD_ACC_STEP="${GRAD_ACC_STEP:-2}"
BATCH_PER_GPU="${BATCH_PER_GPU:-16}"
TOTAL_BATCH_SIZE=$((WORLD_SIZE * BATCH_PER_GPU))
echo "WORLD_SIZE: ${WORLD_SIZE}, TOTAL_BATCH_SIZE: ${TOTAL_BATCH_SIZE}"

for path in "$LLM_VERSION" "$data_path" "$image_folder" "$ecg_folder" "$ecg_tower" "$deepspeed_config"; do
    if [ ! -e "$path" ]; then
        echo "Error: required path not found: $path" >&2
        exit 1
    fi
done

torchrun \
    --nproc_per_node "$GPUS_PER_NODE" \
    --master_addr "$MASTER_ADDR" \
    --node_rank "$NODE_RANK" \
    --master_port "$MASTER_PORT" \
    --nnodes "$NNODES" \
    "$ROOT_DIR/llava/train/train_mem.py" \
    --deepspeed "$deepspeed_config" \
    --model_name_or_path "$LLM_VERSION" \
    --version "$version" \
    --data_path "$data_path" \
    --ecg_folder "$ecg_folder" \
    --ecg_tower "$ecg_tower" \
    --open_clip_config coca_ViT-B-32 \
    --image_folder "$image_folder" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio ori \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir "$ROOT_DIR/checkpoints/${BASE_RUN_NAME}" \
    --num_train_epochs "$num_epochs" \
    --per_device_train_batch_size "$BATCH_PER_GPU" \
    --per_device_eval_batch_size "$BATCH_PER_GPU" \
    --gradient_accumulation_steps "$GRAD_ACC_STEP" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.2 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 64 \
    --lazy_preprocess True \
    --report_to "$report_to"
