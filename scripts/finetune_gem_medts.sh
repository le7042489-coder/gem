#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Defaults can be overridden via env vars, e.g.:
#   MODEL_NAME_OR_PATH=... OUTPUT_DIR=... bash scripts/finetune_gem_medts.sh
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./checkpoints/GEM-7B}"
DATA_PATH="${DATA_PATH:-data/mixed_train.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-.}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/gem-medts-v1}"

ECG_TOWER="${ECG_TOWER:-ecg_coca/open_clip/checkpoint/cpt_wfep_epoch_20.pt}"
OPEN_CLIP_CONFIG="${OPEN_CLIP_CONFIG:-coca_ViT-B-32}"
VISION_TOWER="${VISION_TOWER:-openai/clip-vit-large-patch14-336}"

# If you run out of VRAM, reduce batch size / increase grad accumulation, and
# consider keeping only: --modules_to_save seg_head
deepspeed llava/train/train_mem.py \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --version llava_v1 \
  --data_path "$DATA_PATH" \
  --image_folder "$IMAGE_FOLDER" \
  --ecg_folder "$IMAGE_FOLDER" \
  --ecg_tower "$ECG_TOWER" \
  --open_clip_config "$OPEN_CLIP_CONFIG" \
  --vision_tower "$VISION_TOWER" \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio ori \
  --group_by_modality_length False \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --lora_enable True --lora_r 128 --lora_alpha 256 \
  --modules_to_save embed_tokens lm_head seg_head ecg_projector \
  --tune_mm_mlp_adapter True \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 5000 \
  --save_total_limit 2 \
  --logging_steps 10 \
  --lr_scheduler_type "cosine" \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --tf32 True \
  --lazy_preprocess True \
  --report_to "none"

