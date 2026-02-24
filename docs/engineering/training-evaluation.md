# Training & Evaluation Runbook / 训练与评测手册

## Scope / 适用范围
This document contains the full training and evaluation workflow for GEM engineering usage.
本手册承接主 README 的训练与评测细节，适用于本仓库当前 `scripts/gem_pipeline.py` 工作流。

## Pipeline Overview / Pipeline 总览
Unified entrypoint:

```bash
python scripts/gem_pipeline.py <subcommand> --config configs/pipelines/gem_default.yaml
```

Supported subcommands:

- `validate-config`
- `train`
- `finetune`
- `eval-generate-ecgbench`
- `eval-generate-grounding`
- `eval-score-ecgbench`
- `eval-score-report`
- `grounding-merge`
- `grounding-gpt-eval`
- `grounding-score`

## Config Validation and Overrides / 配置校验与覆盖
Validate config before any heavy job:

```bash
python scripts/gem_pipeline.py validate-config --config configs/pipelines/gem_default.yaml
```

Dry-run a command and print effective config:

```bash
python scripts/gem_pipeline.py train \
  --config configs/pipelines/gem_default.yaml \
  --dry-run \
  --print-effective-config
```

Override config values inline (`--set key=value` can be repeated):

```bash
python scripts/gem_pipeline.py train \
  --config configs/pipelines/gem_default.yaml \
  --set train.output_dir=checkpoints/gem-train-exp1 \
  --set train.num_train_epochs=3 \
  --dry-run
```

## Training / 训练
### Train from scratch / 从头训练
```bash
python scripts/gem_pipeline.py train --config configs/pipelines/gem_default.yaml
```

### Fine-tune with LoRA / LoRA 微调
```bash
python scripts/gem_pipeline.py finetune --config configs/pipelines/gem_default.yaml \
  --set finetune.model_name_or_path=./checkpoints/GEM-7B \
  --set finetune.data_path=data/mixed_train.json \
  --set finetune.image_folder=. \
  --set finetune.output_dir=./checkpoints/gem-medts-v1
```

### Legacy wrappers (compatibility only) / 兼容脚本（仅兼容）
```bash
bash scripts/train_gem.sh
bash scripts/finetune_gem_medts.sh
```

## Training Data Fields / 训练数据字段
For JSON training records:

- `image`: path to ECG image (relative to `--image_folder` or absolute).
- `time_series` (optional): `.npy` path, shape `(12, L)` or `(L, 12)`.
- `ecg` (optional): wfdb record path, used when `time_series` is missing.
- `mask_path` (optional): segmentation label `.npy` path; aligned to length `5000`, empty positions filled with `-1`.

LoRA note:

- `--modules_to_save` can keep specific non-LoRA modules trainable/savable (for example `seg_head`, `ecg_projector`).

## Evaluation / 评测
### ECG-Grounding flow
1. Generate interpretations.

```bash
python scripts/gem_pipeline.py eval-generate-grounding \
  --config configs/pipelines/gem_default.yaml \
  --set evaluation.grounding.question_files_glob='data/ecg_bench/chunks/*.json'
```

2. Merge model outputs.

```bash
python scripts/gem_pipeline.py grounding-merge --config configs/pipelines/gem_default.yaml
```

3. Generate GPT evaluation files.

```bash
python scripts/gem_pipeline.py grounding-gpt-eval --config configs/pipelines/gem_default.yaml
```

4. Aggregate grounding scores.

```bash
python scripts/gem_pipeline.py grounding-score --config configs/pipelines/gem_default.yaml
```

### ECG-Bench flow
1. Generate answers.

```bash
python scripts/gem_pipeline.py eval-generate-ecgbench --config configs/pipelines/gem_default.yaml
```

2. Score benchmark answers.

```bash
python scripts/gem_pipeline.py eval-score-ecgbench --config configs/pipelines/gem_default.yaml
```

3. Evaluate report quality.

```bash
python scripts/gem_pipeline.py eval-score-report --config configs/pipelines/gem_default.yaml
```

## Output Conventions / 输出约定
Default outputs are configured in `configs/pipelines/gem_default.yaml`, including:

- `eval_outputs/gem/ecgbench*`
- `eval_outputs/gem/grounding_*`
- `eval_outputs/gem/report_scores`

## OpenAI-dependent Steps / 依赖 OpenAI 的步骤
The following steps require `OPENAI_API_KEY` (or your configured alternative):

- `eval-score-report`
- `grounding-gpt-eval`

Example:

```bash
export OPENAI_API_KEY=<your_key>
```

## Compatibility Notes / 兼容性说明
- `evaluation/gem_bench/bench_ecgbench.sh` and `evaluation/gem_bench/bench_ecggrounding.sh` remain compatibility wrappers.
- Legacy LLaVA migration reference: `docs/migrations/llava_scripts_legacy.md`.
