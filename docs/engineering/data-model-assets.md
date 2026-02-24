# Data and Model Assets Guide / 数据与模型资产指南

## Scope / 适用范围
This guide defines how to prepare data and model assets required by GEM pipelines and backend services.
本指南用于规范 GEM 训练、评测与服务运行所需的数据和模型资产目录。

## Required Assets at a Glance / 资产清单总览
- ECG time-series datasets (multiple sources)
- ECG image datasets and generated images
- JSON manifests (`mixed_train.json`, grounding/eval jsons)
- ECG encoder checkpoint (`cpt_wfep_epoch_20.pt`)
- Base/fine-tuned MLLM checkpoints (for train/fine-tune/inference)

## Recommended Directory Layout / 推荐目录结构
Organize assets under `./data` as follows:

```bash
.
├── ecg_timeseries
│   ├── champan-shaoxing
│   ├── code15
│   ├── cpsc2018
│   ├── ptbxl
│   ├── georgia
│   └── mimic-iv
├── ecg_images
│   ├── cod15_v4
│   ├── csn_aug_all_layout_papersize
│   ├── csn_ori_layout_papersize
│   ├── csn_part_noise_layout_papersize
│   ├── gen_images
│   │   ├── mimic_gen
│   │   └── ptb-xl-gen
│   ├── mimic
│   ├── mimic_v4
│   └── ptb-xl
├── ecg_bench
│   ├── images
│   ├── ecg-grounding-test-mimiciv.json
│   └── ecg-grounding-test-ptbxl.json
└── ecg_jsons
    └── ECG_Grounding_30k.json
```

## External Data Sources / 外部数据来源
ECG time series:

- [MIMIC-IV](https://physionet.org/content/mimic-iv-ecg/1.0/)
- [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)
- [Code-15%](https://zenodo.org/records/4916206)
- [CPSC 2018](https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/)
- [CSN](https://physionet.org/content/ecg-arrhythmia/1.0.0/)
- [G12E](https://physionet.org/content/challenge-2020/1.0.2/training/georgia/)

Image / benchmark assets:

- [ECG-Grounding dataset](https://huggingface.co/datasets/LANSG/ECG-Grounding)
- [ECG-Instruct](https://huggingface.co/datasets/PULSE-ECG/ECGInstruct/tree/main)
- [ECG-Bench](https://huggingface.co/datasets/PULSE-ECG/ECGBench)

## Model Asset Placement / 模型资产放置
### ECG encoder checkpoint
Download `cpt_wfep_epoch_20.pt` and place it at:

```text
ecg_coca/open_clip/checkpoint/cpt_wfep_epoch_20.pt
```

This path is used by default pipeline configs (`paths.ecg_tower`).

### MLLM checkpoints
Common options used in this repo:

- [PULSE-7B](https://huggingface.co/PULSE-ECG/PULSE-7B)
- [LLaVA v1.6 Vicuna 7B](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b)
- Fine-tuned `GEM-7B` (place under `checkpoints/` and update config accordingly)

If loading a downloaded GEM checkpoint, ensure `mm_ecg_tower` points to a valid ECG tower checkpoint path.

## Optional Asset Generation / 可选资产生成
If your training manifest points image paths to `.npy` samples, you can generate ECG `.png` files:

```bash
python scripts/generate_images.py --processed-dir processed_data --json-path data/mixed_train.json
```

Extra dependencies are listed in:

```text
gem_generation/ecg-image-generator/requirements.txt
```

## Backend Runtime Data Files / 后端运行时数据文件
Backend defaults (override via environment variables when needed):

- `LOCAL_INDEX_PATH` -> `data/local_index.json`
- `LOCAL_INFER_PATH` -> `data/local_infer.json`
- `MIXED_TRAIN_PATH` -> `data/mixed_train.json`

Ensure these files exist for `/samples` and `predict_plus/by_id` workflows.

## Validation Checklist / 校验清单
Before running train/eval or backend inference:

```bash
test -f ecg_coca/open_clip/checkpoint/cpt_wfep_epoch_20.pt
test -f configs/pipelines/gem_default.yaml
test -f data/mixed_train.json
```

Then run config validation:

```bash
python scripts/gem_pipeline.py validate-config --config configs/pipelines/gem_default.yaml
```
