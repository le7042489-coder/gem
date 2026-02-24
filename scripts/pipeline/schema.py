from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


class ConfigValidationError(Exception):
    """Raised when pipeline configuration is malformed."""


DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "repo_root": ".",
        "checkpoints_dir": "checkpoints",
        "data_root": "data",
        "eval_output_root": "eval_outputs",
        "ecg_tower": "ecg_coca/open_clip/checkpoint/cpt_wfep_epoch_20.pt",
    },
    "train": {
        "enabled": True,
        "launcher": "torchrun",
        "model_name_or_path": "checkpoints/GEM-7B",
        "data_path": "data/mixed_train.json",
        "image_folder": ".",
        "ecg_folder": ".",
        "output_dir": "checkpoints/gem-train",
        "deepspeed_config": "scripts/zero2.json",
        "version": "llava_v1",
        "report_to": "none",
        "gpus_per_node": 1,
        "nnodes": 1,
        "node_rank": 0,
        "master_addr": "127.0.0.1",
        "master_port": 1234,
        "num_train_epochs": 1,
        "grad_acc_step": 2,
        "batch_per_gpu": 16,
        "dataloader_num_workers": 64,
        "save_steps": 0.2,
    },
    "finetune": {
        "enabled": True,
        "launcher": "deepspeed",
        "model_name_or_path": "checkpoints/GEM-7B",
        "data_path": "data/mixed_train.json",
        "image_folder": ".",
        "ecg_folder": ".",
        "output_dir": "checkpoints/gem-medts-v1",
        "ecg_tower": "ecg_coca/open_clip/checkpoint/cpt_wfep_epoch_20.pt",
        "open_clip_config": "coca_ViT-B-32",
        "vision_tower": "openai/clip-vit-large-patch14-336",
        "deepspeed_config": "scripts/zero2.json",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
    },
    "evaluation": {
        "datasets_config": "evaluation/config/datasets.yaml",
        "ecgbench": {
            "enabled": True,
            "model_path": "checkpoints/GEM-7B",
            "model_base": None,
            "split": "ecg-grounding-test-mimiciv",
            "question_file": "",
            "answers_dir": "eval_outputs/gem/ecgbench",
            "answers_file": "",
            "image_folder": "",
            "ecg_folder": "",
            "conv_mode": "llava_v1",
            "open_clip_config": "coca_ViT-B-32",
            "ecg_tower": "",
            "temperature": 0.0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 1024,
        },
        "grounding": {
            "enabled": True,
            "model_path": "checkpoints/GEM-7B",
            "model_base": None,
            "question_files_glob": "",
            "question_files": [],
            "answers_dir": "eval_outputs/gem/ecg-grounding-test",
            "image_folder": "",
            "ecg_folder": "",
            "conv_mode": "llava_v1",
            "open_clip_config": "coca_ViT-B-32",
            "ecg_tower": "",
            "gpus": [0],
            "temperature": 0.0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 1024,
        },
        "score_ecgbench": {
            "pred_root": "eval_outputs/gem",
            "output_json": "eval_outputs/gem/ecgbench_scores.json",
            "splits": [],
        },
        "report_eval": {
            "predictions_jsonl": "",
            "gold_json": "",
            "output_dir": "eval_outputs/gem/report_scores",
            "openai_model": "gpt-4o-2024-08-06",
            "max_workers": 8,
            "resume": True,
        },
    },
    "grounding": {
        "merge": {
            "result_jsonl": "",
            "test_file": "",
            "output_json": "eval_outputs/gem/grounding_merged.json",
        },
        "gpt_eval": {
            "input_json": "",
            "output_dir": "eval_outputs/gem/grounding_gpt_eval",
            "template_file": "gem_evaluation/prompts_evaluation.txt",
            "openai_model": "gpt-4o-2024-08-06",
            "start": 0,
            "end": -1,
        },
        "score": {
            "input_dir": "eval_outputs/gem/grounding_gpt_eval",
            "output_json": "eval_outputs/gem/grounding_scores.json",
            "per_id_json": "eval_outputs/gem/grounding_scores_per_id.json",
        },
    },
    "legacy": {
        "deprecation_message": (
            "[DEPRECATED] scripts/llava_scripts compatibility wrappers will be removed "
            "in the next major release. Use legacy/llava_scripts directly."
        )
    },
}


BOOL_FIELDS = {
    "train.enabled",
    "finetune.enabled",
    "evaluation.ecgbench.enabled",
    "evaluation.grounding.enabled",
    "evaluation.report_eval.resume",
}


LIST_INT_FIELDS = {"evaluation.grounding.gpus"}


PATH_FIELDS = {
    "paths.repo_root",
    "paths.checkpoints_dir",
    "paths.data_root",
    "paths.eval_output_root",
    "paths.ecg_tower",
    "train.model_name_or_path",
    "train.data_path",
    "train.image_folder",
    "train.ecg_folder",
    "train.output_dir",
    "train.deepspeed_config",
    "finetune.model_name_or_path",
    "finetune.data_path",
    "finetune.image_folder",
    "finetune.ecg_folder",
    "finetune.output_dir",
    "finetune.ecg_tower",
    "finetune.deepspeed_config",
    "evaluation.datasets_config",
    "evaluation.ecgbench.model_path",
    "evaluation.ecgbench.question_file",
    "evaluation.ecgbench.answers_dir",
    "evaluation.ecgbench.answers_file",
    "evaluation.ecgbench.image_folder",
    "evaluation.ecgbench.ecg_folder",
    "evaluation.ecgbench.ecg_tower",
    "evaluation.grounding.model_path",
    "evaluation.grounding.answers_dir",
    "evaluation.grounding.image_folder",
    "evaluation.grounding.ecg_folder",
    "evaluation.grounding.ecg_tower",
    "evaluation.score_ecgbench.pred_root",
    "evaluation.score_ecgbench.output_json",
    "evaluation.report_eval.predictions_jsonl",
    "evaluation.report_eval.gold_json",
    "evaluation.report_eval.output_dir",
    "grounding.merge.result_jsonl",
    "grounding.merge.test_file",
    "grounding.merge.output_json",
    "grounding.gpt_eval.input_json",
    "grounding.gpt_eval.output_dir",
    "grounding.gpt_eval.template_file",
    "grounding.score.input_dir",
    "grounding.score.output_json",
    "grounding.score.per_id_json",
}


COMMAND_PATH_REQUIREMENTS = {
    "train": [
        "train.model_name_or_path",
        "train.data_path",
        "train.image_folder",
        "train.ecg_folder",
        "paths.ecg_tower",
        "train.deepspeed_config",
    ],
    "finetune": [
        "finetune.model_name_or_path",
        "finetune.data_path",
        "finetune.image_folder",
        "finetune.ecg_folder",
        "finetune.ecg_tower",
        "finetune.deepspeed_config",
    ],
    "eval-generate-ecgbench": [
        "evaluation.ecgbench.model_path",
        "evaluation.ecgbench.question_file",
    ],
    "eval-generate-grounding": [
        "evaluation.grounding.model_path",
    ],
    "eval-score-ecgbench": [
        "evaluation.score_ecgbench.pred_root",
        "evaluation.datasets_config",
    ],
    "eval-score-report": [
        "evaluation.report_eval.predictions_jsonl",
        "evaluation.report_eval.gold_json",
    ],
    "grounding-merge": [
        "grounding.merge.result_jsonl",
        "grounding.merge.test_file",
    ],
    "grounding-gpt-eval": [
        "grounding.gpt_eval.input_json",
        "grounding.gpt_eval.template_file",
    ],
    "grounding-score": [
        "grounding.score.input_dir",
    ],
}


def _merge_dict(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def merge_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    return _merge_dict(DEFAULT_CONFIG, config)


def get_nested(config: Dict[str, Any], dotted_key: str) -> Any:
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def validate_types(config: Dict[str, Any]) -> None:
    for key in BOOL_FIELDS:
        value = get_nested(config, key)
        if not isinstance(value, bool):
            raise ConfigValidationError(f"{key} must be boolean, got {type(value).__name__}")

    for key in LIST_INT_FIELDS:
        value = get_nested(config, key)
        if not isinstance(value, list) or not value:
            raise ConfigValidationError(f"{key} must be a non-empty list of ints")
        if any(not isinstance(item, int) for item in value):
            raise ConfigValidationError(f"{key} must contain ints only")

    for key in PATH_FIELDS:
        value = get_nested(config, key)
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            raise ConfigValidationError(f"{key} must be a string path")


def resolve_path(path_value: str, repo_root: Path) -> str:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return str(candidate)
    return str((repo_root / candidate).resolve())


def resolve_paths(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    cfg = dict(config)
    repo_root_raw = get_nested(cfg, "paths.repo_root")
    if repo_root_raw in (None, ""):
        repo_root = project_root
    else:
        repo_root = Path(repo_root_raw)
        if not repo_root.is_absolute():
            repo_root = (project_root / repo_root).resolve()

    set_nested(cfg, "paths.repo_root", str(repo_root))

    for key in sorted(PATH_FIELDS):
        if key == "paths.repo_root":
            continue
        value = get_nested(cfg, key)
        if not value:
            continue
        set_nested(cfg, key, resolve_path(value, repo_root))

    q_glob = get_nested(cfg, "evaluation.grounding.question_files_glob")
    if isinstance(q_glob, str) and q_glob.strip():
        set_nested(cfg, "evaluation.grounding.question_files_glob", resolve_path(q_glob, repo_root))

    q_files = get_nested(cfg, "evaluation.grounding.question_files")
    if isinstance(q_files, list):
        set_nested(
            cfg,
            "evaluation.grounding.question_files",
            [resolve_path(str(item), repo_root) for item in q_files if str(item).strip()],
        )

    return cfg


def required_paths_for_command(command: str) -> List[str]:
    return list(COMMAND_PATH_REQUIREMENTS.get(command, []))


def collect_missing_paths(config: Dict[str, Any], required_keys: Iterable[str]) -> List[Tuple[str, str]]:
    missing: List[Tuple[str, str]] = []
    for key in required_keys:
        value = get_nested(config, key)
        if not isinstance(value, str) or not value.strip():
            missing.append((key, "empty"))
            continue
        if key.endswith("question_files_glob"):
            matches = list(Path().glob(value))
            if not matches:
                missing.append((key, value))
            continue
        if not Path(value).exists():
            missing.append((key, value))
    return missing
