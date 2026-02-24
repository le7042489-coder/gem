#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Sequence

from pipeline.config import (
    ConfigError,
    PathValidationError,
    dump_config,
    load_effective_config,
    validate_required_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _ensure_parent(path_str: str) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)


def _run(cmd: Sequence[str], dry_run: bool = False, env: Dict[str, str] | None = None) -> int:
    printable = " ".join(cmd)
    print(f"[pipeline] command: {printable}")
    if dry_run:
        return 0

    proc = subprocess.run(list(cmd), cwd=str(PROJECT_ROOT), env=env)
    return proc.returncode


def _python_cmd(script_rel_path: str, *args: str) -> List[str]:
    return [sys.executable, str(PROJECT_ROOT / script_rel_path), *args]


def _handle_validate_config(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    _ = cfg
    print("[pipeline] config validation passed")
    return 0


def _build_train_cmd(cfg: Dict[str, Any]) -> List[str]:
    section = _get(cfg, "train", {})

    gpus_per_node = int(section["gpus_per_node"])
    nnodes = int(section["nnodes"])
    node_rank = int(section["node_rank"])
    master_addr = str(section["master_addr"])
    master_port = int(section["master_port"])

    batch_per_gpu = int(section["batch_per_gpu"])

    output_dir = str(section["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    return [
        section["launcher"],
        "--nproc_per_node",
        str(gpus_per_node),
        "--master_addr",
        master_addr,
        "--node_rank",
        str(node_rank),
        "--master_port",
        str(master_port),
        "--nnodes",
        str(nnodes),
        str(PROJECT_ROOT / "llava" / "train" / "train_mem.py"),
        "--deepspeed",
        str(section["deepspeed_config"]),
        "--model_name_or_path",
        str(section["model_name_or_path"]),
        "--version",
        str(section["version"]),
        "--data_path",
        str(section["data_path"]),
        "--ecg_folder",
        str(section["ecg_folder"]),
        "--ecg_tower",
        str(_get(cfg, "paths.ecg_tower")),
        "--open_clip_config",
        "coca_ViT-B-32",
        "--image_folder",
        str(section["image_folder"]),
        "--vision_tower",
        "openai/clip-vit-large-patch14-336",
        "--mm_projector_type",
        "mlp2x_gelu",
        "--mm_vision_select_layer",
        "-2",
        "--mm_use_im_start_end",
        "False",
        "--mm_use_im_patch_token",
        "False",
        "--image_aspect_ratio",
        "ori",
        "--group_by_modality_length",
        "False",
        "--bf16",
        "True",
        "--output_dir",
        output_dir,
        "--num_train_epochs",
        str(section["num_train_epochs"]),
        "--per_device_train_batch_size",
        str(batch_per_gpu),
        "--per_device_eval_batch_size",
        str(batch_per_gpu),
        "--gradient_accumulation_steps",
        str(section["grad_acc_step"]),
        "--evaluation_strategy",
        "no",
        "--save_strategy",
        "steps",
        "--save_steps",
        str(section["save_steps"]),
        "--save_total_limit",
        "5",
        "--learning_rate",
        "2e-5",
        "--weight_decay",
        "0.",
        "--warmup_ratio",
        "0.03",
        "--lr_scheduler_type",
        "cosine",
        "--logging_steps",
        "1",
        "--tf32",
        "True",
        "--model_max_length",
        "4096",
        "--gradient_checkpointing",
        "True",
        "--dataloader_num_workers",
        str(section["dataloader_num_workers"]),
        "--lazy_preprocess",
        "True",
        "--report_to",
        str(section["report_to"]),
    ]


def _handle_train(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    cmd = _build_train_cmd(cfg)

    env = os.environ.copy()
    env.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

    return _run(cmd, dry_run=args.dry_run, env=env)


def _build_finetune_cmd(cfg: Dict[str, Any]) -> List[str]:
    section = _get(cfg, "finetune", {})
    output_dir = str(section["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    return [
        section["launcher"],
        str(PROJECT_ROOT / "llava" / "train" / "train_mem.py"),
        "--deepspeed",
        str(section["deepspeed_config"]),
        "--model_name_or_path",
        str(section["model_name_or_path"]),
        "--version",
        "llava_v1",
        "--data_path",
        str(section["data_path"]),
        "--image_folder",
        str(section["image_folder"]),
        "--ecg_folder",
        str(section["ecg_folder"]),
        "--ecg_tower",
        str(section["ecg_tower"]),
        "--open_clip_config",
        str(section["open_clip_config"]),
        "--vision_tower",
        str(section["vision_tower"]),
        "--mm_projector_type",
        "mlp2x_gelu",
        "--mm_vision_select_layer",
        "-2",
        "--mm_use_im_start_end",
        "False",
        "--mm_use_im_patch_token",
        "False",
        "--image_aspect_ratio",
        "ori",
        "--group_by_modality_length",
        "False",
        "--bf16",
        "True",
        "--output_dir",
        output_dir,
        "--num_train_epochs",
        str(section["num_train_epochs"]),
        "--per_device_train_batch_size",
        str(section["per_device_train_batch_size"]),
        "--gradient_accumulation_steps",
        str(section["gradient_accumulation_steps"]),
        "--learning_rate",
        str(section["learning_rate"]),
        "--lora_enable",
        "True",
        "--lora_r",
        "128",
        "--lora_alpha",
        "256",
        "--modules_to_save",
        "embed_tokens",
        "lm_head",
        "seg_head",
        "ecg_projector",
        "--tune_mm_mlp_adapter",
        "True",
        "--model_max_length",
        "4096",
        "--gradient_checkpointing",
        "True",
        "--evaluation_strategy",
        "no",
        "--save_strategy",
        "steps",
        "--save_steps",
        "5000",
        "--save_total_limit",
        "2",
        "--logging_steps",
        "10",
        "--lr_scheduler_type",
        "cosine",
        "--warmup_ratio",
        "0.03",
        "--weight_decay",
        "0.0",
        "--tf32",
        "True",
        "--lazy_preprocess",
        "True",
        "--report_to",
        "none",
    ]


def _handle_finetune(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    cmd = _build_finetune_cmd(cfg)
    return _run(cmd, dry_run=args.dry_run)


def _default_answers_file(answers_dir: str, split: str) -> str:
    return str((Path(answers_dir) / split / "step-final.jsonl").resolve())


def _handle_eval_generate_ecgbench(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    section = _get(cfg, "evaluation.ecgbench", {})
    split = str(section.get("split") or "ecgbench")

    answers_file = section.get("answers_file")
    if not answers_file:
        answers_file = _default_answers_file(section["answers_dir"], split)
    _ensure_parent(str(answers_file))

    cmd = _python_cmd(
        "llava/eval/model_ecg_resume.py",
        "--model-path",
        str(section["model_path"]),
        "--model-base",
        str(section.get("model_base") or ""),
        "--image-folder",
        str(section.get("image_folder") or ""),
        "--question-file",
        str(section["question_file"]),
        "--answers-file",
        str(answers_file),
        "--conv-mode",
        str(section.get("conv_mode") or "llava_v1"),
        "--ecg-folder",
        str(section.get("ecg_folder") or ""),
        "--ecg_tower",
        str(section.get("ecg_tower") or ""),
        "--open_clip_config",
        str(section.get("open_clip_config") or "coca_ViT-B-32"),
        "--temperature",
        str(section.get("temperature", 0.0)),
        "--num_beams",
        str(section.get("num_beams", 1)),
        "--max_new_tokens",
        str(section.get("max_new_tokens", 1024)),
    )
    top_p = section.get("top_p")
    if top_p is not None:
        cmd.extend(["--top_p", str(top_p)])

    return _run(cmd, dry_run=args.dry_run)


def _resolve_grounding_question_files(cfg: Dict[str, Any]) -> List[str]:
    section = _get(cfg, "evaluation.grounding", {})
    files: List[str] = []

    for item in section.get("question_files", []):
        if item and Path(item).exists():
            files.append(str(Path(item).resolve()))

    pattern = section.get("question_files_glob")
    if pattern:
        files.extend(sorted(glob(pattern)))

    unique: List[str] = []
    seen = set()
    for item in files:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _handle_eval_generate_grounding(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    section = _get(cfg, "evaluation.grounding", {})
    question_files = _resolve_grounding_question_files(cfg)
    if not question_files:
        raise ConfigError("No grounding question files resolved from question_files/question_files_glob")

    gpus = section.get("gpus", [0])
    answers_dir = Path(section["answers_dir"])
    answers_dir.mkdir(parents=True, exist_ok=True)

    procs: List[subprocess.Popen[Any]] = []
    for idx, question_file in enumerate(question_files):
        gpu_id = int(gpus[idx % len(gpus)])
        stem = Path(question_file).stem
        answers_file = str((answers_dir / f"{stem}-step-final.jsonl").resolve())

        cmd = _python_cmd(
            "llava/eval/model_ecg_resume.py",
            "--model-path",
            str(section["model_path"]),
            "--model-base",
            str(section.get("model_base") or ""),
            "--image-folder",
            str(section.get("image_folder") or ""),
            "--question-file",
            question_file,
            "--answers-file",
            answers_file,
            "--conv-mode",
            str(section.get("conv_mode") or "llava_v1"),
            "--ecg-folder",
            str(section.get("ecg_folder") or ""),
            "--ecg_tower",
            str(section.get("ecg_tower") or ""),
            "--open_clip_config",
            str(section.get("open_clip_config") or "coca_ViT-B-32"),
            "--temperature",
            str(section.get("temperature", 0.0)),
            "--num_beams",
            str(section.get("num_beams", 1)),
            "--max_new_tokens",
            str(section.get("max_new_tokens", 1024)),
        )

        top_p = section.get("top_p")
        if top_p is not None:
            cmd.extend(["--top_p", str(top_p)])

        if args.dry_run:
            _run(cmd, dry_run=True)
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[pipeline] launching grounding chunk on GPU {gpu_id}: {question_file}")
        procs.append(subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env))

    if args.dry_run:
        return 0

    failed = 0
    for proc in procs:
        rc = proc.wait()
        if rc != 0:
            failed += 1

    return 0 if failed == 0 else 4


def _handle_eval_score_ecgbench(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    section = _get(cfg, "evaluation.score_ecgbench", {})
    datasets_config = _get(cfg, "evaluation.datasets_config")

    cmd = _python_cmd(
        "evaluation/evaluate_ecgbench.py",
        "--pred-root",
        str(section["pred_root"]),
        "--datasets-config",
        str(datasets_config),
        "--output-json",
        str(section["output_json"]),
    )

    splits = section.get("splits") or []
    if splits:
        cmd.extend(["--splits", *[str(x) for x in splits]])

    return _run(cmd, dry_run=args.dry_run)


def _handle_eval_score_report(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    section = _get(cfg, "evaluation.report_eval", {})

    cmd = _python_cmd(
        "evaluation/eval_report.py",
        "--predictions-jsonl",
        str(section["predictions_jsonl"]),
        "--gold-json",
        str(section["gold_json"]),
        "--output-dir",
        str(section["output_dir"]),
        "--model",
        str(section["openai_model"]),
        "--max-workers",
        str(section.get("max_workers", 8)),
    )
    if section.get("resume", True):
        cmd.append("--resume")

    return _run(cmd, dry_run=args.dry_run)


def _handle_grounding_merge(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    section = _get(cfg, "grounding.merge", {})
    cmd = _python_cmd(
        "gem_evaluation/process_gem_outputs.py",
        "--result-jsonl",
        str(section["result_jsonl"]),
        "--test-file",
        str(section["test_file"]),
        "--output-json",
        str(section["output_json"]),
    )
    return _run(cmd, dry_run=args.dry_run)


def _handle_grounding_gpt_eval(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    section = _get(cfg, "grounding.gpt_eval", {})
    cmd = _python_cmd(
        "gem_evaluation/generate_gpt_eval.py",
        "--input-json",
        str(section["input_json"]),
        "--output-dir",
        str(section["output_dir"]),
        "--template-file",
        str(section["template_file"]),
        "--model",
        str(section["openai_model"]),
        "--start",
        str(section.get("start", 0)),
        "--end",
        str(section.get("end", -1)),
    )
    return _run(cmd, dry_run=args.dry_run)


def _handle_grounding_score(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    section = _get(cfg, "grounding.score", {})
    cmd = _python_cmd(
        "gem_evaluation/process_grounding_scores.py",
        "--input-dir",
        str(section["input_dir"]),
        "--output-json",
        str(section["output_json"]),
        "--per-id-json",
        str(section["per_id_json"]),
    )
    return _run(cmd, dry_run=args.dry_run)


HANDLERS = {
    "validate-config": _handle_validate_config,
    "train": _handle_train,
    "finetune": _handle_finetune,
    "eval-generate-ecgbench": _handle_eval_generate_ecgbench,
    "eval-generate-grounding": _handle_eval_generate_grounding,
    "eval-score-ecgbench": _handle_eval_score_ecgbench,
    "eval-score-report": _handle_eval_score_report,
    "grounding-merge": _handle_grounding_merge,
    "grounding-gpt-eval": _handle_grounding_gpt_eval,
    "grounding-score": _handle_grounding_score,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GEM unified pipeline CLI")

    sub = parser.add_subparsers(dest="command", required=True)
    commands = [
        "validate-config",
        "train",
        "finetune",
        "eval-generate-ecgbench",
        "eval-generate-grounding",
        "eval-score-ecgbench",
        "eval-score-report",
        "grounding-merge",
        "grounding-gpt-eval",
        "grounding-score",
    ]

    for cmd in commands:
        p = sub.add_parser(cmd)
        p.add_argument("--config", default="configs/pipelines/gem_default.yaml")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--print-effective-config", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_effective_config(args.config, args.set)
        if args.command != "validate-config":
            validate_required_paths(cfg, args.command)

        if args.print_effective_config:
            print(dump_config(cfg))

        handler = HANDLERS[args.command]
        rc = handler(cfg, args)
        if rc != 0:
            return 4
        return 0
    except PathValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] runtime error: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
