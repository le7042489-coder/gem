#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


def _accuracy_score(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return float(correct / len(y_true))


@dataclass
class FileScore:
    file: str
    step: int
    matched: int
    gold_total: int
    metrics: Dict[str, float]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_dataset_config(path: Path) -> Dict[str, Dict[str, Any]]:
    obj = _load_yaml(path)
    datasets = obj.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"Invalid datasets config: missing 'datasets' mapping in {path}")
    return {str(k): dict(v) for k, v in datasets.items() if isinstance(v, dict)}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return data


def _extract_qid(row: Mapping[str, Any]) -> Optional[str]:
    for key in ("question_id", "id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _extract_prediction_text(row: Mapping[str, Any]) -> str:
    for key in ("text", "response"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def _find_prediction_files(pred_root: Path, include_match: str, excludes: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for path in pred_root.rglob("*.jsonl"):
        path_str = str(path)
        if include_match not in path_str:
            continue
        if any(token and token in path_str for token in excludes):
            continue
        files.append(path)
    return sorted(files)


def _extract_step_num(path: Path) -> int:
    stem = path.stem
    m = re.search(r"step-(\d+|final)", stem)
    if m:
        token = m.group(1)
        return 99999 if token == "final" else int(token)

    m = re.search(r"(\d+)", stem)
    if m:
        return int(m.group(1))
    return -1


def _extract_option(text: str, options: Sequence[str]) -> Optional[str]:
    clean = text.strip()
    if clean in options:
        return clean

    for opt in options:
        if f"{opt}." in clean:
            return opt

    if "The correct option is " in clean:
        candidate = clean.split("The correct option is ")[-1].strip()[:1]
        if candidate in options:
            return candidate

    if "Answer:" in clean:
        answer = clean.split("Answer:")[-1]
        for opt in options:
            if opt in answer:
                return opt

    return None


def _labels_from_text(text: str, label_space: Sequence[str], include_norm_abnormal: bool = False) -> List[str]:
    picked = [label for label in label_space if label in text]
    if include_norm_abnormal:
        if "NORM" in text and "ABNORMAL" not in text:
            picked = ["NORM", *picked]
        elif "ABNORMAL" in text:
            picked = ["ABNORMAL", *picked]
    out: List[str] = []
    seen = set()
    for item in picked:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _option_candidates_from_prompt(prompt: Any) -> List[str]:
    if isinstance(prompt, dict):
        prompt_text = str(prompt.get("prompt", ""))
    else:
        prompt_text = str(prompt)

    if "Options:" not in prompt_text:
        return []
    fragment = prompt_text.split("Options:")[-1]
    fragment = fragment.replace("Only answer based on the given Options without any explanation.", "")
    return [item.strip() for item in fragment.split(",") if item.strip()]


def _compute_multilabel_metrics(pred: List[List[str]], gold: List[List[str]]) -> Dict[str, float]:
    try:
        from sklearn.metrics import f1_score, hamming_loss, roc_auc_score
        from sklearn.preprocessing import MultiLabelBinarizer
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn is required for multilabel ECG-Bench scoring. Install scikit-learn first."
        ) from exc

    mlb = MultiLabelBinarizer()
    gold_bin = mlb.fit_transform(gold)
    pred_bin = mlb.transform(pred)

    macro_f1 = float(f1_score(gold_bin, pred_bin, average="macro", zero_division=0))
    hl = float(hamming_loss(gold_bin, pred_bin))

    auc_values: List[float] = []
    for i in range(gold_bin.shape[1]):
        try:
            auc_values.append(float(roc_auc_score(gold_bin[:, i], pred_bin[:, i])))
        except ValueError:
            continue
    macro_auc = float(mean(auc_values)) if auc_values else 0.0

    return {
        "macro_f1": macro_f1,
        "macro_auc": macro_auc,
        "hamming_loss": hl,
    }


def _evaluate_multichoice(
    pred_rows: List[Dict[str, Any]],
    gold_rows: List[Dict[str, Any]],
    options: Sequence[str],
    seed: int,
) -> Tuple[int, int, Dict[str, float]]:
    rng = random.Random(seed)

    pred_map = {
        _extract_qid(row): _extract_prediction_text(row)
        for row in pred_rows
        if _extract_qid(row) is not None
    }

    y_true: List[str] = []
    y_pred: List[str] = []

    for row in gold_rows:
        qid = _extract_qid(row)
        if not qid:
            continue
        true_value = row.get("conversations", [{}, {"value": ""}])[1].get("value")
        if not isinstance(true_value, str):
            continue

        pred_text = pred_map.get(qid, "")
        pred_option = _extract_option(pred_text, options)
        if pred_option is None:
            pred_option = rng.choice(list(options))

        y_true.append(true_value)
        y_pred.append(pred_option)

    if not y_true:
        return 0, 0, {"accuracy": 0.0}

    return len(y_true), len(gold_rows), {"accuracy": _accuracy_score(y_true, y_pred)}


def _evaluate_multilabel(
    pred_rows: List[Dict[str, Any]],
    gold_rows: List[Dict[str, Any]],
    label_space: Sequence[str],
    include_norm_abnormal: bool,
) -> Tuple[int, int, Dict[str, float]]:
    pred_map = {
        _extract_qid(row): _extract_prediction_text(row)
        for row in pred_rows
        if _extract_qid(row) is not None
    }

    y_true: List[List[str]] = []
    y_pred: List[List[str]] = []

    for row in gold_rows:
        qid = _extract_qid(row)
        if not qid:
            continue

        true_value = row.get("conversations", [{}, {"value": ""}])[1].get("value")
        if not isinstance(true_value, str):
            continue

        y_true.append(
            _labels_from_text(
                true_value,
                label_space,
                include_norm_abnormal=include_norm_abnormal,
            )
        )
        y_pred.append(
            _labels_from_text(
                pred_map.get(qid, ""),
                label_space,
                include_norm_abnormal=include_norm_abnormal,
            )
        )

    if not y_true:
        return 0, 0, {"macro_f1": 0.0, "macro_auc": 0.0, "hamming_loss": 1.0}

    return len(y_true), len(gold_rows), _compute_multilabel_metrics(y_pred, y_true)


def _evaluate_option_set(
    pred_rows: List[Dict[str, Any]],
    gold_rows: List[Dict[str, Any]],
) -> Tuple[int, int, Dict[str, float]]:
    pred_map = {qid: row for row in pred_rows if (qid := _extract_qid(row))}

    matched = 0
    total = 0
    for row in gold_rows:
        qid = _extract_qid(row)
        if not qid:
            continue

        true_value = row.get("conversations", [{}, {"value": ""}])[1].get("value")
        if not isinstance(true_value, str):
            continue

        pred_row = pred_map.get(qid)
        pred_text = _extract_prediction_text(pred_row or {})
        candidates = _option_candidates_from_prompt((pred_row or {}).get("prompt", ""))
        pred_tokens = [token for token in candidates if token.lower() in pred_text.lower()]
        pred_joined = "".join(pred_tokens)

        matched += int(set(pred_joined) == set(true_value))
        total += 1

    accuracy = float(matched / total) if total else 0.0
    return total, len(gold_rows), {"accuracy": accuracy}


def _score_split(
    split: str,
    split_cfg: Dict[str, Any],
    pred_root: Path,
    seed: int,
) -> Dict[str, Any]:
    task = str(split_cfg.get("task", "")).strip()
    if task not in {"multichoice", "multilabel", "option_set"}:
        raise ValueError(f"Unsupported task '{task}' for split '{split}'")

    gold_file = Path(str(split_cfg["gold_file"]))
    if not gold_file.is_absolute():
        gold_file = (Path.cwd() / gold_file).resolve()
    if not gold_file.exists():
        raise FileNotFoundError(f"Gold file not found for split '{split}': {gold_file}")

    gold_rows = _load_json(gold_file)
    if not isinstance(gold_rows, list):
        raise ValueError(f"Gold file must contain a JSON list: {gold_file}")

    include_match = str(split_cfg.get("prediction_match") or split)
    excludes = split_cfg.get("exclude_match") or []
    pred_files = _find_prediction_files(pred_root, include_match, excludes)

    file_scores: List[FileScore] = []
    for pred_file in pred_files:
        pred_rows = _load_jsonl(pred_file)

        if task == "multichoice":
            options = split_cfg.get("options") or ["A", "B", "C", "D", "E", "F", "G", "H"]
            matched, gold_total, metrics = _evaluate_multichoice(pred_rows, gold_rows, options, seed)
        elif task == "multilabel":
            label_space = split_cfg.get("label_space") or []
            matched, gold_total, metrics = _evaluate_multilabel(
                pred_rows,
                gold_rows,
                label_space=label_space,
                include_norm_abnormal=bool(split_cfg.get("include_norm_abnormal", False)),
            )
        else:
            matched, gold_total, metrics = _evaluate_option_set(pred_rows, gold_rows)

        file_scores.append(
            FileScore(
                file=str(pred_file),
                step=_extract_step_num(pred_file),
                matched=matched,
                gold_total=gold_total,
                metrics=metrics,
            )
        )

    primary_metric = "accuracy" if task in {"multichoice", "option_set"} else "macro_f1"
    file_scores.sort(key=lambda x: (x.step, x.metrics.get(primary_metric, 0.0)))

    best = max(file_scores, key=lambda x: x.metrics.get(primary_metric, -1.0), default=None)

    payload: Dict[str, Any] = {
        "task": task,
        "gold_file": str(gold_file),
        "gold_count": len(gold_rows),
        "prediction_files": [
            {
                "file": item.file,
                "step": item.step,
                "matched": item.matched,
                "gold_total": item.gold_total,
                "coverage": float(item.matched / item.gold_total) if item.gold_total else 0.0,
                "metrics": item.metrics,
            }
            for item in file_scores
        ],
        "primary_metric": primary_metric,
    }

    if best is not None:
        payload["best"] = {
            "file": best.file,
            "step": best.step,
            "metrics": best.metrics,
        }
    else:
        payload["best"] = None

    return payload


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GEM ECG-Bench results")
    parser.add_argument("--pred-root", required=True, help="Root directory containing prediction JSONL files")
    parser.add_argument("--datasets-config", required=True, help="Path to evaluation/config/datasets.yaml")
    parser.add_argument("--splits", nargs="*", default=[], help="Optional subset of split names")
    parser.add_argument("--output-json", default="", help="Optional output JSON path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON output")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    pred_root = Path(args.pred_root).expanduser().resolve()
    datasets_cfg_path = Path(args.datasets_config).expanduser().resolve()

    if not pred_root.exists():
        raise FileNotFoundError(f"pred_root not found: {pred_root}")
    if not datasets_cfg_path.exists():
        raise FileNotFoundError(f"datasets_config not found: {datasets_cfg_path}")

    datasets = _load_dataset_config(datasets_cfg_path)
    requested_splits = args.splits or sorted(datasets.keys())

    report: Dict[str, Any] = {
        "pred_root": str(pred_root),
        "datasets_config": str(datasets_cfg_path),
        "splits": {},
    }

    for split in requested_splits:
        if split not in datasets:
            raise KeyError(f"Split '{split}' not found in datasets config")
        report["splits"][split] = _score_split(split, datasets[split], pred_root, seed=args.seed)

    json_text = json.dumps(report, indent=2 if args.pretty else None, ensure_ascii=False)
    print(json_text)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(json_text)
            if args.pretty:
                f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
