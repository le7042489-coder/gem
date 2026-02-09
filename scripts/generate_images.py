#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ECG_TOOL_DIR = REPO_ROOT / "gem_generation" / "ecg-image-generator"

STANDARD_12_INPUT_LEADS = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ECG .png images from processed_data/*.npy using "
            "gem_generation/ecg-image-generator, then update data/mixed_train.json."
        )
    )
    parser.add_argument("--processed-dir", type=str, default="processed_data")
    parser.add_argument("--json-path", type=str, default="data/mixed_train.json")
    parser.add_argument("--sample-rate", type=int, default=500)
    parser.add_argument("--layout", type=str, choices=("3x4", "12x1"), default="3x4")
    parser.add_argument("--resolution", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--input-lead-order",
        type=str,
        choices=("standard", "config"),
        default="standard",
        help=(
            "How to interpret the 12 rows in each .npy file: "
            "'standard' assumes [I, II, III, aVR, aVL, aVF, V1..V6]; "
            "'config' uses leadNames_12 from ecg-image-generator/config.yaml."
        ),
    )
    return parser.parse_args()


def _iter_npy_files(processed_dir: Path) -> list[Path]:
    subdirs = [processed_dir / "ludb", processed_dir / "ptbxl", processed_dir / "mimic"]
    npy_files: list[Path] = []
    for subdir in subdirs:
        if not subdir.exists():
            continue
        for path in subdir.rglob("*.npy"):
            if path.name.endswith("_mask.npy"):
                continue
            npy_files.append(path)
    npy_files.sort()
    return npy_files


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'pyyaml'. Install it (e.g. `pip install pyyaml`)."
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_numpy(path: Path):
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'numpy'. Install it (e.g. `pip install numpy`)."
        ) from exc
    return np.load(path)


def _prepare_signal(signal, expected_len: int):
    import numpy as np  # type: ignore

    if signal.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={signal.shape}.")

    if signal.shape[0] == 12:
        signal_12t = signal
    elif signal.shape[1] == 12:
        signal_12t = signal.T
    else:
        raise ValueError(f"Expected shape (12, T) or (T, 12), got {signal.shape}.")

    t = int(signal_12t.shape[1])
    if t > expected_len:
        return signal_12t[:, :expected_len]
    if t < expected_len:
        padded = np.zeros((12, expected_len), dtype=signal_12t.dtype)
        padded[:, :t] = signal_12t
        return padded
    return signal_12t


def _get_input_leads(configs: dict[str, Any], input_lead_order: str) -> list[str]:
    if input_lead_order == "standard":
        return list(STANDARD_12_INPUT_LEADS)
    if input_lead_order == "config":
        leads = configs.get("leadNames_12")
        if not isinstance(leads, list) or len(leads) != 12:
            raise ValueError("config.yaml missing valid 'leadNames_12' (expected list of 12).")
        return [str(x) for x in leads]
    raise ValueError(f"Unsupported input_lead_order={input_lead_order!r}.")


def _plot_leads_and_data(
    full_ecg: dict[str, Any],
    *,
    paper_len: float,
    expected_len: int,
    layout: str,
):
    import numpy as np  # type: ignore

    if layout == "3x4":
        segment_secs = paper_len / 4.0
        segment_len = expected_len // 4

        col1 = ("I", "II", "III")
        col2 = ("aVR", "aVL", "aVF")
        col3 = ("V1", "V2", "V3")
        col4 = ("V4", "V5", "V6")
        lead_to_offset = {lead: 0 for lead in col1}
        lead_to_offset.update({lead: 1 for lead in col2})
        lead_to_offset.update({lead: 2 for lead in col3})
        lead_to_offset.update({lead: 3 for lead in col4})

        plot_leads = [
            col1[2],
            col2[2],
            col3[2],
            col4[2],
            col1[1],
            col2[1],
            col3[1],
            col4[1],
            col1[0],
            col2[0],
            col3[0],
            col4[0],
        ]

        ecg_to_plot: dict[str, Any] = {}
        for lead in plot_leads:
            if lead not in full_ecg:
                raise KeyError(f"Missing lead {lead!r} in ECG data (available={sorted(full_ecg)}).")
            offset = lead_to_offset[lead]
            start = offset * segment_len
            end = start + segment_len
            seg = full_ecg[lead][start:end]
            if len(seg) != segment_len:
                seg = np.pad(seg, (0, max(0, segment_len - len(seg))), mode="constant")
            ecg_to_plot[lead] = seg

        return plot_leads, ecg_to_plot, segment_secs, 4

    if layout == "12x1":
        plot_leads = list(reversed(STANDARD_12_INPUT_LEADS))
        ecg_to_plot: dict[str, Any] = {}
        for lead in plot_leads:
            if lead not in full_ecg:
                raise KeyError(f"Missing lead {lead!r} in ECG data (available={sorted(full_ecg)}).")
            ecg_to_plot[lead] = full_ecg[lead]
        return plot_leads, ecg_to_plot, paper_len, 1

    raise ValueError(f"Unsupported layout={layout!r}.")


def generate_png(
    npy_path: Path,
    configs: dict[str, Any],
    args: argparse.Namespace,
    ecg_plot_func: Any,
) -> bool:
    png_path = npy_path.with_suffix(".png")
    if png_path.exists() and not args.overwrite:
        return False

    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'numpy'. Install it (e.g. `pip install numpy`)."
        ) from exc

    paper_len = float(configs.get("paper_len", 10.0))
    expected_len = int(args.sample_rate * paper_len)

    signal = _load_numpy(npy_path)
    signal = _prepare_signal(signal, expected_len)

    input_leads = _get_input_leads(configs, args.input_lead_order)
    if len(input_leads) != 12:
        raise ValueError(f"Expected 12 input leads, got {len(input_leads)}.")

    full_ecg = {lead: np.asarray(signal[i], dtype=np.float32) for i, lead in enumerate(input_leads)}

    plot_leads, ecg_to_plot, lead_len_secs, columns = _plot_leads_and_data(
        full_ecg, paper_len=paper_len, expected_len=expected_len, layout=args.layout
    )

    configs_plot = dict(configs)
    configs_plot["leadNames_12"] = plot_leads

    ecg_plot_func(
        ecg=ecg_to_plot,
        configs=configs_plot,
        sample_rate=args.sample_rate,
        columns=columns,
        rec_file_name=npy_path.stem,
        output_dir=str(npy_path.parent),
        resolution=args.resolution,
        pad_inches=0,
        lead_index=plot_leads,
        full_mode="None",
        store_text_bbox=False,
        full_header_file="",
        show_lead_name=True,
        show_grid=True,
        show_dc_pulse=True,
        style="colour",
        standard_colours=5,
        print_txt=False,
        json_dict={},
        start_index=0,
        lead_length_in_seconds=lead_len_secs,
    )

    return png_path.exists()


def update_mixed_train_json(json_path: Path) -> int:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {json_path}, got {type(data).__name__}.")

    updated = 0
    for sample in data:
        if not isinstance(sample, dict):
            continue
        img = sample.get("image")
        if isinstance(img, str) and img.endswith(".npy"):
            sample["image"] = img[:-4] + ".png"
            updated += 1

    tmp_path = json_path.with_suffix(json_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\r\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, json_path)
    return updated


def main() -> int:
    args = parse_args()

    if not ECG_TOOL_DIR.exists():
        print(f"Error: missing directory: {ECG_TOOL_DIR}", file=sys.stderr)
        return 2

    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_absolute():
        processed_dir = REPO_ROOT / processed_dir
    processed_dir = processed_dir.resolve()
    if not processed_dir.exists():
        print(f"Error: missing processed dir: {processed_dir}", file=sys.stderr)
        return 2

    config_path = ECG_TOOL_DIR / "config.yaml"
    if not config_path.exists():
        print(f"Error: missing config file: {config_path}", file=sys.stderr)
        return 2

    configs = _load_yaml(config_path)

    npy_files = _iter_npy_files(processed_dir)
    if args.limit != -1:
        npy_files = npy_files[: max(0, args.limit)]

    try:
        from tqdm import tqdm  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'tqdm'. Install it (e.g. `pip install tqdm`)."
        ) from exc

    if str(ECG_TOOL_DIR) not in sys.path:
        sys.path.insert(0, str(ECG_TOOL_DIR))
    try:
        from ecg_plot import ecg_plot  # type: ignore
    except ModuleNotFoundError as exc:
        print(
            "Error: failed to import ecg_plot from ecg-image-generator. "
            "Make sure required dependencies are installed (numpy, matplotlib, pillow, wfdb, pyyaml, tqdm).",
            file=sys.stderr,
        )
        print(f"Details: {exc}", file=sys.stderr)
        return 2

    generated = 0
    for npy_path in tqdm(npy_files, desc="Generating ECG PNGs"):
        try:
            if generate_png(npy_path, configs, args, ecg_plot):
                generated += 1
        except Exception as e:
            print(f"[WARN] Failed: {npy_path}: {e}", file=sys.stderr)

    json_path = Path(args.json_path)
    if not json_path.is_absolute():
        json_path = REPO_ROOT / json_path
    json_path = json_path.resolve()
    if not json_path.exists():
        print(f"Error: missing json file: {json_path}", file=sys.stderr)
        return 2
    update_mixed_train_json(json_path)

    print(f"Generated {generated} images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
