#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:  # pragma: no cover
    from openai import OpenAI as _OpenAIClient
except ModuleNotFoundError:  # pragma: no cover
    _OpenAIClient = None

OpenAI = _OpenAIClient


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _render_prompt(template_raw: str, generated: str, groundtruth: str) -> str:
    return (
        template_raw.replace("{{generated}}", generated).replace("{{groundtruth}}", groundtruth)
    )


def _request_with_retry(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int,
) -> str:
    last_error: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = completion.choices[0].message.content
            return content.strip() if isinstance(content, str) else ""
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"OpenAI request failed after {max_retries} retries: {last_error}")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT evaluation for grounding outputs")
    parser.add_argument("--input-json", required=True, help="Path to merged input JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for per-id output JSON files")
    parser.add_argument("--template-file", default="gem_evaluation/prompts_evaluation.txt")
    parser.add_argument("--model", default=os.getenv("EVAL_MODEL", "gpt-4o-2024-08-06"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--generated-key", default="GEM_generated")
    parser.add_argument("--groundtruth-key", default="GPT4o_generated")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if not args.api_key:
        raise RuntimeError("Missing API key: pass --api-key or set OPENAI_API_KEY")
    if OpenAI is None:
        raise RuntimeError("Missing dependency 'openai'. Install it with: pip install openai")

    input_path = Path(args.input_json).expanduser().resolve()
    template_path = Path(args.template_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"input-json not found: {input_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"template-file not found: {template_path}")

    data = _load_json(input_path)
    if not isinstance(data, list):
        raise ValueError("input-json must contain a JSON list")

    template_raw = template_path.read_text(encoding="utf-8")

    start = max(0, int(args.start))
    end = len(data) if int(args.end) < 0 else min(len(data), int(args.end))
    batch = data[start:end]

    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=args.api_key)

    processed = 0
    skipped = 0
    for idx, row in enumerate(batch, start=1):
        if not isinstance(row, dict):
            continue

        rid = str(row.get("id") or f"row_{start + idx - 1}")
        out_path = output_dir / f"{rid}.json"
        if args.resume and out_path.exists():
            skipped += 1
            continue

        generated = str(row.get(args.generated_key, ""))
        groundtruth = str(row.get(args.groundtruth_key, ""))
        prompt = _render_prompt(template_raw, generated, groundtruth)

        result = _request_with_retry(client, args.model, prompt, args.max_retries)

        payload = {
            "id": rid,
            "results": result,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        processed += 1
        print(f"[{processed}] saved {out_path}")

    print(
        json.dumps(
            {
                "input_json": str(input_path),
                "output_dir": str(output_dir),
                "model": args.model,
                "start": start,
                "end": end,
                "processed": processed,
                "skipped": skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
