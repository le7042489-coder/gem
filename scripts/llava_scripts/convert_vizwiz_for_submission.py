#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO_ROOT = None
for parent in [HERE.parent, *HERE.parents]:
    if (parent / ".git").exists():
        REPO_ROOT = parent
        break

if REPO_ROOT is None:
    raise SystemExit("Could not locate repository root from wrapper path")

TARGET = REPO_ROOT / "legacy" / "llava_scripts" / "convert_vizwiz_for_submission.py"

print(
    "[DEPRECATED] scripts/llava_scripts compatibility wrappers will be removed in the next major release. Use legacy/llava_scripts directly.",
    file=sys.stderr,
)

raise SystemExit(subprocess.call([sys.executable, str(TARGET), *sys.argv[1:]]))
