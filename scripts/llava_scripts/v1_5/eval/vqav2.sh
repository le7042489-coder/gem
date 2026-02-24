#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [[ "$REPO_ROOT" != "/" && ! -d "$REPO_ROOT/.git" ]]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done

if [[ ! -d "$REPO_ROOT/.git" ]]; then
  echo "Could not locate repository root from wrapper path" >&2
  exit 1
fi

TARGET="$REPO_ROOT/legacy/llava_scripts/v1_5/eval/vqav2.sh"

echo "[DEPRECATED] scripts/llava_scripts compatibility wrappers will be removed in the next major release. Use legacy/llava_scripts directly." >&2
exec bash "$TARGET" "$@"
