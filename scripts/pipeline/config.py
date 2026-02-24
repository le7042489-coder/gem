from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import yaml  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency 'pyyaml'. Install it with: pip install pyyaml"
    ) from exc

from .schema import (
    ConfigValidationError,
    collect_missing_paths,
    merge_with_defaults,
    required_paths_for_command,
    resolve_paths,
    set_nested,
    validate_types,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipelines" / "gem_default.yaml"


class ConfigError(Exception):
    """Raised for invalid CLI config input."""


class PathValidationError(Exception):
    """Raised when required paths for a command are missing."""


def _parse_override_value(raw: str) -> Any:
    stripped = raw.strip()
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    if stripped.lower() in {"none", "null"}:
        return None

    for caster in (int, float):
        try:
            if stripped.startswith("0") and len(stripped) > 1 and stripped[1].isdigit():
                break
            return caster(stripped)
        except ValueError:
            continue

    if (
        (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("[") and stripped.endswith("]"))
        or (stripped.startswith('"') and stripped.endswith('"'))
    ):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    return raw


def _apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    out = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Invalid --set value '{item}'. Expected key=value")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"Invalid --set value '{item}'. Key cannot be empty")
        set_nested(out, key, _parse_override_value(raw_value))
    return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError("Pipeline config must be a YAML mapping")
    return data


def load_effective_config(
    config_path: str | Path | None,
    overrides: Iterable[str],
) -> Dict[str, Any]:
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()

    raw = _load_yaml(path)
    merged = merge_with_defaults(raw)
    merged = _apply_overrides(merged, overrides)

    try:
        validate_types(merged)
    except ConfigValidationError as exc:
        raise ConfigError(str(exc)) from exc

    resolved = resolve_paths(merged, PROJECT_ROOT)
    return resolved


def validate_required_paths(config: Dict[str, Any], command: str) -> None:
    required = required_paths_for_command(command)
    missing = collect_missing_paths(config, required)

    if command == "eval-generate-grounding":
        q_files = config.get("evaluation", {}).get("grounding", {}).get("question_files", [])
        q_glob = config.get("evaluation", {}).get("grounding", {}).get("question_files_glob", "")
        has_input = bool(q_files) or bool(q_glob)
        if not has_input:
            missing.append(("evaluation.grounding.question_files_glob", "empty"))
        elif q_glob:
            from glob import glob

            if not glob(q_glob):
                missing.append(("evaluation.grounding.question_files_glob", q_glob))
        if q_files:
            for item in q_files:
                if not Path(str(item)).exists():
                    missing.append(("evaluation.grounding.question_files[]", str(item)))

    if missing:
        lines = ["Missing required paths:"]
        for key, value in missing:
            lines.append(f"- {key}: {value}")
        raise PathValidationError("\n".join(lines))


def dump_config(config: Dict[str, Any]) -> str:
    return json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True)
