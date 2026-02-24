#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-gem}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_BACKEND="${INSTALL_BACKEND:-1}"
FLASH_ATTN_MODE="${FLASH_ATTN_MODE:-auto}"

BACKEND_REQUIREMENTS="${ROOT_DIR}/backend/requirements.txt"
ENV_KIND=""
ACTIVATE_HINT=""
VENV_PATH=""

log() {
    printf '[setup] %s\n' "$1"
}

warn() {
    printf '[setup][warn] %s\n' "$1" >&2
}

die() {
    printf '[setup][error] %s\n' "$1" >&2
    exit 1
}

to_lower() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

is_truthy() {
    case "$(to_lower "$1")" in
        1|true|yes|y|on) return 0 ;;
        0|false|no|n|off) return 1 ;;
        *)
            die "Invalid boolean value: $1 (expected 1/0, true/false, yes/no)"
            ;;
    esac
}

resolve_venv_path() {
    if [[ "$VENV_DIR" = /* ]]; then
        printf '%s' "$VENV_DIR"
    else
        printf '%s' "${ROOT_DIR}/${VENV_DIR}"
    fi
}

pick_python_for_venv() {
    if command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
        printf '%s' "python${PYTHON_VERSION}"
        return
    fi
    if command -v python3 >/dev/null 2>&1; then
        printf '%s' "python3"
        return
    fi
    if command -v python >/dev/null 2>&1; then
        printf '%s' "python"
        return
    fi
    die "No Python interpreter found for venv fallback."
}

run_pip() {
    python -m pip "$@"
}

if command -v conda >/dev/null 2>&1; then
    log "Detected conda. Preparing environment '${ENV_NAME}' (python ${PYTHON_VERSION})."
    eval "$(conda shell.bash hook)"
    if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        log "Conda environment '${ENV_NAME}' already exists. Reusing."
    else
        log "Creating conda environment '${ENV_NAME}'."
        conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
    fi
    conda activate "$ENV_NAME"
    ENV_KIND="conda"
    ACTIVATE_HINT="conda activate ${ENV_NAME}"
else
    warn "Conda not found. Falling back to python venv."
    VENV_PATH="$(resolve_venv_path)"
    PYTHON_BIN="$(pick_python_for_venv)"
    if [[ -d "$VENV_PATH" ]]; then
        log "Virtual environment already exists at '${VENV_PATH}'. Reusing."
    else
        log "Creating virtual environment at '${VENV_PATH}' with ${PYTHON_BIN}."
        "$PYTHON_BIN" -m venv "$VENV_PATH"
    fi
    # shellcheck disable=SC1090
    source "${VENV_PATH}/bin/activate"
    ENV_KIND="venv"
    if [[ "$VENV_DIR" = /* ]]; then
        ACTIVATE_HINT="source ${VENV_DIR}/bin/activate"
    else
        ACTIVATE_HINT="cd ${ROOT_DIR} && source ${VENV_DIR}/bin/activate"
    fi
fi

ACTIVE_PYTHON_MM="$(python -c 'import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")')"
if [[ "$ACTIVE_PYTHON_MM" != "$PYTHON_VERSION" ]]; then
    warn "Active Python is ${ACTIVE_PYTHON_MM}, expected ${PYTHON_VERSION}."
fi

log "Upgrading pip."
run_pip install --upgrade pip

log "Installing project and training dependencies."
run_pip install -e ".[train]"

if is_truthy "$INSTALL_BACKEND"; then
    [[ -f "$BACKEND_REQUIREMENTS" ]] || die "Missing file: ${BACKEND_REQUIREMENTS}"
    log "Installing backend dependencies from backend/requirements.txt."
    run_pip install -r "$BACKEND_REQUIREMENTS"
else
    log "Skipping backend dependencies (INSTALL_BACKEND=${INSTALL_BACKEND})."
fi

log "Pinning peft to 0.10.0 for compatibility."
run_pip install "peft==0.10.0"

case "$(to_lower "$FLASH_ATTN_MODE")" in
    auto)
        log "Installing flash-attn (auto mode)."
        if run_pip install flash-attn --no-build-isolation; then
            log "flash-attn installed successfully."
        else
            warn "flash-attn install failed; continuing because FLASH_ATTN_MODE=auto."
        fi
        ;;
    strict)
        log "Installing flash-attn (strict mode)."
        run_pip install flash-attn --no-build-isolation
        ;;
    skip)
        log "Skipping flash-attn install (FLASH_ATTN_MODE=skip)."
        ;;
    *)
        die "Invalid FLASH_ATTN_MODE: ${FLASH_ATTN_MODE} (expected: auto|strict|skip)"
        ;;
esac

log "Setup completed."
printf '\n'
printf 'Environment type: %s\n' "$ENV_KIND"
printf 'Project root: %s\n' "$ROOT_DIR"
printf 'Activation command: %s\n' "$ACTIVATE_HINT"
