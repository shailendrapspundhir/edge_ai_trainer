#!/usr/bin/env bash
# prerequisites.sh — Edge AI Trainer
#
# Checks dependencies, installs anything missing, creates/sources the project
# virtual environment, and prints the commands to start the stack.
#
# Usage:
#   source ./prerequisites.sh    # recommended — leaves you inside the venv
#   ./prerequisites.sh           # also works, but the venv won't persist in your shell

# --- Allow being sourced or executed --------------------------------------
(return 0 2>/dev/null) && SOURCED=1 || SOURCED=0
if [ "$SOURCED" -eq 1 ]; then
    set +e
else
    set -e
fi

# --- Colors ---------------------------------------------------------------
if [ -t 1 ]; then
    C_RESET="\033[0m"; C_BOLD="\033[1m"
    C_RED="\033[31m"; C_GREEN="\033[32m"; C_YELLOW="\033[33m"; C_BLUE="\033[34m"
else
    C_RESET=""; C_BOLD=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""
fi

info()  { printf "${C_BLUE}[info]${C_RESET}  %s\n" "$*"; }
ok()    { printf "${C_GREEN}[ ok ]${C_RESET}  %s\n" "$*"; }
warn()  { printf "${C_YELLOW}[warn]${C_RESET}  %s\n" "$*"; }
err()   { printf "${C_RED}[fail]${C_RESET}  %s\n" "$*"; }
hdr()   { printf "\n${C_BOLD}== %s ==${C_RESET}\n" "$*"; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

# --- Helpers --------------------------------------------------------------
has_cmd() { command -v "$1" >/dev/null 2>&1; }

detect_pkg_mgr() {
    if   has_cmd apt-get; then echo "apt"
    elif has_cmd dnf;     then echo "dnf"
    elif has_cmd pacman;  then echo "pacman"
    elif has_cmd brew;    then echo "brew"
    else echo ""
    fi
}

PKG_MGR="$(detect_pkg_mgr)"

apt_install() {
    sudo apt-get update -y
    sudo apt-get install -y "$@"
}

install_system_pkg() {
    local pkg="$1"
    case "$PKG_MGR" in
        apt)    apt_install "$pkg" ;;
        dnf)    sudo dnf install -y "$pkg" ;;
        pacman) sudo pacman -S --noconfirm "$pkg" ;;
        brew)   brew install "$pkg" ;;
        *)      err "No supported package manager found. Install '$pkg' manually."; return 1 ;;
    esac
}

python_version_ok() {
    local py="$1"
    "$py" - <<'PY' 2>/dev/null
import sys
ok = (3, 10) <= sys.version_info[:2] < (3, 13)
sys.exit(0 if ok else 1)
PY
}

find_python() {
    for candidate in python3.12 python3.11 python3.10 python3 python; do
        if has_cmd "$candidate" && python_version_ok "$candidate"; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

# --- 1. System tooling ----------------------------------------------------
hdr "Checking system tooling"

for tool in curl git; do
    if has_cmd "$tool"; then
        ok "$tool found"
    else
        warn "$tool missing — attempting install"
        install_system_pkg "$tool" || { err "Could not install $tool"; }
    fi
done

# Docker is only required for `make redis`; warn if missing, don't fail.
if has_cmd docker; then
    ok "docker found ($(docker --version 2>/dev/null | head -1))"
else
    warn "docker not found — needed for 'make redis' (the job queue). Install from https://docs.docker.com/engine/install/"
fi

# --- 2. uv ----------------------------------------------------------------
# Install uv first — it can also manage Python interpreters, which avoids
# relying on the distro shipping a compatible python3.x.
hdr "Checking uv (package manager)"

if has_cmd uv; then
    ok "uv found ($(uv --version))"
else
    info "Installing uv from astral.sh"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installer drops the binary in ~/.local/bin or ~/.cargo/bin
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if has_cmd uv; then
        ok "uv installed ($(uv --version))"
    else
        err "uv install did not place a binary on PATH. Open a new shell and re-run, or install manually."
        [ "$SOURCED" -eq 1 ] && return 1 || exit 1
    fi
fi

# --- 3. Python (>=3.10, <3.13) -------------------------------------------
hdr "Checking Python (>=3.10, <3.13)"

REQUIRED_PY="3.12"
PYTHON_BIN="$(find_python || true)"

if [ -n "$PYTHON_BIN" ]; then
    ok "Using system Python: $PYTHON_BIN ($("$PYTHON_BIN" --version))"
else
    warn "No compatible system Python found. Trying distro packages first, then uv-managed Python."

    # Try distro packages in order of preference. Failures here are non-fatal —
    # we fall back to uv's managed Python afterwards.
    case "$PKG_MGR" in
        apt)
            for v in 3.12 3.11 3.10; do
                if apt-cache show "python$v" >/dev/null 2>&1; then
                    info "Installing python$v + python$v-venv via apt"
                    if apt_install "python$v" "python$v-venv"; then
                        break
                    fi
                fi
            done
            ;;
        dnf)
            for v in 3.12 3.11 3.10; do
                sudo dnf install -y "python$v" 2>/dev/null && break
            done
            ;;
        pacman)
            sudo pacman -S --noconfirm python 2>/dev/null || true
            ;;
        brew)
            brew install python@3.12 2>/dev/null || brew install python@3.11 2>/dev/null || true
            ;;
    esac

    PYTHON_BIN="$(find_python || true)"

    if [ -z "$PYTHON_BIN" ]; then
        info "Falling back to uv-managed Python ($REQUIRED_PY)"
        if uv python install "$REQUIRED_PY"; then
            PYTHON_BIN="$(uv python find "$REQUIRED_PY" 2>/dev/null || true)"
        fi
    fi

    if [ -z "$PYTHON_BIN" ]; then
        # Last resort: let `uv venv` pick/download a Python on its own.
        warn "Could not resolve a Python binary path; will let uv pick one when creating the venv."
        PYTHON_BIN="$REQUIRED_PY"
    fi
    ok "Using Python: $PYTHON_BIN"
fi

# --- 4. Virtual environment ----------------------------------------------
hdr "Virtual environment"

cd "$PROJECT_ROOT"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    ok "Existing venv at $VENV_DIR"
else
    info "Creating venv via uv at $VENV_DIR (python=$PYTHON_BIN)"
    uv venv --python "$PYTHON_BIN" "$VENV_DIR"
    ok "venv created"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "venv activated ($(python --version))"

# --- 5. Project dependencies ---------------------------------------------
hdr "Syncing project dependencies (uv sync)"
uv sync
ok "Base dependencies installed"

# --- 6. .env --------------------------------------------------------------
hdr "Environment file"
if [ ! -f "$PROJECT_ROOT/.env" ] && [ -f "$PROJECT_ROOT/.env.example" ]; then
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    ok "Copied .env.example -> .env (edit it before running real training)"
else
    ok ".env present (or no .env.example to copy from)"
fi

# --- 7. Final instructions ------------------------------------------------
hdr "Ready"
cat <<EOF

${C_GREEN}Edge AI Trainer is ready.${C_RESET}

${C_BOLD}If you ran this script directly${C_RESET} (./prerequisites.sh), activate the venv now:
    source .venv/bin/activate

${C_BOLD}Start the stack (each in a separate terminal):${C_RESET}
    make redis            # Redis job queue (needs Docker)
    make orchestrator     # FastAPI control plane on :8765
    make worker-cpu       # RQ worker for the cpu queue
    make dashboard        # Streamlit dashboard on :8501

${C_BOLD}Smoke test (no GPU needed):${C_RESET}
    make smoke

${C_BOLD}For real training (adds torch + transformers + peft + trl + bnb):${C_RESET}
    make install-train
    make worker-gpu
    eat run --project food_health_coach --recipe qlora_text

See README.md and PLAN.md for more.
EOF

[ "$SOURCED" -eq 1 ] && return 0 || exit 0
