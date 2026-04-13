#!/bin/bash
# AniGen setup script — supports Python 3.10+ with uv (or pip fallback)
#
# Usage:
#   . ./setup.sh --all              # Full install (recommended)
#   . ./setup.sh --all --tsinghua   # Full install with Tsinghua PyPI mirror
#   . ./setup.sh --basic            # Core deps only (no demo UI)
#   . ./setup.sh --demo             # Add Gradio demo deps
#
# The script auto-detects CUDA version and installs matching wheels.
# If uv is available it will be used for faster installs; otherwise falls back to pip.

set -e

# ─── Parse arguments ────────────────────────────────────────────────────────

HELP=false; NEW_ENV=false; TORCH=false; BASIC=false; ALL=false
DEMO=false; FLASH_ATTN=false; XFORMERS=false; TSINGHUA=false; TRAIN=false

TEMP=$(getopt -o h --long help,new-env,torch,basic,all,demo,flash-attn,xformers,tsinghua,train -n 'setup.sh' -- "$@") || { echo "Invalid arguments"; return 1; }
eval set -- "$TEMP"
while true; do
    case "$1" in
        -h|--help)      HELP=true;      shift ;;
        --new-env)      NEW_ENV=true;   shift ;;
        --torch)        TORCH=true;     shift ;;
        --basic)        BASIC=true;     shift ;;
        --all)          ALL=true;       shift ;;
        --demo)         DEMO=true;      shift ;;
        --flash-attn)   FLASH_ATTN=true; shift ;;
        --xformers)     XFORMERS=true;  shift ;;
        --tsinghua)     TSINGHUA=true;  shift ;;
        --train)        TRAIN=true;     shift ;;
        --)             shift; break ;;
        *)              break ;;
    esac
done

if [ "$ALL" = true ]; then TORCH=true; BASIC=true; DEMO=true; XFORMERS=true; FLASH_ATTN=true; fi
if [ "$DEMO" = true ]; then TORCH=true; BASIC=true; fi

if [ "$HELP" = true ] || [ "$#" -eq 0 -a "$ALL" = false -a "$BASIC" = false -a "$TORCH" = false -a "$DEMO" = false -a "$NEW_ENV" = false -a "$FLASH_ATTN" = false -a "$XFORMERS" = false -a "$TRAIN" = false ]; then
    echo "Usage: . ./setup.sh [OPTIONS]"
    echo
    echo "Options:"
    echo "  --new-env       Create a new virtual environment (requires uv or conda)"
    echo "  --torch         Install PyTorch (auto-detects CUDA version)"
    echo "  --basic         Install core dependencies"
    echo "  --all           Install everything (torch + basic + demo + xformers + flash-attn)"
    echo "  --demo          Install Gradio demo dependencies"
    echo "  --flash-attn    Install flash-attention (optional, improves speed)"
    echo "  --xformers      Install xformers (optional, improves speed)"
    echo "  --tsinghua      Use Tsinghua PyPI mirror"
    echo "  --train         Install training dependencies"
    echo "  -h, --help      Show this help"
    echo
    echo "Environment variables:"
    echo "  ANIGEN_PYTHON=/path/to/python   Use a specific Python interpreter"
    echo "  TORCH_VERSION=2.4.0             Override PyTorch version (default: 2.4.0 for Python <=3.12, 2.5.0 for 3.13+)"
    return 0
fi

# ─── Resolve Python interpreter ─────────────────────────────────────────────

WORKDIR="$(pwd)"

if [ -n "${ANIGEN_PYTHON:-}" ]; then
    PYTHON_BIN="$ANIGEN_PYTHON"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
    PYTHON_BIN="$(command -v python3 2>/dev/null || command -v python 2>/dev/null)"
fi

# ─── Helper: choose pip backend ─────────────────────────────────────────────

if command -v uv >/dev/null 2>&1; then
    _INSTALLER=uv
    echo "[SETUP] Using uv (fast mode)"
else
    _INSTALLER=pip
    echo "[SETUP] uv not found, using pip (install uv for faster setup: https://docs.astral.sh/uv/)"
fi

_pip_install() {
    if [ "$_INSTALLER" = "uv" ]; then
        uv pip install "$@"
    else
        "$PYTHON_BIN" -m pip install "$@"
    fi
}

_pip_install_quiet() {
    _pip_install "$@" 2>&1 | tail -3
}

# ─── Helper: download with retry ────────────────────────────────────────────

_download() {
    local url="$1" out="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -L --retry 3 --connect-timeout 15 --max-time 300 -o "$out" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --timeout=30 -O "$out" "$url"
    else
        echo "[ERROR] Neither curl nor wget available"; return 1
    fi
}

_git_clone() {
    git clone "$@" 2>/dev/null || git -c http.proxy= -c https.proxy= clone "$@"
}

# ─── Helper: detect CUDA version from PyTorch ───────────────────────────────

_detect_cuda() {
    "$PYTHON_BIN" -c "import torch; print(torch.version.cuda or '')" 2>/dev/null
}

_detect_torch_version() {
    "$PYTHON_BIN" -c "import torch; print(torch.__version__.split('+')[0].split('a')[0])" 2>/dev/null
}

_detect_python_version() {
    "$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')" 2>/dev/null
}

# ─── Tsinghua mirror ────────────────────────────────────────────────────────

if [ "$TSINGHUA" = true ]; then
    TSINGHUA_URL=https://pypi.tuna.tsinghua.edu.cn/simple
    export PIP_INDEX_URL=$TSINGHUA_URL
    export PIP_EXTRA_INDEX_URL=$TSINGHUA_URL
    export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
    export PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT:-120}
    if [ "$_INSTALLER" = "uv" ]; then
        export UV_INDEX_URL=$TSINGHUA_URL
        export UV_EXTRA_INDEX_URL=$TSINGHUA_URL
    fi
    echo "[SETUP] Using Tsinghua PyPI mirror"
fi

# ─── Step 0: Create environment ─────────────────────────────────────────────

if [ "$NEW_ENV" = true ]; then
    if [ "$_INSTALLER" = "uv" ]; then
        echo "[SETUP] Creating virtual environment with uv..."
        uv venv --python ">=3.10" .venv
        # shellcheck disable=SC1091
        . .venv/bin/activate
    elif command -v conda >/dev/null 2>&1; then
        echo "[SETUP] Creating conda environment..."
        conda create -n anigen "python>=3.10" -y
        conda activate anigen
    else
        echo "[SETUP] Creating venv with stdlib..."
        "$PYTHON_BIN" -m venv .venv
        # shellcheck disable=SC1091
        . .venv/bin/activate
    fi
    PYTHON_BIN="$(command -v python3 2>/dev/null || command -v python)"
    echo "[SETUP] Python: $("$PYTHON_BIN" --version)"
    TORCH=true
fi

# ─── Step 1: Install PyTorch ────────────────────────────────────────────────

if [ "$TORCH" = true ]; then
    # Detect CUDA toolkit version
    CUDA_VER=""
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_VER=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | head -1)
    elif [ -n "${CUDA_HOME:-}" ]; then
        CUDA_VER=$(cat "${CUDA_HOME}/version.json" 2>/dev/null | "$PYTHON_BIN" -c "import json,sys; print(json.load(sys.stdin).get('cuda',{}).get('version',''))" 2>/dev/null || true)
    fi
    CUDA_MAJOR=$(echo "${CUDA_VER}" | cut -d'.' -f1)

    # Auto-select default PyTorch version based on Python version
    PYVER_NUM=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "10")
    if [ -z "${TORCH_VERSION:-}" ]; then
        if [ "$PYVER_NUM" -ge 13 ]; then
            TORCH_VER="2.5.0"
        else
            TORCH_VER="2.4.0"
        fi
    else
        TORCH_VER="$TORCH_VERSION"
    fi

    if [ "${CUDA_MAJOR}" = "12" ]; then
        TORCH_INDEX=https://download.pytorch.org/whl/cu121
    elif [ "${CUDA_MAJOR}" = "11" ]; then
        TORCH_INDEX=https://download.pytorch.org/whl/cu118
    else
        echo "[SETUP] Could not detect CUDA; defaulting to cu121"
        TORCH_INDEX=https://download.pytorch.org/whl/cu121
    fi

    echo "[SETUP] Installing PyTorch ${TORCH_VER} (CUDA index: ${TORCH_INDEX})"
    _pip_install --upgrade pip setuptools wheel
    _pip_install "torch==${TORCH_VER}" "torchvision" --index-url "$TORCH_INDEX"
fi

# ─── Step 2: Install base requirements ──────────────────────────────────────

if [ "$BASIC" = true ]; then
    # Verify PyTorch is available
    if ! "$PYTHON_BIN" -c "import torch" 2>/dev/null; then
        echo "[ERROR] PyTorch not installed. Run with --torch first, or install PyTorch manually."
        return 1
    fi

    CUDA_VERSION=$(_detect_cuda)
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d'.' -f1)
    TORCH_VERSION=$(_detect_torch_version)
    PYVER=$(_detect_python_version)
    echo "[SETUP] Detected: Python ${PYVER}, PyTorch ${TORCH_VERSION}, CUDA ${CUDA_VERSION}"

    echo "[SETUP] Installing base dependencies..."
    _pip_install -r requirements.txt

    # ── spconv (try pre-built CUDA-specific wheel, fall back to generic source build) ──
    echo "[SETUP] Installing spconv..."
    case "$CUDA_MAJOR" in
        11) _pip_install spconv-cu118 2>/dev/null || _pip_install spconv ;;
        12) _pip_install spconv-cu121 2>/dev/null || _pip_install spconv ;;
        *)  _pip_install spconv ;;
    esac

    # ── pytorch3d ──
    echo "[SETUP] Installing pytorch3d..."
    # Try pre-built wheel first (available for py310 + cu121 + select PyTorch versions)
    PT3D_LINKS="https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py${PYVER}_cu${CUDA_MAJOR}1_pyt${TORCH_VERSION//./}/download.html"
    if ! _pip_install pytorch3d --find-links "$PT3D_LINKS" --no-build-isolation 2>/dev/null; then
        echo "[SETUP] Pre-built pytorch3d wheel not available; building from source..."
        mkdir -p /tmp/anigen_extensions
        if [ ! -f /tmp/anigen_extensions/pytorch3d/setup.py ]; then
            rm -rf /tmp/anigen_extensions/pytorch3d
            _download https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.7.8.tar.gz /tmp/anigen_extensions/pytorch3d.tar.gz
            tar -xzf /tmp/anigen_extensions/pytorch3d.tar.gz -C /tmp/anigen_extensions
            mv /tmp/anigen_extensions/pytorch3d-0.7.8 /tmp/anigen_extensions/pytorch3d
        fi
        _pip_install /tmp/anigen_extensions/pytorch3d --no-build-isolation
    fi

    # ── nvdiffrast ──
    echo "[SETUP] Installing nvdiffrast..."
    if ! "$PYTHON_BIN" -c "import nvdiffrast.torch" 2>/dev/null; then
        mkdir -p /tmp/anigen_extensions
        # Try pre-built wheel first
        NVDR_WHL="/tmp/anigen_extensions/nvdiffrast-0.3.3-cp${PYVER}-cp${PYVER}-linux_x86_64.whl"
        NVDR_WHL_URL="https://huggingface.co/spaces/JeffreyXiang/TRELLIS/resolve/main/wheels/nvdiffrast-0.3.3-cp${PYVER}-cp${PYVER}-linux_x86_64.whl?download=true"
        if _download "$NVDR_WHL_URL" "$NVDR_WHL" 2>/dev/null && _pip_install "$NVDR_WHL" 2>/dev/null; then
            echo "[SETUP] nvdiffrast installed from pre-built wheel"
        else
            echo "[SETUP] Pre-built nvdiffrast wheel not available; building from source..."
            if [ ! -f /tmp/anigen_extensions/nvdiffrast/setup.py ]; then
                rm -rf /tmp/anigen_extensions/nvdiffrast
                _download https://github.com/NVlabs/nvdiffrast/archive/refs/tags/v0.3.3.tar.gz /tmp/anigen_extensions/nvdiffrast.tar.gz
                tar -xzf /tmp/anigen_extensions/nvdiffrast.tar.gz -C /tmp/anigen_extensions
                mv /tmp/anigen_extensions/nvdiffrast-0.3.3 /tmp/anigen_extensions/nvdiffrast
            fi
            _pip_install /tmp/anigen_extensions/nvdiffrast --no-build-isolation
        fi
    else
        echo "[SETUP] nvdiffrast already installed"
    fi

    echo "[SETUP] Core installation complete."
fi

# ─── Step 3: Optional — flash-attn ──────────────────────────────────────────

if [ "$FLASH_ATTN" = true ]; then
    echo "[SETUP] Installing flash-attn (this may take a while if building from source)..."
    _pip_install flash-attn --no-build-isolation || echo "[WARN] flash-attn install failed — pipeline will fall back to SDPA"
fi

# ─── Step 4: Optional — xformers ────────────────────────────────────────────

if [ "$XFORMERS" = true ]; then
    echo "[SETUP] Installing xformers..."
    TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}
    _pip_install xformers --index-url "$TORCH_INDEX" || echo "[WARN] xformers install failed — pipeline will fall back to SDPA"
fi

# ─── Step 5: Training dependencies ──────────────────────────────────────────

if [ "$TRAIN" = true ]; then
    echo "[SETUP] Installing training dependencies..."
    _pip_install tensorboard pandas lpips

    # ── CUBVH (local extension, used by training datasets) ──
    if [ -d "${WORKDIR}/extensions/CUBVH" ]; then
        echo "[SETUP] Building CUBVH..."
        _pip_install "${WORKDIR}/extensions/CUBVH" --no-build-isolation
    fi
fi

# ─── Step 6: Demo (Gradio) dependencies ─────────────────────────────────────

if [ "$DEMO" = true ]; then
    echo "[SETUP] Installing demo dependencies..."
    _pip_install gradio==4.44.1 gradio_litmodel3d==0.0.1 fastapi==0.112.2 starlette==0.38.6 jinja2==3.1.5 pydantic==2.10.6 "huggingface_hub<0.25"
fi

echo "[SETUP] Done."
