#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TEST_ROOT=$(mktemp -d)
trap 'rm -rf "$TEST_ROOT"' EXIT

PYTHON_STUB="$TEST_ROOT/python"
cat >"$PYTHON_STUB" <<'PYTHON'
#!/bin/sh
if [ "$1" = "-c" ]; then
    case "$2" in
        *"sys.version_info.major"*) printf '%s\n' "312" ;;
        *"sys.version_info.minor"*) printf '%s\n' "12" ;;
        *"torch.version.cuda"*) printf '%s\n' "12.8" ;;
        *"torch.__version__"*) printf '%s\n' "${STUB_TORCH_VERSION:-2.7.0}" ;;
    esac
elif [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
    shift 2
    printf 'PIP_INSTALL %s\n' "$*"
    printf 'ARCHS %s %s\n' "${TORCH_CUDA_ARCH_LIST:-}" "${FLASH_ATTN_CUDA_ARCHS:-}"
elif [ "$1" = "--version" ]; then
    printf '%s\n' "Python 3.12.0"
fi
PYTHON
chmod +x "$PYTHON_STUB"

run_setup() {
    local capability=$1
    shift
    # shellcheck disable=SC2016
    env \
        -u ANIGEN_TORCH_INDEX_URL \
        -u CUDA_HOME \
        -u FLASH_ATTN_CUDA_ARCHS \
        -u TORCH_CUDA_ARCH_LIST \
        -u TORCH_VERSION \
        -u VIRTUAL_ENV \
        ANIGEN_CUDA_CAPABILITY="$capability" \
        ANIGEN_PYTHON="$PYTHON_STUB" \
        PATH=/usr/bin:/bin \
        bash -c 'cd "$1"; shift; . ./setup.sh "$@"' _ "$REPO_ROOT" "$@"
}

blackwell_output=$(run_setup 12.0 --torch)
grep -Fq "Installing PyTorch 2.7.0 (CUDA index: https://download.pytorch.org/whl/cu128)" <<<"$blackwell_output"
grep -Fq "ARCHS 12.0 120" <<<"$blackwell_output"

legacy_output=$(run_setup 8.6 --torch)
grep -Fq "Installing PyTorch 2.4.0 (CUDA index: https://download.pytorch.org/whl/cu121)" <<<"$legacy_output"

# shellcheck disable=SC2016
if old_torch_output=$(env \
    -u ANIGEN_TORCH_INDEX_URL \
    -u CUDA_HOME \
    -u FLASH_ATTN_CUDA_ARCHS \
    -u TORCH_CUDA_ARCH_LIST \
    -u VIRTUAL_ENV \
    ANIGEN_CUDA_CAPABILITY=12.0 \
    ANIGEN_PYTHON="$PYTHON_STUB" \
    PATH=/usr/bin:/bin \
    TORCH_VERSION=2.6.0 \
    bash -c 'cd "$1"; . ./setup.sh --torch' _ "$REPO_ROOT" 2>&1); then
    echo "Expected PyTorch 2.6 to be rejected on Blackwell" >&2
    exit 1
fi
grep -Fq "requires PyTorch >=2.7 with CUDA 12.8 support" <<<"$old_torch_output"

echo "Blackwell setup selection tests passed"
