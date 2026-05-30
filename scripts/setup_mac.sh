#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY310="${PY310:-/opt/homebrew/opt/python@3.10/bin/python3.10}"

"$PY310" -m venv "$ROOT/.venv-mac"
"$ROOT/.venv-mac/bin/pip" install -U pip wheel setuptools
"$ROOT/.venv-mac/bin/pip" install -r "$ROOT/requirements-mac.txt"

# Vendored Metal packages (Task 2 must have populated extern/).
for p in mtlbvh mtlmesh mtlgemm mtldiffrast; do
  if [ -d "$ROOT/extern/$p" ]; then
    "$ROOT/.venv-mac/bin/pip" install -e "$ROOT/extern/$p" --no-build-isolation
  fi
done
echo "setup_mac.sh complete"
