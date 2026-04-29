#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "[venv] Creating .venv..."
    PYTHON=$(command -v python3 || command -v python)
    "$PYTHON" -m venv "$SCRIPT_DIR/.venv"
fi

echo "[deps] Installing project dependencies..."
"$SCRIPT_DIR/.venv/bin/pip" install -e "$SCRIPT_DIR"

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source .venv/bin/activate"
