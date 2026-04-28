#!/bin/bash

# Detect if being sourced or executed directly.
# When sourced, the venv is activated at the end.
(return 0 2>/dev/null) && SOURCED=1 || SOURCED=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run all setup in a subshell so set -e failures don't kill the parent shell.
(
    set -euo pipefail

    # ── Virtual environment ────────────────────────────────────────────────────

    if [ ! -d "$SCRIPT_DIR/.venv" ]; then
        echo "[venv] Creating .venv..."
        PYTHON=$(command -v python3 || command -v python)
        "$PYTHON" -m venv "$SCRIPT_DIR/.venv"
    fi

    PIP="$SCRIPT_DIR/.venv/bin/pip"

    # ── Python dependencies ────────────────────────────────────────────────────

    echo "[deps] Installing project dependencies..."
    "$PIP" install -q -e "$SCRIPT_DIR"

    # Detect available accelerator and install the matching torch build.
    # XPU check: look for Intel GPU via sycl-ls (oneAPI) or /dev/dri with i915/xe.
    if command -v sycl-ls &>/dev/null && sycl-ls 2>/dev/null | grep -qi "intel.*gpu"; then
        ACCEL=xpu
    elif [ -d /dev/dri ] && (lsmod 2>/dev/null | grep -qE "^(i915|xe) "); then
        ACCEL=xpu
    elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        ACCEL=cuda
    else
        ACCEL=cpu
    fi

    echo "[deps] Detected accelerator: $ACCEL"

    if [ "$ACCEL" = "xpu" ]; then
        echo "[deps] Installing torch (XPU build)..."
        "$PIP" install -q \
            torch==2.10.0+xpu \
            torchvision==0.25.0+xpu \
            torchaudio==2.10.0+xpu \
            --index-url https://download.pytorch.org/whl/xpu
    elif [ "$ACCEL" = "cuda" ]; then
        echo "[deps] Installing torch (CUDA build)..."
        "$PIP" install -q torch torchvision torchaudio
    else
        echo "[deps] Installing torch (CPU build)..."
        "$PIP" install -q torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cpu
    fi

    # ── Data ──────────────────────────────────────────────────────────────────

    DEST="$SCRIPT_DIR/data/01_raw"
    MARKER="$DEST/train_raw.csv"

    if [ -f "$MARKER" ]; then
        echo "[data] Raw data already present — skipping download."
    else
        echo "[data] Downloading raw data from Kaggle..."

        if [ ! -f "$SCRIPT_DIR/keys.md" ]; then
            echo "Error: keys.md not found. Create it with KAGGLE_API_TOKEN=<token> before running." >&2
            exit 1
        fi

        source "$SCRIPT_DIR/keys.md"
        TOKEN="$KAGGLE_API_TOKEN"
        COMPETITION="novozymes-enzyme-stability-prediction"

        mkdir -p "$DEST"

        FILES=(
            sample_submission.csv
            test.csv
            test_labels.csv
            train.csv
            train_updates_20220929.csv
            wildtype_structure_prediction_af2.pdb
        )

        for f in "${FILES[@]}"; do
            echo "  Downloading $f..."
            curl -sL -H "Authorization: Bearer $TOKEN" \
                "https://www.kaggle.com/api/v1/competitions/data/download/$COMPETITION/$f" \
                -o "$DEST/$f"
        done

        "$SCRIPT_DIR/.venv/bin/python3" - <<EOF
import zipfile, io

path = "$DEST/train.csv"
with open(path, "rb") as f:
    data = f.read()

if data[:2] == b"PK":
    z = zipfile.ZipFile(io.BytesIO(data))
    content = z.read("train.csv")
    with open("$DEST/train_raw.csv", "wb") as f:
        f.write(content)
    print("  Extracted train.csv -> train_raw.csv")
EOF

        echo "[data] Done. Files saved to $DEST"
    fi

) || { echo "Setup failed." >&2; return 1 2>/dev/null || exit 1; }

echo ""
if [ "$SOURCED" = "1" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo "Setup complete. Virtualenv activated."
else
    echo "Setup complete. Activate the environment with:"
    echo "  source .venv/bin/activate"
fi
