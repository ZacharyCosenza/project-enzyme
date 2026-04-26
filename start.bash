#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Virtual environment ────────────────────────────────────────────────────────

if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "[venv] Creating .venv..."
    python3 -m venv "$SCRIPT_DIR/.venv"
fi

PIP="$SCRIPT_DIR/.venv/bin/pip"

# ── Python dependencies ────────────────────────────────────────────────────────

echo "[deps] Installing project dependencies..."
"$PIP" install -q -e "$SCRIPT_DIR"

echo "[deps] Installing torch (XPU build)..."
"$PIP" install -q \
    torch==2.10.0+xpu \
    torchvision==0.25.0+xpu \
    torchaudio==2.10.0+xpu \
    --index-url https://download.pytorch.org/whl/xpu

# ── Data ───────────────────────────────────────────────────────────────────────

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

    # train.csv is served as a zip — extract it in-place
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

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source .venv/bin/activate"
