#!/usr/bin/env bash
# One-command full run for ConfSleepNet on an L4 GPU (GCP DL image or HF Jobs l4x1).
#
# Assumes: the SleepEDF-20 EDF files are already at data/sleepedf/.../sleep-cassette/
# (see README "Data"). train.py auto-selects cuda when present — no code change.
#
# Cost sanity: ~10 GPU-h for edf20 (both fusions share weights), ~+2.3x for edf78.
# Est. ~$35 on L4, under the $200 / 2-week challenge cap.
set -euo pipefail
cd "$(dirname "$0")"

# Path to the extracted sleep-cassette dir (39 PSG = edf20 subset locally; the
# edf78 step below needs the FULL cassette extraction, subjects 00-82).
SRC="${SRC:-../../data/sleepedf/sleep-edf-database-expanded-1.0.0/sleep-cassette}"

# --- torch (skip if the image already ships a CUDA build) --------------------
# Comment this block out on HF Jobs / Colab / GCP DL images that preinstall torch.
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo ">> installing CUDA torch (no GPU torch found)"
  pip install --index-url https://download.pytorch.org/whl/cu121 torch
fi
pip install -r requirements.txt

python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# --- SleepEDF-20: preprocess + full 20-fold LOSO -----------------------------
python src/preprocess.py --src "$SRC" --max-subject 20 --out data/edf20
python src/train.py --data data/edf20 --out runs/edf20_full --epochs 100 --folds 0

# --- SleepEDF-78: LOSO superset of the paper's 10-fold (note the deviation) ---
# Requires the full cassette extraction (subjects 00-82) at $SRC, not just the
# edf20 subset. Comment out if only the 39-recording subset is present.
python src/preprocess.py --src "$SRC" --max-subject 83 --out data/edf78
python src/train.py --data data/edf78 --out runs/edf78_full --epochs 100 --folds 0

echo ">> done. verdict numbers in runs/edf20_full/results.json and runs/edf78_full/results.json"
