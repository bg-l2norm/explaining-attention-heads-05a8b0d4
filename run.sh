#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "[run.sh] installing deps"
pip install -q -r requirements.txt
python -m spacy download en_core_web_sm >/dev/null 2>&1 || true

echo "[run.sh] running minimal GPT-2 repro"
python3 repro.py
