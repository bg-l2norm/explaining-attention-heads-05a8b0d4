#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "[run.sh] installing deps"
pip install -q -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -c "import spacy; spacy.load('en_core_web_sm'); print('[run.sh] spaCy model OK')"

echo "[run.sh] running minimal GPT-2 repro"
python3 repro.py
