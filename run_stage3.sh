#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r requirements.txt
python run_stage3.py --data-dir "${1:-$HOME/Downloads/archive}" --output-dir results
