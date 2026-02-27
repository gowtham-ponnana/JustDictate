#!/usr/bin/env bash
# Launch ParakeetSTT using uv (handles Python version + deps automatically)
set -e

cd "$(dirname "$0")"

echo "Starting ParakeetSTT..."
echo "First run will install dependencies and download the model (~2.5 GB)."
echo ""

uv run python parakeet_stt.py
