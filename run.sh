#!/usr/bin/env bash
# Launch JustDictate using uv (handles Python version + deps automatically)
set -e

cd "$(dirname "$0")"

echo "Starting JustDictate..."
echo "First run will install dependencies and download the model (~2.5 GB)."
echo ""

uv run python just_dictate.py
