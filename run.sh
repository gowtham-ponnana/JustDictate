#!/usr/bin/env bash
# Launch JustDictate using uv (handles Python version + deps automatically)
set -e

cd "$(dirname "$0")"

# Install Homebrew if missing
if ! command -v brew &>/dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# Install uv if missing
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    brew install uv
fi

echo "Starting JustDictate..."
echo "First run will install dependencies and download the model (~2.5 GB)."
echo ""

uv run python just_dictate.py
