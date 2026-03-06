#!/usr/bin/env bash
# Build JustDictate.app and install to /Applications.
# Handles all prerequisites: Homebrew, uv, Python deps, PyInstaller.
# Safe to re-run — always removes the old app, rebuilds from scratch, and copies fresh.
set -e

cd "$(dirname "$0")"

echo "=== JustDictate Installer ==="
echo ""

# 1. Remove existing app first (clean slate)
if [ -d "/Applications/JustDictate.app" ]; then
    echo "Removing existing /Applications/JustDictate.app..."
    rm -rf /Applications/JustDictate.app
fi

# 2. Install Homebrew if missing
if ! command -v brew &>/dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# 3. Install uv if missing
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    brew install uv
fi

# 4. Install Python dependencies
echo "Installing Python dependencies..."
uv sync

# 5. Install PyInstaller into the venv
echo "Installing PyInstaller..."
uv pip install pyinstaller

# 6. Clean stale build artifacts
rm -rf dist/ build/

# 7. Build with PyInstaller (pyproject.toml must be moved — see CLAUDE.md gotcha)
mv pyproject.toml pyproject.toml.bak
echo "Building JustDictate.app..."
uv run pyinstaller JustDictate.spec --clean
mv pyproject.toml.bak pyproject.toml

# 8. Copy to /Applications
echo "Copying to /Applications..."
cp -R dist/JustDictate.app /Applications/

echo ""
echo "=== Done! ==="
echo "JustDictate.app has been installed to /Applications."
echo ""
echo "IMPORTANT: Since this is a new binary, you must (re-)grant permissions"
echo "in System Settings > Privacy & Security:"
echo "  - Microphone"
echo "  - Accessibility"
echo "  - Input Monitoring"
echo ""
echo "If JustDictate already appears in those lists, toggle it OFF then ON,"
echo "or remove it and re-add it."
