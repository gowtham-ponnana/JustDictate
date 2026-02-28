#!/usr/bin/env bash
# Build JustDictate.app and install to /Applications.
# Handles all prerequisites: Homebrew, uv, Python deps, PyInstaller.
set -e

cd "$(dirname "$0")"

echo "=== JustDictate Installer ==="
echo ""

# 1. Install Homebrew if missing
if ! command -v brew &>/dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# 2. Install uv if missing
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    brew install uv
fi

# 3. Install Python dependencies
echo "Installing Python dependencies..."
uv sync

# 4. Install PyInstaller into the venv
echo "Installing PyInstaller..."
uv pip install pyinstaller

# 5. Clean stale build artifacts
rm -rf dist/ build/

# 6. Build with PyInstaller (pyproject.toml must be moved — see CLAUDE.md gotcha)
mv pyproject.toml pyproject.toml.bak
echo "Building JustDictate.app..."
uv run pyinstaller JustDictate.spec --clean
mv pyproject.toml.bak pyproject.toml

# 7. Install to /Applications
if [ -d "/Applications/JustDictate.app" ]; then
    echo "Removing old /Applications/JustDictate.app..."
    rm -rf /Applications/JustDictate.app
fi
echo "Copying to /Applications..."
cp -R dist/JustDictate.app /Applications/

echo ""
echo "=== Done! ==="
echo "JustDictate.app has been installed to /Applications."
echo ""
echo "Grant these permissions in System Settings > Privacy & Security:"
echo "  - Microphone"
echo "  - Accessibility"
echo "  - Input Monitoring"
