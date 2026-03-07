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
    # Homebrew installs to /opt/homebrew (Apple Silicon) or /usr/local (Intel)
    if [ -f /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -f /usr/local/bin/brew ]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
fi

# 3. Install uv if missing
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    brew install uv
fi

# 4. Ensure a compatible Python is available (onnxruntime requires <3.14)
echo "Ensuring compatible Python version..."
uv python install 3.13

# 5. Install Python dependencies
echo "Installing Python dependencies..."
uv sync --python 3.13

# 6. Install PyInstaller into the venv
echo "Installing PyInstaller..."
uv pip install pyinstaller

# 7. Clean stale build artifacts
rm -rf dist/ build/

# 8. Build with PyInstaller
#    pyproject.toml is moved aside so PyInstaller doesn't pick it up as build config.
#    Trap ensures it's restored even if the build fails.
mv pyproject.toml pyproject.toml.bak
trap 'mv pyproject.toml.bak pyproject.toml 2>/dev/null' EXIT
echo "Building JustDictate.app..."
.venv/bin/pyinstaller JustDictate.spec --clean
mv pyproject.toml.bak pyproject.toml
trap - EXIT

# 9. Copy to /Applications
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
