#!/usr/bin/env bash
# Build, verify, and install JustDictate while retaining the previous app.
set -euo pipefail

cd "$(dirname "$0")"

mode="install"
case "${1:-}" in
    "") ;;
    --build-only) mode="build-only" ;;
    --install-only) mode="install-only" ;;
    -h|--help)
        echo "Usage: ./install.sh [--build-only | --install-only]"
        echo "Default: build, verify, and install to /Applications."
        echo "--install-only: install the existing verified dist/JustDictate.app."
        exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then
    echo "Pass at most one option; use --help for usage." >&2
    exit 2
fi
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "JustDictate requires macOS on Apple Silicon." >&2
    exit 1
fi

target_app="/Applications/JustDictate.app"
backup_app="/Applications/.JustDictate.previous.app"
stage_dir=""
target_moved=0
backup_moved=0
new_installed=0
committed=0

cleanup() {
    local status="$?"
    local recovery_failed=0
    trap - EXIT HUP INT TERM
    set +e
    if [ "$committed" -eq 0 ] && [ -n "$stage_dir" ]; then
        if [ "$new_installed" -eq 1 ] && [ -e "$target_app" ]; then
            mv "$target_app" "$stage_dir/failed-new.app" || recovery_failed=1
        fi
        if [ "$target_moved" -eq 1 ] && [ -e "$backup_app" ]; then
            if [ ! -e "$target_app" ]; then
                mv "$backup_app" "$target_app" || recovery_failed=1
            else
                recovery_failed=1
            fi
        fi
        if [ "$backup_moved" -eq 1 ] && [ -e "$stage_dir/older-backup.app" ]; then
            if [ ! -e "$backup_app" ]; then
                mv "$stage_dir/older-backup.app" "$backup_app" || recovery_failed=1
            else
                recovery_failed=1
            fi
        fi
    fi
    if [ "$recovery_failed" -eq 1 ]; then
        echo "Automatic rollback could not finish. Recoverable bundles remain at:" >&2
        echo "  $backup_app" >&2
        echo "  $stage_dir" >&2
        status=1
    elif [[ "$stage_dir" == /Applications/.JustDictate-install.* ]]; then
        # This directory was created by this invocation with mktemp. It contains
        # only our staged copy and, after success, the superseded older backup.
        rm -rf "$stage_dir" || echo "Could not remove installer staging directory: $stage_dir" >&2
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

check_identity() {
    local app="$1"
    if [ -L "$app" ] || [ ! -d "$app" ] ||
       [ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$app/Contents/Info.plist" 2>/dev/null)" != "com.gowtham.just-dictate" ]; then
        echo "Refusing to replace or remove an unexpected bundle: $app" >&2
        return 1
    fi
}

verify_bundle() {
    local app="$1"
    check_identity "$app"
    /usr/bin/plutil -lint "$app/Contents/Info.plist"
    if [ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$app/Contents/Info.plist")" != "JustDictate" ] ||
       [ ! -x "$app/Contents/MacOS/JustDictate" ]; then
        echo "Built app is missing its expected executable." >&2
        return 1
    fi
    /usr/bin/lipo -verify_arch arm64 "$app/Contents/MacOS/JustDictate"
    /usr/bin/codesign --verify --deep --strict --verbose=2 "$app"
}

echo "Building and installing JustDictate ($mode)."

if [ "$mode" != "install-only" ]; then
    if ! command -v uv >/dev/null 2>&1; then
        if ! command -v brew >/dev/null 2>&1; then
            echo "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
        brew install uv
    fi

    echo "Installing build dependencies with Python 3.13..."
    uv python install 3.13
    uv sync --python 3.13
    uv pip install --python .venv/bin/python pyinstaller

    # A spec is already the complete PyInstaller configuration. Keep project
    # metadata in place, including when a build fails or is interrupted.
    echo "Building JustDictate.app..."
    .venv/bin/pyinstaller --clean --noconfirm JustDictate.spec
fi

verify_bundle "dist/JustDictate.app"
if [ "$mode" = "build-only" ]; then
    echo "Verified build: $PWD/dist/JustDictate.app"
    echo "Install it later with: ./install.sh --install-only"
    exit 0
fi
if [ ! -x .venv/bin/python ]; then
    echo "A project Python environment is needed to request a graceful app shutdown." >&2
    echo "Run ./install.sh to prepare it and build the app." >&2
    exit 1
fi

# Stage on the same volume as the destination so replacement uses renames.
# Build/copy/signature failures leave the installed app running and untouched.
stage_dir="$(mktemp -d /Applications/.JustDictate-install.XXXXXX)"
/usr/bin/ditto "dist/JustDictate.app" "$stage_dir/JustDictate.app"
verify_bundle "$stage_dir/JustDictate.app"
if [ -e "$target_app" ] || [ -L "$target_app" ]; then
    check_identity "$target_app"
fi
if [ -e "$backup_app" ] || [ -L "$backup_app" ]; then
    check_identity "$backup_app"
fi

echo "The new app is staged. Requesting graceful shutdown of the installed app..."
.venv/bin/python - "$target_app" <<'PY'
import os
import sys
import time

from AppKit import NSRunningApplication
from Foundation import NSDate, NSRunLoop

target = os.path.realpath(sys.argv[1])
apps = [app for app in NSRunningApplication.runningApplicationsWithBundleIdentifier_(
    "com.gowtham.just-dictate")
    if app.bundleURL() is not None and os.path.realpath(str(app.bundleURL().path())) == target]
for app in apps:
    if not app.terminate():
        raise SystemExit("JustDictate declined to quit. Finish your dictation, quit it, then run ./install.sh --install-only.")
deadline = time.monotonic() + 15
while any(not app.isTerminated() for app in apps) and time.monotonic() < deadline:
    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))
if any(not app.isTerminated() for app in apps):
    raise SystemExit("JustDictate is still running. No app was replaced. Quit it, then run ./install.sh --install-only.")
PY

if [ -e "$target_app" ]; then
    if [ -e "$backup_app" ]; then
        backup_moved=1
        mv "$backup_app" "$stage_dir/older-backup.app"
    fi
    target_moved=1
    mv "$target_app" "$backup_app"
fi
new_installed=1
mv "$stage_dir/JustDictate.app" "$target_app"
verify_bundle "$target_app"
committed=1

echo "Installed and verified: $target_app"
if [ -e "$backup_app" ]; then
    echo "Previous version retained at: $backup_app"
fi
echo "Open JustDictate from /Applications when ready."
echo "If macOS requests access again, grant Microphone, Accessibility, and Input Monitoring"
echo "in System Settings > Privacy & Security."
