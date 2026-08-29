# JustDictate — Developer Guide

This file is for AI assistants (Claude Code, Copilot, etc.) working on this codebase.

## What is this?

JustDictate is a macOS menu bar speech-to-text app. Hold a hotkey, speak, release — your words are transcribed and auto-typed into the focused app. Think SuperWhisper but open source, using NVIDIA's Parakeet TDT 0.6B v3 via MLX on Apple Silicon.

## Architecture

```
just_dictate.py      ← Entry point. rumps menu bar app. Ties everything together.
├── model_manager.py ← Loads Parakeet via parakeet-mlx (Metal GPU), handles transcription.
├── dictation_engine.py ← pynput hotkey listener + sounddevice recording + clipboard paste + Escape-to-cancel + Escape-to-undo + input device selection.
├── floating_window.py  ← PyObjC NSWindow overlay with waveform animation.
└── config_manager.py   ← JSON config + stats at ~/.config/just-dictate/
```

## Key Technical Decisions

### Why parakeet-mlx (not ONNX / PyTorch / whisper.cpp)?
- Same NVIDIA Parakeet TDT 0.6B v3 model, running on the Apple Silicon GPU via MLX
- Typically 2–3× faster than the previous ONNX INT8 CPU path on M-series, especially for long dictation
- `parakeet-mlx` is Apple Silicon only — that is intentional for this app right now
- Weights are BF16 in memory; Hugging Face cache is `~/.cache/just-dictate/parakeet-mlx/` (~2.5 GB download)
- Audio longer than 90s is chunked (60s windows, 5s overlap) in-process. Do not call `model.transcribe(path)` — that path needs ffmpeg. Feed numpy PCM through `get_logmel` + `generate`.

### Why rumps + PyObjC (not tkinter/Qt)?
- Homebrew Python 3.13+ does not ship tkinter
- rumps is purpose-built for macOS menu bar apps
- PyObjC gives native NSWindow for the floating overlay (rumps is too limited for custom windows)

### Why NSSound for completion feedback (not subprocess)?
- `NSSound.soundNamed_("Tink")` plays built-in macOS system sounds with zero latency
- No subprocess overhead, no file path needed — the sound name is resolved by AppKit
- Already have PyObjC as a dependency, so no new imports

### Why CGEvent for paste (not osascript)?
- `osascript` requires its own Accessibility permission and has timeout issues
- CGEvent works with the same Accessibility permission that pynput already needs

### Why uv?
- Pins Python to >=3.11,<3.14 so the MLX / numpy / rumps stack stays compatible
- uv auto-manages Python version and dependencies

## Known Gotchas

These are bugs that took hours to solve. **Do not change** these patterns without understanding why:

1. **Apple Silicon only.** `parakeet-mlx` will not run on Intel Macs. `ModelManager.load()` checks `platform.machine() == "arm64"` and fails with a clear error otherwise.

2. **Do not use `model.transcribe(path)`.** That helper shells out to ffmpeg. Dictation already has a float32 numpy buffer — convert with `mx.array(audio, dtype=mx.float32)` and run `get_logmel` + `model.generate`. Do not cast the waveform to bfloat16; `get_logmel` views the STFT as float32 and will crash.

3. **PyInstaller must ship `mlx/lib/libmlx.dylib` and `mlx/lib/mlx.metallib`.** The metallib is ~180 MB and is not a Python import, so `collect_all('mlx')` is required. Missing it fails at runtime with Metal shader errors.

4. **`rumps.App.quit_button` is a property**, not a regular attribute. Assigning `self.quit_button = MenuItem(...)` calls the property setter. Don't add a custom quit button to `self.menu` — let rumps handle it via the `quit_button="Quit"` constructor arg.

5. **`rumps.Timer(0, callback)` crashes** with `ValueError: depythonifying 'double', got 'function'`. Use `PyObjCTools.AppHelper.callAfter()` for thread-safe UI updates instead.

6. **PyObjCTools is a namespace package** (no `__init__.py`). py2app's `imp.find_module` can't find it. If packaging with py2app, put it in `includes`, not `packages`.

7. **Escape cancel uses a `_cancelled` flag**, not just `_cancel_recording()`. The flag is needed because `_on_release` fires after `_cancel_recording` when the user releases the hotkey — without the flag, it would still try to transcribe. `_cancel_recording()` sets `_cancelled = True` and `_on_release()` checks/resets it.

8. **Escape has three behaviors** depending on state: (1) during recording → cancel, (2) within 5s after paste → undo via Cmd+Z, (3) otherwise → pass through. All routed through `_handle_escape()`. The undo sends CGEvent Cmd+Z (keycode 6), which works in GUI apps but not terminals (where Cmd+Z sends SIGTSTP).

9. **`rumps.MenuItem.clear()` crashes during `__init__`** because the internal `_menu` (NSMenu) is `None` before the menu is attached to the app. Guard with `if self.mic_menu._menu is not None` before calling `clear()`.

10. **`sd.query_devices()` returns a cached device list**. When Bluetooth devices (AirPods) connect after app launch, they won't appear. Must call `sd._terminate()` + `sd._initialize()` to force PortAudio to re-query Core Audio before listing devices.

11. **Device warm-up eliminates recording delay for external mics**. External devices (iPhone Continuity, AirPods, USB mics) take 1-2s to initialize on first `sd.InputStream()` open. A brief open→close cycle in `_warm_up_device()` pre-initializes the OS driver so subsequent opens are near-instant. The mic is NOT left open — only opened during actual recording.

## Running Locally

```bash
# Requires macOS and uv (brew install uv)
uv run python just_dictate.py
```

First run downloads the MLX model (~2.5 GB) to `~/.cache/just-dictate/parakeet-mlx/`. Requires Apple Silicon.

## Building the .app Bundle

Use the install script to build and copy to `/Applications`:

```bash
./install.sh
# Or manually: see install.sh for the full sequence
```

The spec file (`JustDictate.spec`) is preconfigured to:
- `collect_all('mlx')` so `libmlx.dylib` and `mlx.metallib` ship in the bundle
- Include `libportaudio.dylib` as a binary
- Set `LSUIElement: True` in `info_plist` (hides dock icon)
- Set `NSMicrophoneUsageDescription` in `info_plist`

## macOS Permissions

The app needs these in System Settings > Privacy & Security:
- **Microphone** — audio recording
- **Accessibility** — hotkey detection + CGEvent paste
- **Input Monitoring** — pynput global key listener

## Config

Stored at `~/.config/just-dictate/config.json`:

```json
{
  "hotkey": "right_cmd",
  "auto_type_method": "clipboard_paste",
  "add_trailing_space": true,
  "input_device": null
}
```

- `input_device`: `null` = system default, or a device name string (e.g., `"MacBook Pro Microphone"`)

Hotkey options: `right_cmd`, `right_alt`, `left_ctrl_left_alt`

Stats at `~/.config/just-dictate/stats.json` (resets on the 1st of each month):

```json
{"month": "2026-03", "total_recording_seconds": 0.0, "total_recordings": 0}
```

- `month`: auto-set to current `"YYYY-MM"`. When the month changes, old stats are archived and counters reset.

Archived stats at `~/.config/just-dictate/stats_history.json`:

```json
{"legacy": {"total_recording_seconds": 433718.69, "total_recordings": 340}, "2026-03": {"total_recording_seconds": 12.5, "total_recordings": 5}}
```

- `"legacy"` key holds pre-monthly-reset data (migrated automatically on first run).

## Model Cache

Downloaded to `~/.cache/just-dictate/`:
- `parakeet-mlx/` — Hugging Face cache for `mlx-community/parakeet-tdt-0.6b-v3` (~2.5 GB)

Old ONNX caches (`parakeet-v3/`, `parakeet-v3-int8/`, `silero-vad/`) are unused and can be deleted.

## Dependencies

All managed via `pyproject.toml`:
- `parakeet-mlx` — NVIDIA Parakeet TDT 0.6B v3 on MLX (pulls `mlx`, `librosa`)
- `sounddevice` — microphone input (via PortAudio)
- `pynput` — global hotkey detection
- `numpy` — audio array processing
- `rumps` — macOS menu bar framework
- `pyobjc-framework-Cocoa` — NSWindow, NSView, etc.
- `pyobjc-framework-Quartz` — CGEvent for keystroke injection
