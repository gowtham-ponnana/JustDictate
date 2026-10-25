"""JSON config read/write at ~/.config/just-dictate/config.json"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "just-dictate"
CONFIG_FILE = CONFIG_DIR / "config.json"
STATS_FILE = CONFIG_DIR / "stats.json"
STATS_HISTORY_FILE = CONFIG_DIR / "stats_history.json"

STATS_DEFAULTS = {"month": "", "total_recording_seconds": 0.0, "total_recordings": 0}

DEFAULTS = {
    "hotkey": "right_cmd",
    "auto_type_method": "clipboard_paste",
    "add_trailing_space": True,
    "input_device": None,
    "personalization_enabled": True,
    "retain_personalization_audio": True,
    "learn_from_edits": True,
    "personalization_retention_days": 90,
}

HOTKEY_PRESETS = {
    "right_cmd": {"keys": ["Key.cmd_r"], "vk_codes": [0x36], "label": "Right Command"},
    "right_alt": {"keys": ["Key.alt_r"], "vk_codes": [0x3D], "label": "Right Alt"},
    "left_ctrl_left_alt": {
        "keys": ["Key.ctrl_l", "Key.alt_l"],
        "vk_codes": [0x3B, 0x3A],
        "label": "Left Ctrl + Left Alt",
    },
}


def load() -> dict:
    """Load config, creating defaults if missing."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        # Merge any new defaults
        for k, v in DEFAULTS.items():
            cfg.setdefault(k, v)
        return cfg
    save(DEFAULTS)
    return dict(DEFAULTS)


def save(cfg: dict) -> None:
    """Persist config to disk."""
    _write_json(CONFIG_FILE, cfg)


def _write_json(path: Path, data: dict) -> None:
    """Replace complete files atomically; an interrupted save keeps the old copy."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _current_month() -> str:
    """Return current month as 'YYYY-MM'."""
    return datetime.now().strftime("%Y-%m")


def _archive_stats(stats: dict) -> None:
    """Append old stats to stats_history.json under their month key (or 'legacy')."""
    month_key = stats.get("month") or "legacy"
    history = {}
    if STATS_HISTORY_FILE.exists():
        try:
            with open(STATS_HISTORY_FILE) as f:
                history = json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass
    history[month_key] = {
        "total_recording_seconds": stats.get("total_recording_seconds", 0.0),
        "total_recordings": stats.get("total_recordings", 0),
    }
    _write_json(STATS_HISTORY_FILE, history)


def load_stats() -> dict:
    """Load recording stats, creating defaults if missing or corrupted.

    Auto-archives and resets stats when the month changes.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    month = _current_month()

    if STATS_FILE.exists():
        try:
            with open(STATS_FILE) as f:
                stats = json.load(f)
            for k, v in STATS_DEFAULTS.items():
                stats.setdefault(k, v)
            # Check if month matches — if not, archive and reset
            if stats.get("month") != month:
                if stats.get("total_recordings", 0) > 0:
                    _archive_stats(stats)
                stats = dict(STATS_DEFAULTS)
                stats["month"] = month
                save_stats(stats)
            return stats
        except (json.JSONDecodeError, ValueError):
            pass

    fresh = dict(STATS_DEFAULTS)
    fresh["month"] = month
    save_stats(fresh)
    return fresh


def save_stats(stats: dict) -> None:
    """Persist recording stats to disk."""
    _write_json(STATS_FILE, stats)


def get_hotkey_preset(name: str) -> dict:
    """Return hotkey preset by name, falling back to right_cmd."""
    return HOTKEY_PRESETS.get(name, HOTKEY_PRESETS["right_cmd"])
