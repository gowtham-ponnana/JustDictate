"""JSON config read/write at ~/.config/just-dictate/config.json"""

import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "just-dictate"
CONFIG_FILE = CONFIG_DIR / "config.json"
STATS_FILE = CONFIG_DIR / "stats.json"

STATS_DEFAULTS = {"total_recording_seconds": 0.0, "total_recordings": 0}

DEFAULTS = {
    "hotkey": "right_cmd",
    "auto_type_method": "clipboard_paste",
    "add_trailing_space": True,
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
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def load_stats() -> dict:
    """Load recording stats, creating defaults if missing or corrupted."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE) as f:
                stats = json.load(f)
            for k, v in STATS_DEFAULTS.items():
                stats.setdefault(k, v)
            return stats
        except (json.JSONDecodeError, ValueError):
            pass
    save_stats(STATS_DEFAULTS)
    return dict(STATS_DEFAULTS)


def save_stats(stats: dict) -> None:
    """Persist recording stats to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)


def get_hotkey_preset(name: str) -> dict:
    """Return hotkey preset by name, falling back to right_cmd."""
    return HOTKEY_PRESETS.get(name, HOTKEY_PRESETS["right_cmd"])
