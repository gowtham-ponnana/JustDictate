"""JSON config read/write at ~/.config/parakeet-stt/config.json"""

import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "parakeet-stt"
CONFIG_FILE = CONFIG_DIR / "config.json"

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


def get_hotkey_preset(name: str) -> dict:
    """Return hotkey preset by name, falling back to right_cmd."""
    return HOTKEY_PRESETS.get(name, HOTKEY_PRESETS["right_cmd"])
