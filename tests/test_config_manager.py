import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import config_manager as config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "config.json"
        paths = patch.multiple(config, CONFIG_DIR=self.path.parent, CONFIG_FILE=self.path)
        paths.start()
        self.addCleanup(paths.stop)

    def test_existing_preferences_survive_new_defaults(self):
        self.path.write_text(json.dumps({"hotkey": "right_alt", "add_trailing_space": False}))
        actual = config.load()
        self.assertEqual(actual["hotkey"], "right_alt")
        self.assertFalse(actual["add_trailing_space"])
        self.assertTrue(actual["personalization_enabled"])
        self.assertTrue(actual["retain_personalization_audio"])

    def test_fresh_config_enables_local_learning_and_audio_without_prompt(self):
        actual = config.load()
        self.assertTrue(actual["personalization_enabled"])
        self.assertTrue(actual["retain_personalization_audio"])
        self.assertEqual(json.loads(self.path.read_text()), actual)

    def test_saved_off_preferences_are_never_overridden_by_defaults(self):
        config.save({"personalization_enabled": False, "retain_personalization_audio": False})
        actual = config.load()
        self.assertFalse(actual["personalization_enabled"])
        self.assertFalse(actual["retain_personalization_audio"])

    def test_failed_save_preserves_previous_file(self):
        config.save({"hotkey": "right_cmd"})
        with patch.object(config.json, "dump", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                config.save({"hotkey": "right_alt"})
        self.assertEqual(json.loads(self.path.read_text()), {"hotkey": "right_cmd"})
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_completed_save_is_private_and_readable(self):
        config.save(config.DEFAULTS)
        self.assertEqual(config.load(), config.DEFAULTS)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
