"""Native menu and private-pasteboard integration; never opens the microphone.

The app event loop is not run. UI dispatch is drained explicitly on the test's
main thread, and all model work, global input, sound, and target-app paste are
replaced. Config, SQLite history, and optional WAVs use temporary directories.
"""

from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np
import AppKit
from Foundation import NSData
from PyObjCTools import AppHelper
import rumps.rumps as rumps_impl

import config_manager
import dictation_engine
import just_dictate


class AppIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.callbacks = queue.SimpleQueue()
        self.patches = []
        self.app = None
        self.addCleanup(self._cleanup)

        for attribute, path in (
            ("CONFIG_DIR", self.directory),
            ("CONFIG_FILE", self.directory / "config.json"),
            ("STATS_FILE", self.directory / "stats.json"),
            ("STATS_HISTORY_FILE", self.directory / "stats_history.json"),
        ):
            self._patch(config_manager, attribute, path)
        # Most tests exercise an existing user's explicit opt-out. Fresh-install
        # defaults are tested separately without this config file.
        config_manager.save(dict(config_manager.DEFAULTS,
                                 personalization_enabled=False,
                                 retain_personalization_audio=False))
        self._patch(just_dictate, "CONFIG_DIR", self.directory)
        self._patch(rumps_impl, "application_support", lambda name: str(self.directory / "support"))
        self._patch(AppHelper, "callAfter", lambda fn, *args, **kwargs:
                    self.callbacks.put((fn, args, kwargs)))
        self._patch(just_dictate, "FloatingOverlay", Mock())
        self._patch(just_dictate.ModelManager, "load", Mock())
        self.transcribe = self._patch(just_dictate.ModelManager, "transcribe", Mock(return_value="Raw words."))
        self._patch(just_dictate.DictationEngine, "start", Mock())
        self._patch(just_dictate.DictationEngine, "_warm_up_device", Mock())
        self.paste = self._patch(just_dictate.DictationEngine, "_auto_type", Mock())
        self._patch(just_dictate.sd, "_terminate", Mock())
        self._patch(just_dictate.sd, "_initialize", Mock())
        self._patch(just_dictate.sd, "query_devices", Mock(return_value=[
            {"name": "Test Microphone", "max_input_channels": 1}
        ]))
        self._patch(just_dictate.JustDictateApp, "_play_completion_sound", Mock())
        self.notifications = self._patch(just_dictate.rumps, "notification", Mock())
        # Leave the real learning worker alive; replace only initial GPU loading
        # so native menu construction and all UI callbacks stay on this thread.
        loader = just_dictate.JustDictateApp._load_model
        with patch.object(just_dictate.JustDictateApp, "_load_model", Mock()):
            self.app = just_dictate.JustDictateApp()
        loader(self.app)
        self.drain()

    def _patch(self, target, name, replacement):
        active = patch.object(target, name, replacement)
        result = active.start()
        self.patches.append(active)
        return result

    def _cleanup(self):
        try:
            if self.app is not None:
                self.app._learning_timer.stop()
                if self.app.engine:
                    self.app.engine.stop()
                self.app.correction_monitor.close()
                self.app.learning.close()
                self.assertFalse(self.app.learning._worker.is_alive())
        finally:
            for active in reversed(self.patches):
                active.stop()
            self.temporary.cleanup()

    def drain(self):
        self.app.learning._queue.join()
        while not self.callbacks.empty():
            fn, args, kwargs = self.callbacks.get_nowait()
            fn(*args, **kwargs)

    def enable(self):
        self.app._toggle_learning(self.app.learning_enabled_item)
        self.drain()
        self.assertTrue(self.app.learning.enabled)

    def save_sample(self, raw="Original words.", delivered="Delivered words."):
        self.assertTrue(self.app.learning.record(raw, delivered, np.ones(6400, dtype=np.float32)))
        self.drain()
        return self.app.personalization.latest()

    def review_item(self, identity):
        self.app._refresh_learning_menu()
        return next(item for item in self.app.review_menu.values()
                    if getattr(item, "_record_id", None) == identity)

    def test_native_menu_startup_disabled_has_no_learning_store(self):
        self.assertIsNotNone(self.app.learning_menu._menu)
        self.assertIn("Personalization", self.app.menu)
        self.assertFalse(self.app.learning.enabled)
        self.assertFalse(self.app.learning.retain_audio)
        self.assertFalse(self.app.learning_enabled_item.state)
        self.assertFalse(self.app.learning_audio_item.state)
        self.assertFalse(self.app.personalization.directory.exists())
        self.assertTrue(config_manager.CONFIG_FILE.exists())
        self.assertTrue(config_manager.STATS_FILE.exists())
        self.assertFalse(self.app.learning.record("private", "private", None))
        self.app._check_learning(None)
        self.drain()
        self.assertFalse(self.app.personalization.directory.exists())
        self.assertIn("Ready", self.app.status_item.title)

    def test_permission_failure_is_visible_and_setup_can_be_retried(self):
        with patch.object(just_dictate.DictationEngine, "start", side_effect=RuntimeError(
                "Enable Accessibility in System Settings")):
            just_dictate.JustDictateApp._load_model(self.app)
        self.drain()
        self.assertIn("Error", self.app.status_item.title)
        self.assertIn("Accessibility", self.app.status_item.title)
        self.assertFalse(self.app.engine.is_listening)
        self.assertFalse(self.app._startup_lock.locked())
        # Simulate the next native setup attempt succeeding after permission is
        # granted; listener acceptance itself is exercised by engine tests.
        just_dictate.JustDictateApp._load_model(self.app)
        self.drain()
        self.assertIn("Ready", self.app.status_item.title)

    def test_fresh_defaults_capture_transcript_audio_and_schedule_without_prompt(self):
        self.app._learning_timer.stop()
        self.app.engine.stop()
        self.app.correction_monitor.close()
        self.app.learning.close()
        self.assertFalse(self.app.learning._worker.is_alive())
        self.app = None
        config_manager.CONFIG_FILE.unlink()
        self.assertFalse((self.directory / "personalization").exists())

        loader = just_dictate.JustDictateApp._load_model
        train_method = just_dictate.PersonalizationManager.train_if_due
        with patch.object(just_dictate.rumps, "Window") as window, \
                patch.object(just_dictate.rumps, "alert") as alert, \
                patch.object(just_dictate.PersonalizationManager, "train_if_due",
                             autospec=True, side_effect=train_method) as train:
            with patch.object(just_dictate.JustDictateApp, "_load_model", Mock()):
                self.app = just_dictate.JustDictateApp()
            loader(self.app)
            self.drain()
            self.assertTrue(self.app.learning.enabled)
            self.assertTrue(self.app.learning.retain_audio)
            self.assertTrue(self.app.learning_enabled_item.state)
            self.assertTrue(self.app.learning_audio_item.state)
            saved = config_manager.load()
            self.assertTrue(saved["personalization_enabled"])
            self.assertTrue(saved["retain_personalization_audio"])
            train.assert_called_once_with(self.app.personalization, force=False)

            # Exercise the real engine -> background service -> SQLite/WAV path;
            # only ASR and paste themselves are replaced, as in the other tests.
            audio = np.full(6400, 0.25, dtype=np.float32)
            self.app.engine._processing = True
            self.app.engine._transcribe_and_type([audio], self.app.engine._generation)
            self.drain()
            self.paste.assert_called_once_with("Raw words. ")
            row = self.app.personalization.latest()
            self.assertEqual((row["raw"], row["delivered"]), ("Raw words.", "Raw words."))
            self.assertIsNone(row["corrected"], "Automatic capture must not invent a verified label")
            self.assertEqual(self.app.personalization.status()["reviewed_count"], 0)
            self.assertIsNotNone(row["audio_name"])
            with wave.open(str(self.app.personalization.audio_directory / row["audio_name"]), "rb") as wav:
                self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()), (16000, 1, 2))
                self.assertEqual(wav.getnframes(), len(audio))
                restored = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2") / 32767
                np.testing.assert_allclose(restored, audio, atol=2 / 32767)

            # Exercise the periodic callback too; neither check needs a prompt.
            self.app._check_learning(None)
            self.drain()
            self.assertEqual(train.call_count, 2)
            window.assert_not_called()
            alert.assert_not_called()
            self.notifications.assert_not_called()

    def test_enable_and_audio_preferences_persist_with_actual_sqlite_history(self):
        self.enable()
        self.assertTrue(config_manager.load()["personalization_enabled"])
        first = self.save_sample()
        self.assertEqual((first["raw"], first["delivered"]), ("Original words.", "Delivered words."))
        self.assertIsNone(first["corrected"])
        self.assertIsNone(first["audio_name"])
        self.assertTrue(self.app.personalization.database.exists())
        self.app._toggle_learning_audio(self.app.learning_audio_item)
        self.drain()
        second = self.save_sample("Other original.", "Other delivered.")
        self.assertTrue(config_manager.load()["retain_personalization_audio"])
        self.assertIsNotNone(second["audio_name"])
        self.assertTrue((self.app.personalization.audio_directory / second["audio_name"]).exists())
        self.app._toggle_learning(self.app.learning_enabled_item)
        self.drain()
        self.assertFalse(config_manager.load()["personalization_enabled"])
        self.assertFalse(self.app.learning.record("not retained", "not retained", None))
        self.assertEqual(self.app.personalization.status()["record_count"], 2)

    def test_engine_personalization_preserves_original_and_delivered_pair(self):
        self.enable()
        manager = self.app.personalization
        for index in range(24):
            raw = f"Please ask Gotham tomorrow about task {index}."
            identity = manager.record(raw, raw)
            manager.review(identity, raw.replace("Gotham", "Gowtham"))
        self.assertEqual(manager.train_if_due(force=True)["status"], "trained")
        raw = "Please ask Gotham tomorrow about lunch."
        self.transcribe.return_value = raw
        self.app.engine._processing = True
        self.app.engine._transcribe_and_type(
            [np.ones(6400, dtype=np.float32)], self.app.engine._generation
        )
        self.drain()
        self.paste.assert_called_once_with("Please ask Gowtham tomorrow about lunch. ")
        row = manager.latest()
        self.assertEqual(row["raw"], raw)
        self.assertEqual(row["delivered"], "Please ask Gowtham tomorrow about lunch.")
        self.assertIsNone(row["corrected"], "Automatic output is never a verified training label")

    def test_review_cancellation_does_not_confirm_or_train(self):
        self.enable()
        record = self.save_sample()
        response = SimpleNamespace(clicked=0, text="An unconfirmed edit")
        with patch.object(just_dictate.rumps, "Window") as window, \
                patch.object(self.app.personalization, "train_if_due", wraps=self.app.personalization.train_if_due) as train:
            window.return_value.run.return_value = response
            self.app._review_dictation(self.review_item(record["id"]))
            window.assert_called_once()
            train.assert_not_called()
        self.assertIsNone(self.app.personalization.latest()["corrected"])
        self.assertFalse(self.app.engine._suspended)
        self.assertFalse(self.app._menu_active)

    def test_explicit_confirmation_saves_review_without_modifying_pasted_text(self):
        self.enable()
        record = self.save_sample()
        response = SimpleNamespace(clicked=1, text="Exactly what I said.")
        with patch.object(just_dictate.rumps, "Window") as window:
            window.return_value.run.return_value = response
            self.app._review_dictation(self.review_item(record["id"]))
        self.assertEqual(self.app.personalization.latest()["corrected"], response.text)
        self.assertEqual(self.app.personalization.status()["reviewed_count"], 1)
        self.paste.assert_not_called()
        self.assertFalse(self.app.engine._suspended)

    def test_review_audio_playback_preserves_edit_and_cancel_does_not_confirm(self):
        self.enable()
        self.app._toggle_learning_audio(self.app.learning_audio_item)
        self.drain()
        record = self.save_sample()
        sound = Mock()
        with patch.object(just_dictate.rumps, "Window") as window, \
                patch.object(AppKit, "NSSound") as sound_class:
            sound_class.alloc.return_value.initWithContentsOfFile_byReference_.return_value = sound
            window.return_value.run.side_effect = [
                SimpleNamespace(clicked=2, text="Unsaved draft"),
                SimpleNamespace(clicked=0, text="Unsaved draft"),
            ]
            self.app._review_dictation(self.review_item(record["id"]))
            window.return_value.add_button.assert_called_once_with("Play Recording")
            self.assertEqual(window.return_value.default_text, "Unsaved draft")
            sound.play.assert_called_once()
            self.assertEqual(sound.stop.call_count, 2)
        self.assertIsNone(self.app.personalization.latest()["corrected"])
        self.assertFalse(self.app.engine._suspended)

    def test_corrupt_learning_store_shows_unavailable_and_dictation_still_works(self):
        self.app._learning_timer.stop()
        self.app.engine.stop()
        self.app.correction_monitor.close()
        self.app.learning.close()
        self.app = None
        cfg = config_manager.load()
        cfg["personalization_enabled"] = True
        config_manager.save(cfg)
        store = self.directory / "personalization"
        store.mkdir()
        (store / "history.sqlite3").write_bytes(b"This is not a SQLite database")
        loader = just_dictate.JustDictateApp._load_model
        with patch.object(just_dictate.JustDictateApp, "_load_model", Mock()):
            self.app = just_dictate.JustDictateApp()
        loader(self.app)
        self.drain()
        self.assertIn("unavailable", self.app.learning_status_item.title)
        self.app.engine._processing = True
        self.app.engine._transcribe_and_type(
            [np.ones(6400, dtype=np.float32)], self.app.engine._generation
        )
        self.drain()
        self.paste.assert_called_once_with("Raw words. ")
        self.assertTrue(self.app.personalization.status()["unavailable"])

    def test_review_exception_stops_playback_and_resumes_hotkeys(self):
        self.enable()
        self.app._toggle_learning_audio(self.app.learning_audio_item)
        self.drain()
        record = self.save_sample()
        sound = Mock()
        with patch.object(just_dictate.rumps, "Window") as window, \
                patch.object(AppKit, "NSSound") as sound_class:
            sound_class.alloc.return_value.initWithContentsOfFile_byReference_.return_value = sound
            window.return_value.run.side_effect = [
                SimpleNamespace(clicked=2, text="Unsaved draft"),
                RuntimeError("Dialog closed unexpectedly"),
            ]
            with self.assertRaises(RuntimeError):
                self.app._review_dictation(self.review_item(record["id"]))
        self.assertEqual(sound.stop.call_count, 2)
        self.assertIsNone(self.app.personalization.latest()["corrected"])
        self.assertFalse(self.app.engine._suspended)
        self.assertFalse(self.app._menu_active)

    def test_busy_engine_prevents_modal_review_and_microphone_refresh(self):
        self.enable()
        record = self.save_sample()
        self.app.engine._processing = True
        try:
            with patch.object(just_dictate.rumps, "Window") as window:
                self.app._review_dictation(self.review_item(record["id"]))
                window.assert_not_called()
            just_dictate.sd._terminate.reset_mock()
            self.app._refresh_devices(None)
            just_dictate.sd._terminate.assert_not_called()
            self.assertFalse(self.app._menu_active)
            self.assertFalse(self.app.engine._suspended)
        finally:
            self.app.engine._processing = False

    def test_monthly_timer_starts_training_only_when_engine_and_menu_are_idle(self):
        self.enable()
        with patch.object(self.app.personalization, "train_if_due", wraps=self.app.personalization.train_if_due) as train:
            self.app.engine._processing = True
            self.app._check_learning(None)
            self.drain()
            train.assert_not_called()
            self.app.engine._processing = False
            self.app._menu_active = True
            self.app._check_learning(None)
            self.drain()
            train.assert_not_called()
            self.app._menu_active = False
            self.app._check_learning(None)
            self.drain()
            train.assert_called_once_with(force=False)


class PrivatePasteboardIntegrationTests(unittest.TestCase):
    def test_native_rich_text_unicode_and_binary_roundtrip_on_private_pasteboard(self):
        # An independently named pasteboard never touches the user's clipboard.
        board = AppKit.NSPasteboard.pasteboardWithUniqueName()
        self.addCleanup(board.releaseGlobally)
        item = AppKit.NSPasteboardItem.alloc().init()
        text = "Élodie → Gowtham 😃"
        self.assertTrue(item.setString_forType_(text, AppKit.NSPasteboardTypeString))
        payloads = {
            "public.rtf": b"{\\rtf1\\ansi rich text}",
            "com.justdictate.test-binary": bytes(range(128)),
        }
        for kind, data in payloads.items():
            self.assertTrue(item.setData_forType_(NSData.dataWithBytes_length_(data, len(data)), kind))
        board.clearContents()
        self.assertTrue(board.writeObjects_([item]))
        with patch.object(AppKit, "NSPasteboard", SimpleNamespace(generalPasteboard=lambda: board)):
            original = dictation_engine._clipboard_snapshot()
            count = dictation_engine._clipboard_set("Temporary dictation")
            self.assertEqual(dictation_engine._clipboard_get(), "Temporary dictation")
            self.assertTrue(dictation_engine._clipboard_restore(original, count))
            self.assertEqual(dictation_engine._clipboard_get(), text)
            restored = dictation_engine._clipboard_snapshot()
            self.assertEqual([dict(types) for types in restored], [dict(types) for types in original])
            for kind, data in payloads.items():
                self.assertEqual(dict(restored[0])[kind], data)
            stale_count = dictation_engine._clipboard_set("Another dictation")
            dictation_engine._clipboard_set("A newer user copy")
            self.assertFalse(dictation_engine._clipboard_restore(original, stale_count))
            self.assertEqual(dictation_engine._clipboard_get(), "A newer user copy")


if __name__ == "__main__":
    unittest.main()
