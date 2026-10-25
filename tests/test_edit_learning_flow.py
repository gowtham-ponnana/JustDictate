"""Real local store/trainer/queue with simulated editor snapshots; no OS input."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from correction_monitor import CorrectionMonitor, TextSnapshot, STABLE_SECONDS, TRACK_SECONDS
from learning_service import LearningService
from personalization import PersonalizationManager


class Editor:
    snapshot = None

    def capture(self):
        return self.snapshot


class EditLearningFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manager = PersonalizationManager(Path(self.temp.name) / "learning",
                                              enabled=True, retain_audio=True)
        self.clock = 0.0
        self.edits_enabled = True
        self.editor = Editor()
        self.service = LearningService(self.manager, enabled=True, retain_audio=True,
                                       is_idle=lambda: not self.monitor.is_tracking)
        self.monitor = CorrectionMonitor(
            self.editor, self.service.record, self.service.auto_correct,
            lambda: self.edits_enabled, clock=lambda: self.clock,
        )
        self.addCleanup(self.service.close)
        self.addCleanup(self.monitor.close)

    def set_editor(self, text, selection=None):
        self.editor.snapshot = TextSnapshot("editor", text, selection or (len(text), 0))

    def dictate_and_correct(self, raw):
        context = "Existing document content is never a training transcript: "
        self.set_editor(context)
        self.monitor.prepare(raw + " ")
        self.assertTrue(self.monitor.complete(raw, raw, np.zeros(6400, dtype=np.float32)))
        self.service._queue.join()
        self.set_editor(context + raw + " ")
        self.monitor.poll()
        self.clock += 1
        self.set_editor(context + raw.replace("Gotham", "Gowtham") + " ")
        self.monitor.poll()
        self.clock += STABLE_SECONDS
        self.monitor.poll()
        self.service._queue.join()

    def test_user_edits_train_without_reviews_then_used_text_and_audio_are_deleted(self):
        for number in range(18):
            self.dictate_and_correct(f"Please ask Gotham tomorrow about task {number}.")
        state = self.manager.status()
        self.assertEqual(state["automatic_count"], 18)
        self.assertEqual(state["manual_count"], 0)
        self.assertEqual(len(list(self.manager.audio_directory.glob("*.wav"))), 18)
        for row in self.manager.recent(20):
            self.assertNotIn("Existing document", row["raw"])
            self.assertNotIn("Existing document", row["corrected"])
        # A still-editable observation must never be trained then deleted.
        self.assertFalse(self.service.schedule_training(force=True))
        self.clock += TRACK_SECONDS + 1
        self.monitor.poll()
        self.assertTrue(self.service.schedule_training(force=True))
        self.service._queue.join()
        state = self.manager.status()
        self.assertEqual(state["last_attempt"]["status"], "trained")
        self.assertEqual(state["last_attempt"]["purged_records"], 18)
        self.assertEqual(state["record_count"], 0)
        self.assertEqual(list(self.manager.audio_directory.glob("*.wav")), [])
        self.assertEqual(self.manager.apply("Please ask Gotham tomorrow about lunch."),
                         "Please ask Gowtham tomorrow about lunch.")
        restored = PersonalizationManager(self.manager.directory, enabled=True)
        self.assertEqual(restored.status()["record_count"], 0)
        self.assertEqual(restored.apply("Please ask Gotham tomorrow about lunch."),
                         "Please ask Gowtham tomorrow about lunch.")

    def test_disabling_edit_observation_keeps_local_history_enabled(self):
        self.edits_enabled = False
        self.set_editor("")
        self.monitor.prepare("Keep this dictation. ")
        self.assertTrue(self.monitor.complete("Keep this dictation.", "Keep this dictation.",
                                              np.zeros(6400, dtype=np.float32)))
        self.service._queue.join()
        row = self.manager.latest()
        self.assertEqual(row["raw"], "Keep this dictation.")
        self.assertIsNone(row["corrected"])
        self.assertFalse(self.monitor.is_tracking)


if __name__ == "__main__":
    unittest.main()
