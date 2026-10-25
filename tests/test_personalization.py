from contextlib import closing
from pathlib import Path
import stat
import sqlite3
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np

import personalization as p


class PersonalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name) / "personalization"
        self.now = datetime(2026, 12, 15, tzinfo=timezone.utc)
        self.clock = patch.object(p, "_now", side_effect=lambda: self.now)
        self.clock.start()
        self.manager = p.PersonalizationManager(self.directory, enabled=True)

    def tearDown(self):
        self.clock.stop()
        self.temp.cleanup()

    def examples(self, count=24):
        ids = []
        for number in range(count):
            raw = f"Please ask Gotham tomorrow about task {number}."
            record = self.manager.record(raw, raw)
            self.assertTrue(self.manager.review(record, raw.replace("Gotham", "Gowtham")))
            ids.append(record)
        return ids

    def test_disabled_is_lazy_and_does_not_capture_or_change_text(self):
        path = Path(self.temp.name) / "disabled"
        manager = p.PersonalizationManager(path)
        self.assertIsNone(manager.record("Hello", "Hello", np.zeros(16000)))
        self.assertEqual(manager.apply("Hello"), "Hello")
        self.assertEqual(manager.train_if_due()["status"], "disabled")
        manager.reset()
        self.assertFalse(path.exists())

    def test_only_reviewed_examples_train_and_rows_survive_restart(self):
        for number in range(24):
            self.manager.record(f"Ask Gotham about item {number}.", "Ask Gowtham.")
        result = self.manager.train_if_due(force=True)
        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(result["reviewed_count"], 0)
        restored = p.PersonalizationManager(self.directory, enabled=True)
        self.assertEqual(restored.status()["record_count"], 24)
        self.assertEqual(restored.status()["reviewed_count"], 0)
        self.assertFalse(restored.latest()["reviewed"])

    def test_monthly_schedule_year_rollover_catchup_and_no_same_month_retrain(self):
        self.examples()
        self.assertEqual(self.manager.train_if_due()["status"], "not_due")
        self.assertEqual(self.manager.status()["next_training_month"], "2027-01")
        self.now = datetime(2027, 1, 2, tzinfo=timezone.utc)
        manager = p.PersonalizationManager(self.directory, enabled=True)
        result = manager.train_if_due()
        self.assertEqual(result["status"], "trained")
        self.assertGreater(result["metrics"]["baseline_errors"], result["metrics"]["candidate_errors"])
        self.assertEqual(manager.train_if_due()["status"], "not_due")
        self.assertEqual(manager.status()["next_training_month"], "2027-02")
        self.assertEqual(manager.status()["version"], 1)
        restored = p.PersonalizationManager(self.directory, enabled=True)
        self.assertEqual(restored.apply("Please ask Gotham tomorrow about lunch."),
                         "Please ask Gowtham tomorrow about lunch.")
        self.now = datetime(2027, 3, 1, tzinfo=timezone.utc)
        self.assertEqual(restored.train_if_due()["status"], "insufficient_data")
        self.assertEqual(restored.status()["record_count"], 0)

    def test_force_does_not_bypass_validation_and_retry_requires_reviewed_data(self):
        self.now = datetime(2027, 1, 2, tzinfo=timezone.utc)
        self.assertEqual(self.manager.train_if_due()["status"], "insufficient_data")
        self.assertEqual(self.manager.train_if_due()["status"], "unchanged")
        self.manager.record("Unreviewed text", "Unreviewed text")
        self.assertEqual(self.manager.train_if_due()["status"], "unchanged")
        self.assertEqual(self.manager.train_if_due(force=True)["status"], "insufficient_data")
        self.examples()
        self.assertEqual(self.manager.train_if_due()["status"], "trained")

    def test_duplicate_rows_do_not_increase_support_or_cross_holdout(self):
        for _ in range(30):
            key = self.manager.record("Please ask Gotham tomorrow.", "Please ask Gotham tomorrow.")
            self.manager.review(key, "Please ask Gowtham tomorrow.")
        result = self.manager.train_if_due(force=True)
        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(result["reviewed_count"], 1)

    def test_exact_context_only_preserves_spacing_unicode_and_does_not_cascade(self):
        self.examples()
        self.assertEqual(self.manager.train_if_due(force=True)["status"], "trained")
        self.assertEqual(self.manager.apply("Please ask Gotham tomorrow about lunch."),
                         "Please ask Gowtham tomorrow about lunch.")
        for text in ("I visited Gotham yesterday.", "Gotham", "Please ask Gotham today.",
                     "Gotham City", "Nothing familiar.", "Please ask gotham tomorrow."):
            self.assertEqual(self.manager.apply(text), text)
        text = "Élodie: Please\nask Gotham\ttomorrow — thanks!"
        self.assertEqual(self.manager.apply(text), text.replace("Gotham", "Gowtham"))
        rules = (p.Rule(("one",), "two", "say", ".", 2),
                 p.Rule(("two",), "three", "say", ".", 2))
        self.assertEqual(p._apply("say one.", p._index(rules)), "say two.")

    def test_conflicting_confirmation_vetoes_rule(self):
        self.examples()
        # Place a conflicting example in training by selecting the deterministic split.
        raw = "Please ask Gotham tomorrow about the city."
        identity = self.manager.record(raw, raw)
        self.manager.review(identity, raw)
        result = self.manager.train_if_due(force=True)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(self.manager.apply(raw), raw)

    def test_ambiguous_identical_audio_text_group_becomes_abstention(self):
        raw = "Please ask Gotham tomorrow."
        rows = [{"raw": raw, "corrected": raw.replace("Gotham", "Gowtham")},
                {"raw": raw, "corrected": raw}]
        grouped = p._groups(rows)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0]["corrected"], raw)

    def test_heldout_regression_rejects_candidate_without_replacing_model(self):
        self.examples()
        self.assertEqual(self.manager.train_if_due(force=True)["status"], "trained")
        old_model = self.manager._model.copy()
        self.examples()
        harmful = p.Rule(("Gotham",), "An invented person", "ask", "tomorrow", 10)
        with patch.object(p, "_learn", return_value=(harmful,)):
            result = self.manager.train_if_due(force=True)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(self.manager._model, old_model)
        self.assertEqual(self.manager.status()["record_count"], 24)

    def test_rollback_to_baseline_persists_and_disable_bypasses_model(self):
        self.examples()
        self.assertEqual(self.manager.train_if_due(force=True)["status"], "trained")
        text = "Please ask Gotham tomorrow about lunch."
        self.manager.set_enabled(False)
        self.assertEqual(self.manager.apply(text), text)
        self.manager.set_enabled(True)
        self.assertNotEqual(self.manager.apply(text), text)
        self.assertTrue(self.manager.rollback())
        self.assertFalse(self.manager.rollback())
        manager = p.PersonalizationManager(self.directory, enabled=True)
        self.assertEqual(manager.apply(text), text)
        self.assertEqual(manager.train_if_due()["status"], "not_due")

    def test_audio_opt_in_permissions_retention_and_full_reset(self):
        audio = np.zeros(16000, dtype=np.float32)
        self.manager.record("No audio", "No audio", audio)
        self.assertFalse(self.manager.audio_directory.exists())
        self.manager.set_retain_audio(True)
        key = self.manager.record("Saved audio", "Saved audio", audio)
        path = self.manager.audio_directory / self.manager.latest()["audio_name"]
        self.assertTrue(path.exists())
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.directory.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(self.manager.database.stat().st_mode), 0o600)
        self.manager.review(key, "Saved audio")
        self.now = datetime(2027, 4, 1, tzinfo=timezone.utc)
        self.assertEqual(self.manager.status()["record_count"], 0)
        self.assertFalse(path.exists())
        self.manager.record("New", "New", audio)
        self.manager.reset()
        self.assertFalse(self.directory.exists())
        self.assertEqual(self.manager.apply("Please ask Gotham tomorrow."), "Please ask Gotham tomorrow.")
        self.assertEqual(self.manager.recent(), [])

    def test_storage_caps_and_delete_record(self):
        self.manager.set_retain_audio(True)
        with patch.object(p, "MAX_RECORDS", 2), patch.object(p, "MAX_AUDIO_BYTES", 33000):
            for n in range(4):
                self.manager.record(str(n), str(n), np.zeros(16000))
            self.assertEqual(len(self.manager.recent()), 2)
            self.assertLessEqual(sum(f.stat().st_size for f in self.manager.audio_directory.glob("*.wav")), 33000)
            key = self.manager.latest()["id"]
            self.assertTrue(self.manager.delete_record(key))
            self.assertFalse(self.manager.delete_record(key))
        self.assertEqual(len(self.manager.recent()), 1)

    def test_reset_during_training_cannot_restore_deleted_data(self):
        self.examples()
        entered, release = threading.Event(), threading.Event()
        real_learn = p._learn
        def delayed(rows):
            entered.set()
            self.assertTrue(release.wait(5))
            return real_learn(rows)
        results = []
        with patch.object(p, "_learn", side_effect=delayed):
            worker = threading.Thread(target=lambda: results.append(self.manager.train_if_due(force=True)))
            worker.start()
            self.assertTrue(entered.wait(5))
            self.manager.reset()
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results[0]["status"], "stale")
        self.assertFalse(self.directory.exists())
        self.assertFalse(self.manager.status()["training"])

    def test_empty_or_oversized_reviews_are_rejected(self):
        key = self.manager.record("Hello", "Hello")
        for value in ("", " ", "x" * (p.MAX_TEXT_CHARS + 1)):
            with self.assertRaises(ValueError):
                self.manager.review(key, value)
        self.assertFalse(self.manager.review(9999, "Missing"))

    def test_corrupted_database_is_preserved_and_optional_feature_fails_open(self):
        self.manager.database.write_bytes(b"not a sqlite database")
        manager = p.PersonalizationManager(self.directory, enabled=True)
        self.assertTrue(manager.status()["unavailable"])
        self.assertEqual(manager.database.read_bytes(), b"not a sqlite database")
        self.assertEqual(manager.apply("Dictation still works"), "Dictation still works")
        self.assertIsNone(manager.record("No capture", "No capture"))
        self.assertFalse(manager.review(1, "Anything"))
        self.assertEqual(manager.train_if_due(force=True)["status"], "unavailable")
        self.assertEqual(manager.recent(), [])
        manager.reset()
        self.assertFalse(manager.status()["unavailable"])
        self.assertIsNotNone(manager.record("Fresh start", "Fresh start"))

    def test_invalid_saved_model_does_not_interrupt_dictation_or_delete_history(self):
        self.manager.record("Keep history", "Keep history")
        with self.manager._connection() as connection:
            self.manager._put(connection, "model", {"version": 1, "rules": [{"source": []}]})
        manager = p.PersonalizationManager(self.directory, enabled=False)
        self.assertTrue(manager.status()["unavailable"])
        self.assertTrue(manager.database.exists())
        manager.set_enabled(True)
        self.assertEqual(manager.apply("Text"), "Text")

    def test_orphan_audio_from_interrupted_write_is_removed_on_restart(self):
        self.manager.audio_directory.mkdir(mode=0o700)
        orphan = self.manager.audio_directory / "orphan.wav"
        orphan.write_bytes(b"orphaned speech")
        manager = p.PersonalizationManager(self.directory, enabled=True)
        self.assertFalse(orphan.exists())
        self.assertEqual(manager.status()["record_count"], 0)

    def test_enrollment_begins_when_enabled_not_when_app_first_launches(self):
        path = Path(self.temp.name) / "later"
        manager = p.PersonalizationManager(path, enabled=False)
        self.now = datetime(2027, 2, 1, tzinfo=timezone.utc)
        manager.set_enabled(True)
        self.assertEqual(manager.status()["next_training_month"], "2027-03")

    def test_disable_during_apply_discards_inflight_personalization(self):
        def disable_inflight(raw, index):
            self.manager.set_enabled(False)
            return "Personalized text"
        with patch.object(p, "_apply", side_effect=disable_inflight):
            self.assertEqual(self.manager.apply("Original text"), "Original text")

    def test_rule_budget_is_applied_before_conflict_evaluation(self):
        rows = []
        for group in range(4):
            for number in range(3):
                raw = f"Please ask Name{group} tomorrow about task {number}."
                rows.append({"raw": raw, "corrected": raw.replace(f"Name{group}", f"Person{group}")})
        with patch.object(p, "MAX_RULES", 2), patch.object(p, "_apply", wraps=p._apply) as apply:
            rules = p._learn(rows)
        self.assertEqual(len(rules), 2)
        self.assertEqual(apply.call_count, 6)  # Two candidates, only three affected rows each.

    def test_indexed_conflict_detection_handles_multiword_and_repeated_occurrences(self):
        rows = [
            {"raw": "Please ask go thumb tomorrow about one.",
             "corrected": "Please ask Gowtham tomorrow about one."},
            {"raw": "Please ask go thumb tomorrow about two.",
             "corrected": "Please ask Gowtham tomorrow about two."},
            {"raw": "Please ask go thumb tomorrow and ask go thumb tomorrow.",
             "corrected": "Please ask go thumb tomorrow and ask go thumb tomorrow."},
        ]
        self.assertEqual(p._learn(rows), ())

    def test_automatic_labels_target_exact_row_and_manual_review_takes_precedence(self):
        first = self.manager.record("Ask Gotham tomorrow.", "Ask Gotham tomorrow.")
        second = self.manager.record("Unrelated next dictation.", "Unrelated next dictation.")
        self.assertTrue(self.manager.auto_correct(first, "Ask Gowtham tomorrow."))
        entries = {row["id"]: row for row in self.manager.recent()}
        self.assertEqual(entries[first]["raw"], "Ask Gotham tomorrow.")
        self.assertEqual(entries[first]["delivered"], "Ask Gotham tomorrow.")
        self.assertEqual(entries[first]["label_source"], "automatic")
        self.assertIsNone(entries[second]["corrected"])
        self.assertEqual(self.manager.status()["automatic_count"], 1)
        self.assertTrue(self.manager.review(first, "Ask Gautam tomorrow."))
        self.assertFalse(self.manager.auto_correct(first, "Ask Another tomorrow."))
        self.assertFalse(self.manager.auto_correct(first, None))
        self.assertEqual(self.manager.status()["manual_count"], 1)
        self.assertEqual(self.manager.status()["automatic_count"], 0)
        self.assertFalse(self.manager.auto_correct(999999, "Missing"))

    def test_automatic_label_can_be_revised_or_revoked(self):
        key = self.manager.record("Ask Gotham tomorrow.", "Ask Gotham tomorrow.")
        self.assertTrue(self.manager.auto_correct(key, "Ask Gowtham tomorrow."))
        self.assertFalse(self.manager.auto_correct(key, "Ask Gowtham tomorrow."))
        self.assertTrue(self.manager.auto_correct(key, "Ask Gautam tomorrow."))
        self.assertTrue(self.manager.auto_correct(key, None))
        self.assertFalse(self.manager.auto_correct(key, None))
        self.assertIsNone(self.manager.latest()["corrected"])
        self.assertIsNone(self.manager.latest()["label_source"])
        self.assertEqual(self.manager.status()["reviewed_count"], 0)

    def test_old_schema_migrates_reviewed_examples_as_manual(self):
        self.manager.reset()
        self.directory.mkdir()
        with closing(sqlite3.connect(self.manager.database)) as connection, connection:
            connection.execute("""CREATE TABLE samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL,
                raw TEXT NOT NULL, delivered TEXT NOT NULL, corrected TEXT, audio_name TEXT)""")
            connection.execute("INSERT INTO samples (created, raw, delivered, corrected) VALUES (?, ?, ?, ?)",
                               (self.now.timestamp(), "Original", "Original", "Reviewed"))
        manager = p.PersonalizationManager(self.directory, enabled=True)
        self.assertEqual(manager.latest()["label_source"], "manual")
        self.assertEqual(manager.status()["manual_count"], 1)
        self.assertFalse(manager.auto_correct(manager.latest()["id"], "Automatic"))

    def test_accepted_training_deletes_used_text_audio_and_duplicates_only(self):
        self.manager.set_retain_audio(True)
        audio = np.zeros(160, dtype=np.float32)
        used_ids = []
        for number in range(24):
            raw = f"Please ask Gotham tomorrow about task {number}."
            key = self.manager.record(raw, raw, audio)
            self.manager.auto_correct(key, raw.replace("Gotham", "Gowtham"))
            used_ids.append(key)
        raw = "Please ask Gotham tomorrow about task 0."
        duplicate = self.manager.record(raw, raw, audio)
        self.manager.review(duplicate, raw.replace("Gotham", "Gowtham"))
        used_ids.append(duplicate)
        unreviewed = self.manager.record("Keep this unreviewed sample", "Keep this unreviewed sample", audio)
        long_text = "too long " * p.MAX_TRAIN_TOKENS
        unusable = self.manager.record(long_text, long_text, audio)
        self.manager.review(unusable, long_text)
        before = {row["id"]: row for row in self.manager.recent(100)}
        result = self.manager.train_if_due(force=True)
        self.assertEqual(result["status"], "trained")
        self.assertEqual(result["purged_records"], 25)
        self.assertEqual({row["id"] for row in self.manager.recent(100)}, {unreviewed, unusable})
        for key in used_ids:
            self.assertFalse((self.manager.audio_directory / before[key]["audio_name"]).exists())
        for key in (unreviewed, unusable):
            self.assertTrue((self.manager.audio_directory / before[key]["audio_name"]).exists())
        manager = p.PersonalizationManager(self.directory, enabled=True)
        self.assertEqual(manager.apply("Please ask Gotham tomorrow about a new task."),
                         "Please ask Gowtham tomorrow about a new task.")
        # SQLite secure_delete removes sample payloads from freed database pages.
        self.assertNotIn(b"about task 23", self.manager.database.read_bytes())

    def test_new_month_carries_previous_learning_after_training_history_is_purged(self):
        self.examples()
        self.assertEqual(self.manager.train_if_due(force=True)["status"], "trained")
        self.assertEqual(self.manager.recent(), [])
        self.now = datetime(2027, 1, 2, tzinfo=timezone.utc)
        for number in range(24):
            raw = f"Please email Alice tomorrow about item {number}."
            key = self.manager.record(raw, raw)
            self.manager.auto_correct(key, raw.replace("Alice", "Alyce"))
        self.assertEqual(self.manager.train_if_due()["status"], "trained")
        self.assertEqual(self.manager.apply("Please ask Gotham tomorrow about lunch."),
                         "Please ask Gowtham tomorrow about lunch.")
        self.assertEqual(self.manager.apply("Please email Alice tomorrow about lunch."),
                         "Please email Alyce tomorrow about lunch.")
        self.assertEqual(self.manager.status()["record_count"], 0)
        self.assertEqual(self.manager.status()["rule_count"], 2)
        self.assertTrue(self.manager.rollback())
        self.assertEqual(self.manager.apply("Please ask Gotham tomorrow about lunch."),
                         "Please ask Gowtham tomorrow about lunch.")
        self.assertEqual(self.manager.apply("Please email Alice tomorrow about lunch."),
                         "Please email Alice tomorrow about lunch.")
        self.assertEqual(self.manager.status()["record_count"], 0)

    def test_new_evidence_can_retire_harmful_rule_even_when_raw_was_correct(self):
        self.examples()
        self.assertEqual(self.manager.train_if_due(force=True)["status"], "trained")
        for number in range(24):
            raw = f"Please ask Gotham tomorrow about city number {number}."
            key = self.manager.record(raw, self.manager.apply(raw))
            self.manager.review(key, raw)
        result = self.manager.train_if_due(force=True)
        self.assertEqual(result["status"], "trained")
        self.assertEqual(result["metrics"]["baseline_errors"], 0)
        self.assertGreater(result["metrics"]["previous_errors"], 0)
        self.assertEqual(result["metrics"]["candidate_errors"], 0)
        self.assertEqual(self.manager.status()["rule_count"], 0)
        self.assertEqual(self.manager.recent(), [])

    def test_same_model_keeps_history_for_a_later_informative_update(self):
        self.examples()
        self.assertEqual(self.manager.train_if_due(force=True)["status"], "trained")
        self.examples()
        result = self.manager.train_if_due(force=True)
        self.assertEqual(result["status"], "unchanged_model")
        self.assertEqual(self.manager.status()["version"], 1)
        self.assertEqual(self.manager.status()["record_count"], 24)

    def test_stale_training_does_not_delete_new_or_changed_samples(self):
        self.examples()
        entered, release = threading.Event(), threading.Event()
        original = p._learn
        def delayed(rows):
            entered.set()
            self.assertTrue(release.wait(5))
            return original(rows)
        results = []
        with patch.object(p, "_learn", side_effect=delayed):
            worker = threading.Thread(target=lambda: results.append(self.manager.train_if_due(force=True)))
            worker.start()
            self.assertTrue(entered.wait(5))
            new_key = self.manager.record("New data", "New data")
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results[0]["status"], "stale")
        self.assertEqual(self.manager.status()["record_count"], 25)
        self.assertEqual(self.manager.latest()["id"], new_key)

    def test_commit_failure_keeps_history_audio_and_previous_model(self):
        self.manager.set_retain_audio(True)
        self.examples()
        raw = "Please ask Gotham tomorrow about an audio example."
        key = self.manager.record(raw, raw, np.zeros(160))
        self.manager.review(key, raw.replace("Gotham", "Gowtham"))
        audio_path = self.manager.audio_directory / self.manager.latest()["audio_name"]
        original = self.manager._put
        def failing_put(connection, key, value):
            if key == "last_attempt":
                raise sqlite3.OperationalError("simulated commit preparation failure")
            return original(connection, key, value)
        with patch.object(self.manager, "_put", side_effect=failing_put):
            with self.assertRaises(sqlite3.OperationalError):
                self.manager.train_if_due(force=True)
        self.assertEqual(self.manager.status()["record_count"], 25)
        self.assertTrue(audio_path.exists())
        self.assertEqual(self.manager.status()["version"], 0)
        restored = p.PersonalizationManager(self.directory, enabled=True)
        self.assertEqual(restored.status()["version"], 0)

    def test_interrupted_audio_cleanup_recovers_on_restart(self):
        self.manager.set_retain_audio(True)
        for number in range(24):
            raw = f"Please ask Gotham tomorrow about task {number}."
            key = self.manager.record(raw, raw, np.zeros(160))
            self.manager.review(key, raw.replace("Gotham", "Gowtham"))
        with patch.object(self.manager, "_remove_audio", side_effect=OSError("simulated busy file")):
            result = self.manager.train_if_due(force=True)
        self.assertEqual(result["status"], "trained")
        self.assertEqual(result["audio_cleanup_pending"], 24)
        self.assertEqual(len(list(self.manager.audio_directory.glob("*.wav"))), 24)
        restored = p.PersonalizationManager(self.directory, enabled=True)
        self.assertEqual(restored.status()["record_count"], 0)
        self.assertEqual(list(self.manager.audio_directory.glob("*.wav")), [])

    def test_successful_training_reclaims_database_space_without_losing_learning(self):
        for number in range(80):
            raw = f"Please ask Gotham tomorrow about task {number}. " + "additional context " * 80
            key = self.manager.record(raw, raw)
            self.manager.review(key, raw.replace("Gotham", "Gowtham"))
        before = self.manager.database.stat().st_size
        self.assertGreater(before, 200000)
        result = self.manager.train_if_due(force=True)
        after = self.manager.database.stat().st_size
        self.assertEqual(result["status"], "trained")
        self.assertLess(after, before // 2)
        self.assertEqual(self.manager.status()["record_count"], 0)
        restored = p.PersonalizationManager(self.directory, enabled=True)
        self.assertEqual(restored.apply("Please ask Gotham tomorrow about lunch."),
                         "Please ask Gowtham tomorrow about lunch.")
        self.assertTrue(restored.status()["can_rollback"])

    def test_failed_compaction_keeps_accepted_model_and_purged_history(self):
        self.examples()
        with patch.object(self.manager, "_compact_database", side_effect=sqlite3.OperationalError("database is busy")):
            result = self.manager.train_if_due(force=True)
        self.assertEqual(result["status"], "trained")
        self.assertTrue(result["storage_cleanup_pending"])
        self.assertEqual(self.manager.status()["record_count"], 0)
        restored = p.PersonalizationManager(self.directory, enabled=True)
        self.assertEqual(restored.status()["version"], 1)
        self.assertEqual(restored.status()["record_count"], 0)


if __name__ == "__main__":
    unittest.main()
