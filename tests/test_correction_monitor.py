from pathlib import Path
import tempfile
import threading
import unittest

from correction_monitor import CorrectionMonitor, TextSnapshot, TRACK_SECONDS, _small_replacement


class Reader:
    snapshot = None

    def capture(self):
        return self.snapshot


class CorrectionMonitorTests(unittest.TestCase):
    def setUp(self):
        self.reader = Reader()
        self.now = 0.0
        self.enabled = True
        self.saved = []
        self.corrections = []
        self.delayed = False
        self.accept_labels = True
        self.monitor = CorrectionMonitor(
            self.reader, self.record, self.correct, lambda: self.enabled,
            clock=lambda: self.now,
        )
        self.addCleanup(self.monitor.close)

    def record(self, raw, delivered, audio, callback):
        self.saved.append((raw, delivered, audio, callback))
        if not self.delayed:
            callback(len(self.saved))
        return True

    def correct(self, record_id, corrected):
        if corrected is not None and not self.accept_labels:
            return False
        self.corrections.append((record_id, corrected))
        return True

    def snapshot(self, value, *, field="field", selection=None):
        if selection is None:
            selection = (len(value), 0)
        self.reader.snapshot = TextSnapshot(field, value, selection)

    def paste(self, *, field="field", prefix="Notes: ", suffix=" Next item.",
              raw="Please ask Gotham tomorrow.", selected="draft", verify=True):
        self.snapshot(prefix + selected + suffix, field=field, selection=(len(prefix), len(selected)))
        self.monitor.prepare(raw + " ")
        self.monitor.complete(raw, raw, b"audio")
        expected = prefix + raw + " " + suffix
        self.snapshot(expected, field=field)
        if verify:
            self.monitor.poll()
        return expected

    def observe(self, value, at, *, field="field"):
        self.assertGreaterEqual(at, self.now, "Fake monotonic time must not move backward")
        self.now = at
        self.snapshot(value, field=field)
        self.monitor.poll()

    def stable_edit(self, expected, *, field="field", at=None):
        at = self.now + 1 if at is None else at
        edited = expected.replace("Gotham", "Gowtham")
        self.observe(edited, at, field=field)
        self.observe(edited, at + 60, field=field)
        return edited

    def test_unchanged_output_is_never_confirmed(self):
        expected = self.paste()
        self.observe(expected, 30)
        self.assertEqual(self.corrections, [])
        self.assertTrue(self.monitor.is_tracking)

    def test_exact_paste_must_be_observed_before_any_edit(self):
        expected = self.paste(verify=False)
        self.observe(expected.replace("Gotham", "Gowtham"), 1)
        self.observe(expected.replace("Gotham", "Gowtham"), 4)
        self.observe(expected.replace("Gotham", "Gowtham"), 6)
        self.assertEqual(self.corrections, [])
        self.assertFalse(self.monitor.is_tracking)

    def test_debounce_and_only_inserted_span_saved(self):
        expected = self.paste()
        edited = expected.replace("Gotham", "Gowtham")
        self.observe(edited, 1)
        self.observe(edited, 60.99)
        self.assertEqual(self.corrections, [])
        self.observe(edited, 61)
        self.assertEqual(self.corrections, [(1, "Please ask Gowtham tomorrow.")])
        self.observe(edited, 80)
        self.assertEqual(len(self.corrections), 1)

    def test_unicode_offsets_are_python_offsets_not_utf16(self):
        expected = self.paste(prefix="😀 café: ", suffix=" 日本語", selected="📝")
        self.stable_edit(expected)
        self.assertEqual(self.corrections, [(1, "Please ask Gowtham tomorrow.")])

    def test_transient_word_deletion_does_not_prevent_completed_typo_fix(self):
        expected = self.paste()
        self.observe(expected.replace("Gotham", ""), 1)
        self.observe(expected.replace("Gotham", ""), 30)
        self.assertTrue(self.monitor.is_tracking)
        self.assertEqual(self.corrections, [])
        self.observe(expected.replace("Gotham", "Gowtham"), 31)
        self.observe(expected.replace("Gotham", "Gowtham"), 90.99)
        self.assertEqual(self.corrections, [])
        self.observe(expected.replace("Gotham", "Gowtham"), 91)
        self.assertEqual(self.corrections, [(1, "Please ask Gowtham tomorrow.")])

    def test_outside_edit_revokes_prior_label_and_drops_range(self):
        expected = self.paste()
        edited = self.stable_edit(expected)
        self.observe("New " + edited, 62)
        self.assertEqual(self.corrections[-1], (1, None))
        self.assertFalse(self.monitor.is_tracking)

    def test_appended_unrelated_text_is_not_a_correction(self):
        expected = self.paste(prefix="", suffix="", selected="")
        self.observe(expected + "Another unrelated sentence.", 1)
        self.observe(expected + "Another unrelated sentence.", 61)
        self.assertEqual(self.corrections, [])
        self.assertFalse(self.monitor.is_tracking)

    def test_wholesale_rewrite_or_delete_is_not_a_correction(self):
        for replacement in ("", "Something completely different."):
            with self.subTest(replacement=replacement):
                self.monitor.cancel()
                self.paste(prefix="", suffix="", selected="")
                changed_at = self.now + 1
                self.observe(replacement, changed_at)
                self.observe(replacement, changed_at + 60)
                self.assertEqual(self.corrections, [])
                self.assertFalse(self.monitor.is_tracking)

    def test_edit_back_immediately_revokes_and_does_not_confirm_original(self):
        expected = self.paste()
        self.stable_edit(expected)
        self.observe(expected, 62)
        self.assertEqual(self.corrections, [(1, "Please ask Gowtham tomorrow."), (1, None)])
        self.observe(expected, 80)
        self.assertEqual(len(self.corrections), 2)

    def test_focus_switch_pauses_and_return_pairs_original_row(self):
        first = self.paste(field="first")
        second = self.paste(field="second")
        self.stable_edit(second, field="second", at=1)
        self.stable_edit(first, field="first")
        self.assertEqual(self.corrections, [
            (2, "Please ask Gowtham tomorrow."), (1, "Please ask Gowtham tomorrow."),
        ])

    def test_delayed_save_callbacks_bind_their_own_contexts(self):
        self.delayed = True
        first = self.paste(field="first")
        second = self.paste(field="second")
        self.saved[1][3](22)
        self.saved[0][3](11)
        self.stable_edit(first, field="first", at=1)
        self.stable_edit(second, field="second")
        self.assertEqual([row[0] for row in self.corrections], [11, 22])

    def test_label_waits_for_save_callback(self):
        self.delayed = True
        expected = self.paste()
        edited = self.stable_edit(expected)
        self.assertEqual(self.corrections, [])
        self.saved[0][3](7)
        self.observe(edited, 62)
        self.assertEqual(self.corrections, [(7, "Please ask Gowtham tomorrow.")])

    def test_cancel_invalidates_delayed_callback_even_if_row_id_is_reused(self):
        self.delayed = True
        first = self.paste(field="old")
        old_callback = self.saved[0][3]
        self.monitor.cancel()
        second = self.paste(field="new")
        old_callback(1)
        self.saved[1][3](1)
        self.stable_edit(first, field="old", at=1)
        self.assertEqual(self.corrections, [])
        self.stable_edit(second, field="new")
        self.assertEqual(self.corrections, [(1, "Please ask Gowtham tomorrow.")])

    def test_cancel_revokes_active_inferred_label(self):
        self.stable_edit(self.paste())
        self.monitor.cancel()
        self.assertEqual(self.corrections[-1], (1, None))
        self.assertFalse(self.monitor.is_tracking)

    def test_disable_revokes_active_label_but_keeps_independent_history_capture(self):
        self.stable_edit(self.paste())
        self.enabled = False
        self.monitor.poll()
        self.assertEqual(self.corrections[-1], (1, None))
        self.paste()
        self.assertEqual(len(self.saved), 2)
        self.assertFalse(self.monitor.is_tracking)

    def test_ttl_finalizes_stable_label_and_drops_delayed_binding(self):
        self.stable_edit(self.paste())
        self.now = TRACK_SECONDS
        self.monitor.poll()
        self.assertFalse(self.monitor.is_tracking)
        self.assertEqual(self.corrections, [(1, "Please ask Gowtham tomorrow.")])
        self.delayed = True
        expected = self.paste(field="late")
        self.stable_edit(expected, field="late", at=TRACK_SECONDS + 1)
        callback = self.saved[-1][3]
        self.now = 2 * TRACK_SECONDS
        self.monitor.poll()
        callback(10)
        self.assertFalse(self.monitor.is_tracking)
        self.assertEqual(len(self.corrections), 1)

    def test_bounded_fields_and_one_insertion_per_field(self):
        expected = None
        for number in range(10):
            expected = self.paste(field=f"field{number}")
        self.assertEqual(len(self.monitor._contexts), 8)
        self.stable_edit(expected, field="field0")
        self.assertEqual(self.corrections, [])
        self.paste(field="field9")
        self.assertEqual(len(self.monitor._contexts), 8)

    def test_refused_label_is_retried_without_losing_observation(self):
        expected = self.paste()
        self.accept_labels = False
        edited = self.stable_edit(expected)
        self.assertEqual(self.corrections, [])
        self.accept_labels = True
        self.observe(edited, 62)
        self.assertEqual(self.corrections, [(1, "Please ask Gowtham tomorrow.")])

    def test_unsupported_field_still_records_without_tracking(self):
        self.reader.snapshot = None
        self.monitor.prepare("hello ")
        self.monitor.complete("hello", "hello", b"audio")
        self.assertEqual(len(self.saved), 1)
        self.assertFalse(self.monitor.is_tracking)

    def test_prepare_does_not_wait_for_observer_lock_or_reuse_old_prepare(self):
        self.snapshot("", selection=(0, 0))
        self.monitor.prepare("Please ask Gotham tomorrow. ")
        acquired = threading.Event()
        release = threading.Event()

        def hold_lock():
            with self.monitor._lock:
                acquired.set()
                release.wait(2)

        worker = threading.Thread(target=hold_lock)
        worker.start()
        self.assertTrue(acquired.wait(1))
        try:
            self.monitor.prepare("Please ask Gotham tomorrow. ")
            self.assertIsNone(self.monitor._prepared)
        finally:
            release.set()
            worker.join(1)
        self.monitor.complete("Please ask Gotham tomorrow.", "Please ask Gotham tomorrow.", b"audio")
        self.assertFalse(self.monitor.is_tracking)

    def test_close_stops_native_capture_and_keeps_stable_label(self):
        self.stable_edit(self.paste())
        self.monitor.close()
        self.reader.capture = lambda: self.fail("capture after close")
        self.monitor.poll()
        self.assertEqual(self.corrections, [(1, "Please ask Gowtham tomorrow.")])

    def test_second_separate_word_correction_replaces_inferred_label(self):
        expected = self.paste(raw="Please ask Gotham about the Sonic release.")
        first = expected.replace("Gotham", "Gowtham")
        self.observe(first, 1)
        self.observe(first, 61)
        second = first.replace("Sonic", "Sona")
        self.observe(second, 62)
        self.assertEqual(self.corrections[-1], (1, None))
        self.observe(second, 121.99)
        self.assertEqual(self.corrections[-1], (1, None))
        self.observe(second, 122)
        self.assertEqual(self.corrections, [
            (1, "Please ask Gowtham about the Sonic release."),
            (1, None),
            (1, "Please ask Gowtham about the Sona release."),
        ])
        self.assertTrue(self.monitor.is_tracking)

    def test_further_word_edit_restarts_full_minute_before_first_label(self):
        expected = self.paste(raw="Please ask Gotham about the Sonic release.")
        first = expected.replace("Gotham", "Gowtham")
        self.observe(first, 1)
        second = first.replace("Sonic", "Sona")
        self.observe(second, 40)
        self.observe(second, 61)  # The first edit's minute has passed, but not the latest edit's.
        self.assertEqual(self.corrections, [])
        self.observe(second, 99.99)
        self.assertEqual(self.corrections, [])
        self.observe(second, 100)
        self.assertEqual(self.corrections, [(1, "Please ask Gowtham about the Sona release.")])

    def test_three_separate_lexical_corrections_are_bounded_by_total_words(self):
        raw = "Ask Gotham about Sonic before calling Ellis later today."
        three = raw.replace("Gotham", "Gowtham").replace("Sonic", "Sona").replace("Ellis", "Alice")
        self.assertTrue(_small_replacement(raw, three))
        self.assertFalse(_small_replacement(raw, three.replace("today", "tomorrow")))
        expanded = raw.replace("Gotham", "Gowtham Kumar Singh").replace("Sonic", "Sona")
        self.assertFalse(_small_replacement(raw, expanded))
        self.assertFalse(_small_replacement("Gotham and Sonic", "Gowtham and Sona"))

    def test_multiple_replacements_still_reject_insertions_deletions_and_punctuation_edits(self):
        raw = "Please ask Gotham about the Sonic release."
        corrected = raw.replace("Gotham", "Gowtham").replace("Sonic", "Sona")
        for unsafe in (corrected + " Send it today.", corrected.replace("about the ", ""),
                       corrected.replace(".", "!"), "Rewrite everything entirely."):
            with self.subTest(updated=unsafe):
                self.assertFalse(_small_replacement(raw, unsafe))

    def test_multiple_correction_labels_train_compatible_contextual_rules(self):
        from personalization import PersonalizationManager

        with tempfile.TemporaryDirectory() as directory:
            manager = PersonalizationManager(Path(directory) / "learning", enabled=True)
            for number in range(24):
                raw = f"Please ask Gotham about the Sonic release task {number}."
                corrected = raw.replace("Gotham", "Gowtham").replace("Sonic", "Sona")
                self.assertTrue(_small_replacement(raw, corrected))
                identity = manager.record(raw, raw)
                self.assertTrue(manager.auto_correct(identity, corrected))
            result = manager.train_if_due(force=True)
            self.assertEqual(result["status"], "trained")
            self.assertEqual(result["rule_count"], 2)
            self.assertEqual(manager.apply("Please ask Gotham about the Sonic release task tomorrow."),
                             "Please ask Gowtham about the Sona release task tomorrow.")


if __name__ == "__main__":
    unittest.main()
