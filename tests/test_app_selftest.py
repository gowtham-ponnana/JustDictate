import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from app_selftest import SelfTestFailure, _exercise_pipeline, _require, run_selftest


class SelfTestReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "report.json"

    def report(self):
        return json.loads(self.path.read_text())

    def test_success_requires_checks_and_produces_absolute_private_report(self):
        def checks(report):
            _require(report, "example", True, count=18)
            report["permissions"] = {
                "accessibility": True, "input_monitoring": True, "microphone": "authorized",
            }

        with patch("app_selftest._perform_checks", checks):
            self.assertEqual(run_selftest(self.path), 0)
        report = self.report()
        self.assertTrue(report["passed"])
        self.assertTrue(Path(report["report_path"]).is_absolute())
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertTrue(report["readiness"]["permissions_ready"])
        self.assertFalse(report["scope"]["live_hardware_end_to_end"])

    def test_failed_check_sets_nonzero_exit_and_false_report(self):
        with patch("app_selftest._perform_checks", lambda report: _require(report, "missing_model", False)):
            self.assertEqual(run_selftest(self.path), 1)
        report = self.report()
        self.assertFalse(report["passed"])
        self.assertEqual(report["checks"], [{"name": "missing_model", "passed": False}])

    def test_unexpected_error_message_is_not_exposed(self):
        with patch("app_selftest._perform_checks", side_effect=ValueError("private field contents")):
            self.assertEqual(run_selftest(self.path), 1)
        report = self.report()
        self.assertEqual(report["error"], {"type": "ValueError"})
        self.assertNotIn("private field contents", self.path.read_text())

    def test_no_checks_cannot_be_reported_as_pass(self):
        with patch("app_selftest._perform_checks", lambda report: None):
            self.assertEqual(run_selftest(self.path), 1)

    def test_missing_permissions_do_not_claim_live_readiness(self):
        def checks(report):
            _require(report, "runtime", True)
            report["permissions"] = {
                "accessibility": False, "input_monitoring": True, "microphone": "unknown",
            }

        with patch("app_selftest._perform_checks", checks):
            self.assertEqual(run_selftest(self.path), 0)
        self.assertFalse(self.report()["readiness"]["permissions_ready"])
        self.assertFalse(self.report()["readiness"]["live_microphone_and_target_app_tested"])


class SelfTestPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "history"

    def test_real_learning_pipeline_captures_trains_purges_and_reopens(self):
        class Clipboard:
            def __init__(self):
                self.writes = []

            def write(self, text):
                self.writes.append(text)

        clipboard = Clipboard()
        report = {"checks": []}
        # The package smoke substitutes an actual cached-model callable here;
        # this unit test isolates the real monitor/service/storage lifecycle.
        import config_manager
        with patch("config_manager.load", side_effect=AssertionError("user config accessed")) as original:
            _exercise_pipeline(report, np.zeros(16000, dtype=np.float32),
                               lambda audio: "A synthetic smoke test.", clipboard, self.directory)
            self.assertIs(config_manager.load, original)
        self.assertEqual(len(clipboard.writes), 19)
        self.assertTrue(all(check["passed"] for check in report["checks"]))
        by_name = {check["name"]: check for check in report["checks"]}
        self.assertEqual(by_name["automatic_edit_capture"]["automatic_labels"], 18)
        self.assertEqual(by_name["used_history_and_audio_deleted"]["purged_records"], 18)
        self.assertEqual(by_name["used_history_and_audio_deleted"]["remaining_wav_files"], 1)

    def test_empty_recognition_fails_instead_of_claiming_runtime_success(self):
        class Clipboard:
            def write(self, text):
                pass

        report = {"checks": []}
        with self.assertRaises(SelfTestFailure):
            _exercise_pipeline(report, np.zeros(16000, dtype=np.float32),
                               lambda audio: "", Clipboard(), self.directory)
        self.assertFalse(report["checks"][-1]["passed"])


if __name__ == "__main__":
    unittest.main()
