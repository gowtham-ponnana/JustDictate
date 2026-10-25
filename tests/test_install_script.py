"""Exercise the installer's real rename/rollback section in temporary folders.

These tests never invoke the installer, build an app, stop applications, or write
to /Applications. Only its extracted transaction and cleanup functions execute.
"""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


INSTALLER = Path(__file__).resolve().parents[1] / "install.sh"


class InstallTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.target = self.directory / "JustDictate.app"
        self.backup = self.directory / ".JustDictate.previous.app"
        self.stage = self.directory / "stage"
        self.stage.mkdir()
        for app, value in ((self.target, "current"), (self.backup, "previous"),
                           (self.stage / "JustDictate.app", "new")):
            app.mkdir()
            (app / "identity").write_text(value)

    def tearDown(self):
        self.temporary.cleanup()

    def transaction(self, mode):
        source = INSTALLER.read_text()
        cleanup = source[source.index("cleanup() {"):source.index("trap cleanup EXIT")]
        start = source.index('if [ -e "$target_app" ]; then\n    if [ -e "$backup_app" ]; then')
        end = source.index("committed=1", start) + len("committed=1")
        transaction = source[start:end]
        script = """set -euo pipefail
target_app="$TEST_TARGET"
backup_app="$TEST_BACKUP"
stage_dir="$TEST_STAGE"
target_moved=0
backup_moved=0
new_installed=0
committed=0
""" + cleanup + """
trap cleanup EXIT
verify_bundle() { [ "$TEST_MODE" != "verify-failure" ]; }
mv() {
    if [ "$TEST_MODE" = "rename-failure" ] && [ "$1" = "$stage_dir/JustDictate.app" ]; then
        return 1
    fi
    command mv "$@"
}
""" + transaction
        environment = dict(os.environ, TEST_TARGET=str(self.target), TEST_BACKUP=str(self.backup),
                           TEST_STAGE=str(self.stage), TEST_MODE=mode)
        return subprocess.run(["bash", "-c", script], env=environment, capture_output=True, text=True)

    def test_verified_install_retains_immediately_previous_bundle(self):
        result = self.transaction("success")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.target / "identity").read_text(), "new")
        self.assertEqual((self.backup / "identity").read_text(), "current")

    def test_failed_final_verification_restores_current_app_and_older_backup(self):
        result = self.transaction("verify-failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.target / "identity").read_text(), "current")
        self.assertEqual((self.backup / "identity").read_text(), "previous")

    def test_failed_replacement_rename_restores_both_prior_bundles(self):
        result = self.transaction("rename-failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.target / "identity").read_text(), "current")
        self.assertEqual((self.backup / "identity").read_text(), "previous")


if __name__ == "__main__":
    unittest.main()
