import threading
import unittest

from learning_service import LearningService


class MemoryManager:
    def __init__(self):
        self.records = []
        self.corrections = []
        self.training = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.block_training = False
        self.fail_record = False

    def record(self, *args):
        if self.fail_record:
            self.fail_record = False
            raise OSError("disk is full")
        self.records.append(args)
        return len(self.records)

    def auto_correct(self, record_id, corrected):
        self.corrections.append((record_id, corrected))
        return True

    def apply(self, raw):
        return raw + " personalized"

    def train_if_due(self, force=False):
        self.training += 1
        self.started.set()
        if self.block_training:
            if not self.release.wait(3):
                raise TimeoutError("test did not release worker")

    def set_enabled(self, enabled):
        pass

    def set_retain_audio(self, enabled):
        pass

    def reset(self):
        self.records.clear()


class LearningServiceTests(unittest.TestCase):
    def setUp(self):
        self.manager = MemoryManager()
        self.idle = True
        self.service = LearningService(self.manager, enabled=True,
                                       is_idle=lambda: self.idle)

    def tearDown(self):
        self.manager.release.set()
        self.service.close()
        self.assertFalse(self.service._worker.is_alive())

    def block_worker(self):
        self.manager.block_training = True
        self.assertTrue(self.service.schedule_training(force=True))
        self.assertTrue(self.manager.started.wait(2))

    def drain(self):
        self.manager.release.set()
        self.service._queue.join()

    def test_reset_drops_pre_reset_captures_but_accepts_future_dictations(self):
        self.block_worker()
        self.service.record("before", "before", object())
        self.service.reset()
        self.service.record("after", "after", None)
        self.drain()
        self.assertEqual(self.manager.records, [("after", "after", None)])

    def test_disable_invalidates_pending_capture_and_apply_immediately(self):
        self.block_worker()
        self.service.record("private", "private", None)
        self.service.set_enabled(False)
        self.assertEqual(self.service.apply("text"), "text")
        self.assertFalse(self.service.record("new", "new", None))
        self.drain()
        self.assertEqual(self.manager.records, [])

    def test_queue_is_bounded_and_never_blocks_paste_producer(self):
        self.block_worker()
        for i in range(16):
            self.assertTrue(self.service.record(str(i), str(i), None))
        self.assertFalse(self.service.record("overflow", "overflow", None))
        self.drain()
        self.assertEqual(len(self.manager.records), 16)

    def test_audio_is_discarded_unless_separately_enabled(self):
        audio = object()
        self.service.record("one", "one", audio)
        self.service._queue.join()
        self.assertIsNone(self.manager.records[0][2])
        self.service.set_retain_audio(True)
        self.service.record("two", "two", audio)
        self.service._queue.join()
        self.assertIs(self.manager.records[1][2], audio)

    def test_capture_failure_does_not_kill_worker_or_later_dictation(self):
        self.manager.fail_record = True
        self.service.record("one", "one", None)
        self.service.record("two", "two", None)
        self.service._queue.join()
        self.assertEqual(self.manager.records, [("two", "two", None)])

    def test_training_deduplicates_checks_and_waits_until_idle(self):
        self.idle = False
        self.assertFalse(self.service.schedule_training())
        self.idle = True
        self.block_worker()
        self.assertFalse(self.service.schedule_training(force=True))
        self.drain()
        self.assertEqual(self.manager.training, 1)

    def test_close_drains_accepted_writes_and_rejects_new_work(self):
        self.service.record("one", "one", None)
        self.service.close()
        self.assertEqual(self.manager.records, [("one", "one", None)])
        self.assertFalse(self.service.record("two", "two", None))
        self.assertFalse(self.service.schedule_training())

    def test_record_receipt_binds_the_saved_row_not_latest(self):
        receipts = []
        self.service.record("one", "one", None, lambda identity: receipts.append(("one", identity)))
        self.service.record("two", "two", None, lambda identity: receipts.append(("two", identity)))
        self.service._queue.join()
        self.assertEqual(receipts, [("one", 1), ("two", 2)])

    def test_retraction_cannot_be_overwritten_by_older_queued_label(self):
        self.block_worker()
        self.assertTrue(self.service.auto_correct(3, "Temporary edit"))
        self.assertTrue(self.service.auto_correct(3, None))
        self.assertEqual(self.manager.corrections, [(3, None)])
        self.drain()
        self.assertEqual(self.manager.corrections, [(3, None)])

    def test_retraction_is_processed_even_when_queue_is_full(self):
        self.block_worker()
        for i in range(16):
            self.service.record(str(i), str(i), None)
        self.assertTrue(self.service.auto_correct(3, None))
        self.assertEqual(self.manager.corrections, [(3, None)])
        self.drain()

    def test_record_failure_delivers_empty_receipt(self):
        self.manager.fail_record = True
        receipts = []
        self.service.record("one", "one", None, receipts.append)
        self.service._queue.join()
        self.assertEqual(receipts, [None])


if __name__ == "__main__":
    unittest.main()
