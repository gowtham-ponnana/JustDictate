"""Bounded background persistence and monthly learning, outside the paste path."""

import logging
import queue
import threading

log = logging.getLogger(__name__)


class LearningService:
    def __init__(self, manager, *, enabled=False, retain_audio=False,
                 is_idle=lambda: True, on_update=lambda: None):
        self.manager = manager
        self.enabled = enabled
        self.retain_audio = retain_audio
        self._is_idle = is_idle
        self._on_update = on_update
        self._lock = threading.RLock()
        self._epoch = 0
        self._closed = False
        self._training_pending = False
        self.last_training_result = None
        self._correction_versions = {}
        self._correction_sequence = 0
        self._queue = queue.Queue(maxsize=16)
        self._worker = threading.Thread(target=self._run, name="local-learning", daemon=True)
        self._worker.start()

    def apply(self, raw):
        return self.manager.apply(raw) if self.enabled and not self._closed else raw

    def record(self, raw, delivered, audio, on_saved=None):
        """Audio belongs to the finished recording and must not be mutated later."""
        with self._lock:
            if not self.enabled or self._closed:
                return False
            return self._enqueue(("record", self._epoch,
                                  (raw, delivered, audio if self.retain_audio else None, on_saved)))

    def auto_correct(self, record_id, corrected):
        """Queue an inferred edit (or revocation) for the exact captured row."""
        with self._lock:
            if not self.enabled or self._closed:
                return False
            self._correction_sequence += 1
            version = self._correction_sequence
            self._correction_versions[record_id] = version
            if len(self._correction_versions) > 4096:
                self._correction_versions.pop(next(iter(self._correction_versions)))
            if corrected is None:
                # Never drop a retraction behind a full capture queue. Its new
                # version also invalidates older correction jobs still queued.
                self.manager.auto_correct(record_id, None)
            else:
                return self._enqueue(("correction", self._epoch, (record_id, corrected, version)))
        self._on_update()
        return True

    def set_enabled(self, enabled):
        with self._lock:
            self._epoch += 1
            self._correction_versions.clear()
            self.enabled = bool(enabled)
            self.manager.set_enabled(self.enabled)
        self._on_update()

    def set_retain_audio(self, enabled):
        with self._lock:
            self.retain_audio = bool(enabled)
            # Discard pending captures when the retention preference changes.
            self._epoch += 1
            self.manager.set_retain_audio(self.retain_audio)
        self._on_update()

    def reset(self):
        with self._lock:
            self._epoch += 1
            self._correction_versions.clear()
            self.last_training_result = None
            self.manager.reset()
        self._on_update()

    def schedule_training(self, *, force=False):
        # The idle predicate consults the correction monitor. Never call it
        # under this lock: the monitor can submit corrections in the opposite
        # direction, creating a lock inversion during normal edits.
        if not self._is_idle():
            return False
        with self._lock:
            if self._closed or not self.enabled or self._training_pending:
                return False
            self._training_pending = True
            if not self._enqueue(("train", self._epoch, force)):
                self._training_pending = False
                return False
            return True

    def close(self):
        """Drain accepted writes on exit; the worker never keeps the process alive."""
        with self._lock:
            self._closed = True
        self._worker.join(timeout=3)

    def _enqueue(self, item):
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            log.warning("Local learning queue is full; skipping this capture or check.")
            return False

    def _run(self):
        while True:
            try:
                kind, epoch, payload = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._closed:
                    return
                continue
            try:
                if kind == "record":
                    # Serialize against reset/disable so queued audio cannot recreate
                    # deleted history or be saved after learning has been disabled.
                    raw, delivered, audio, on_saved = payload
                    saved_id = None
                    try:
                        with self._lock:
                            if epoch == self._epoch and self.enabled:
                                saved_id = self.manager.record(raw, delivered, audio)
                    finally:
                        # Release the lock before a receipt re-enters tracking.
                        if on_saved is not None:
                            on_saved(saved_id)
                elif kind == "correction":
                    record_id, corrected, version = payload
                    with self._lock:
                        if (epoch == self._epoch and self.enabled and
                                version == self._correction_versions.get(record_id)):
                            self.manager.auto_correct(record_id, corrected)
                elif kind == "train":
                    with self._lock:
                        valid = epoch == self._epoch and self.enabled and not self._closed
                    if valid and self._is_idle():
                        result = self.manager.train_if_due(force=payload)
                        if isinstance(result, dict) and result.get("status") not in (
                                "not_due", "unchanged", "disabled", "busy"):
                            self.last_training_result = result
                self._on_update()
            except Exception:
                # Exception messages may contain transcript text or a private path.
                log.error("Local learning operation failed; dictation is unaffected.")
            finally:
                if kind == "train":
                    with self._lock:
                        self._training_pending = False
                self._queue.task_done()
