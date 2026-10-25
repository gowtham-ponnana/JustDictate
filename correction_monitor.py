"""Infer bounded edits to our own pasted span; keep document context in RAM.

Only exact paste verification followed by a small, stable replacement can create
an inferred correction. Unchanged output is never a training label. The separate
personalization trainer still requires repeated evidence and held-out validation.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
import logging
import re
import threading
import time

log = logging.getLogger(__name__)
_TOKEN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
MAX_DOCUMENT_CHARS = 100_000
MAX_SPAN_CHARS = 16_000
MAX_SPAN_TOKENS = 256
MAX_CONTEXTS = 8
TRACK_SECONDS = 15 * 60
PASTE_VERIFY_SECONDS = 5
# Wait a full minute after the most recently observed edit before saving a label.
STABLE_SECONDS = 60


@dataclass(frozen=True)
class TextSnapshot:
    element: object
    value: str
    selection: tuple[int, int]  # Python character offsets: (start, length)


@dataclass(eq=False)
class _Context:
    element: object
    prefix: str
    suffix: str
    pasted: str
    expected: str
    created: float
    epoch: int
    record_id: int | None = None
    verified: bool = False
    last_value: str | None = None
    candidate: str | None = None
    changed_at: float = 0
    labelled: str | None = None
    invalid: bool = False


def _same_element(first, second):
    try:
        return bool(first == second)
    except Exception:
        return False


def _small_replacement(original, updated):
    """Allow at most three replaced words across three lexical spans.

    Insertions/deletions and full rewrites are deliberately ambiguous. Whitespace
    alone and unchanged recognition cannot be used as evidence of correctness.
    """
    if max(len(original), len(updated)) > MAX_SPAN_CHARS:
        return False
    before, after = _TOKEN.findall(original), _TOKEN.findall(updated)
    if max(len(before), len(after)) > MAX_SPAN_TOKENS:
        return False
    if before == after or not before or not after:
        return False
    operations = SequenceMatcher(
        None, before, after, autojunk=False
    ).get_opcodes()
    changes = [operation for operation in operations if operation[0] != "equal"]
    if not 1 <= len(changes) <= 3:
        return False
    lexical = lambda token: any(character.isalnum() for character in token)
    changed_words = 0
    replacement_budget = 0
    for tag, i, j, a, b in changes:
        if (tag != "replace" or not (1 <= j - i <= 3 and 1 <= b - a <= 3) or
                not all(lexical(token) for token in before[i:j] + after[a:b])):
            return False
        changed_words += j - i
        # A one-word recognition expanded to three words consumes the full
        # budget; counting source words alone would permit a much larger edit.
        replacement_budget += max(j - i, b - a)
    total_words = sum(lexical(token) for token in before)
    unchanged_words = sum(lexical(token) for tag, i, j, _a, _b in operations
                          if tag == "equal" for token in before[i:j])
    return (0 < replacement_budget <= 3 and unchanged_words > 0 and
            changed_words <= max(1, total_words // 2))


class CorrectionMonitor:
    def __init__(self, reader, record_fn, correction_fn, enabled_fn,
                 on_update=lambda: None, *, clock=time.monotonic):
        self.reader = reader
        self._record = record_fn
        self._correct = correction_fn
        self._enabled = enabled_fn
        self._on_update = on_update
        self._clock = clock
        self._lock = threading.RLock()
        self._contexts = []
        self._prepared = None
        self._epoch = 0
        self._stop = threading.Event()
        self._thread = None

    @property
    def is_tracking(self):
        with self._lock:
            self._expire(self._clock())
            return self._prepared is not None or bool(self._contexts)

    def start(self):
        with self._lock:
            if self._thread is not None or self._stop.is_set():
                return
            self._thread = threading.Thread(target=self._run, name="correction-monitor", daemon=True)
            self._thread.start()

    def _run(self):
        while not self._stop.wait(0.5):
            try:
                self.poll()
            except Exception:
                # Never log document values or exceptions containing AX data.
                log.warning("Automatic correction observation failed.")

    def _capture(self):
        try:
            snapshot = self.reader.capture()
            if not isinstance(snapshot, TextSnapshot) or not isinstance(snapshot.value, str):
                return None
            start, length = snapshot.selection
            if (snapshot.element is None or len(snapshot.value) > MAX_DOCUMENT_CHARS or
                    not isinstance(start, int) or not isinstance(length, int) or
                    start < 0 or length < 0 or start + length > len(snapshot.value)):
                return None
            return snapshot
        except Exception:
            return None

    def prepare(self, text):
        """Capture the target immediately before paste; no document data is saved."""
        # The engine serializes prepare/complete. Clearing this atomic reference
        # before a nonblocking acquisition also invalidates an older preparation
        # when an AX observation or retraction is currently using the lock.
        self._prepared = None
        if not self._lock.acquire(blocking=False):
            return
        try:
            if (self._stop.is_set() or not self._enabled() or not isinstance(text, str) or
                    not text.strip() or len(text) > MAX_SPAN_CHARS or
                    len(_TOKEN.findall(text)) > MAX_SPAN_TOKENS):
                return
            snapshot = self._capture()
            if snapshot is None:
                # The native reader supplies fixed diagnostic codes, never
                # document contents. Emit once per dictation, not on each poll.
                log.info("Edit tracking skipped: %s.",
                         getattr(self.reader, "last_rejection", None) or "snapshot_unavailable")
                return
            start, length = snapshot.selection
            prefix, suffix = snapshot.value[:start], snapshot.value[start + length:]
            # A new known insertion finalizes an older stable edit in this field.
            # Other fields remain observable when the user returns to them.
            for context in tuple(self._contexts):
                if _same_element(context.element, snapshot.element):
                    self._drop(context, revoke=False)
            self._prepared = _Context(
                snapshot.element, prefix, suffix, text, prefix + text + suffix,
                self._clock(), self._epoch,
            )
        finally:
            self._lock.release()

    def complete(self, raw, delivered, audio):
        """Persist this result, then bind observation to its asynchronously saved ID."""
        with self._lock:
            context, self._prepared = self._prepared, None
            if self._stop.is_set():
                return False
            if not self._enabled():
                # History capture is an independent preference. The recording
                # service decides whether to retain it when edit learning is off.
                context = None
            if context is not None:
                if context.epoch != self._epoch or context.pasted.rstrip() != delivered.rstrip():
                    context = None
                else:
                    while len(self._contexts) >= MAX_CONTEXTS:
                        self._drop(self._contexts[0], revoke=False)
                    self._contexts.append(context)
            epoch = self._epoch

            def on_saved(record_id):
                with self._lock:
                    # Reset can reuse SQLite row IDs: both epoch and context
                    # identity must still match before binding a delayed save.
                    if (epoch != self._epoch or self._stop.is_set() or
                            context is None or context not in self._contexts or
                            context.record_id is not None):
                        return
                    if isinstance(record_id, int) and record_id > 0:
                        context.record_id = record_id
                    else:
                        self._drop(context, revoke=False)

        # Persistence has its own queue/lock; never acquire it while holding the
        # monitor lock (the completion callback can arrive on another thread).
        try:
            accepted = self._record(raw, delivered, audio, on_saved)
        except Exception:
            accepted = False
            log.warning("Dictation could not be queued for local learning.")
        if accepted is False and context is not None:
            with self._lock:
                self._drop(context, revoke=False)
        return accepted is not False

    def _revoke(self, context):
        """The integration must synchronously process retractions, never drop them."""
        if context.labelled is not None and context.record_id is not None:
            if self._correct(context.record_id, None) is False:
                return False
            context.labelled = None
        return True

    def _drop(self, context, *, revoke):
        if revoke:
            if not self._revoke(context):
                return False
        if context in self._contexts:
            self._contexts.remove(context)
        return True

    def _expire(self, now):
        if self._prepared is not None and now - self._prepared.created >= PASTE_VERIFY_SECONDS:
            self._prepared = None
        for context in tuple(self._contexts):
            # Expiration finalizes an already stable inferred label, but never
            # invents a label for an unobserved or unchanged value.
            if (now - context.created >= TRACK_SECONDS or
                    (not context.verified and now - context.created >= PASTE_VERIFY_SECONDS)):
                self._drop(context, revoke=False)

    def poll(self, now=None):
        now = self._clock() if now is None else now
        with self._lock:
            if self._stop.is_set():
                return
            if not self._enabled():
                self.cancel()
                return
            self._expire(now)
            if not self._contexts:
                return
            snapshot = self._capture()
            if snapshot is None:
                return
            for context in tuple(self._contexts):
                if not _same_element(context.element, snapshot.element):
                    continue
                value = snapshot.value
                if not context.verified:
                    if value == context.expected:
                        context.verified = True
                        context.last_value = value
                    return
                if value != context.last_value:
                    if not self._revoke(context):
                        return
                    context.last_value = value
                    context.changed_at = now
                    context.candidate = None
                    context.invalid = False
                    if (not value.startswith(context.prefix) or not value.endswith(context.suffix) or
                            len(value) < len(context.prefix) + len(context.suffix)):
                        self._drop(context, revoke=False)
                        return
                    end = len(value) - len(context.suffix) if context.suffix else len(value)
                    edited = value[len(context.prefix):end]
                    if edited.strip() == context.pasted.strip():
                        return
                    if not _small_replacement(context.pasted, edited):
                        # Selecting a word and typing its replacement can briefly
                        # look like deletion. Wait for the same debounce period
                        # before abandoning an internally ambiguous span.
                        context.invalid = True
                        return
                    context.candidate = edited.strip()
                elif context.invalid and now - context.changed_at >= STABLE_SECONDS:
                    self._drop(context, revoke=False)
                elif (context.candidate is not None and context.record_id is not None and
                      context.labelled != context.candidate and now - context.changed_at >= STABLE_SECONDS):
                    accepted = self._correct(context.record_id, context.candidate)
                    if accepted is not False:
                        context.labelled = context.candidate
                        self._on_update()
                return

    def cancel(self):
        """Undo/reset/disable invalidates all pending observations and inferred labels."""
        with self._lock:
            self._epoch += 1
            self._prepared = None
            for context in tuple(self._contexts):
                self._drop(context, revoke=True)
            self._on_update()

    def close(self):
        """Stop observation and preserve previously stable inferred corrections."""
        self._stop.set()
        with self._lock:
            self._epoch += 1
            self._prepared = None
            self._contexts.clear()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1)
