"""Private, local correction learning; never modifies the speech model.

Manual reviews and conservatively captured edits train this model. It learns short
substitutions with their immediate token context, not free-form sentence generation.
A grouped holdout must improve before a candidate replaces the active model. Used
training transcripts/audio are then deleted. Audio is optional reference material
and is not used by this text correction trainer.
"""

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
from types import MappingProxyType
import unicodedata
import uuid
import wave


MIN_REVIEWED = 12
MIN_SUPPORT = 2
MIN_VALIDATION = 3
MAX_RULE_TOKENS = 3
MAX_RULES = 500
MAX_RULE_CHARS = 128
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_RECORDS = 2000
MAX_TEXT_CHARS = 16000
MAX_TRAIN_TOKENS = 256
MAX_AUDIO_SECONDS = 120
MAX_AUDIO_BYTES = 256 * 1024 * 1024
SAMPLE_RATE = 16000
TOKEN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)


def _now():
    return datetime.now().astimezone()


def _next_month(month):
    year, number = map(int, month.split("-"))
    return f"{year + (number == 12):04d}-{number % 12 + 1:02d}"


def _normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@dataclass(frozen=True)
class Rule:
    source: tuple[str, ...]
    replacement: str
    left: str | None
    right: str | None
    support: int


def _index(rules):
    index = defaultdict(list)
    for rule in rules:
        index[rule.source[0]].append(rule)
    return MappingProxyType({
        key: tuple(sorted(value, key=lambda rule: (-len(rule.source), -rule.support)))
        for key, value in index.items()
    })


def _apply(text, index):
    if not index or len(text) > MAX_TEXT_CHARS:
        return text
    spans = list(TOKEN.finditer(text))
    tokens = [span.group() for span in spans]
    output = []
    cursor = 0
    i = 0
    while i < len(tokens):
        matched = False
        for rule in index.get(tokens[i], ()):
            end = i + len(rule.source)
            if tuple(tokens[i:end]) != rule.source:
                continue
            left = tokens[i - 1] if i else None
            right = tokens[end] if end < len(tokens) else None
            if left != rule.left or right != rule.right:
                continue
            output.extend((text[cursor:spans[i].start()], rule.replacement))
            cursor = spans[end - 1].end()
            i = end
            matched = True
            break
        if not matched:
            i += 1
    output.append(text[cursor:])
    return "".join(output)


def _word_errors(reference, hypothesis):
    # Include case and punctuation: changing a confirmed spelling must count.
    reference = TOKEN.findall(reference)
    hypothesis = TOKEN.findall(hypothesis)
    row = list(range(len(hypothesis) + 1))
    for i, word in enumerate(reference, 1):
        next_row = [i]
        for j, other in enumerate(hypothesis, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1,
                                row[j - 1] + (word != other)))
        row = next_row
    return row[-1]


def _proposals(raw, corrected):
    source = list(TOKEN.finditer(raw))
    target = list(TOKEN.finditer(corrected))
    before, after = [m.group() for m in source], [m.group() for m in target]
    for tag, i, j, a, b in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if tag != "replace" or not (1 <= j - i <= MAX_RULE_TOKENS and
                                     1 <= b - a <= MAX_RULE_TOKENS):
            continue
        left = before[i - 1] if i else None
        right = before[j] if j < len(before) else None
        # Require a lexical neighbor; a bare name alone is insufficient context.
        if not any(value and any(c.isalnum() for c in value) for value in (left, right)):
            continue
        if not all(any(c.isalnum() for c in token) for token in before[i:j] + after[a:b]):
            continue
        replacement = corrected[target[a].start():target[b - 1].end()]
        if (any(len(token) > MAX_RULE_CHARS for token in before[i:j] + after[a:b]) or
                any(value and len(value) > MAX_RULE_CHARS for value in (left, right)) or
                len(replacement) > MAX_RULE_CHARS * MAX_RULE_TOKENS + 16):
            continue
        yield (tuple(before[i:j]), left, right), replacement


def _learn(rows):
    votes = defaultdict(lambda: defaultdict(set))
    for row in rows:
        for key, replacement in _proposals(row["raw"], row["corrected"]):
            votes[key][replacement].add(_normalized(row["raw"]))
    proposals = []
    for (source, left, right), replacements in votes.items():
        # Conflicting target spellings never become an automatic replacement.
        if len(replacements) != 1:
            continue
        replacement, supporting = next(iter(replacements.items()))
        if len(supporting) < MIN_SUPPORT:
            continue
        proposals.append(Rule(source, replacement, left, right, len(supporting)))
    # Bound training work as well as the resulting model. Prefer rules with the
    # strongest independent support before spending time checking conflicts.
    proposals = sorted(proposals, key=lambda rule: (-rule.support, rule.source,
                                                   rule.replacement))[:MAX_RULES]
    return _nonharmful_rules(proposals, rows)


def _nonharmful_rules(proposals, rows):
    """Filter rules against new labels without rescanning every row per rule."""
    applicable = {(rule.source, rule.left, rule.right): set() for rule in proposals}
    lengths = defaultdict(set)
    for rule in proposals:
        lengths[rule.source[0]].add(len(rule.source))
    # Tokenize each row once to locate affected contexts. Scanning every row and
    # retokenizing it for every candidate made full retained histories expensive.
    for row_number, row in enumerate(rows):
        tokens = TOKEN.findall(row["raw"])
        for start, token in enumerate(tokens):
            for length in lengths.get(token, ()):
                end = start + length
                if end > len(tokens):
                    continue
                key = (tuple(tokens[start:end]), tokens[start - 1] if start else None,
                       tokens[end] if end < len(tokens) else None)
                if key in applicable:
                    applicable[key].add(row_number)
    candidates = []
    base_errors = {}
    for rule in proposals:
        index = _index((rule,))
        # Confirmed correct text and conflicting examples veto harmful edits.
        harmful = False
        for row_number in applicable[(rule.source, rule.left, rule.right)]:
            row = rows[row_number]
            updated = _apply(row["raw"], index)
            if row_number not in base_errors:
                base_errors[row_number] = _word_errors(row["corrected"], row["raw"])
            if _word_errors(row["corrected"], updated) > base_errors[row_number]:
                harmful = True
                break
        if harmful:
            continue
        candidates.append(rule)
    return tuple(candidates)


def _incremental_rules(previous, learned, rows):
    retained = _nonharmful_rules(previous, rows)
    new_by_context = {(rule.source, rule.left, rule.right): rule for rule in learned}
    combined = []
    for rule in retained:
        key = (rule.source, rule.left, rule.right)
        combined.append(new_by_context.pop(key, rule))
    # Existing, uncontradicted learning takes precedence when capacity is full.
    combined.extend(new_by_context.values())
    return tuple(combined[:MAX_RULES])


def _behavior(rules):
    return frozenset((rule.source, rule.left, rule.right, rule.replacement) for rule in rules)


def _eligible(row):
    return (len(TOKEN.findall(row["raw"])) <= MAX_TRAIN_TOKENS and
            len(TOKEN.findall(row["corrected"])) <= MAX_TRAIN_TOKENS)


def _groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        if _eligible(row):
            grouped[_normalized(row["raw"])].append(row)
    result = []
    for key, values in grouped.items():
        labels = {" ".join(value["corrected"].split()) for value in values}
        row = dict(values[0])
        # Contradictory labels become an abstention example, not extra support.
        if len(labels) != 1:
            row["corrected"] = row["raw"]
        row["group"] = key
        result.append(row)
    return sorted(result, key=lambda row: hashlib.sha256(row["group"].encode()).digest())


def _decode_rules(model):
    if (not isinstance(model, dict) or not isinstance(model.get("rules"), list) or
            len(model["rules"]) > MAX_RULES or type(model.get("version")) is not int or
            model["version"] < 0 or
            (model.get("trained_at") is not None and not isinstance(model["trained_at"], str))):
        raise ValueError("Invalid personalization model")
    result = []
    for item in model["rules"]:
        if not isinstance(item, dict) or not isinstance(item.get("source"), (list, tuple)):
            raise ValueError("Invalid personalization model")
        source = tuple(item["source"])
        if not 1 <= len(source) <= MAX_RULE_TOKENS or not all(
                isinstance(s, str) and 0 < len(s) <= MAX_RULE_CHARS and TOKEN.fullmatch(s) for s in source):
            raise ValueError("Invalid personalization model")
        replacement = item["replacement"]
        if (not isinstance(replacement, str) or not replacement.strip() or
                len(replacement) > MAX_RULE_CHARS * MAX_RULE_TOKENS + 16 or
                not 1 <= len(TOKEN.findall(replacement)) <= MAX_RULE_TOKENS or
                type(item.get("support")) is not int or item["support"] < MIN_SUPPORT or
                any(value is not None and (not isinstance(value, str) or not value or
                                           len(value) > MAX_RULE_CHARS or not TOKEN.fullmatch(value))
                    for value in (item["left"], item["right"]))):
            raise ValueError("Invalid personalization model")
        if not any(value and any(c.isalnum() for c in value) for value in (item["left"], item["right"])):
            raise ValueError("Invalid personalization model context")
        result.append(Rule(source, replacement, item["left"], item["right"], item["support"]))
    return tuple(result)


class PersonalizationManager:
    def __init__(self, directory: Path, enabled=False, retain_audio=False, retention_days=90):
        self.directory = Path(directory)
        self.database = self.directory / "history.sqlite3"
        self.audio_directory = self.directory / "audio"
        self.enabled = bool(enabled)
        self.retain_audio = bool(retain_audio)
        self.retention_days = max(1, int(retention_days))
        self._lock = threading.RLock()
        self._generation = 0
        self._training = False
        self._unavailable = False
        self._model = {"version": 0, "rules": [], "trained_at": None}
        self._index = _index(())
        if self.enabled or self.database.exists():
            try:
                with self._lock, self._connection() as connection:
                    model = self._get(connection, "model", self._model)
                    index = _index(_decode_rules(model))
                    self._prune(connection)
                    self._model, self._index = model, index
            except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, AttributeError):
                # Preserve unreadable data for inspection or explicit deletion.
                # A broken optional learning store must never stop dictation.
                self._unavailable = True

    @contextmanager
    def _connection(self):
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.directory.chmod(0o700)
        fd = os.open(self.database, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
        self.database.chmod(0o600)
        connection = sqlite3.connect(self.database, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA secure_delete = ON")
            with connection:
                # Include schema migrations in rollback, not just subsequent DML.
                connection.execute("BEGIN")
                connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                connection.execute("""CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL,
                    raw TEXT NOT NULL, delivered TEXT NOT NULL, corrected TEXT,
                    audio_name TEXT, label_source TEXT)""")
                columns = {row[1] for row in connection.execute("PRAGMA table_info(samples)")}
                if "label_source" not in columns:
                    connection.execute("ALTER TABLE samples ADD COLUMN label_source TEXT")
                    connection.execute("UPDATE samples SET label_source = 'manual' WHERE corrected IS NOT NULL")
                if self._get(connection, "enrolled_month") is None:
                    self._put(connection, "enrolled_month", _now().strftime("%Y-%m"))
                    self._put(connection, "revision", 0)
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _get(connection, key, default=None):
        row = connection.execute("SELECT value FROM meta WHERE key = ? AND length(value) <= ?",
                                 (key, MAX_METADATA_BYTES)).fetchone()
        if row is None and connection.execute("SELECT 1 FROM meta WHERE key = ?", (key,)).fetchone():
            raise ValueError("Oversized personalization metadata")
        return json.loads(row[0]) if row else default

    @staticmethod
    def _put(connection, key, value):
        connection.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))

    def _changed(self, connection, training=True):
        self._generation += 1
        if training:
            self._put(connection, "revision", self._get(connection, "revision", 0) + 1)

    def _remove_audio(self, name):
        if name and Path(name).name == name:
            (self.audio_directory / name).unlink(missing_ok=True)

    def _compact_database(self):
        # DELETE releases SQLite pages for reuse, but VACUUM returns their space
        # to the filesystem. Open only an existing file and run outside a
        # transaction, after the accepted model and row purge have committed.
        connection = sqlite3.connect(self.database.resolve().as_uri() + "?mode=rw", uri=True, timeout=5)
        try:
            connection.execute("VACUUM")
        finally:
            connection.close()

    def _prune(self, connection):
        cutoff = (_now() - timedelta(days=self.retention_days)).timestamp()
        rows = connection.execute("""SELECT id, audio_name, corrected FROM samples WHERE created < ?
            OR id NOT IN (SELECT id FROM samples ORDER BY id DESC LIMIT ?)""",
                                  (cutoff, MAX_RECORDS)).fetchall()
        if rows:
            for row in rows:
                self._remove_audio(row["audio_name"])
            connection.executemany("DELETE FROM samples WHERE id = ?", [(row["id"],) for row in rows])
            self._changed(connection, training=any(row["corrected"] is not None for row in rows))
        referenced = {row[0] for row in connection.execute(
            "SELECT audio_name FROM samples WHERE audio_name IS NOT NULL")}
        files = []
        for path in self.audio_directory.glob("*.wav"):
            if path.name not in referenced:
                # A process killed between saving WAV and committing its row can
                # leave an orphan; it must not outlive the history retention policy.
                path.unlink(missing_ok=True)
            else:
                files.append(path)
        files.sort(key=lambda path: path.stat().st_mtime)
        total = sum(path.stat().st_size for path in files)
        for path in files:
            if total <= MAX_AUDIO_BYTES:
                break
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total -= size
            connection.execute("UPDATE samples SET audio_name = NULL WHERE audio_name = ?", (path.name,))

    def set_enabled(self, enabled):
        with self._lock:
            self.enabled = bool(enabled)
            self._generation += 1
            if self.enabled and not self._unavailable:
                with self._connection() as connection:
                    self._prune(connection)

    def set_retain_audio(self, retain_audio):
        with self._lock:
            self.retain_audio = bool(retain_audio)
            self._generation += 1

    def _save_audio(self, audio):
        import numpy as np

        audio = np.asarray(audio)
        if audio.ndim != 1 or not 0 < len(audio) <= SAMPLE_RATE * MAX_AUDIO_SECONDS:
            return None
        if not np.all(np.isfinite(audio)):
            return None
        self.audio_directory.mkdir(mode=0o700, exist_ok=True)
        self.audio_directory.chmod(0o700)
        name = uuid.uuid4().hex + ".wav"
        path = self.audio_directory / name
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
        try:
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
                with wave.open(stream, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(SAMPLE_RATE)
                    wav.writeframes(pcm.tobytes())
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return name

    def record(self, raw, delivered, audio=None):
        if not isinstance(raw, str) or not isinstance(delivered, str):
            raise TypeError("Transcripts must be text")
        if not raw.strip() or max(len(raw), len(delivered)) > MAX_TEXT_CHARS:
            return None
        with self._lock:
            if not self.enabled or self._unavailable:
                return None
            audio_name = None
            try:
                with self._connection() as connection:
                    if self.retain_audio and audio is not None:
                        audio_name = self._save_audio(audio)
                    cursor = connection.execute(
                        "INSERT INTO samples (created, raw, delivered, audio_name) VALUES (?, ?, ?, ?)",
                        (_now().timestamp(), raw.strip(), delivered.strip(), audio_name))
                    self._changed(connection, training=False)
                    self._prune(connection)
                    return cursor.lastrowid
            except Exception:
                self._remove_audio(audio_name)
                raise

    def recent(self, limit=20):
        with self._lock:
            if self._unavailable or not self.database.exists():
                return []
            with self._connection() as connection:
                self._prune(connection)
                rows = connection.execute("SELECT * FROM samples ORDER BY id DESC LIMIT ?",
                                          (max(1, min(int(limit), MAX_RECORDS)),)).fetchall()
                return [dict(row) | {"reviewed": row["corrected"] is not None} for row in rows]

    def latest(self):
        rows = self.recent(1)
        return rows[0] if rows else None

    def review(self, record_id, corrected):
        if not isinstance(corrected, str) or not corrected.strip() or len(corrected) > MAX_TEXT_CHARS:
            raise ValueError("A nonempty transcript of at most 16000 characters is required")
        with self._lock:
            if not self.enabled or self._unavailable or not self.database.exists():
                return False
            with self._connection() as connection:
                cursor = connection.execute("UPDATE samples SET corrected = ?, label_source = 'manual' WHERE id = ?",
                                            (corrected.strip(), record_id))
                if cursor.rowcount:
                    self._changed(connection)
                    return True
                return False

    def auto_correct(self, record_id, corrected):
        """Label the exact captured dictation; automatic edits never override review.

        The editor monitor is responsible for proving the edit belongs to this
        dictation and is a small, stable correction. None revokes an automatic
        label when that observation later becomes ambiguous or is undone.
        """
        if corrected is not None and (not isinstance(corrected, str) or not corrected.strip() or
                                      len(corrected) > MAX_TEXT_CHARS):
            raise ValueError("A nonempty transcript of at most 16000 characters is required")
        with self._lock:
            if not self.enabled or self._unavailable or not self.database.exists():
                return False
            with self._connection() as connection:
                row = connection.execute("SELECT corrected, label_source FROM samples WHERE id = ?", (record_id,)).fetchone()
                if row is None or row["label_source"] == "manual":
                    return False
                value = corrected.strip() if corrected is not None else None
                source = "automatic" if value is not None else None
                if row["corrected"] == value and row["label_source"] == source:
                    return False
                connection.execute("UPDATE samples SET corrected = ?, label_source = ? WHERE id = ?",
                                   (value, source, record_id))
                self._changed(connection)
                return True

    def delete_record(self, record_id):
        with self._lock:
            if self._unavailable or not self.database.exists():
                return False
            with self._connection() as connection:
                row = connection.execute("SELECT audio_name FROM samples WHERE id = ?", (record_id,)).fetchone()
                if row is None:
                    return False
                self._remove_audio(row[0])
                connection.execute("DELETE FROM samples WHERE id = ?", (record_id,))
                self._changed(connection)
                return True

    def apply(self, raw):
        # Attribute snapshots and immutable index avoid SQLite/worker locks here.
        if not self.enabled or self._unavailable:
            return raw
        generation = self._generation
        output = _apply(raw, self._index)
        return output if self.enabled and not self._unavailable and generation == self._generation else raw

    def status(self):
        with self._lock:
            status = {"enabled": self.enabled, "retain_audio": self.retain_audio,
                      "unavailable": self._unavailable,
                      "reason": "Local personalization data is unreadable; reset to start fresh" if self._unavailable else None,
                      "training": self._training, "record_count": 0, "reviewed_count": 0,
                      "manual_count": 0, "automatic_count": 0,
                      "rule_count": len(self._model.get("rules", [])),
                      "last_trained": self._model.get("trained_at"),
                      "version": self._model.get("version", 0), "can_rollback": False,
                      "next_training_month": None, "last_attempt": None,
                      "metrics": self._model.get("metrics")}
            if self._unavailable or not self.database.exists():
                return status
            with self._connection() as connection:
                self._prune(connection)
                (status["record_count"], status["reviewed_count"], status["manual_count"],
                 status["automatic_count"]) = connection.execute(
                    """SELECT COUNT(*), COUNT(corrected),
                    COUNT(CASE WHEN label_source = 'manual' THEN 1 END),
                    COUNT(CASE WHEN label_source = 'automatic' THEN 1 END) FROM samples""").fetchone()
                base_month = self._get(connection, "completed_month", self._get(connection, "enrolled_month"))
                status["next_training_month"] = _next_month(base_month)
                status["can_rollback"] = self._get(connection, "previous_model") is not None
                status["last_attempt"] = self._get(connection, "last_attempt")
            return status

    def train_if_due(self, force=False):
        with self._lock:
            if self._unavailable:
                return {"status": "unavailable", "reason": "Local personalization data is unreadable; reset to start fresh"}
            if not self.enabled:
                return {"status": "disabled", "reason": "Personalization is off"}
            if self._training:
                return {"status": "busy", "reason": "Training is already running"}
            with self._connection() as connection:
                self._prune(connection)
                month = _now().strftime("%Y-%m")
                base_month = self._get(connection, "completed_month", self._get(connection, "enrolled_month"))
                if not force and month <= base_month:
                    return {"status": "not_due", "reason": "Next monthly update is not due"}
                revision = self._get(connection, "revision", 0)
                attempt = self._get(connection, "last_attempt", {})
                if not force and attempt.get("month") == month and attempt.get("revision") == revision:
                    return {"status": "unchanged", "reason": "Waiting for new corrected examples"}
                rows = [dict(row) for row in connection.execute(
                    "SELECT id, raw, corrected, audio_name FROM samples WHERE corrected IS NOT NULL")]
                generation = self._generation
                incumbent = self._index
                previous_rules = _decode_rules(self._model)
                self._training = True
        try:
            used_rows = [row for row in rows if _eligible(row)]
            grouped = _groups(used_rows)
            count = len(grouped)
            result = {"status": "insufficient_data", "reason": f"Need at least {MIN_REVIEWED} distinct corrected or reviewed transcripts",
                      "reviewed_count": count, "month": month, "revision": revision}
            model = None
            if count >= MIN_REVIEWED:
                validation_count = max(MIN_VALIDATION, math.ceil(count * 0.2))
                validation, training = grouped[:validation_count], grouped[validation_count:]
                rules = _incremental_rules(previous_rules, _learn(training), training)
                candidate = _index(rules)
                base_errors, active_errors, candidate_errors = [], [], []
                for row in validation:
                    base_errors.append(_word_errors(row["corrected"], row["raw"]))
                    active_errors.append(_word_errors(row["corrected"], _apply(row["raw"], incumbent)))
                    candidate_errors.append(_word_errors(row["corrected"], _apply(row["raw"], candidate)))
                metrics = {"training_examples": len(training), "validation_examples": len(validation),
                           "baseline_errors": sum(base_errors), "previous_errors": sum(active_errors),
                           "candidate_errors": sum(candidate_errors),
                           "validation_tokens": sum(len(TOKEN.findall(row["corrected"])) for row in validation)}
                changed = _behavior(rules) != _behavior(previous_rules)
                accepted = (changed and
                            (sum(candidate_errors) < sum(base_errors) or sum(candidate_errors) < sum(active_errors)) and
                            all(c <= min(b, a) for c, b, a in zip(candidate_errors, base_errors, active_errors)))
                unchanged = not changed and self._model.get("version", 0) > 0
                outcome = "trained" if accepted else ("unchanged_model" if unchanged else "rejected")
                reason = ("Validated improvement; used training history deleted" if accepted else
                          "No new correction learned; training history retained" if unchanged else
                          "No validated improvement without regressions; training history retained")
                result.update(status=outcome, reason=reason,
                              metrics=metrics, rule_count=len(rules))
                if accepted:
                    model = {"version": self._model.get("version", 0) + 1,
                             "trained_at": _now().isoformat(), "rules": [asdict(rule) for rule in rules],
                             "metrics": metrics}
            with self._lock:
                if generation != self._generation or not self.enabled:
                    return {"status": "stale", "reason": "Data or settings changed during training; retry later"}
                with self._connection() as connection:
                    if model is not None:
                        self._put(connection, "previous_model", self._model)
                        self._put(connection, "model", model)
                        self._put(connection, "completed_month", month)
                        connection.executemany("DELETE FROM samples WHERE id = ?", [(row["id"],) for row in used_rows])
                        self._changed(connection)
                        result["purged_records"] = len(used_rows)
                        result["revision"] = self._get(connection, "revision", 0)
                    self._put(connection, "last_attempt", result)
                if model is not None:
                    self._model = model
                    self._index = _index(_decode_rules(model))
                    pending = 0
                    for row in used_rows:
                        try:
                            self._remove_audio(row["audio_name"])
                        except OSError:
                            pending += 1
                    # A crash or unlink failure leaves unreferenced WAVs; normal
                    # retention maintenance removes those on the next pass.
                    if pending:
                        result["audio_cleanup_pending"] = pending
                        result["reason"] = "Validated improvement; audio cleanup will retry"
                    try:
                        self._compact_database()
                    except (OSError, sqlite3.Error):
                        # Compaction is independent of model promotion. Never
                        # report a failed training run or resurrect purged rows.
                        result["storage_cleanup_pending"] = True
                        result["reason"] = "Validated improvement; some storage cleanup is pending"
                return result
        finally:
            with self._lock:
                self._training = False

    def rollback(self):
        with self._lock:
            if self._unavailable or not self.database.exists():
                return False
            with self._connection() as connection:
                previous = self._get(connection, "previous_model")
                if previous is None:
                    return False
                rules = _decode_rules(previous)
                self._put(connection, "model", previous)
                self._put(connection, "previous_model", None)
                # Keep this month's completed marker to avoid immediately reinstating it.
                self._model = previous
                self._index = _index(rules)
                self._generation += 1
                return True

    def reset(self):
        with self._lock:
            self._generation += 1
            self._unavailable = True
            self._model = {"version": 0, "rules": [], "trained_at": None}
            self._index = _index(())
            if self.audio_directory.exists():
                for path in self.audio_directory.glob("*.wav"):
                    path.unlink(missing_ok=True)
                if not any(self.audio_directory.iterdir()):
                    self.audio_directory.rmdir()
            for suffix in ("", "-journal", "-wal", "-shm"):
                Path(str(self.database) + suffix).unlink(missing_ok=True)
            if self.directory.exists() and not any(self.directory.iterdir()):
                self.directory.rmdir()
            self._unavailable = False
