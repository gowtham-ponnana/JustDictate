"""Exercise local personalization with real ASR of synthetic speech.

Run: .venv/bin/python tools/benchmark_personalized_speech.py
Requires macOS say and the already-cached Parakeet model. Uses no microphone,
network, or personal history. Known TTS scripts are reference labels for this
isolated fixture only; production labels come from user edits or explicit review.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_manager import ModelManager  # noqa: E402
from personalization import PersonalizationManager, _word_errors  # noqa: E402


TRAIN_TOPICS = (
    "the quarterly budget", "our next meeting", "the project deadline",
    "the security review", "the database backup", "the customer survey",
    "the product launch", "the server upgrade", "the invoice number",
    "the hiring plan", "the vacation schedule", "the network outage",
    "the design proposal", "the training session", "the presentation slides",
    "the delivery address", "the test results", "the release notes",
)
TEST_TOPICS = (
    "the office keys", "the travel arrangements", "the annual report",
    "the support request", "the performance issue",
)
CONTROLS = (
    "Batman protects Gotham City from criminals.",
    "The train to Gotham City leaves tomorrow.",
    "Please ask Sarah tomorrow about the quarterly budget.",
)


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    started = time.monotonic()
    model = ModelManager()
    model.load()
    with tempfile.TemporaryDirectory(prefix="justdictate-personalized-speech-") as temporary:
        root = Path(temporary)
        manager = PersonalizationManager(root / "learning", enabled=True)

        def transcribe(script):
            path = root / "utterance.wav"
            subprocess.run(
                ["/usr/bin/say", "-o", str(path), "--data-format=LEI16@16000", script],
                check=True,
            )
            with wave.open(str(path)) as stream:
                audio = np.frombuffer(
                    stream.readframes(stream.getnframes()), dtype="<i2"
                ).astype(np.float32) / 32768
            return model.transcribe(audio)

        training = []
        for topic in TRAIN_TOPICS:
            reference = f"Please ask Gowtham tomorrow about {topic}."
            raw = transcribe(reference)
            record_id = manager.record(raw, raw)
            manager.review(record_id, reference)
            training.append({
                "reference": reference, "raw": raw,
                "errors": _word_errors(reference, raw),
            })
        result = manager.train_if_due(force=True)
        unseen = []
        scripts = [f"Please ask Gowtham tomorrow about {topic}." for topic in TEST_TOPICS]
        for reference in scripts + list(CONTROLS):
            raw = transcribe(reference)
            personalized = manager.apply(raw)
            unseen.append({
                "reference": reference, "raw": raw, "personalized": personalized,
                "baseline_errors": _word_errors(reference, raw),
                "personalized_errors": _word_errors(reference, personalized),
                "control": reference in CONTROLS,
            })
        report = {
            "training_count": len(training),
            "training_raw_error_total": sum(row["errors"] for row in training),
            "training": training,
            "training_result": result,
            "unseen": unseen,
            "unseen_name_baseline_errors": sum(
                row["baseline_errors"] for row in unseen if not row["control"]
            ),
            "unseen_name_personalized_errors": sum(
                row["personalized_errors"] for row in unseen if not row["control"]
            ),
            "controls_unchanged": all(
                row["raw"] == row["personalized"] for row in unseen if row["control"]
            ),
            "elapsed_seconds": time.monotonic() - started,
        }
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
