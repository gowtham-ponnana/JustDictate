"""Compare warm Parakeet revisions on a reproducible speaker-balanced LibriSpeech sample.

Unpack OpenSLR 12 test-clean.tar.gz and test-other.tar.gz into the same directory.
Run: .venv/bin/python tools/benchmark_speech.py /path/to/LibriSpeech --baseline-ref 3e6dce4
The first hash-ranked utterance per speaker is reserved for calibration.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_manager import ModelManager
from personalization import _word_errors
from tools.benchmark_model import load_baseline


def words(text):
    # LibriSpeech references omit punctuation and capitalization distinctions.
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text.lower().replace("’", "'")))


def select(corpus):
    cases = []
    for split in ("test-clean", "test-other"):
        by_speaker = defaultdict(list)
        for transcript in sorted((corpus / split).rglob("*.trans.txt")):
            for line in transcript.read_text().splitlines():
                uid, reference = line.split(" ", 1)
                path = transcript.parent / f"{uid}.flac"
                rank = hashlib.sha256(("JustDictate-v1/" + uid).encode()).hexdigest()
                by_speaker[uid.split("-")[0]].append((rank, path, words(reference)))
        if not by_speaker:
            raise ValueError(f"No LibriSpeech transcripts in {corpus / split}")
        for speaker, utterances in sorted(by_speaker.items()):
            selected = sorted(utterances)[1:6]
            for index, (_, path, reference) in enumerate(selected):
                cases.append((split, speaker, path, reference, index == len(selected) - 1))
    random.Random(514).shuffle(cases)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--baseline-ref", default="3e6dce4")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    current = ModelManager()
    current.load()
    baseline = load_baseline(args.baseline_ref, current)
    paths = [("baseline", baseline), ("installed_source", current)]
    records = []
    for i, (split, speaker, path, reference, check_quiet) in enumerate(select(args.corpus)):
        pcm, rate = sf.read(path, dtype="float32")
        if rate != 16000 or pcm.ndim != 1:
            raise ValueError(f"Expected 16 kHz mono audio at {path}")
        cases = [("original", pcm)]
        if check_quiet:
            cases.append(("minus_30_dB", pcm * 10**(-30 / 20)))
        for level, audio in cases:
            record = dict(utterance=path.stem, speaker=speaker, split=split, level=level,
                          reference_words=len(reference.split()), audio_seconds=len(audio) / rate)
            order = paths if i % 2 == 0 else reversed(paths)
            for name, model in order:
                start = time.perf_counter()
                text = model.transcribe(audio, rate)
                record[name] = dict(seconds=time.perf_counter() - start,
                                    word_errors=_word_errors(reference, words(text)))
            records.append(record)
    summary = {}
    for level in ("original", "minus_30_dB"):
        rows = [r for r in records if r["level"] == level]
        count = sum(r["reference_words"] for r in rows)
        result = dict(clips=len(rows), speakers=len({r["speaker"] for r in rows}),
                      words=count, audio_seconds=sum(r["audio_seconds"] for r in rows))
        for name, _ in paths:
            times = [r[name]["seconds"] for r in rows]
            errors = sum(r[name]["word_errors"] for r in rows)
            result[name] = dict(errors=errors, wer_percent=100 * errors / count,
                                median_ms=1000 * statistics.median(times),
                                p95_ms=1000 * float(np.quantile(times, 0.95)))
        summary[level] = result
    print(json.dumps(dict(baseline_ref=args.baseline_ref, summary=summary, cases=records), indent=2))


if __name__ == "__main__":
    main()
