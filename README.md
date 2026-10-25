# JustDictate

**Fast, local dictation for Macs with Apple Silicon.** Hold Right Command, speak,
and release to paste your words into the focused app. JustDictate uses NVIDIA
Parakeet TDT 0.6B v3 on the Apple Silicon GPU and learns recurring text corrections
from your edits in supported apps.

Audio, transcripts, and personal learning stay on your Mac. No cloud transcription,
account, or API key is required.

## Install as a Mac app

Requires **macOS 14+ and an Apple Silicon Mac** for the current MLX dependencies.
These commands build and install the app locally. Run in Terminal:

```bash
git clone https://github.com/gowtham-ponnana/JustDictate.git
cd JustDictate
chmod +x install.sh
./install.sh
open /Applications/JustDictate.app
```

The installer prepares dependencies, builds the app, and installs it in
`/Applications`. The first launch downloads the model, approximately **2.5 GB**.
An existing app is replaced only after the new build is verified; the previous
version is kept at `/Applications/.JustDictate.previous.app`.

In **System Settings → Privacy & Security**, enable JustDictate under
**Microphone**, **Accessibility**, and **Input Monitoring**. Wait for the menu
bar status to say **Ready**.

After rebuilding, macOS may require fresh permissions. Quit JustDictate, remove
and re-add `/Applications/JustDictate.app` in Accessibility, enable the relevant
permissions, and reopen it. **Retry Hotkey Setup** retries startup after a grant.

To update, run these commands from your existing JustDictate checkout:

```bash
git pull --ff-only
./install.sh
open /Applications/JustDictate.app
```

## How to use it

1. Click the text field where you want to write.
2. Hold **Right Command**, speak, then release. A waveform shows recording;
   a completion sound signals transcription has finished.
3. Correct a word in the pasted sentence when needed. In compatible editors,
   small corrections are saved as learning examples after **one minute without
   further edits**. Each new edit restarts that minute; text can be revisited
   within the 15-minute tracking window.
4. Open **Personalization → Review Recent Dictations** to see entries marked
   **edit captured**, or review a transcript manually.

The menu lets you change the microphone and hotkey. **Escape** cancels a recording;
within five seconds after a paste, it sends the focused app's Undo command.
Automatic correction capture has been tested in TextEdit and native Cocoa text
fields. Other editors depend on their Accessibility support.

## Measured performance

On an **Apple M1 Max**, the included benchmark measured **15–32% less transcription
time** than the previous JustDictate implementation:

| Synthetic audio | Previous JustDictate | Optimized JustDictate | Speedup |
|---|---:|---:|---:|
| Short sentence, 3.3 seconds | 77.3 ms | **65.3 ms** | 1.18× |
| Numbers, 5.4 seconds | 123.5 ms | **95.7 ms** | 1.29× |
| Technical sentence, 6.4 seconds | 140.1 ms | **95.9 ms** | 1.46× |
| Repeated speech, 114 seconds | 2.398 s | **1.658 s** | 1.45× |

These are medians of three alternating trials after warmup, using the same BF16
model with `parakeet-mlx 0.5.2` and `MLX 0.32.2`, compared with commit `37d562a`.
The tested text, token IDs, and timestamps matched. Times cover transcription;
startup, recording, and delivery into another app are outside this measurement.
[Raw results](benchmarks/model_m1_max.json) · [Benchmark script](tools/benchmark_model.py)

The decoder reuses unchanged predictor work, reduces GPU decision reads, and
skips redundant overlap processing in long recordings. Model warmup happens at
startup. Clipboard restoration runs after completion feedback. Edit capture and
its bounded retry share the existing wait before paste.

### Compared with Wispr Flow and superwhisper

**A matching end-to-end speed comparison with JustDictate is still pending.**
For reference, the [Talkify-maintained benchmark](https://github.com/tornikegomareli/talkify-latency-bench/blob/main/RESULTS.md)
reported these measurements from August 2026: 20 trials of one short phrase on an
**M4 Max, macOS 26**, timed from stopping recording to final visible text.

| App and mode tested | Median wait | P95 wait |
|---|---:|---:|
| JustDictate | Not measured with this harness | — |
| Wispr Flow 1.6.492, cloud | 587 ms | 693 ms |
| superwhisper 2.16.1, Standard EN local | 216 ms | 247 ms |

Those are another product developer's measurements of specific versions.
Their hardware, audio, and timing boundaries differ from our transcription
benchmark above, so the two tables cannot establish a speed or accuracy ranking
for JustDictate. P95 means 95% of measured waits were at or below that value.

## Automatic local learning

JustDictate updates a **small personal text-correction model** from corrected or
reviewed transcripts. Parakeet's speech-recognition weights stay unchanged;
recordings are kept for playback and review, not acoustic training. Learned
corrections apply to future dictations after recognition.

For example, repeated corrections can teach `ask Gotham tomorrow` →
`ask Gowtham tomorrow`, while keeping `visit Gotham yesterday` unchanged.
Learning aims to reduce repeated editing; the speed improvements above come
from the recognition and delivery pipeline.

- **Calendar-month schedule:** the first automatic update becomes eligible in
  the month after enrollment. Starting October 25 makes November the first
  eligible month; this is not a rolling 30-day timer.
- **Automatic checks:** on startup and every minute while the app is running.
  Training waits until dictation and correction tracking are idle. Tracking can
  last up to 15 minutes after a paste. A closed app catches up after reopening.
- **Enough evidence first:** at least **12 distinct corrected or reviewed
  transcripts** are required. Repeating the same sentence does not create more
  independent examples. New corrections need support from multiple examples.
- **Validation before activation:** at least 20% of examples, with a minimum of
  three, are held out from training. An update must reduce their errors without
  worsening any held-out example. Otherwise, the current model stays active.
- **Cleanup after success:** used training and validation transcripts and their
  audio are deleted. Learned corrections and one previous version remain.

No monthly approval prompt is needed. An individual edit does not immediately
change the model, and an update is not guaranteed every month if evidence is
insufficient. **Personalization → Learning Details…** shows the status and next
check. **Check for Learning Update** requests an earlier check when idle, using
the same validation requirements. **Restore Previous Learning** rolls back rules;
it cannot restore deleted training data.

## Local data and controls

Local history, audio retention, and automatic edit learning are **on by default**
for new settings. They can be switched off in **Personalization**; existing off
preferences are respected. History lives in
`~/.config/just-dictate/personalization/`.

- Unused history is retained for up to **90 days**, with limits of **2,000
  dictations** and **256 MiB of audio**. Audio is saved only for clips up to two
  minutes; older audio may be removed before its transcript. Maintenance runs
  while the app is open.
- Surrounding editor text is used in memory to verify corrections to a pasted
  span. Only the dictation and its correction enter learning history. Automatic
  edit observation skips secure, read-only, and unsupported fields.
- Disabling learning preserves existing data. **Delete Learning Data** removes
  recordings, transcripts, learned corrections, and rollback state. Local files
  have restricted user permissions and are not separately encrypted by the app.

Setup downloads dependencies and model files. Cached transcription works offline;
model loading may contact Hugging Face for files or metadata. Dictation audio and
transcripts are not uploaded.

## License

MIT. The adapted Parakeet decoder retains its upstream
[Apache-2.0 license](licenses/parakeet-mlx.txt).
