"""Model loading and transcription using parakeet-mlx on Apple Silicon."""

import logging
import platform
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"
CACHE_DIR = Path.home() / ".cache" / "just-dictate"
MLX_CACHE_DIR = CACHE_DIR / "parakeet-mlx"

# Single GPU pass is fine for typical dictation. Chunk only very long holds
# so we do not blow GPU memory, and stitch with overlap.
LONG_AUDIO_S = 90.0
CHUNK_DURATION_S = 60.0
OVERLAP_DURATION_S = 5.0
TARGET_SAMPLE_RATE = 16000


class ModelManager:
    def __init__(self, *, fast_decode: bool = True):
        self._model = None
        self._loading = False
        self._infer_lock = threading.Lock()
        self._dtype = None
        self._fast_decode_requested = fast_decode
        self._fast_decode = False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load Parakeet TDT 0.6B v3 via MLX. Downloads on first run (~2.5 GB)."""
        if self._model is not None:
            return
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError(
                "The MLX backend requires Apple Silicon (M-series). "
                f"Got {platform.system()} {platform.machine()}."
            )

        self._loading = True
        try:
            import mlx.core as mx
            from parakeet_mlx import from_pretrained

            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            MLX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

            log.info(
                "Loading MLX Parakeet from %s (first run downloads ~2.5 GB)...",
                MODEL_ID,
            )
            self._dtype = mx.bfloat16
            self._model = from_pretrained(
                MODEL_ID,
                dtype=self._dtype,
                cache_dir=str(MLX_CACHE_DIR),
            )
            self._model.eval()
            # This adapter is verified against 0.5.2's exact greedy algorithm.
            # New library versions retain the upstream path until revalidated.
            self._fast_decode = (
                self._fast_decode_requested
                and _supports_fast_decode(self._model)
            )
            log.info("MLX model loaded on %s.", mx.default_device())

            # Pay Metal/kernel init now so the first real dictation is fast.
            warmup = np.zeros(int(0.4 * TARGET_SAMPLE_RATE), dtype=np.float32)
            self._transcribe_unlocked(warmup, TARGET_SAMPLE_RATE)
            log.info("MLX warmup done.")
        finally:
            self._loading = False

    def transcribe(self, audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> str:
        """Transcribe a numpy audio array (float32, mono) to text."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        audio = np.asarray(audio)
        if audio.ndim not in (1, 2) or (audio.ndim == 2 and audio.shape[1] == 0):
            raise ValueError("Audio must be a mono array or a frames-by-channels array.")
        if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
            raise ValueError("Sample rate must be a positive integer.")
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains non-finite samples.")
        if audio.size == 0:
            return ""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != TARGET_SAMPLE_RATE:
            audio = _resample_linear(audio, sample_rate, TARGET_SAMPLE_RATE)
            sample_rate = TARGET_SAMPLE_RATE

        duration = len(audio) / sample_rate
        t0 = time.perf_counter()
        with self._infer_lock:
            text = self._transcribe_unlocked(audio, sample_rate)
        elapsed = time.perf_counter() - t0
        rtf = duration / elapsed if elapsed > 0 else 0.0
        log.info(
            "Transcribed %.1fs audio in %.2fs (RTFx %.1f, mlx-bf16)",
            duration,
            elapsed,
            rtf,
        )
        return text

    def _transcribe_unlocked(self, audio: np.ndarray, sample_rate: int) -> str:
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        hop = self._model.preprocessor_config.hop_length
        if len(audio) < hop:
            return ""

        # parakeet-mlx's file loader always feeds float32 PCM into get_logmel.
        # BF16 here breaks the STFT view (complex64 vs bf16 packing).
        audio_mx = mx.array(audio, dtype=mx.float32)
        duration = len(audio) / sample_rate
        if duration <= LONG_AUDIO_S:
            mel = get_logmel(audio_mx, self._model.preprocessor_config)
            result = self._generate(mel)
            return (result.text or "").strip()
        return self._transcribe_chunked(audio_mx, sample_rate)

    def _transcribe_chunked(self, audio_mx, sample_rate: int) -> str:
        from parakeet_mlx.alignment import (
            merge_longest_common_subsequence,
            merge_longest_contiguous,
            sentences_to_result,
            tokens_to_sentences,
        )
        from parakeet_mlx.audio import get_logmel

        cfg = self._model.preprocessor_config
        chunk_samples = int(CHUNK_DURATION_S * sample_rate)
        overlap_samples = int(OVERLAP_DURATION_S * sample_rate)
        hop = max(chunk_samples - overlap_samples, cfg.hop_length)
        n = int(audio_mx.shape[0])
        all_tokens = []

        for start in range(0, n, hop):
            end = min(start + chunk_samples, n)
            if end - start < cfg.hop_length:
                break
            chunk = audio_mx[start:end]
            result = self._generate(get_logmel(chunk, cfg))
            offset = start / sample_rate
            for sentence in result.sentences:
                for token in sentence.tokens:
                    token.start += offset
                    token.end = token.start + token.duration
            if all_tokens:
                try:
                    all_tokens = merge_longest_contiguous(
                        all_tokens,
                        result.tokens,
                        overlap_duration=OVERLAP_DURATION_S,
                    )
                except RuntimeError:
                    all_tokens = merge_longest_common_subsequence(
                        all_tokens,
                        result.tokens,
                        overlap_duration=OVERLAP_DURATION_S,
                    )
            else:
                all_tokens = list(result.tokens)
            # A final window may already include the entire remaining tail.
            # Iterating once more would decode only its overlap, duplicating
            # work and potentially replacing a well-contextualized ending.
            if end == n:
                break

        if not all_tokens:
            return ""
        return sentences_to_result(tokens_to_sentences(all_tokens)).text.strip()

    def _generate(self, mel):
        if not self._fast_decode:
            return self._model.generate(mel)[0]
        return _generate_greedy_text(self._model, mel)


def _supports_fast_decode(model) -> bool:
    from importlib.metadata import PackageNotFoundError, version

    try:
        if version("parakeet-mlx") != "0.5.2":
            return False
        from parakeet_mlx.parakeet import ParakeetTDT
    except (PackageNotFoundError, ImportError):
        # Frozen app bundles may lack distribution metadata. A future library
        # may move its classes. Both cases must retain working upstream inference.
        return False
    return type(model) is ParakeetTDT


def _generate_greedy_text(model, mel):
    """Single-utterance TDT greedy inference, equivalent to parakeet-mlx 0.5.2.

    Adapted from senstella/parakeet-mlx's ParakeetTDT.decode_greedy (Apache-2.0).
    Upstream license: licenses/parakeet-mlx.txt.

    Preserve token selection, duration selection, BF16 state, and the decoder's
    zero-duration guard. Reuse the predictor only across blanks (its inputs
    remain identical), and fetch both argmax decisions in one synchronization.
    We do not compute confidence entropy: JustDictate only consumes text and
    token timestamps. The returned internal alignment confidence is therefore
    not a measured confidence and must never be used to authorize corrections.
    """
    import mlx.core as mx
    from parakeet_mlx.alignment import (
        AlignedToken,
        sentences_to_result,
        tokens_to_sentences,
    )
    from parakeet_mlx.tokenizer import decode

    features, lengths = model.encoder(mel)
    mx.eval(features, lengths)
    if features.shape[0] != 1:
        raise ValueError("Fast decoding supports one utterance at a time.")
    length = int(lengths[0])
    blank = len(model.vocabulary)
    step = 0
    new_symbols = 0
    last_token = None
    hidden_state = None
    decoder_out = None
    hypothesis = []
    while step < length:
        if decoder_out is None:
            decoder_out, (hidden, cell) = model.decoder(
                mx.array([[last_token]]) if last_token is not None else None,
                hidden_state,
            )
            decoder_out = decoder_out.astype(features.dtype)
            decoder_hidden = (hidden.astype(features.dtype), cell.astype(features.dtype))

        joint_out = model.joint(features[:, step : step + 1], decoder_out)
        pred_token, decision = mx.stack(
            [
                mx.argmax(joint_out[0, 0, :, : blank + 1]),
                mx.argmax(joint_out[0, 0, :, blank + 1 :]),
            ]
        ).tolist()
        duration = model.durations[decision]
        if pred_token != blank:
            hypothesis.append(
                AlignedToken(
                    pred_token,
                    text=decode([pred_token], model.vocabulary),
                    start=step * model.time_ratio,
                    duration=duration * model.time_ratio,
                )
            )
            last_token = pred_token
            hidden_state = decoder_hidden
            decoder_out = None

        step += duration
        new_symbols += 1
        if duration != 0:
            new_symbols = 0
        elif model.max_symbols is not None and model.max_symbols <= new_symbols:
            step += 1
            new_symbols = 0

    return sentences_to_result(tokens_to_sentences(hypothesis))


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio
    duration = audio.shape[0] / src_rate
    dst_len = int(round(duration * dst_rate))
    if dst_len <= 1:
        return audio[:1]
    src_x = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)
