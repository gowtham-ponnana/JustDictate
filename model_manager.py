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
    def __init__(self):
        self._model = None
        self._loading = False
        self._infer_lock = threading.Lock()
        self._dtype = None

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
            result = self._model.generate(mel)[0]
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
            result = self._model.generate(get_logmel(chunk, cfg))[0]
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

        if not all_tokens:
            return ""
        return sentences_to_result(tokens_to_sentences(all_tokens)).text.strip()


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
