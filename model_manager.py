"""Model loading and transcription using onnx-asr with Parakeet TDT 0.6B v3."""

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
CACHE_DIR = Path.home() / ".cache" / "just-dictate"
MODEL_DIR = CACHE_DIR / "parakeet-v3"
VAD_DIR = CACHE_DIR / "silero-vad"


class ModelManager:
    def __init__(self):
        self._model = None
        self._vad = None
        self._loading = False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the Parakeet model. Downloads on first run (~2.5 GB)."""
        if self._model is not None:
            return
        self._loading = True
        try:
            import onnx_asr

            # Only pass path if model files already exist there.
            # If path exists but is empty, onnx_asr skips download.
            model_path = None
            if MODEL_DIR.exists() and any(MODEL_DIR.glob("*.onnx")):
                model_path = str(MODEL_DIR)
                log.info("Loading model from cache %s ...", MODEL_DIR)
            else:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                model_path = str(MODEL_DIR)
                # Remove empty dir so onnx_asr triggers download
                if MODEL_DIR.exists() and not any(MODEL_DIR.iterdir()):
                    MODEL_DIR.rmdir()
                log.info("Downloading model to %s (first run, ~2.5 GB)...", MODEL_DIR)

            self._model = onnx_asr.load_model(
                MODEL_NAME,
                model_path,
                providers=["CPUExecutionProvider"],
            )
            log.info("Model loaded.")

            # Load VAD — API varies across onnx_asr versions
            vad_path = str(VAD_DIR) if VAD_DIR.exists() and any(VAD_DIR.glob("*.onnx")) else None
            if vad_path is None and VAD_DIR.exists() and not any(VAD_DIR.iterdir()):
                VAD_DIR.rmdir()
            try:
                self._vad = onnx_asr.load_vad("silero", str(VAD_DIR))
                log.info("VAD loaded with path.")
            except TypeError:
                try:
                    self._vad = onnx_asr.load_vad("silero")
                    log.info("VAD loaded without path.")
                except Exception as e:
                    log.warning("VAD unavailable: %s", e)
                    self._vad = None
        finally:
            self._loading = False

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a numpy audio array (float32, mono) to text."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Flatten to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        result = self._model.recognize(audio, sample_rate=sample_rate)

        if isinstance(result, str):
            return result.strip()
        # Some versions return an object with .text
        if hasattr(result, "text"):
            return result.text.strip()
        return str(result).strip()
