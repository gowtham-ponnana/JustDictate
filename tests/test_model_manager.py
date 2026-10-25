import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from model_manager import ModelManager, _generate_greedy_text, _supports_fast_decode


class ModelInputTests(unittest.TestCase):
    def setUp(self):
        self.manager = ModelManager()
        self.manager._model = object()
        self.manager._transcribe_unlocked = Mock(return_value="hello")

    def test_invalid_samples_and_shapes_do_not_reach_gpu(self):
        for audio in (np.array([np.nan]), np.array([np.inf]), np.zeros((2, 2, 2)), np.zeros((10, 0))):
            with self.subTest(shape=audio.shape), self.assertRaises(ValueError):
                self.manager.transcribe(audio)
        self.manager._transcribe_unlocked.assert_not_called()

    def test_invalid_sample_rates_do_not_reach_gpu(self):
        for rate in (0, -16000, 44100.5):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                self.manager.transcribe(np.zeros(1000), rate)
        self.manager._transcribe_unlocked.assert_not_called()

    def test_empty_input_skips_inference(self):
        self.assertEqual(self.manager.transcribe(np.array([], dtype=np.float32)), "")
        self.manager._transcribe_unlocked.assert_not_called()

    def test_stereo_float64_is_downmixed_before_transcription(self):
        audio = np.array([[0.1, 0.3], [0.5, -0.1]], dtype=np.float64)
        self.assertEqual(self.manager.transcribe(audio), "hello")
        converted, rate = self.manager._transcribe_unlocked.call_args.args
        np.testing.assert_allclose(converted, [0.2, 0.2])
        self.assertEqual(converted.dtype, np.float32)
        self.assertEqual(rate, 16000)

    def test_default_falls_back_until_model_support_is_verified(self):
        self.manager._model = Mock()
        self.manager._model.generate.return_value = [SimpleNamespace(text="upstream")]
        self.assertEqual(self.manager._generate("mel").text, "upstream")
        self.manager._model.generate.assert_called_once_with("mel")


class ChunkingTests(unittest.TestCase):
    def test_end_is_transcribed_once_without_dropping_new_tail(self):
        manager = ModelManager()
        manager._model = SimpleNamespace(preprocessor_config=SimpleNamespace(hop_length=1))
        # Small sample rate makes coverage checks cheap; use the same durations.
        for seconds, expected in (
            (91, [(0, 60), (55, 91)]),
            (110, [(0, 60), (55, 110)]),
            (111, [(0, 60), (55, 111)]),
            (115, [(0, 60), (55, 115)]),
            (116, [(0, 60), (55, 115), (110, 116)]),
            (170, [(0, 60), (55, 115), (110, 170)]),
            (171, [(0, 60), (55, 115), (110, 170), (165, 171)]),
        ):
            calls = []

            def get_logmel(audio, cfg):
                calls.append((int(audio[0]), int(audio[-1]) + 1))
                return audio

            manager._generate = Mock(return_value=SimpleNamespace(sentences=[], tokens=[]))
            with self.subTest(seconds=seconds), patch("parakeet_mlx.audio.get_logmel", get_logmel):
                self.assertEqual(manager._transcribe_chunked(np.arange(seconds), 1), "")
                self.assertEqual(calls, expected)


class FastGreedyTests(unittest.TestCase):
    def test_missing_packaged_metadata_keeps_upstream_decoder_available(self):
        from importlib.metadata import PackageNotFoundError
        with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            self.assertFalse(_supports_fast_decode(object()))

    def test_version_and_model_guards(self):
        class Supported:
            pass

        class Subclass(Supported):
            pass

        with patch("parakeet_mlx.parakeet.ParakeetTDT", Supported):
            with patch("importlib.metadata.version", return_value="0.5.2"):
                self.assertTrue(_supports_fast_decode(Supported()))
                self.assertFalse(_supports_fast_decode(Subclass()))
                self.assertFalse(_supports_fast_decode(object()))
            with patch("importlib.metadata.version", return_value="0.5.3"):
                self.assertFalse(_supports_fast_decode(Supported()))

    def test_blank_predictor_reuse_and_zero_duration_guard_match_upstream(self):
        import mlx.core as mx
        from parakeet_mlx.parakeet import ParakeetTDT

        class FakeTDT:
            vocabulary = ["▁one", "▁two"]
            durations = [0, 1]
            max_symbols = 2
            time_ratio = 0.08

            def __init__(self):
                self.decoder_calls = 0
                self.joint_calls = 0
                self.features = mx.zeros((1, 5, 1))

            def encoder(self, mel):
                return self.features, mx.array([5])

            def decoder(self, token, hidden_state):
                self.decoder_calls += 1
                value = 0 if hidden_state is None else int(hidden_state[0][0]) + 1
                return mx.array([[[float(value)]]]), (mx.array([value]), mx.array([value]))

            def joint(self, feature, decoder_out):
                # Two blanks reuse predictor 0; tokens then advance it to 1 and
                # 2; another blank retains predictor 2. Assert actual state so
                # this fixture fails if the adapter reuses a stale predictor.
                expected_predictors = [0, 0, 0, 1, 2, 2]
                expected = expected_predictors[self.joint_calls]
                if int(decoder_out[0, 0, 0]) != expected:
                    raise AssertionError("Wrong predictor state")
                token, duration = [(2, 1), (2, 1), (0, 0), (1, 0), (2, 1), (0, 1)][self.joint_calls]
                self.joint_calls += 1
                logits = np.full((1, 1, 1, 5), -20.0, dtype=np.float32)
                logits[0, 0, 0, token] = 20
                logits[0, 0, 0, 3 + duration] = 20
                return mx.array(logits)

        reference = FakeTDT()
        expected, _ = ParakeetTDT.decode_greedy(reference, reference.features, mx.array([5]))
        optimized = FakeTDT()
        actual = _generate_greedy_text(optimized, None)
        signature = lambda tokens: [(t.id, t.text, t.start, t.duration) for t in tokens]
        self.assertEqual(signature(actual.tokens), signature(expected[0]))
        self.assertEqual(reference.decoder_calls, 6)
        self.assertEqual(optimized.decoder_calls, 3)
        self.assertEqual([t.start for t in actual.tokens], [0.16, 0.16, 0.32])

    def test_batch_input_is_rejected(self):
        import mlx.core as mx
        fake = SimpleNamespace(encoder=lambda mel: (mx.zeros((2, 5, 1)), mx.array([5, 5])))
        with self.assertRaisesRegex(ValueError, "one utterance"):
            _generate_greedy_text(fake, None)


if __name__ == "__main__":
    unittest.main()
