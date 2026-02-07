"""Tests for voice biometrics pipeline."""

import numpy as np

from friendlyface.core.models import EventType
from friendlyface.recognition.voice import (
    VoiceEmbedding,
    VoiceInferenceResult,
    VoiceMatch,
    _cosine_similarity,
    _mel_filterbank,
    extract_mfcc,
    extract_voice_embedding,
    run_voice_inference,
)


def _make_pcm(duration_s: float = 0.5, sample_rate: int = 16000, freq: float = 440.0) -> bytes:
    """Generate a sine-wave PCM int16 audio buffer."""
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    signal = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
    return signal.tobytes()


class TestMelFilterbank:
    def test_shape(self):
        fb = _mel_filterbank(n_filters=26, n_fft=512, sample_rate=16000)
        assert fb.shape == (26, 257)  # (n_filters, n_fft//2 + 1)

    def test_non_negative(self):
        fb = _mel_filterbank(n_filters=26, n_fft=512, sample_rate=16000)
        assert np.all(fb >= 0)


class TestExtractMfcc:
    def test_output_shape(self):
        audio = _make_pcm(duration_s=0.5, sample_rate=16000)
        mfcc = extract_mfcc(audio, sample_rate=16000, n_mfcc=13)
        assert mfcc.shape == (13,)

    def test_custom_n_mfcc(self):
        audio = _make_pcm(duration_s=0.5)
        mfcc = extract_mfcc(audio, n_mfcc=20)
        assert mfcc.shape == (20,)

    def test_different_signals_produce_different_mfcc(self):
        audio_low = _make_pcm(freq=200.0)
        audio_high = _make_pcm(freq=4000.0)
        mfcc_low = extract_mfcc(audio_low)
        mfcc_high = extract_mfcc(audio_high)
        # Different frequencies should yield different MFCC vectors
        assert not np.allclose(mfcc_low, mfcc_high)

    def test_empty_audio_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Empty audio"):
            extract_mfcc(b"")

    def test_deterministic(self):
        audio = _make_pcm()
        a = extract_mfcc(audio)
        b = extract_mfcc(audio)
        np.testing.assert_array_equal(a, b)


class TestVoiceEmbedding:
    def test_has_artifact_hash(self):
        audio = _make_pcm(duration_s=1.0)
        emb = extract_voice_embedding(audio, sample_rate=16000, n_mfcc=13)
        assert isinstance(emb, VoiceEmbedding)
        assert len(emb.artifact_hash) == 64  # SHA-256 hex digest
        assert emb.artifact_hash != ""

    def test_embedding_shape(self):
        audio = _make_pcm()
        emb = extract_voice_embedding(audio, n_mfcc=13)
        assert emb.embedding.shape == (13,)
        assert emb.n_mfcc == 13

    def test_duration_computed(self):
        audio = _make_pcm(duration_s=1.0, sample_rate=16000)
        emb = extract_voice_embedding(audio, sample_rate=16000)
        assert abs(emb.duration_seconds - 1.0) < 0.01

    def test_sample_rate_stored(self):
        audio = _make_pcm(sample_rate=8000)
        emb = extract_voice_embedding(audio, sample_rate=8000)
        assert emb.sample_rate == 8000

    def test_same_audio_same_hash(self):
        audio = _make_pcm()
        emb1 = extract_voice_embedding(audio)
        emb2 = extract_voice_embedding(audio)
        assert emb1.artifact_hash == emb2.artifact_hash


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, b)) < 1e-9

    def test_zero_vector(self):
        a = np.array([1.0, 2.0])
        z = np.array([0.0, 0.0])
        assert _cosine_similarity(a, z) == 0.0

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-9


class TestRunVoiceInference:
    def _make_ref_embeddings(self) -> dict[str, np.ndarray]:
        """Create reference embeddings for testing."""
        np.random.seed(42)
        return {
            "alice": np.random.randn(13),
            "bob": np.random.randn(13),
            "charlie": np.random.randn(13),
        }

    def test_returns_inference_result(self):
        audio = _make_pcm()
        refs = self._make_ref_embeddings()
        result = run_voice_inference(audio, refs)
        assert isinstance(result, VoiceInferenceResult)

    def test_matches_sorted_by_confidence(self):
        audio = _make_pcm()
        refs = self._make_ref_embeddings()
        result = run_voice_inference(audio, refs)
        confidences = [m.confidence for m in result.matches]
        assert confidences == sorted(confidences, reverse=True)

    def test_top_k_limits_matches(self):
        audio = _make_pcm()
        refs = self._make_ref_embeddings()
        result = run_voice_inference(audio, refs, top_k=2)
        assert len(result.matches) == 2

    def test_match_labels_from_references(self):
        audio = _make_pcm()
        refs = self._make_ref_embeddings()
        result = run_voice_inference(audio, refs)
        labels = {m.label for m in result.matches}
        assert labels.issubset(set(refs.keys()))

    def test_input_hash_computed(self):
        audio = _make_pcm()
        refs = self._make_ref_embeddings()
        result = run_voice_inference(audio, refs)
        assert len(result.input_hash) == 64

    def test_embedding_returned(self):
        audio = _make_pcm()
        refs = self._make_ref_embeddings()
        result = run_voice_inference(audio, refs, n_mfcc=13)
        assert isinstance(result.embedding, VoiceEmbedding)
        assert result.embedding.embedding.shape == (13,)

    def test_forensic_event_created(self):
        audio = _make_pcm()
        refs = self._make_ref_embeddings()
        result = run_voice_inference(audio, refs)
        event = result.forensic_event
        assert event.event_type == EventType.INFERENCE_RESULT
        assert event.event_hash != ""
        assert event.verify()
        assert event.payload["modality"] == "voice"
        assert event.payload["input_hash"] == result.input_hash
        assert event.payload["sample_rate"] == 16000
        assert event.payload["n_mfcc"] == 13

    def test_provenance_node_created(self):
        audio = _make_pcm()
        refs = self._make_ref_embeddings()
        result = run_voice_inference(audio, refs)
        node = result.provenance_node
        assert node.entity_type == "inference"
        assert node.node_hash != ""
        assert node.metadata["modality"] == "voice"

    def test_match_is_voice_match(self):
        audio = _make_pcm()
        refs = self._make_ref_embeddings()
        result = run_voice_inference(audio, refs)
        for m in result.matches:
            assert isinstance(m, VoiceMatch)
            assert isinstance(m.label, str)
            assert isinstance(m.confidence, float)

    def test_confidences_non_negative(self):
        audio = _make_pcm()
        refs = self._make_ref_embeddings()
        result = run_voice_inference(audio, refs)
        for m in result.matches:
            assert m.confidence >= 0.0

    def test_empty_references(self):
        audio = _make_pcm()
        result = run_voice_inference(audio, {})
        assert result.matches == []
