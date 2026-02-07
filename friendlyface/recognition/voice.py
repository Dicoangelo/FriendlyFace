"""Voice biometrics pipeline with MFCC embedding extraction.

Extracts fixed-size voice embeddings from raw PCM audio using MFCC features
(numpy-only DSP, no external audio libraries), and runs speaker identification
via cosine similarity against reference embeddings with forensic logging.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from friendlyface.core.models import (
    EventType,
    ForensicEvent,
    ProvenanceNode,
    ProvenanceRelation,
)


@dataclass
class VoiceEmbedding:
    """Fixed-size voice embedding vector with provenance metadata."""

    embedding: np.ndarray
    sample_rate: int
    duration_seconds: float
    n_mfcc: int
    artifact_hash: str


@dataclass
class VoiceMatch:
    """A single speaker match with confidence score."""

    label: str
    confidence: float


@dataclass
class VoiceInferenceResult:
    """Container for voice inference outputs with forensic metadata."""

    matches: list[VoiceMatch]
    input_hash: str
    embedding: VoiceEmbedding
    forensic_event: ForensicEvent
    provenance_node: ProvenanceNode


def _hash_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def _mel_filterbank(n_filters: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """Build a mel-scale triangular filterbank matrix.

    Returns shape (n_filters, n_fft // 2 + 1).
    """

    # Mel conversion helpers
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    low_mel = hz_to_mel(0.0)
    high_mel = hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])

    # Convert Hz points to FFT bin indices
    n_bins = n_fft // 2 + 1
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_filters, n_bins))
    for i in range(n_filters):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # Rising slope
        if center > left:
            filterbank[i, left:center] = (np.arange(left, center) - left) / (center - left)
        # Falling slope
        if right > center:
            filterbank[i, center:right] = (right - np.arange(center, right)) / (right - center)

    return filterbank


def _dct_matrix(n_out: int, n_in: int) -> np.ndarray:
    """Build a type-II DCT matrix of shape (n_out, n_in)."""
    k = np.arange(n_out).reshape(-1, 1)
    n = np.arange(n_in).reshape(1, -1)
    return np.cos(np.pi * k * (2.0 * n + 1.0) / (2.0 * n_in))


def extract_mfcc(
    audio_data: bytes,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    *,
    frame_size_ms: int = 25,
    frame_step_ms: int = 10,
    n_fft: int = 512,
    n_mel_filters: int = 26,
) -> np.ndarray:
    """Extract mean MFCC vector from raw int16 PCM audio bytes.

    Pipeline: frame -> window -> FFT -> mel filterbank -> log -> DCT -> mean.

    Parameters
    ----------
    audio_data:
        Raw bytes of int16 PCM audio.
    sample_rate:
        Audio sample rate in Hz.
    n_mfcc:
        Number of MFCC coefficients to retain.
    frame_size_ms:
        Frame length in milliseconds.
    frame_step_ms:
        Frame step (hop) in milliseconds.
    n_fft:
        FFT size.
    n_mel_filters:
        Number of mel filterbank channels.

    Returns
    -------
    Mean MFCC vector of shape (n_mfcc,).
    """
    # Parse raw int16 PCM
    signal = np.frombuffer(audio_data, dtype=np.int16).astype(np.float64)
    if signal.size == 0:
        raise ValueError("Empty audio data")

    # Pre-emphasis
    signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

    # Framing
    frame_len = int(sample_rate * frame_size_ms / 1000)
    frame_step = int(sample_rate * frame_step_ms / 1000)

    if frame_len < 1:
        raise ValueError(f"frame_size_ms too small: {frame_size_ms}")
    if frame_step < 1:
        raise ValueError(f"frame_step_ms too small: {frame_step_ms}")

    # Pad signal to fill last frame
    n_frames = max(1, 1 + (len(signal) - frame_len) // frame_step)
    pad_len = (n_frames - 1) * frame_step + frame_len - len(signal)
    if pad_len > 0:
        signal = np.append(signal, np.zeros(pad_len))

    # Build frames matrix
    indices = np.arange(frame_len).reshape(1, -1) + np.arange(n_frames).reshape(-1, 1) * frame_step
    frames = signal[indices]

    # Hamming window
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_len) / (frame_len - 1))
    frames = frames * window

    # FFT -> power spectrum
    fft_result = np.fft.rfft(frames, n=n_fft)
    power_spectrum = (np.abs(fft_result) ** 2) / n_fft

    # Mel filterbank
    mel_fb = _mel_filterbank(n_mel_filters, n_fft, sample_rate)
    mel_energies = power_spectrum @ mel_fb.T

    # Log (with floor to avoid log(0))
    mel_energies = np.maximum(mel_energies, 1e-10)
    log_mel = np.log(mel_energies)

    # DCT to get MFCCs
    dct = _dct_matrix(n_mfcc, n_mel_filters)
    mfcc = log_mel @ dct.T

    # Return mean across frames
    return np.mean(mfcc, axis=0)


def extract_voice_embedding(
    audio_data: bytes,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    *,
    frame_size_ms: int = 25,
    frame_step_ms: int = 10,
    n_fft: int = 512,
    n_mel_filters: int = 26,
) -> VoiceEmbedding:
    """Extract a voice embedding with artifact hashing.

    Wraps extract_mfcc and computes a SHA-256 hash over the raw audio
    for forensic traceability.

    Returns a VoiceEmbedding with the mean MFCC vector and metadata.
    """
    mfcc = extract_mfcc(
        audio_data,
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        frame_size_ms=frame_size_ms,
        frame_step_ms=frame_step_ms,
        n_fft=n_fft,
        n_mel_filters=n_mel_filters,
    )
    artifact_hash = _hash_bytes(audio_data)
    n_samples = len(audio_data) // 2  # int16 = 2 bytes per sample
    duration = n_samples / sample_rate

    return VoiceEmbedding(
        embedding=mfcc,
        sample_rate=sample_rate,
        duration_seconds=duration,
        n_mfcc=n_mfcc,
        artifact_hash=artifact_hash,
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def run_voice_inference(
    audio_data: bytes,
    reference_embeddings: dict[str, np.ndarray],
    *,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    top_k: int = 5,
    actor: str = "voice_inference_engine",
    previous_hash: str = "GENESIS",
    sequence_number: int = 0,
    model_provenance_id: str | None = None,
) -> VoiceInferenceResult:
    """Run voice speaker identification on raw PCM audio.

    Parameters
    ----------
    audio_data:
        Raw bytes of int16 PCM audio.
    reference_embeddings:
        Dict mapping speaker label to reference MFCC embedding vector.
    sample_rate:
        Audio sample rate in Hz.
    n_mfcc:
        Number of MFCC coefficients.
    top_k:
        Number of top matches to return.
    actor:
        Actor identity for forensic event logging.
    previous_hash:
        Previous hash in the event chain.
    sequence_number:
        Sequence position in the event chain.
    model_provenance_id:
        Optional UUID of a parent provenance node.

    Returns
    -------
    VoiceInferenceResult with matches, input hash, embedding,
    forensic event, and provenance node.
    """
    input_hash = _hash_bytes(audio_data)

    embedding = extract_voice_embedding(audio_data, sample_rate=sample_rate, n_mfcc=n_mfcc)

    # Compute cosine similarity against all reference embeddings
    scores: dict[str, float] = {}
    for label, ref_emb in reference_embeddings.items():
        sim = _cosine_similarity(embedding.embedding, ref_emb)
        # Clamp to [0, 1] for confidence interpretation
        scores[label] = max(0.0, sim)

    # Sort by confidence descending, take top-K
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    actual_k = min(top_k, len(sorted_scores))
    matches = [VoiceMatch(label=lbl, confidence=conf) for lbl, conf in sorted_scores[:actual_k]]

    # Log forensic event
    forensic_event = ForensicEvent(
        event_type=EventType.INFERENCE_RESULT,
        actor=actor,
        payload={
            "modality": "voice",
            "input_hash": input_hash,
            "sample_rate": sample_rate,
            "n_mfcc": n_mfcc,
            "duration_seconds": embedding.duration_seconds,
            "artifact_hash": embedding.artifact_hash,
            "top_k": actual_k,
            "matches": [{"label": m.label, "confidence": m.confidence} for m in matches],
        },
        previous_hash=previous_hash,
        sequence_number=sequence_number,
    ).seal()

    # Create provenance node
    parent_ids = []
    relations = []
    if model_provenance_id is not None:
        parent_ids = [model_provenance_id]
        relations = [ProvenanceRelation.GENERATED_BY]

    provenance_node = ProvenanceNode(
        entity_type="inference",
        entity_id=str(forensic_event.id),
        metadata={
            "modality": "voice",
            "input_hash": input_hash,
            "top_k": actual_k,
            "confidence": matches[0].confidence if matches else 0.0,
            "sample_rate": sample_rate,
            "n_mfcc": n_mfcc,
            "duration_seconds": embedding.duration_seconds,
        },
        parents=parent_ids,
        relations=relations,
    ).seal()

    return VoiceInferenceResult(
        matches=matches,
        input_hash=input_hash,
        embedding=embedding,
        forensic_event=forensic_event,
        provenance_node=provenance_node,
    )
