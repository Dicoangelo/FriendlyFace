"""Unified recognition pipeline composing detection → embedding → gallery → liveness → calibration (US-084).

Provides ``RecognitionPipeline`` that wires the five ML modules into a
single end-to-end flow: detect faces → align → extract embeddings →
search gallery → liveness check → calibrate confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from friendlyface.recognition.calibration import (
    CalibratedScore,
    PlattCalibrator,
    calibrate_with_quality,
)
from friendlyface.recognition.detection import DetectedFace, FaceDetector
from friendlyface.recognition.embeddings import EmbeddingExtractor, FaceEmbedding
from friendlyface.recognition.gallery import FaceGallery, GalleryMatch
from friendlyface.recognition.liveness import LivenessResult, check_liveness

logger = logging.getLogger("friendlyface.recognition.pipeline")


@dataclass
class PipelineResult:
    """Full result from one run of the recognition pipeline."""

    faces_detected: int
    best_face: DetectedFace | None
    embedding: FaceEmbedding | None
    gallery_matches: list[GalleryMatch]
    liveness: LivenessResult | None
    calibration: CalibratedScore | None
    quality_score: float
    details: dict[str, Any] = field(default_factory=dict)


class RecognitionPipeline:
    """End-to-end recognition pipeline.

    Composes: FaceDetector → EmbeddingExtractor → FaceGallery (search) →
    check_liveness → calibrate_with_quality.

    Parameters
    ----------
    gallery : FaceGallery
        Database-backed gallery for 1:N search.
    detector : FaceDetector | None
        Face detector. Created with defaults if *None*.
    extractor : EmbeddingExtractor | None
        Embedding extractor. Created with defaults if *None*.
    calibrator : PlattCalibrator | None
        Optional Platt calibrator (fitted). Quality-only if *None*.
    liveness_threshold : float
        Minimum liveness score to pass.
    """

    def __init__(
        self,
        gallery: FaceGallery,
        detector: FaceDetector | None = None,
        extractor: EmbeddingExtractor | None = None,
        calibrator: PlattCalibrator | None = None,
        liveness_threshold: float = 0.5,
    ) -> None:
        self.gallery = gallery
        self.detector = detector or FaceDetector()
        self.extractor = extractor or EmbeddingExtractor()
        self.calibrator = calibrator
        self.liveness_threshold = liveness_threshold

    async def run(
        self,
        image: np.ndarray,
        top_k: int = 5,
        skip_liveness: bool = False,
    ) -> PipelineResult:
        """Execute the full pipeline on a single image.

        Parameters
        ----------
        image : np.ndarray
            Input image (H, W, 3) RGB uint8.
        top_k : int
            Maximum gallery matches to return.
        skip_liveness : bool
            If *True*, skip the liveness check.

        Returns
        -------
        PipelineResult
            Aggregated result from every pipeline stage.
        """
        # 1. Detect faces
        faces = self.detector.detect(image)
        if not faces:
            return PipelineResult(
                faces_detected=0,
                best_face=None,
                embedding=None,
                gallery_matches=[],
                liveness=None,
                calibration=None,
                quality_score=0.0,
                details={"error": "no_faces_detected"},
            )

        best = faces[0]  # highest confidence
        quality = best.quality_score

        # 2. Extract embedding from the aligned face crop
        aligned = best.aligned
        if aligned is None:
            return PipelineResult(
                faces_detected=len(faces),
                best_face=best,
                embedding=None,
                gallery_matches=[],
                liveness=None,
                calibration=None,
                quality_score=quality,
                details={"error": "alignment_failed"},
            )

        embedding = self.extractor.extract(aligned)

        # 3. Gallery search
        gallery_matches = await self.gallery.search(embedding, top_k=top_k)

        # 4. Liveness check (on original crop, not aligned)
        liveness: LivenessResult | None = None
        if not skip_liveness:
            h, w = image.shape[:2]
            x, y, bw, bh = best.bbox
            face_crop = image[
                max(y, 0) : min(y + bh, h),
                max(x, 0) : min(x + bw, w),
            ]
            if face_crop.size > 0:
                liveness = check_liveness(face_crop, threshold=self.liveness_threshold)

        # 5. Calibrate the top match confidence (if any matches)
        calibration: CalibratedScore | None = None
        if gallery_matches:
            raw_score = gallery_matches[0].similarity
            calibration = calibrate_with_quality(
                raw_score=raw_score,
                quality_score=quality,
                calibrator=self.calibrator,
            )

        return PipelineResult(
            faces_detected=len(faces),
            best_face=best,
            embedding=embedding,
            gallery_matches=gallery_matches,
            liveness=liveness,
            calibration=calibration,
            quality_score=quality,
            details={
                "detector_backend": self.detector.backend,
                "embedding_backend": self.extractor.backend,
                "embedding_model": self.extractor.model_version,
            },
        )
