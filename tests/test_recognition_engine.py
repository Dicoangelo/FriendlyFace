"""Tests for the pluggable recognition engine interface (US-007)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from friendlyface.recognition.engine import (
    EngineMatch,
    EngineResult,
    FallbackEngine,
    ONNXEngine,
    ProxyEngine,
    RecognitionEngine,
    get_engine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_image_bytes(size: tuple[int, int] = (112, 112)) -> bytes:
    """Create a minimal grayscale PNG as bytes."""
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _train_fallback(tmp_path: Path) -> FallbackEngine:
    """Train a FallbackEngine with synthetic data and return it."""
    engine = FallbackEngine()
    # Create 6 grayscale 112x112 images (2 per class)
    images = [np.random.rand(112, 112).astype(np.float64) for _ in range(6)]
    labels = [0, 0, 1, 1, 2, 2]
    engine.train(images, labels, output_dir=tmp_path / "models")
    return engine


# ---------------------------------------------------------------------------
# test_fallback_engine_wraps_pipeline
# ---------------------------------------------------------------------------


class TestFallbackEngineWrapsPipeline:
    """FallbackEngine delegates to the existing PCA+SVM inference code."""

    def test_predict_uses_pca_svm(self, tmp_path: Path) -> None:
        engine = _train_fallback(tmp_path)
        image_bytes = _make_test_image_bytes()
        result = engine.predict(image_bytes, top_k=3)

        assert isinstance(result, EngineResult)
        assert result.engine_name == "fallback"
        assert len(result.matches) <= 3
        assert all(isinstance(m, EngineMatch) for m in result.matches)
        # Forensic event present
        assert result.inference_event is not None
        assert result.provenance_node is not None

    def test_predict_without_models_raises(self) -> None:
        engine = FallbackEngine()
        with pytest.raises(RuntimeError, match="model paths must be set"):
            engine.predict(b"fake")

    def test_train_produces_model_files(self, tmp_path: Path) -> None:
        engine = _train_fallback(tmp_path)
        info = engine.get_model_info()
        assert info["ready"] is True
        assert Path(info["pca_model_path"]).exists()
        assert Path(info["svm_model_path"]).exists()


# ---------------------------------------------------------------------------
# test_fallback_engine_default
# ---------------------------------------------------------------------------


class TestFallbackEngineDefault:
    """FallbackEngine is the default when no engine is specified."""

    def test_default_engine_is_fallback(self) -> None:
        with patch("friendlyface.config.settings") as mock_settings:
            mock_settings.recognition_engine = "fallback"
            engine = get_engine()
            assert isinstance(engine, FallbackEngine)

    def test_auto_resolves_to_fallback(self) -> None:
        engine = get_engine("auto")
        assert isinstance(engine, FallbackEngine)

    def test_explicit_fallback(self) -> None:
        engine = get_engine("fallback")
        assert isinstance(engine, FallbackEngine)


# ---------------------------------------------------------------------------
# test_onnx_engine_missing_runtime_raises
# ---------------------------------------------------------------------------


class TestONNXEngineMissingRuntime:
    """ONNXEngine raises ImportError with helpful message when onnxruntime is absent."""

    def test_import_error_message(self) -> None:
        import sys

        # Temporarily hide onnxruntime
        real_module = sys.modules.get("onnxruntime")
        sys.modules["onnxruntime"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="onnxruntime"):
                # We need to re-import the class to trigger the import check
                # but the class itself does the check in __init__
                # Force the import to fail inside __init__
                with patch.dict("sys.modules", {"onnxruntime": None}):
                    import builtins

                    original_import = builtins.__import__

                    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
                        if name == "onnxruntime":
                            raise ImportError("No module named 'onnxruntime'")
                        return original_import(name, *args, **kwargs)

                    with patch.object(builtins, "__import__", side_effect=mock_import):
                        ONNXEngine()
        finally:
            if real_module is not None:
                sys.modules["onnxruntime"] = real_module
            else:
                sys.modules.pop("onnxruntime", None)


# ---------------------------------------------------------------------------
# test_proxy_engine_delegates
# ---------------------------------------------------------------------------


class TestProxyEngineDelegates:
    """ProxyEngine forwards to the upstream URL."""

    def test_predict_calls_upstream(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "matches": [
                {"label": 42, "confidence": 0.95},
                {"label": 7, "confidence": 0.80},
            ]
        }

        with patch("httpx.post", return_value=mock_response) as mock_post:
            engine = ProxyEngine(upstream_url="https://api.example.com/recognize")
            result = engine.predict(_make_test_image_bytes(), top_k=5)

            mock_post.assert_called_once()
            assert result.engine_name == "proxy"
            assert len(result.matches) == 2
            assert result.matches[0].label == 42
            assert result.matches[0].confidence == 0.95

    def test_predict_no_url_raises(self) -> None:
        engine = ProxyEngine(upstream_url="")
        with pytest.raises(RuntimeError, match="upstream URL not configured"):
            engine.predict(b"fake")

    def test_train_raises_not_implemented(self) -> None:
        engine = ProxyEngine(upstream_url="https://example.com")
        with pytest.raises(NotImplementedError, match="does not support local training"):
            engine.train([], [])

    def test_model_info(self) -> None:
        engine = ProxyEngine(upstream_url="https://api.example.com")
        info = engine.get_model_info()
        assert info["engine"] == "proxy"
        assert info["ready"] is True
        assert info["upstream_url"] == "https://api.example.com"


# ---------------------------------------------------------------------------
# test_engine_factory_returns_correct_type
# ---------------------------------------------------------------------------


class TestEngineFactory:
    """get_engine() returns the correct engine type."""

    def test_fallback(self) -> None:
        assert isinstance(get_engine("fallback"), FallbackEngine)

    def test_proxy(self) -> None:
        engine = get_engine("proxy", upstream_url="https://example.com")
        assert isinstance(engine, ProxyEngine)

    def test_unknown_engine_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown recognition engine"):
            get_engine("nonexistent")

    def test_case_insensitive(self) -> None:
        assert isinstance(get_engine("FALLBACK"), FallbackEngine)
        assert isinstance(get_engine("Proxy", upstream_url="https://x.com"), ProxyEngine)

    def test_reads_settings_when_none(self) -> None:
        with patch("friendlyface.config.settings") as mock_settings:
            mock_settings.recognition_engine = "fallback"
            engine = get_engine(None)
            assert isinstance(engine, FallbackEngine)


# ---------------------------------------------------------------------------
# test_all_engines_have_same_interface
# ---------------------------------------------------------------------------


class TestAllEnginesHaveSameInterface:
    """All engine classes implement the full RecognitionEngine ABC."""

    @pytest.mark.parametrize("cls", [FallbackEngine, ProxyEngine])
    def test_is_subclass(self, cls: type) -> None:
        assert issubclass(cls, RecognitionEngine)

    @pytest.mark.parametrize("method", ["train", "predict", "get_model_info"])
    def test_fallback_has_method(self, method: str) -> None:
        assert callable(getattr(FallbackEngine, method))

    @pytest.mark.parametrize("method", ["train", "predict", "get_model_info"])
    def test_proxy_has_method(self, method: str) -> None:
        assert callable(getattr(ProxyEngine, method))

    @pytest.mark.parametrize("method", ["train", "predict", "get_model_info"])
    def test_onnx_has_method(self, method: str) -> None:
        assert callable(getattr(ONNXEngine, method))

    def test_onnx_is_subclass(self) -> None:
        assert issubclass(ONNXEngine, RecognitionEngine)

    def test_engine_result_fields(self, tmp_path: Path) -> None:
        """EngineResult from FallbackEngine has all required fields."""
        engine = _train_fallback(tmp_path)
        result = engine.predict(_make_test_image_bytes())

        assert hasattr(result, "matches")
        assert hasattr(result, "input_hash")
        assert hasattr(result, "engine_name")
        assert hasattr(result, "model_info")
        assert hasattr(result, "inference_event")
        assert hasattr(result, "provenance_node")

    def test_get_model_info_has_required_keys(self) -> None:
        """All engines return at least 'engine' and 'ready' from get_model_info."""
        fallback_info = FallbackEngine().get_model_info()
        assert "engine" in fallback_info
        assert "ready" in fallback_info

        proxy_info = ProxyEngine(upstream_url="https://example.com").get_model_info()
        assert "engine" in proxy_info
        assert "ready" in proxy_info
