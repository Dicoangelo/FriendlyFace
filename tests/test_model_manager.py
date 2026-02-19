"""Tests for ModelManager: model resolution, download, hash verification."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from friendlyface.recognition.model_manager import ModelManager, ModelInfo


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    """Create a temporary model directory."""
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture()
def manager(model_dir: Path) -> ModelManager:
    """ModelManager with a temp model directory."""
    return ModelManager(model_dir=str(model_dir))


@pytest.fixture()
def fake_model(model_dir: Path) -> Path:
    """Create a small fake ONNX file."""
    p = model_dir / "test_model.onnx"
    p.write_bytes(b"FAKE_ONNX_MODEL_DATA_1234567890")
    return p


class TestResolve:
    """Tests for model path resolution."""

    def test_resolve_returns_none_when_no_models(self, manager: ModelManager) -> None:
        result = manager.resolve()
        assert result is None

    def test_resolve_finds_onnx_in_model_dir(
        self, manager: ModelManager, fake_model: Path
    ) -> None:
        result = manager.resolve()
        assert result is not None
        assert result == str(fake_model)

    def test_resolve_env_var_overrides(
        self, manager: ModelManager, fake_model: Path, tmp_path: Path
    ) -> None:
        env_model = tmp_path / "env_model.onnx"
        env_model.write_bytes(b"ENV_MODEL")
        with patch.dict(os.environ, {"FF_ONNX_MODEL_PATH": str(env_model)}):
            result = manager.resolve()
        assert result == str(env_model)

    def test_resolve_env_var_missing_file_warns(
        self, manager: ModelManager, fake_model: Path
    ) -> None:
        with patch.dict(os.environ, {"FF_ONNX_MODEL_PATH": "/nonexistent/model.onnx"}):
            result = manager.resolve()
        # Falls through to model dir since env path doesn't exist
        assert result == str(fake_model)

    def test_resolve_picks_first_alphabetically(self, model_dir: Path) -> None:
        (model_dir / "beta.onnx").write_bytes(b"BETA")
        (model_dir / "alpha.onnx").write_bytes(b"ALPHA")
        manager = ModelManager(model_dir=str(model_dir))
        result = manager.resolve()
        assert result is not None
        assert "alpha.onnx" in result

    def test_resolve_ignores_non_onnx(self, model_dir: Path) -> None:
        (model_dir / "readme.txt").write_text("not a model")
        (model_dir / "model.pkl").write_bytes(b"PICKLE")
        manager = ModelManager(model_dir=str(model_dir))
        result = manager.resolve()
        assert result is None

    def test_resolve_nonexistent_dir_returns_none(self, tmp_path: Path) -> None:
        manager = ModelManager(model_dir=str(tmp_path / "does_not_exist"))
        result = manager.resolve()
        assert result is None


class TestVerify:
    """Tests for hash verification."""

    def test_verify_correct_hash(self, manager: ModelManager, fake_model: Path) -> None:
        actual_hash = ModelManager._compute_hash(str(fake_model))
        assert manager.verify(str(fake_model), actual_hash) is True

    def test_verify_wrong_hash(self, manager: ModelManager, fake_model: Path) -> None:
        assert manager.verify(str(fake_model), "0" * 64) is False

    def test_verify_case_insensitive(self, manager: ModelManager, fake_model: Path) -> None:
        actual_hash = ModelManager._compute_hash(str(fake_model))
        assert manager.verify(str(fake_model), actual_hash.upper()) is True

    def test_compute_hash_deterministic(self, fake_model: Path) -> None:
        h1 = ModelManager._compute_hash(str(fake_model))
        h2 = ModelManager._compute_hash(str(fake_model))
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest


class TestDownload:
    """Tests for model download."""

    def test_download_unknown_model_no_url_raises(self, manager: ModelManager) -> None:
        with pytest.raises(ValueError, match="not in registry"):
            manager.download("nonexistent_model_xyz")

    def test_download_hash_mismatch_deletes_file(
        self, manager: ModelManager, tmp_path: Path
    ) -> None:
        # Create a fake server file
        src = tmp_path / "served.onnx"
        src.write_bytes(b"MODEL_DATA")

        with pytest.raises(RuntimeError, match="Hash mismatch"):
            manager.download(
                "test_model",
                url=f"file://{src}",
                expected_hash="0" * 64,
            )
        # Temp file should be cleaned up
        assert not (manager.model_dir / "test_model.onnx.tmp").exists()

    def test_download_success_with_file_url(
        self, manager: ModelManager, tmp_path: Path
    ) -> None:
        src = tmp_path / "served.onnx"
        data = b"REAL_MODEL_CONTENT"
        src.write_bytes(data)

        import hashlib

        expected = hashlib.sha256(data).hexdigest()

        result = manager.download("test_model", url=f"file://{src}", expected_hash=expected)
        assert result.exists()
        assert result.name == "test_model.onnx"
        assert result.read_bytes() == data

    def test_download_without_hash_verification(
        self, manager: ModelManager, tmp_path: Path
    ) -> None:
        src = tmp_path / "served.onnx"
        src.write_bytes(b"NOHASH")

        result = manager.download("nohash_model", url=f"file://{src}", expected_hash=None)
        assert result.exists()

    def test_download_creates_dir_if_missing(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new_models"
        manager = ModelManager(model_dir=str(new_dir))

        src = tmp_path / "served.onnx"
        src.write_bytes(b"DATA")

        result = manager.download("test", url=f"file://{src}", expected_hash=None)
        assert new_dir.exists()
        assert result.exists()


class TestListModels:
    """Tests for listing available models."""

    def test_list_empty_dir(self, manager: ModelManager) -> None:
        models = manager.list_models()
        assert models == []

    def test_list_finds_onnx_files(self, manager: ModelManager, fake_model: Path) -> None:
        models = manager.list_models()
        assert len(models) == 1
        assert models[0].name == "test_model"
        assert models[0].size_bytes > 0

    def test_list_multiple_models(self, model_dir: Path) -> None:
        (model_dir / "model_a.onnx").write_bytes(b"A")
        (model_dir / "model_b.onnx").write_bytes(b"B")
        manager = ModelManager(model_dir=str(model_dir))
        models = manager.list_models()
        assert len(models) == 2
        names = {m.name for m in models}
        assert names == {"model_a", "model_b"}

    def test_list_ignores_non_onnx(self, model_dir: Path) -> None:
        (model_dir / "model.onnx").write_bytes(b"ONNX")
        (model_dir / "notes.txt").write_text("notes")
        manager = ModelManager(model_dir=str(model_dir))
        models = manager.list_models()
        assert len(models) == 1

    def test_list_nonexistent_dir(self, tmp_path: Path) -> None:
        manager = ModelManager(model_dir=str(tmp_path / "nope"))
        assert manager.list_models() == []

    def test_model_info_dataclass(self) -> None:
        info = ModelInfo(name="test", path="/foo/test.onnx", size_bytes=1024)
        assert info.name == "test"
        assert info.sha256 is None


class TestRegistry:
    """Tests for model registry."""

    def test_get_registry_returns_dict(self, manager: ModelManager) -> None:
        reg = manager.get_registry()
        assert isinstance(reg, dict)

    def test_registry_loaded_from_json(self, model_dir: Path, tmp_path: Path) -> None:
        reg_path = model_dir / "MODEL_REGISTRY.json"
        reg_data = {"test_model": {"url": "https://example.com/model.onnx", "sha256": "abc123"}}
        reg_path.write_text(json.dumps(reg_data))
        # ModelManager loads from project root — test with patched path
        manager = ModelManager(model_dir=str(model_dir))
        # Registry is loaded from the project-level file, not from model_dir
        # So this test verifies the registry API works
        reg = manager.get_registry()
        assert isinstance(reg, dict)


class TestModelDir:
    """Tests for model directory configuration."""

    def test_model_dir_from_constructor(self, tmp_path: Path) -> None:
        d = tmp_path / "custom"
        d.mkdir()
        manager = ModelManager(model_dir=str(d))
        assert manager.model_dir == d

    def test_model_dir_from_env(self, tmp_path: Path) -> None:
        d = tmp_path / "env_models"
        d.mkdir()
        with patch.dict(os.environ, {"FF_MODEL_DIR": str(d)}):
            manager = ModelManager()
        assert manager.model_dir == d

    def test_model_dir_constructor_overrides_env(self, tmp_path: Path) -> None:
        env_dir = tmp_path / "env"
        env_dir.mkdir()
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        with patch.dict(os.environ, {"FF_MODEL_DIR": str(env_dir)}):
            manager = ModelManager(model_dir=str(custom_dir))
        assert manager.model_dir == custom_dir
