"""Tests for the AppConfig dataclass and YAML configuration loader."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from backend.app.config import AppConfig, load_config


class TestAppConfigDefaults:
    """Verify all default values match the design specification."""

    def test_rag_defaults(self):
        cfg = AppConfig()
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 50
        assert cfg.retrieval_top_k == 5
        assert cfg.embedding_model == "all-MiniLM-L6-v2"

    def test_llm_defaults(self):
        cfg = AppConfig()
        assert cfg.llm_model == "llama3.2:3b"
        assert cfg.llm_base_url == "http://localhost:11434"
        assert cfg.llm_temperature == 0.1
        assert cfg.llm_max_tokens == 512

    def test_tts_defaults(self):
        cfg = AppConfig()
        assert cfg.tts_model_path == "./models/piper/en_US-lessac-medium.onnx"
        assert cfg.tts_sample_rate == 22050

    def test_avatar_defaults(self):
        cfg = AppConfig()
        assert cfg.avatar_checkpoint_dir == "./models/sadtalker/"
        assert cfg.avatar_fps == 25
        assert cfg.avatar_resolution == (256, 256)

    def test_storage_defaults(self):
        cfg = AppConfig()
        assert cfg.chroma_persist_dir == "./data/chroma"
        assert cfg.media_output_dir == "./data/media"

    def test_validation_defaults(self):
        cfg = AppConfig()
        assert cfg.max_image_size_mb == 10
        assert cfg.max_pdf_size_mb == 50
        assert cfg.min_image_resolution == (256, 256)
        assert cfg.allowed_image_formats == ["png", "jpg", "jpeg"]


class TestLoadConfig:
    """Test YAML configuration loading with various scenarios."""

    def test_missing_file_returns_defaults(self, tmp_path):
        cfg = load_config(str(tmp_path / "nonexistent.yaml"))
        assert cfg == AppConfig()

    def test_empty_file_returns_defaults(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        cfg = load_config(str(empty))
        assert cfg == AppConfig()

    def test_partial_override(self, tmp_path):
        partial = tmp_path / "partial.yaml"
        partial.write_text(yaml.dump({"chunk_size": 1024, "retrieval_top_k": 10}))
        cfg = load_config(str(partial))
        assert cfg.chunk_size == 1024
        assert cfg.retrieval_top_k == 10
        # Other fields stay at defaults
        assert cfg.chunk_overlap == 50
        assert cfg.llm_model == "llama3.2:3b"

    def test_full_override(self, tmp_path):
        data = {
            "chunk_size": 256,
            "chunk_overlap": 25,
            "retrieval_top_k": 3,
            "embedding_model": "custom-model",
            "llm_model": "mistral:7b",
            "llm_base_url": "http://remote:11434",
            "llm_temperature": 0.5,
            "llm_max_tokens": 1024,
            "tts_model_path": "/opt/models/tts.onnx",
            "tts_sample_rate": 44100,
            "avatar_checkpoint_dir": "/opt/models/avatar/",
            "avatar_fps": 30,
            "avatar_resolution": [512, 512],
            "chroma_persist_dir": "/data/chroma",
            "media_output_dir": "/data/media",
            "max_image_size_mb": 20,
            "max_pdf_size_mb": 100,
            "min_image_resolution": [128, 128],
            "allowed_image_formats": ["png", "webp"],
        }
        full = tmp_path / "full.yaml"
        full.write_text(yaml.dump(data))
        cfg = load_config(str(full))

        assert cfg.chunk_size == 256
        assert cfg.llm_model == "mistral:7b"
        assert cfg.avatar_resolution == (512, 512)
        assert cfg.min_image_resolution == (128, 128)
        assert cfg.allowed_image_formats == ["png", "webp"]
        assert cfg.tts_sample_rate == 44100

    def test_unknown_keys_are_ignored(self, tmp_path):
        f = tmp_path / "extra.yaml"
        f.write_text(yaml.dump({"chunk_size": 100, "unknown_key": "value"}))
        cfg = load_config(str(f))
        assert cfg.chunk_size == 100
        assert not hasattr(cfg, "unknown_key")

    def test_invalid_yaml_returns_defaults(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(": : : not valid yaml [[[")
        cfg = load_config(str(bad))
        assert cfg == AppConfig()

    def test_tuple_fields_from_list(self, tmp_path):
        f = tmp_path / "tuples.yaml"
        f.write_text(yaml.dump({"avatar_resolution": [640, 480]}))
        cfg = load_config(str(f))
        assert cfg.avatar_resolution == (640, 480)
        assert isinstance(cfg.avatar_resolution, tuple)

    def test_loads_real_config_file(self):
        """Verify the shipped config/config.yaml loads without error."""
        cfg = load_config("config/config.yaml")
        assert cfg.chunk_size == 512
        assert cfg.avatar_resolution == (256, 256)
