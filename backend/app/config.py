"""Application configuration with YAML loading and sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    """All tunable parameters for the Live Talking Head Avatar system."""

    # RAG settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"

    # LLM settings
    llm_model: str = "llama3.2:3b"
    llm_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512

    # TTS settings
    tts_model_path: str = "./models/piper/en_US-lessac-medium.onnx"
    tts_sample_rate: int = 22050

    # Avatar settings
    avatar_checkpoint_dir: str = "./models/sadtalker/"
    avatar_fps: int = 25
    avatar_resolution: tuple[int, int] = (256, 256)

    # Storage
    chroma_persist_dir: str = "./data/chroma"
    media_output_dir: str = "./data/media"

    # Validation limits
    max_image_size_mb: int = 10
    max_pdf_size_mb: int = 50
    min_image_resolution: tuple[int, int] = (256, 256)
    allowed_image_formats: list[str] = field(
        default_factory=lambda: ["png", "jpg", "jpeg"]
    )


def _apply_overrides(config: AppConfig, data: dict[str, Any]) -> AppConfig:
    """Apply a flat or nested dict of overrides onto an AppConfig instance."""
    for key, value in data.items():
        if not hasattr(config, key):
            continue
        current = getattr(config, key)
        # Convert lists to tuples where the default is a tuple
        if isinstance(current, tuple) and isinstance(value, (list, tuple)):
            value = tuple(value)
        setattr(config, key, value)
    return config


def load_config(path: str = "config/config.yaml") -> AppConfig:
    """Load configuration from a YAML file, falling back to defaults.

    Parameters
    ----------
    path:
        Path to the YAML configuration file. If the file does not exist
        or cannot be parsed, a default ``AppConfig`` is returned.

    Returns
    -------
    AppConfig
        Populated configuration dataclass.
    """
    config = AppConfig()
    config_path = Path(path)

    if not config_path.is_file():
        return config

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except (yaml.YAMLError, OSError):
        return config

    if not isinstance(raw, dict):
        return config

    return _apply_overrides(config, raw)
