"""Application configuration via environment variables (.env supported)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VISTA_",
        extra="ignore",
    )

    # --- Paths ---
    data_dir: Path = Path("data")

    # --- LLM ---
    llm_provider: str = "rule-based"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    embedding_model: str = "text-embedding-3-small"
    # Minimum cosine similarity for a semantic object match to be accepted.
    semantic_retrieval_threshold: float = 0.85

    # --- Detection ---
    # "synthetic" (ColorDetector, deterministic tests/demo) or
    # "real" (YOLO ONNX via cv2.dnn).
    detector_mode: str = "synthetic"
    yolo_model_path: str | None = None
    model_size: str = "yolov8s"
    detection_confidence_threshold: float = 0.25

    # --- Tracking ---
    tracking_iou_threshold: float = 0.3
    occlusion_tolerance_frames: int = 30
    reid_similarity_threshold: float = 0.6

    # --- Frame sampling ---
    sampling_strategy: str = "adaptive"
    sampling_fps: float = 2.0

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def evidence_dir(self) -> Path:
        return self.data_dir / "evidence"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    def ensure_dirs(self) -> None:
        for directory in (
            self.data_dir,
            self.uploads_dir,
            self.evidence_dir,
            self.artifacts_dir,
            self.models_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
