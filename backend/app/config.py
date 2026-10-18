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

    # --- Storage ---
    # "local": local disk only. "s3": also mirror video, memory DB and
    # evidence frames to Amazon S3 (credentials from the AWS default chain).
    storage: str = "local"
    s3_bucket: str | None = None
    s3_prefix: str = "vista"
    aws_region: str | None = None

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8000
    # Directory of the built frontend to serve from the API (single container).
    frontend_dir: Path = Path("frontend/dist")

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
    # "real"    : YOLO ONNX via cv2.dnn (required for real footage)
    # "synthetic": ColorDetector, deterministic tests + generated demo only
    # A missing model in "real" mode is an explicit error, never a silent
    # downgrade to the synthetic detector.
    detector_mode: str = "real"
    yolo_model_path: str | None = None
    yolo_class_names_path: str | None = None
    # Optional JSON file mapping raw detector labels -> canonical categories.
    detector_label_map: str | None = None
    # Open-vocabulary (YOLO-World) text labels, comma-separated.
    detector_labels: str = (
        "notebook,desk,shelf,coin,mug,backpack,laptop,phone,person,chair,bottle,tablet"
    )
    openvocab_model: str = "yolov8s-worldv2.pt"
    owlv2_model: str = "google/owlv2-base-patch16-ensemble"
    model_size: str = "yolov8s"
    detection_confidence_threshold: float = 0.25

    @property
    def detector_label_list(self) -> list[str]:
        return [s.strip() for s in self.detector_labels.split(",") if s.strip()]

    # --- Tracking ---
    tracking_iou_threshold: float = 0.3
    occlusion_tolerance_frames: int = 30
    reid_similarity_threshold: float = 0.6

    # --- Frame sampling ---
    sampling_strategy: str = "adaptive"
    sampling_fps: float = 2.0

    # --- Scene change ---
    scene_change_threshold: float = 0.15
    # Mean optical-flow magnitude above which the camera is considered to be
    # moving and scene-change detection is suppressed.
    scene_change_motion_gate: float = 2.0

    # --- Generic unknown-object fallback ---
    # When the configured detector finds nothing, propose persistent salient
    # regions (MOG2 foreground) as unclassified "object" detections. Off by
    # default: MOG2 assumes a static camera and produces false objects on
    # handheld footage.
    saliency_fallback: bool = False
    saliency_min_area: int = 600
    saliency_min_persistence_overlap: float = 0.1

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
