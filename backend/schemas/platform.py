from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DatasetStatus(str, Enum):
    uploaded = "uploaded"
    validating = "validating"
    valid = "valid"
    failed = "failed"
    indexed = "indexed"
    training = "training"
    ready = "ready"


class PredictionStatus(str, Enum):
    queued = "queued"
    preprocessing = "preprocessing"
    segmenting = "segmenting"
    embedding = "embedding"
    ocr = "ocr"
    correcting = "correcting"
    translating = "translating"
    completed = "completed"
    needs_review = "needs_review"
    failed = "failed"


class DatasetStats(BaseModel):
    total_files: int = 0
    valid_images: int = 0
    scanned_pdfs: int = 0
    annotations: int = 0
    corrupted_files: int = 0
    duplicate_files: int = 0
    formats: Dict[str, int] = Field(default_factory=dict)
    resolutions: Dict[str, int] = Field(default_factory=dict)
    mean_width: Optional[float] = None
    mean_height: Optional[float] = None
    low_light_ratio: Optional[float] = None
    blur_ratio: Optional[float] = None
    estimated_script_count: Optional[int] = None


class DatasetCreateResponse(BaseModel):
    dataset_id: str
    status: DatasetStatus
    stats: DatasetStats
    warnings: List[str] = Field(default_factory=list)
    metadata_path: str


class PipelineStage(BaseModel):
    name: str
    status: str
    confidence: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    prediction_id: str
    status: PredictionStatus
    raw_text: str = ""
    corrected_text: str = ""
    translated_text: str = ""
    confidence: float = 0.0
    unknown_symbol_ratio: float = 0.0
    anomaly_score: float = 0.0
    nearest_symbols: List[Dict[str, Any]] = Field(default_factory=list)
    stages: List[PipelineStage] = Field(default_factory=list)
    requires_human_review: bool = False


class TrainingRequest(BaseModel):
    dataset_id: str
    base_model_version: Optional[str] = None
    strategy: str = "few_shot_domain_adaptation"
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 3e-5
    mixed_precision: bool = True
    enable_self_supervised_warmup: bool = True
    enable_contrastive_learning: bool = True
    freeze_encoder_epochs: int = 2


class TrainingJobResponse(BaseModel):
    job_id: str
    dataset_id: str
    status: str
    message: str


class AnnotationCorrection(BaseModel):
    prediction_id: str
    corrected_text: str
    corrected_symbols: List[Dict[str, Any]] = Field(default_factory=list)
    notes: Optional[str] = None
