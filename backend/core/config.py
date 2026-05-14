from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    app_name: str = "Ancient Script AI Platform"
    api_version: str = "v1"
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(True, env="DEBUG")

    secret_key: str = Field("change-me-in-production", env="SECRET_KEY")
    access_token_expire_minutes: int = 60
    allowed_origins: List[str] = ["*"]

    database_url: str = Field("postgresql+asyncpg://ancient:ancient@localhost:5432/ancient_ai", env="DATABASE_URL")
    mongodb_url: str = Field("mongodb://localhost:27017", env="MONGODB_URL")
    mongodb_name: str = "ancient_ai"
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")

    storage_root: Path = Path("storage")
    raw_dataset_dir: Path = Path("storage/datasets/raw")
    processed_dataset_dir: Path = Path("storage/datasets/processed")
    model_registry_dir: Path = Path("storage/models")
    vector_index_dir: Path = Path("storage/vector_indexes")
    max_upload_mb: int = 2048

    vector_backend: str = Field("faiss", env="VECTOR_BACKEND")
    embedding_dim: int = 768

    default_ocr_backend: str = "hybrid"
    enable_wandb: bool = Field(False, env="ENABLE_WANDB")
    wandb_project: str = "ancient-script-ai"
    tensorboard_dir: Path = Path("storage/runs")

    @validator("debug", pre=True)
    def _parse_debug(cls, value):
        if isinstance(value, bool):
            return value
        if value is None:
            return True
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "y", "on", "debug", "dev", "development"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", "release", "prod", "production"}:
            return False
        return False

    @validator("storage_root", "raw_dataset_dir", "processed_dataset_dir", "model_registry_dir", "vector_index_dir", pre=True)
    def _to_path(cls, value):
        return Path(value)

    def ensure_directories(self) -> None:
        for path in [
            self.storage_root,
            self.raw_dataset_dir,
            self.processed_dataset_dir,
            self.model_registry_dir,
            self.vector_index_dir,
            self.tensorboard_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
