"""
Configuration management for RTVT-LipSync application.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Storage Configuration
    storage_path: Path = Path("./storage")
    max_upload_size: int = 104857600  # 100MB
    
    # Models Configuration
    models_path: Path = Path("./app/models")
    whisper_model: str = "base"  # tiny, base, small, medium, large
    device: str = "cpu"  # cpu, cuda, mps
    
    # Processing Configuration
    max_workers: int = 2
    job_timeout: int = 3600  # 1 hour
    max_concurrent_jobs: int = 5
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    
    # Optional: Redis Configuration
    redis_url: Optional[str] = None
    redis_max_connections: int = 10
    
    # Optional: MinIO Configuration
    minio_endpoint: Optional[str] = None
    minio_access_key: Optional[str] = None
    minio_secret_key: Optional[str] = None
    minio_bucket: str = "rtvt-lipsync"
    minio_secure: bool = False
    
    # Optional: Database Configuration
    database_url: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create storage directories if they don't exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.storage_path / "uploads",
            self.storage_path / "processing",
            self.storage_path / "outputs",
            self.storage_path / "jobs",
            self.storage_path / "logs",
            self.models_path,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def uploads_dir(self) -> Path:
        """Directory for uploaded files."""
        return self.storage_path / "uploads"
    
    @property
    def processing_dir(self) -> Path:
        """Directory for intermediate processing files."""
        return self.storage_path / "processing"
    
    @property
    def outputs_dir(self) -> Path:
        """Directory for final output files."""
        return self.storage_path / "outputs"
    
    @property
    def jobs_dir(self) -> Path:
        """Directory for job metadata files."""
        return self.storage_path / "jobs"
    
    @property
    def logs_dir(self) -> Path:
        """Directory for log files."""
        return self.storage_path / "logs"
    
    @property
    def use_redis(self) -> bool:
        """Check if Redis is configured."""
        return self.redis_url is not None
    
    @property
    def use_minio(self) -> bool:
        """Check if MinIO is configured."""
        return all([
            self.minio_endpoint,
            self.minio_access_key,
            self.minio_secret_key,
        ])


# Global settings instance
settings = Settings()
