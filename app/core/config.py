import os

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration (type-safe, validated)."""

    # Database
    database_url: str = Field(
        default="sqlite:///./test.db", description="Database connection string"
    )

    #Data
    data_dir: str = Field(
        default=os.path.join(os.path.dirname(__file__), "data"),
        description="Directory for storing raw/preprocessed/prediction data",
    )

    # ML Model
    model_path: str = Field(
        default=os.path.join(
            os.path.dirname(__file__), "models", "house_price_model.joblib"
        ),
        description="Path to trained model file",
    )

    # Logger settings
    log_dir: str = Field(
        default=os.path.join(os.path.dirname(__file__), "logs"),
        description="Path to log file",
    )

    # Application behavior
    log_level: str = Field(
        default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )

    class Config:
        env_file = ".env"  # Load from .env
        case_sensitive = False  # Allow DATABASE_URL or database_url


# Global settings instance
settings = Settings()
