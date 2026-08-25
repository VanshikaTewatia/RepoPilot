"""Application configuration using Pydantic Settings (Pydantic v2).

Loads configuration from environment variables and an optional .env file with
type validation, sensible defaults, and environment-aware helpers.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Union
import json

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application configuration and environment variables."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Application & Environment
    # -------------------------------------------------------------------------
    app_name: str = Field(
        default="RepoPilot",
        description="Name of the application",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Current application version",
    )
    environment: Literal["development", "staging", "production", "test"] = Field(
        default="development",
        description="Deployment environment runtime mode",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode for verbose error messages",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging severity level",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Host interface to bind the API server",
    )
    port: int = Field(
        default=8000,
        description="Port to bind the API server",
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins for frontend communication",
    )

    # -------------------------------------------------------------------------
    # Gemini AI / LLM Configuration
    # -------------------------------------------------------------------------
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key for LLM reasoning and embeddings",
    )
    gemini_model_name: str = Field(
        default="gemini-3.6-flash",
        description="Default Gemini model for agent reasoning and patch generation",
    )
    gemini_embedding_model: str = Field(
        default="gemini-embedding-2",
        description="Gemini model used for vector embedding generation",
    )
    gemini_embedding_dimension: int = Field(
        default=3072,
        description="Dimension of Gemini embedding vectors",
    )
    gemini_embedding_batch_size: int = Field(
        default=20,
        description="Number of text chunks sent per embedding API request",
    )
    gemini_embedding_tpm_limit: int = Field(
        default=30000,
        description="Gemini embedding tokens-per-minute quota to stay under when pacing requests",
    )
    gemini_embedding_rate_limit_window_seconds: float = Field(
        default=60.0,
        description="Sliding window size in seconds used to enforce the TPM budget",
    )
    gemini_embedding_max_retries: int = Field(
        default=5,
        description="Maximum attempts for a single embedding API call before giving up",
    )
    gemini_embedding_retry_base_seconds: float = Field(
        default=2.0,
        description="Base delay for exponential backoff between embedding retry attempts",
    )
    gemini_embedding_retry_max_seconds: float = Field(
        default=60.0,
        description="Upper bound on any single retry delay (including Retry-After) in seconds",
    )

    # -------------------------------------------------------------------------
    # Database (PostgreSQL 16 + pgvector)
    # -------------------------------------------------------------------------
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/repopilot",
        description="Async SQLAlchemy database connection URL (PostgreSQL + asyncpg)",
    )
    db_echo: bool = Field(
        default=False,
        description="Whether SQLAlchemy should log SQL queries to stdout",
    )
    db_pool_size: int = Field(
        default=10,
        description="Number of persistent database connections to maintain in the pool",
    )
    db_max_overflow: int = Field(
        default=20,
        description="Maximum number of overflow connections beyond db_pool_size",
    )

    # -------------------------------------------------------------------------
    # GitHub Integration
    # -------------------------------------------------------------------------
    github_token: str = Field(
        default="",
        description="GitHub Personal Access Token for repo cloning and PR creation",
    )
    github_api_base_url: str = Field(
        default="https://api.github.com",
        description="GitHub REST API base URL (overridable in tests to point at a mock server)",
    )

    # -------------------------------------------------------------------------
    # Workspace & Docker Sandbox Execution
    # -------------------------------------------------------------------------
    workspace_dir: Path = Field(
        default=Path("./storage/workspaces"),
        description="Filesystem root path for cloned repositories and task workspaces",
    )
    docker_sandbox_image: str = Field(
        default="python:3.11-slim",
        description="Base Docker image used for sandbox test execution",
    )
    sandbox_timeout_seconds: int = Field(
        default=45,
        description="Hard execution timeout in seconds for container test runs",
    )
    sandbox_max_cpu: float = Field(
        default=2.0,
        description="Maximum CPU cores allocated to a sandbox container",
    )
    sandbox_max_memory_mb: int = Field(
        default=2048,
        description="Maximum memory in Megabytes allocated to a sandbox container",
    )
    sandbox_network_mode: str = Field(
        default="none",
        description="Container network isolation mode (default: 'none' for locked down sandbox)",
    )

    # -------------------------------------------------------------------------
    # Agent Reasoning Loop & Search
    # -------------------------------------------------------------------------
    agent_max_iterations: int = Field(
        default=3,
        description="Maximum retry attempts in the iterative self-correction loop",
    )
    vector_top_k: int = Field(
        default=10,
        description="Default number of code chunks returned from vector retrieval",
    )

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------
    @field_validator("cors_origins", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Support comma-separated strings or JSON arrays in environment variables."""
        if isinstance(v, str):
            v_trimmed = v.strip()
            if not v_trimmed:
                return []
            if v_trimmed.startswith("[") and v_trimmed.endswith("]"):
                try:
                    parsed = json.loads(v_trimmed)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed]
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in v_trimmed.split(",") if item.strip()]
        return v

    @field_validator("workspace_dir", mode="before")
    @classmethod
    def resolve_workspace_path(cls, v: Union[str, Path]) -> Path:
        """Convert string paths to Path instances."""
        if isinstance(v, str):
            return Path(v)
        return v

    # -------------------------------------------------------------------------
    # Convenience Properties
    # -------------------------------------------------------------------------
    @property
    def is_development(self) -> bool:
        """Check if currently running in development environment."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if currently running in production environment."""
        return self.environment == "production"

    @property
    def is_test(self) -> bool:
        """Check if currently running in test environment."""
        return self.environment == "test"

    @property
    def sync_database_url(self) -> str:
        """Return synchronous database URL for tools like Alembic migrations."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://")


@lru_cache()
def get_settings() -> Settings:
    """Create and cache application settings instance."""
    return Settings()


# Global settings instance for easy direct import
settings: Settings = get_settings()
