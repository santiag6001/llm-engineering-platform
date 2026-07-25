"""Typed environment-driven settings."""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class Settings(BaseModel):
    """Validated process configuration."""

    model_config = ConfigDict(frozen=True)

    llama_server_base_url: HttpUrl = HttpUrl("http://127.0.0.1:8080")
    llama_server_timeout_seconds: float = Field(default=120.0, gt=0)
    llama_server_stream_idle_timeout_seconds: float = Field(default=30.0, gt=0)
    llama_server_stream_event_max_bytes: int = Field(default=1_048_576, gt=0)
    public_model: str = Field(default="local-model", min_length=1)

    @classmethod
    def from_environment(cls) -> Settings:
        """Load settings from supported environment variables."""

        values: dict[str, str] = {}
        variable_names = {
            "LLAMA_SERVER_BASE_URL": "llama_server_base_url",
            "LLAMA_SERVER_TIMEOUT_SECONDS": "llama_server_timeout_seconds",
            "LLAMA_SERVER_STREAM_IDLE_TIMEOUT_SECONDS": (
                "llama_server_stream_idle_timeout_seconds"
            ),
            "LLAMA_SERVER_STREAM_EVENT_MAX_BYTES": (
                "llama_server_stream_event_max_bytes"
            ),
            "LLM_PLATFORM_MODEL": "public_model",
        }
        for environment_name, field_name in variable_names.items():
            if value := os.environ.get(environment_name):
                values[field_name] = value
        return cls.model_validate(values)

    @property
    def backend_base_url(self) -> str:
        """Return the normalized base URL without a trailing slash."""

        return str(self.llama_server_base_url).rstrip("/")
