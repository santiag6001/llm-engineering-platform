from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_platform.config import Settings


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llama_server_timeout_seconds", 0),
        ("llama_server_stream_idle_timeout_seconds", 0),
        ("llama_server_stream_event_max_bytes", 0),
    ],
)
def test_non_positive_stream_limits_are_rejected(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({field: value})
