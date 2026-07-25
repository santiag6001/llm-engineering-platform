from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_platform.config import Settings


def test_non_positive_timeout_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(llama_server_timeout_seconds=0)
