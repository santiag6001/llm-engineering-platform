from __future__ import annotations

from pathlib import Path

import pytest
from scripts.experiment_registry_smoke import run_smoke


@pytest.mark.asyncio
async def test_offline_registry_smoke(tmp_path: Path) -> None:
    await run_smoke(tmp_path)
