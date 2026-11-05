import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from biaslens.config import load_config  # noqa: E402
from biaslens.db import get_sessionmaker  # noqa: E402

@pytest.fixture
def tmp_cfg(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_file}")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = load_config(offline=True)
    return cfg

@pytest.fixture
def session(tmp_cfg):
    return get_sessionmaker(tmp_cfg.db_url)()
