from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class ModelSpec:
    id: str
    label: str

@dataclass
class Config:
    models: list[ModelSpec]
    temperature: float = 0.0
    n_reps: int = 1
    max_tokens: int = 256
    concurrency: int = 4
    request_pause_s: float = 0.2
    max_retries: int = 5
    base_url: str = "https://api.groq.com/openai/v1"
    templates: list[str] = field(default_factory=list)
    bootstrap_samples: int = 2000
    random_seed: int = 42
    sources: dict = field(default_factory=dict)
    groq_api_key: str | None = None
    db_url: str = "sqlite:///biaslens.db"
    offline: bool = False

    @property
    def template_paths(self) -> list[Path]:
        return [PROJECT_ROOT / t for t in self.templates]

def load_config(path: str | Path | None = None, offline: bool = False) -> Config:
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}

    models = [ModelSpec(id=m["id"], label=m.get("label", m["id"])) for m in raw.get("models", [])]
    cfg = Config(
        models=models,
        temperature=float(raw.get("temperature", 0.0)),
        n_reps=int(raw.get("n_reps", 1)),
        max_tokens=int(raw.get("max_tokens", 256)),
        concurrency=int(raw.get("concurrency", 4)),
        request_pause_s=float(raw.get("request_pause_s", 0.2)),
        max_retries=int(raw.get("max_retries", 5)),
        base_url=raw.get("base_url", "https://api.groq.com/openai/v1"),
        templates=raw.get("templates", []),
        bootstrap_samples=int(raw.get("bootstrap_samples", 2000)),
        random_seed=int(raw.get("random_seed", 42)),
        sources=raw.get("sources", {}),
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        db_url=os.environ.get("DB_URL", raw.get("db_url", "sqlite:///biaslens.db")),
        offline=offline,
    )
    return cfg
