from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class Condition:
    id: str
    confidence: str
    style: str
    perspective: str
    context: str
    target: str
    wrong_suggestion: bool


@dataclass(frozen=True)
class Mitigation:
    id: str
    type: str
    instruction: str | None = None


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    seed: int
    models: list[ModelSpec]
    conditions: list[Condition]
    mitigations: list[Mitigation]

    @property
    def dataset(self) -> dict[str, Any]:
        return self.raw["dataset"]

    @property
    def experiment(self) -> dict[str, Any]:
        return self.raw["experiment"]


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"dataset", "experiment", "models", "conditions", "mitigations"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    models = [ModelSpec(**item) for item in raw["models"]]
    conditions = [Condition(**item) for item in raw["conditions"]]
    mitigations = [Mitigation(**item) for item in raw["mitigations"]]
    if not any(m.type == "none" for m in mitigations):
        raise ValueError("At least one mitigation with type=none is required")
    if len({m.id for m in models}) != len(models):
        raise ValueError("Model ids must be unique")
    if len({c.id for c in conditions}) != len(conditions):
        raise ValueError("Condition ids must be unique")

    return Config(
        raw=raw,
        path=path,
        seed=int(raw.get("seed", 7650)),
        models=models,
        conditions=conditions,
        mitigations=mitigations,
    )


def resolve_path(config: Config, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config.path.parent / path
    return path.resolve()

