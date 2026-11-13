"""Experiment config: one YAML file holds every constant a result depends on."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    full = (ROOT / path).resolve()
    cfg = yaml.safe_load(full.read_text(encoding="utf-8"))
    cfg["config_path"] = str(full.relative_to(ROOT)) if full.is_relative_to(ROOT) else full.name
    return cfg


def load_phrases(cfg: dict, set_name: str) -> list[dict]:
    """Phrases of one evaluation set, each with id, s1, s2 and switch_word."""
    raw = json.loads((ROOT / cfg["sets"][set_name]).read_text(encoding="utf-8"))
    return raw["phrases"]


def set_role(cfg: dict, set_name: str) -> str:
    """'dev' or 'val', as declared in the phrase file itself."""
    return json.loads((ROOT / cfg["sets"][set_name]).read_text(encoding="utf-8"))["role"]
