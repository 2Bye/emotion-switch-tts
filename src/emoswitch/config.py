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


def phrase_file(cfg: dict, set_name: str) -> str:
    """Path of a set's phrase file; a set whose audio is not released yet is under `planned_sets`."""
    return {**cfg.get("planned_sets", {}), **cfg["sets"]}[set_name]


def load_phrases(cfg: dict, set_name: str) -> list[dict]:
    """Phrases of one evaluation set, each with id, s1, s2 and switch_word."""
    return json.loads((ROOT / phrase_file(cfg, set_name)).read_text(encoding="utf-8"))["phrases"]


def set_role(cfg: dict, set_name: str) -> str:
    """'dev', 'val' or 'test', as declared in the phrase file itself."""
    return json.loads((ROOT / phrase_file(cfg, set_name)).read_text(encoding="utf-8"))["role"]
