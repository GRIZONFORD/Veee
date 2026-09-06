"""config.py — Carga de configuracion YAML con sobrescritura por variables de entorno."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("config/config.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Lee el YAML de configuracion. `VEEE_DB` sobrescribe la ruta de la base."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"No se encontro {p}. Copie config/config.example.yaml a config/config.yaml."
        )
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if (db := os.getenv("VEEE_DB")):
        cfg.setdefault("storage", {})["db_path"] = db
    return cfg


def model_params_from_cfg(cfg: dict[str, Any]):
    """Instancia ModelParams a partir del bloque `model` del YAML."""
    from .model import ModelParams
    return ModelParams(**{k: v for k, v in (cfg.get("model") or {}).items()
                          if k in ModelParams.__dataclass_fields__})
