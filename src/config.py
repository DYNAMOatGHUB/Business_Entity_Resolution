"""Configuration loader and seed utilities."""

from pathlib import Path
import os
import random
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: str | Path) -> Path:
    """Return ``path`` as-is if it exists (or is absolute), else resolve it against the repo root."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p


def load_config(path: str = "configs/default.yaml") -> dict:
    """Load configuration from a YAML file (relative paths fall back to the repo root)."""
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(resolved, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
