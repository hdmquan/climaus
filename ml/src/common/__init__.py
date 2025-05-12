from pathlib import Path

from .logger import setup_logger
from .config import load_config, save_config, _validate_config
from .versioning import ModelCheckpoint

__all__ = [
    "setup_logger",
    "load_config",
    "save_config",
    "_validate_config",
    "ModelCheckpoint",
]


class PATH:
    ROOT = Path(__file__).parents[2]
    DATA = ROOT / "data"
    R_DATA = DATA / "raw"
    P_DATA = DATA / "processed"
