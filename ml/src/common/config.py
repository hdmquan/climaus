import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from .logger import setup_logger

logger = setup_logger()

REQUIRED_KEYS = {
    "model": ["name"],
    "train": ["epochs", "batch_size", "learning_rate"],
}

OPTIONAL_SECTIONS = {
    "output": {
        "log_dir": "outputs/logs",
        "model_dir": "models/weights",
        "version": None,
        "note": "",
    }
}


def _validate_config(config: Dict[str, Any]):
    logger.debug("🔍 Validating config structure...")

    for section, keys in REQUIRED_KEYS.items():

        if section not in config:
            raise ValueError(f"Missing required section: '{section}'")

        for key in keys:

            if key not in config[section]:
                raise ValueError(f"Missing required key '{key}' in section '{section}'")

    for section, defaults in OPTIONAL_SECTIONS.items():

        if section not in config:
            logger.warning(f"Optional section '{section}' missing, using defaults.")
            config[section] = defaults

        else:
            for key, default in defaults.items():
                config[section].setdefault(key, default)

    known_sections = set(REQUIRED_KEYS.keys()) | set(OPTIONAL_SECTIONS.keys())

    for section in config:
        if section not in known_sections:
            logger.warning(f"Unknown config section: '{section}'")

    logger.debug("Config structure validated.")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Invalid config format — must be a YAML dict")

    logger.info(f"📘 Loaded config from {path}")
    _validate_config(config)

    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]):

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    logger.info(f"Config saved to {path}")
