import pytest
from src.common.config import load_config, save_config, _validate_config
from pathlib import Path
import tempfile
import yaml

valid_config = {
    "model": {"name": "TestModel"},
    "train": {"epochs": 10, "batch_size": 4, "learning_rate": 0.001},
}

invalid_config_missing_key = {
    "model": {},
    "train": {"epochs": 10, "batch_size": 4, "learning_rate": 0.001},
}

invalid_config_missing_section = {
    "train": {"epochs": 10, "batch_size": 4, "learning_rate": 0.001},
}


def test_valid_config_passes():
    _validate_config(valid_config.copy())


def test_missing_required_key_raises():
    with pytest.raises(
        ValueError, match="Missing required key 'name' in section 'model'"
    ):
        _validate_config(invalid_config_missing_key)


def test_missing_required_section_raises():
    with pytest.raises(ValueError, match="Missing required section: 'model'"):
        _validate_config(invalid_config_missing_section)


def test_load_config_reads_and_validates(tmp_path: Path):
    config_file = tmp_path / "valid.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(valid_config, f)

    config = load_config(config_file)
    assert config["model"]["name"] == "TestModel"


def test_save_config_writes_file(tmp_path: Path):
    path = tmp_path / "saved_config.yaml"
    save_config(valid_config, path)

    assert path.exists()

    with open(path) as f:
        data = yaml.safe_load(f)
