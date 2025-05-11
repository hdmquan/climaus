import pytest
import torch
from pathlib import Path
from src.common.versioning import ModelCheckpoint


@pytest.fixture
def dummy_model():

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 10)

        def forward(self, x):
            return self.fc(x)

    return DummyModel()


@pytest.fixture
def temp_model_dir(tmp_path):
    return tmp_path / "models"


def test_save_and_load_auto_version(dummy_model, temp_model_dir):
    ckpt = ModelCheckpoint(base_dir=temp_model_dir)
    meta = {"foo": "bar"}

    save_path = ckpt.save(dummy_model, meta)
    assert save_path.exists()
    assert (save_path / "model.pt").exists()
    assert (save_path / "meta.json").exists()

    loaded_meta = ckpt.load(dummy_model)
    assert loaded_meta["foo"] == "bar"


def test_save_manual_version(dummy_model, temp_model_dir):
    ckpt = ModelCheckpoint(base_dir=temp_model_dir)
    ckpt.save(dummy_model, {"test": True}, version="v9.9.9")

    manual_path = temp_model_dir / "DummyModel" / "v9.9.9"
    assert manual_path.exists()
    assert (manual_path / "model.pt").exists()


def test_load_specific_version(dummy_model, temp_model_dir):
    ckpt = ModelCheckpoint(base_dir=temp_model_dir)
    ckpt.save(dummy_model, {"version": "one"}, version="v0.1.0")
    ckpt.save(dummy_model, {"version": "two"}, version="v0.2.0")

    meta = ckpt.load(dummy_model, version="v0.1.0")
    assert meta["version"] == "one"

    meta = ckpt.load(dummy_model, version="v0.2.0")
    assert meta["version"] == "two"


def test_load_fails_no_versions(dummy_model, tmp_path):
    ckpt = ModelCheckpoint(base_dir=tmp_path / "models")
    with pytest.raises(FileNotFoundError):
        ckpt.load(dummy_model)


def test_load_fails_missing_weights(dummy_model, temp_model_dir):
    ckpt = ModelCheckpoint(base_dir=temp_model_dir)
    model_dir = temp_model_dir / "DummyModel" / "v0.1.0"
    model_dir.mkdir(parents=True)
    with open(model_dir / "meta.json", "w") as f:
        f.write("{}")

    with pytest.raises(FileNotFoundError):
        ckpt.load(dummy_model, version="v0.1.0")


def test_increment_version_logic():
    ckpt = ModelCheckpoint()
    assert ckpt._increment_version("v0.0.0") == "v0.1.0"
    assert ckpt._increment_version("v0.3.0") == "v0.4.0"
    assert ckpt._increment_version("v1.9.0") == "v1.10.0"
