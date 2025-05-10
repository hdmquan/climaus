import json
import torch
from pathlib import Path
from typing import Optional
from .logger import setup_logger

logger = setup_logger()


class ModelCheckpoint:
    def __init__(self, base_dir="models/weights"):
        self.base_dir = Path(base_dir)

    def _get_model_dir(self, model_class_name: str) -> Path:
        return self.base_dir / model_class_name

    def _get_existing_versions(self, model_dir: Path) -> list:
        if not model_dir.exists():
            return []
        # Find latest version
        return sorted(
            [
                d.name
                for d in model_dir.iterdir()
                if d.is_dir() and d.name.startswith("v")
            ]
        )

    def _get_latest_version(self, versions: list) -> Optional[str]:
        return versions[-1] if versions else None

    def _increment_version(self, version: str) -> str:
        try:
            major, minor, *_ = version.lstrip("v").split(".")
            return f"v{major}.{int(minor)+1}.0"
        except Exception:
            logger.warning(f"Invalid version string: {version}, resetting to v0.1.0")
            return "v0.1.0"

    def save(
        self,
        model,
        meta: dict,
        version: Optional[str] = None,
        auto_increment: bool = True,
    ) -> Path:
        model_name = model.__class__.__name__
        model_dir = self._get_model_dir(model_name)
        versions = self._get_existing_versions(model_dir)

        if version:
            v = version
            logger.info(f"Saving model {model_name} manually as {v}")
        else:
            if auto_increment:
                v = self._increment_version(
                    self._get_latest_version(versions)
                    or "v0.0.0"  # Pretend no version exists i.e. v0.1.0 is first version
                )
                logger.info(f"Auto-incremented version to {v} for model {model_name}")
            else:
                v = self._get_latest_version(versions) or "v0.1.0"
                logger.info(f"Using existing/latest version {v} for model {model_name}")

        save_path = model_dir / v
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), save_path / "model.pt")
        with open(save_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.success(f"Model {model_name} saved at {save_path}")
        return save_path

    def load(self, model, version: Optional[str] = None) -> dict:
        """
        Example:

            ```python
                model = Model()
                checkpoint = ModelCheckpoint()
                meta = checkpoint.load(model, version="v0.1.0")
            ```
        """

        model_name = model.__class__.__name__

        model_dir = self._get_model_dir(model_name)
        versions = self._get_existing_versions(model_dir)

        if not versions:
            logger.error(
                f"No saved versions found for model {model_name} in {model_dir}"
            )
            raise FileNotFoundError("No saved model versions found.")

        v = version or self._get_latest_version(versions)
        load_path = model_dir / v

        if not (load_path / "model.pt").exists():
            raise FileNotFoundError(f"Model weights not found at {load_path}/model.pt")

        # Model mutated in-place
        model.load_state_dict(torch.load(load_path / "model.pt"))

        meta_file = load_path / "meta.json"
        if meta_file.exists():
            with open(meta_file, "r") as f:
                meta = json.load(f)
        else:
            meta = {}

        logger.success(f"Loaded model {model_name} from version {v}")
        return meta
