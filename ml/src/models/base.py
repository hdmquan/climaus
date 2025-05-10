from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl
import json


class BaseLightningModel(pl.LightningModule, ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.save_hyperparameters(self.config)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError(
            "You must define optimizer(s) in configure_optimizers"
        )

    def training_step(self, batch: Any, batch_idx: int):
        x = self.to_device(batch["input"])
        y_hat = self(x)
        loss = self.loss_function(y_hat, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x = self.to_device(batch["input"])
        y_hat = self(x)
        loss = self.loss_function(y_hat, x)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def loss_function(self, preds, targets):
        return nn.MSELoss()(preds, targets)

    def save_model(self, path: Union[str, Path]):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        with open(path / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

    def load_model(self, path: Union[str, Path]):

        path = Path(path)
        self.load_state_dict(torch.load(path / "model.pt", map_location=self._device))

        return self

    def to_device(self, data: Any):
        if isinstance(data, torch.Tensor):
            return data.to(self._device)

        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}

        elif isinstance(data, (list, tuple)):
            return type(data)(self.to_device(x) for x in data)

        return data

    def summary(self) -> str:
        return f"{self.__class__.__name__} with {sum(p.numel() for p in self.parameters()):,} parameters"
