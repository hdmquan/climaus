from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.common import ModelCheckpoint


class BaseLightningModel(pl.LightningModule, ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()

        self.config = config or {}
        self._ckpt = ModelCheckpoint()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)

        self.save_hyperparameters(self.config)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError(
            "You must define optimizer(s) in configure_optimizers"
        )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x = self.to_device(batch["input"])

        y_hat = self(x)
        loss = self.loss_function(y_hat, x)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x = self.to_device(batch["input"])

        y_hat = self(x)
        loss = self.loss_function(y_hat, x)

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def loss_function(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.MSELoss()(preds, targets)

    def save_model(
        self,
        meta: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        auto_increment: bool = True,
    ) -> Path:

        meta = meta or {}
        meta["config"] = self.config

        return self._ckpt.save(
            self, meta, version=version, auto_increment=auto_increment
        )

    def load_model(self, version: Optional[str] = None) -> Dict[str, Any]:
        return self._ckpt.load(self, version)

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
