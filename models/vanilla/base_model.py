from abc import ABC, abstractmethod

import torch
from torch import nn
from models.model_type import HiveModelType


class BaseModel(ABC, nn.Module):
    encoder: nn.Module
    decoder: nn.Module
    model_type: HiveModelType

    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type

    @abstractmethod
    def loss_fn(self, x, y) -> nn.Module:
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def get_latent(self, x) -> torch.Tensor:
        latent = self.encoder(x)
        return latent
