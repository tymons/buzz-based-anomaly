from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseModel(ABC, nn.Module):
    encoder: nn.Module
    decoder: nn.Module

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
