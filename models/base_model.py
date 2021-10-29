from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseModel(ABC, nn.Module):
    @abstractmethod
    def loss_fn(self, x, y) -> nn.Module:
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def inference(self, x) -> torch.Tensor:
        pass
