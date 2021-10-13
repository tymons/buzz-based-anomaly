from abc import ABC, abstractmethod

from torch import nn
from features.contrastive_feature_dataset import ContrastiveInput, ContrastiveOutput


class ContrastiveBaseModel(ABC, nn.Module):
    @abstractmethod
    def loss_fn(self, x, y, discriminator) -> nn.Module:
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass

    @abstractmethod
    def forward(self, x: ContrastiveInput) -> ContrastiveOutput:
        pass
