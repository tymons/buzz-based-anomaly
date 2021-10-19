from abc import ABC, abstractmethod

from torch import nn
from features.contrastive_feature_dataset import ContrastiveOutput


class ContrastiveVariationalBaseModel(ABC, nn.Module):
    @abstractmethod
    def loss_fn(self, target, background, model_output: ContrastiveOutput, discriminator) -> nn.Module:
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass

    @abstractmethod
    def forward(self, target, background) -> ContrastiveOutput:
        pass
