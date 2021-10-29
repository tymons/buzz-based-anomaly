from abc import ABC, abstractmethod

import torch
from torch import nn
from features.contrastive_feature_dataset import ContrastiveOutput


class ContrastiveBaseModel(ABC, nn.Module):
    s_encoder: nn.Module
    z_encoder: nn.Module
    decoder: nn.Module

    @abstractmethod
    def loss_fn(self, target, background, model_output: ContrastiveOutput) -> nn.Module:
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass

    @abstractmethod
    def forward(self, target, background) -> ContrastiveOutput:
        pass

    def get_latent(self, data) -> torch.Tensor:
        latent = self.s_encoder(data)
        return latent
