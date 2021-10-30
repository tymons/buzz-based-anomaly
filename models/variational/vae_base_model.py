from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class VaeOutput:
    output: torch.Tensor
    log_var: torch.Tensor
    mean: torch.Tensor


def kld_loss(mean, log_var):
    """ KLD loss for normal distribution"""
    return -0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp())


def reparameterize(mu, log_var):
    """ Function for reparametrization trick """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


class Flattener(nn.Module):
    def __init__(self, conv_encoder: nn.Module):
        super().__init__()
        self.conv_encoder = conv_encoder
        self.flatten = nn.Flatten()

    def forward(self, x):
        y = self.conv_encoder(x)
        y = self.flatten(y)
        return y


class VaeBaseModel(ABC, nn.Module):
    encoder: nn.Module
    decoder: nn.Module
    linear_means: nn.Module
    linear_log_var: nn.Module

    @abstractmethod
    def loss_fn(self, x, y) -> nn.Module:
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def get_latent(self, data) -> torch.Tensor:
        y = self.encoder(data)
        latent_mean, latent_var = self.linear_means(y), self.linear_log_var(y)
        latent = reparameterize(latent_mean, latent_var)
        return latent