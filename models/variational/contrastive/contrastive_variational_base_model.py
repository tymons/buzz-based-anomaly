from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor
from features.contrastive_feature_dataset import ContrastiveOutput
from models.variational.vae_base_model import reparameterize


def latent_permutation(latent_batch: Tensor, inplace: bool = False):
    """
    Function for latent permutation.
    :param latent_batch: concatenated batch of z's and s's
    :param inplace: flag for inplace operation
    :return:
    """
    latent_batch = latent_batch.squeeze()

    data = latent_batch.clone() if inplace is False else latent_batch
    rand_indices = torch.randperm(data[:, 0:data.shape[1] // 2].shape[0])
    data[:, 0:data.shape[1] // 2] = data[:, 0:data.shape[1] // 2][rand_indices]
    rand_indices = torch.randperm(data[:, data.shape[1] // 2:].shape[0])
    data[:, data.shape[1] // 2:] = data[:, data.shape[1] // 2:][rand_indices]

    return data


class ContrastiveVariationalBaseModel(ABC, nn.Module):
    s_encoder: nn.Module
    z_encoder: nn.Module
    decoder: nn.Module
    s_linear_means: nn.Module
    s_linear_log_var: nn.Module

    @abstractmethod
    def loss_fn(self, target, background, model_output: ContrastiveOutput, discriminator) -> nn.Module:
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass

    @abstractmethod
    def forward(self, target, background) -> ContrastiveOutput:
        pass

    def get_latent(self, data) -> torch.Tensor:
        y = self.s_encoder(data)
        latent_mean, latent_var = self.s_linear_means(y), self.s_linear_log_var(y)
        latent = reparameterize(latent_mean, latent_var)
        return latent
