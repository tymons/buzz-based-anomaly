from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn, Tensor
from features.contrastive_feature_dataset import ContrastiveOutput
from models.variational.vae_base_model import reparameterize


def kl_closed_form(means_p, covs_p, means_q, covs_q):
    """
    Method for calculating kl divergence in closed form.
    See page 13 on https://stanford.edu/~jduchi/projects/general_notes.pdf
    :param means_p:
    :param covs_p:
    :param means_q:
    :param covs_q:
    :return:
    """
    det_covs_q = np.linalg.det(covs_q)
    det_covs_p = np.linalg.det(covs_p)
    n = covs_p.shape[0]
    mean_diff = (means_q - means_p).reshape(n, -1)
    return 0.5 * (np.log(det_covs_q / det_covs_p) - n + np.trace(np.linalg.inv(covs_q) * covs_p)
                  + (np.matmul(np.matmul(mean_diff.transpose(), np.linalg.inv(covs_q)), mean_diff))).squeeze()


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
        latent_mean, _ = self.s_linear_means(y), self.s_linear_log_var(y)
        return latent_mean
