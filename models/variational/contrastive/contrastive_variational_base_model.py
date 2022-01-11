from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, Tensor
from features.contrastive_feature_dataset import VaeContrastiveOutput
from models.model_type import HiveModelType
from models.discriminator import Discriminator
from models.variational.vae_base_model import kld_loss


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
    model_type: HiveModelType

    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type

    def loss_fn(self, target, background, model_output: VaeContrastiveOutput, discriminator: Discriminator):
        """
        Method for variational loss fn
        :param target:
        :param background:
        :param model_output:
        :param discriminator:
        """
        # reconstruction loss for target and background
        loss = F.mse_loss(target, model_output.target, reduction='mean')
        loss += F.mse_loss(background, model_output.background, reduction='mean')
        # KLD losses
        loss += kld_loss(model_output.target_qs_mean.squeeze(), model_output.target_qs_log_var.squeeze())
        loss += kld_loss(model_output.target_qz_mean.squeeze(), model_output.target_qz_log_var.squeeze())
        loss += kld_loss(model_output.background_qz_mean.squeeze(), model_output.background_qz_log_var.squeeze())

        # total correction loss
        # with torch.no_grad():
        q = torch.cat((model_output.target_qs_latent.squeeze(), model_output.target_qz_latent.squeeze()), dim=-1)
        q_bar = latent_permutation(q)
        q_score = discriminator(q)
        q_bar_score = discriminator(q_bar)
        tc_loss = torch.mean(torch.logit(q_score))
        loss += tc_loss

        disc_loss = discriminator.loss_fn(torch.ones_like(q_score), q_score)
        disc_loss += discriminator.loss_fn(torch.zeros_like(q_bar_score), q_bar_score)
        loss += disc_loss

        return loss

    @abstractmethod
    def get_params(self) -> dict:
        pass

    @abstractmethod
    def forward(self, target, background) -> VaeContrastiveOutput:
        pass

    def get_latent(self, data) -> torch.Tensor:
        y = self.s_encoder(data)
        latent_mean, _ = self.s_linear_means(y), self.s_linear_log_var(y)
        return latent_mean
