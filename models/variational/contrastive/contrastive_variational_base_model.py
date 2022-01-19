from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, Tensor
from models.model_type import HiveModelType
from models.variational.vae_base_model import kld_loss

_fields = ['target', 'background', 'target_mean', 'target_log_var', 'target_latent', 'background_mean',
           'background_log_var', 'background_latent']
VaeContrastiveOutput = namedtuple('VaeContrastiveOutput', _fields, defaults=(None,) * len(_fields))


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


def latent_permutation(latent_batch: Tensor, inplace: bool = False, indices=None):
    """
    Function for latent permutation.
    :param indices: force indices
    :param latent_batch: concatenated batch of z's and s's
    :param inplace: flag for inplace operation
    :return:
    """
    latent_batch = latent_batch.squeeze()

    data = latent_batch.clone().detach() if inplace is False else latent_batch
    rand_indices = torch.randperm(latent_batch.shape[0]) if indices is None else indices
    data[:, (latent_batch.shape[1] // 2):] = latent_batch[:, (latent_batch.shape[1] // 2):][rand_indices]

    return data, rand_indices


class ContrastiveVariationalBaseModel(ABC, nn.Module):
    encoder: nn.Module
    decoder: nn.Module
    linear_means: nn.Module
    linear_log_var: nn.Module
    model_type: HiveModelType
    alpha: float

    def __init__(self, model_type, alpha):
        super().__init__()
        self.model_type = model_type
        self.alpha = alpha

    def loss_fn(self, target, background, model_output: VaeContrastiveOutput, discriminator):
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
        recon_loss = loss.item()

        # KLD losses
        loss += kld_loss(model_output.target_mean.squeeze(dim=1), model_output.target_log_var.squeeze(dim=1))
        loss += kld_loss(model_output.background_mean.squeeze(dim=1), model_output.background_log_var.squeeze(dim=1))

        # total correction loss
        q = torch.cat((model_output.target_latent.squeeze(dim=1), model_output.background_latent.squeeze(dim=1)), dim=-1)
        q_bar, indices = latent_permutation(q)

        q_score, q_bar_score = discriminator(q, q_bar)
        tc_loss = self.alpha * -torch.mean(torch.logit(q_score, eps=1e-34))
        loss += tc_loss

        disc_loss = discriminator.loss_fn(q_score, q_bar_score)
        loss += disc_loss

        return loss, (recon_loss, disc_loss.item(), tc_loss.item()), indices

    @abstractmethod
    def get_params(self) -> dict:
        pass

    @abstractmethod
    def forward(self, target, background):
        pass

    def get_latent(self, data) -> torch.Tensor:
        y = self.encoder(data)
        latent_mean, _ = self.linear_means(y), self.linear_log_var(y)
        return latent_mean
