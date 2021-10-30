import torch
import torch.nn.functional as F
import models.variational.contrastive.contrastive_variational_base_model as cvbm

from torch import nn, Tensor
from typing import List
from models.discriminator import Discriminator
from models.vanilla.conv1d_ae import Conv1DEncoder, Conv1DDecoder
from models.conv_utils import convolutional_to_mlp
from models.variational.vae_base_model import reparameterize, kld_loss, Flattener
from features.contrastive_feature_dataset import ContrastiveOutput


class ContrastiveConv1DVAE(cvbm.ContrastiveVariationalBaseModel):
    def __init__(self, features: List[int], dropout_probs: List[float], kernel_size: int, padding: int, max_pool: int,
                 latent_size: int, input_size: int):
        super().__init__()

        self._feature_map = features
        self._dropout_probs = dropout_probs
        self._kernel_size = kernel_size
        self._padding = padding
        self._latent = latent_size
        self._max_pool = max_pool

        connector_size, conv_temporal = convolutional_to_mlp(input_size, len(features), kernel_size, padding, max_pool)
        self.s_encoder = Flattener(Conv1DEncoder(features, dropout_probs, kernel_size, padding, max_pool))
        self.z_encoder = Flattener(Conv1DEncoder(features, dropout_probs, kernel_size, padding, max_pool))
        self.decoder = Conv1DDecoder(features[::-1], dropout_probs[::-1], kernel_size, padding, 2 * latent_size,
                                     features[-1] * connector_size, conv_temporal[::-1])

        self.flatten = nn.Flatten()
        self.s_linear_means = nn.Linear(features[-1] * connector_size, latent_size)
        self.s_linear_log_var = nn.Linear(features[-1] * connector_size, latent_size)

        self.z_linear_means = nn.Linear(features[-1] * connector_size, latent_size)
        self.z_linear_log_var = nn.Linear(features[-1] * connector_size, latent_size)

    def loss_fn(self, target, background, model_output: ContrastiveOutput,
                discriminator: Discriminator) -> torch.Tensor:
        """
        Method for calculating loss function for pytorch model
        :param model_output:
        :param discriminator:
        :param target: target input
        :param background: background data
        :return: loss
        """
        # reconstruction loss for target and background
        loss = F.mse_loss(target, model_output.target, reduction='mean')
        loss += F.mse_loss(background, model_output.background, reduction='mean')
        # KLD losses
        loss += kld_loss(model_output.target_qs_mean, model_output.target_qs_log_var)
        loss += kld_loss(model_output.target_qz_mean, model_output.target_qz_log_var)
        loss += kld_loss(model_output.background_qz_mean, model_output.background_qz_log_var)

        # total correction loss
        with torch.no_grad():
            q = torch.cat((model_output.target_qs_latent, model_output.target_qz_latent), dim=-1).squeeze()
            q_bar = cvbm.latent_permutation(q)
            q_score, q_bar_score = discriminator(q, q_bar)
            disc_loss = discriminator.loss_fn(q_score, q_bar_score)
            loss += disc_loss

        return loss

    def forward(self, target, background) -> ContrastiveOutput:
        """
        Method for performing forward pass
        :param target: target data for contrastive autoencoder
        :param background: background data for contrastive autoencoder
        :return: ContrastiveData with reconstructed background and target
        """
        target_s = self.s_encoder(target)
        target_s = self.flatten(target_s)
        tg_s_mean, tg_s_log_var = self.s_linear_means(target_s), self.s_linear_log_var(target_s)

        target_z = self.z_encoder(target)
        target_z = self.flatten(target_z)
        tg_z_mean, tg_z_log_var = self.z_linear_means(target_z), self.z_linear_log_var(target_z)

        background_z = self.z_encoder(background)
        background_z = self.flatten(background_z)
        bg_z_mean, bg_z_log_var = self.z_linear_means(background_z), self.z_linear_log_var(background_z)

        tg_s: Tensor = reparameterize(tg_s_mean, tg_s_log_var)
        tg_z: Tensor = reparameterize(tg_z_mean, tg_z_log_var)
        bg_z: Tensor = reparameterize(bg_z_mean, bg_z_log_var)

        tg_output = self.decoder(torch.cat(tensors=[tg_s, tg_z], dim=-1))
        bg_output = self.decoder(torch.cat(tensors=[torch.zeros_like(tg_s), bg_z], dim=-1))

        return ContrastiveOutput(target=tg_output, background=bg_output,
                                 target_qs_mean=tg_s_mean, target_qs_log_var=tg_s_log_var,
                                 target_qz_mean=tg_z_mean, target_qz_log_var=tg_z_log_var,
                                 background_qz_mean=bg_z_mean, background_qz_log_var=bg_z_log_var,
                                 target_qs_latent=tg_s, target_qz_latent=tg_z)

    def get_params(self) -> dict:
        """
        Method for returning model layer sizes
        :return: dictionary with model layer sizes
        """
        return {
            'model_feature_map': self._feature_map,
            'model_dropouts': self._dropout_probs,
            'model_kernel_size': self._kernel_size,
            'model_padding': self._padding,
            'model_latent': self._latent,
            'model_max_pool': self._max_pool
        }
