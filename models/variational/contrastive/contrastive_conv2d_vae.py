from torch import nn, Tensor

from typing import List

import models.variational.contrastive.contrastive_variational_base_model as cvbm
from models.model_type import HiveModelType
from models.vanilla.conv2d_ae import Conv2DEncoder, Conv2DDecoder
from models.conv_utils import convolutional_to_mlp
from models.variational.vae_base_model import reparameterize, Flattener

from features.contrastive_feature_dataset import VaeContrastiveOutput


class ContrastiveConv2DVAE(cvbm.ContrastiveVariationalBaseModel):
    def __init__(self, model_type: HiveModelType, features: List[int], dropout_probs: List[float], kernel_size: int,
                 padding: int, max_pool: int, latent_size: int, input_size: tuple, tc_alpha: float = 0.1,
                 tc_loss: bool = False):
        super().__init__(model_type, tc_alpha, tc_loss)

        self._feature_map = features
        self._dropout_probs = dropout_probs
        self._kernel_size = kernel_size
        self._padding = padding
        self._latent = latent_size
        self._max_pool = max_pool

        connector_size, conv_temporal = convolutional_to_mlp(input_size, len(features), kernel_size, padding, max_pool)
        self.encoder = Flattener(Conv2DEncoder(features, dropout_probs, kernel_size, padding, max_pool))
        self.decoder = Conv2DDecoder(features[::-1], dropout_probs[::-1], kernel_size, padding, latent_size,
                                     features[-1] * connector_size, conv_temporal[::-1])

        self.flatten = nn.Flatten()
        self.linear_means = nn.Linear(features[-1] * connector_size, latent_size)
        self.linear_log_var = nn.Linear(features[-1] * connector_size, latent_size)

    def forward(self, target, background) -> VaeContrastiveOutput:
        """
        Method for performing forward pass
        :param target: target data for contrastive autoencoder
        :param background: background data for contrastive autoencoder
        :return: ContrastiveData with reconstructed background and target
        """
        target = self.encoder(target)
        target = self.flatten(target)
        tg_mean, tg_log_var = self.linear_means(target), self.linear_log_var(target)

        background = self.encoder(background)
        background = self.flatten(background)
        bg_mean, bg_log_var = self.linear_means(background), self.linear_log_var(background)

        tg_latent: Tensor = reparameterize(tg_mean, tg_log_var)
        bg_latent: Tensor = reparameterize(bg_mean, bg_log_var)

        tg_output = self.decoder(tg_latent)
        bg_output = self.decoder(bg_latent)

        return VaeContrastiveOutput(target=tg_output, background=bg_output,
                                    target_latent=tg_latent, background_latent=bg_latent,
                                    target_mean=tg_mean, target_log_var=tg_log_var,
                                    background_mean=bg_mean, background_log_var=bg_log_var)

    def get_params(self) -> dict:
        """
        Function for returning model layer sizes
        :return: dictionary with model layer sizes
        """
        return {
            'model_feature_map': self._feature_map,
            'model_dropouts': self._dropout_probs,
            'model_kernel_size': self._kernel_size,
            'model_padding': self._padding,
            'model_latent': self._latent,
            'model_max_pool': self._max_pool,
            'model_tc_alpha': self.tc_alpha,
            'model_tc_loss': self.tc_component
        }
