import torch

from torch import nn, functional

from models.conv_utils import convolutional_to_mlp
from typing import List

import models.variational.vae_base_model as vbm
from models.model_type import HiveModelType
from models.vanilla.conv2d_ae import Conv2DDecoder, Conv2DEncoder


class Conv2DVAE(vbm.VaeBaseModel):
    def __init__(self, model_type: HiveModelType, features: List[int], dropout_probs: List[float], kernel_size: int,
                 padding: int, max_pool: int, latent: int, input_size: tuple):
        super().__init__(model_type)
        self._feature_map = features
        self._dropout_probs = dropout_probs
        self._kernel_size = kernel_size
        self._padding = padding
        self._latent = latent
        self._max_pool = max_pool

        connector_size, conv_temporal = convolutional_to_mlp(input_size, len(features), kernel_size, padding, max_pool)
        self.encoder = vbm.Flattener(Conv2DEncoder(features, dropout_probs, kernel_size, padding, max_pool))
        self.decoder = Conv2DDecoder(features[::-1], dropout_probs[::-1], kernel_size, padding, latent,
                                     features[-1] * connector_size, conv_temporal[::-1])

        self.flatten = nn.Flatten()
        self.linear_means = nn.Linear(features[-1] * connector_size, latent)
        self.linear_log_var = nn.Linear(features[-1] * connector_size, latent)

    def loss_fn(self, x, y) -> torch.Tensor:
        """
        Loss function for convolutional 2d vae
        :param x: model input
        :param y: model output
        :return:
        """
        mse = functional.F.mse_loss(y.output, x, reduction='mean')
        kld = vbm.kld_loss(y.mean, y.log_var)
        return mse + kld

    def get_params(self) -> dict:
        return {
            'model_feature_map': self._feature_map,
            'model_dropouts': self._dropout_probs,
            'model_kernel_size': self._kernel_size,
            'model_padding': self._padding,
            'model_latent': self._latent,
            'model_max_pool': self._max_pool
        }

    def forward(self, x):
        """
        Method for performing forward pass
        :param x: input data
        :return: model output VaeOutput
        """
        y = self.encoder(x)
        means = self.linear_means(y)
        log_var = self.linear_log_var(y)
        z = vbm.reparameterize(means, log_var)
        recon_x = self.decoder(z)
        return vbm.VaeOutput(output=recon_x, mean=means, log_var=log_var)
