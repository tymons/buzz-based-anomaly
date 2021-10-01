import torch

from torch import nn, functional

from models.base_model import BaseModel
from models.ae import Encoder, Decoder

from typing import List, Union
from dataclasses import dataclass


@dataclass
class VaeOutput:
    output: torch.Tensor
    log_var: torch.Tensor
    mean: torch.Tensor


def _kld_loss(mean, log_var):
    """ KLD loss for normal distribution"""
    return -0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp())


def reparameterize(mu, log_var):
    """ Function for reparametrization trick """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)

    return mu + eps * std


class VAE(BaseModel):
    def __init__(self, layers: List[int], latent_size: int, input_size: int,
                 dropout: Union[List[float], float] = 0.2):
        super().__init__()

        self._layers = layers
        self._latent_size = latent_size
        self._dropout = dropout

        self.latent_size = latent_size
        self.encoder = Encoder(layers, dropout, input_size)
        self.decoder = Decoder(layers[::-1], latent_size, dropout, input_size)

        self.linear_means = nn.Linear(layers[-1], latent_size)
        self.linear_log_var = nn.Linear(layers[-1], latent_size)

    def loss_fn(self, x: torch.Tensor, y: VaeOutput) -> nn.Module:
        """
        Method for returning loss for vae
        :param x: model input
        :param y: model output
        :return: loss
        """
        mse = functional.F.mse_loss(y.output, x, reduction='mean')
        kld = _kld_loss(y.mean, y.log_var)
        return mse + kld

    def get_params(self) -> dict:
        """
        Method for getting model params
        :return: dictionary
        """
        return {
            'model_layers': self._layers,
            'model_latent': self._latent_size,
            'model_dropouts': self._dropout
        }

    def forward(self, x):
        """
        Method for performing forward pass
        :param x:
        :return:
        """
        y = self.encoder(x)
        means = self.linear_means(y)
        log_var = self.linear_log_var(y)
        z = reparameterize(means, log_var)
        recon_x = self.decoder(z)
        return VaeOutput(output=recon_x, mean=means, log_var=log_var)
