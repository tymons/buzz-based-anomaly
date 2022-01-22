from torch import nn, Tensor

from typing import Union, List

import models.variational.contrastive.contrastive_variational_base_model as cvbm
from models.model_type import HiveModelType
from models.vanilla.ae import Encoder, Decoder
from models.variational.vae_base_model import reparameterize

from features.contrastive_feature_dataset import VaeContrastiveOutput


class ContrastiveVAE(cvbm.ContrastiveVariationalBaseModel):
    def __init__(self, model_type: HiveModelType, layers: List[int], latent_size: int, input_size: int,
                 dropout: Union[List[float], float] = 0.2, tc_alpha: float = 0.1, tc_loss: bool = False):
        super().__init__(model_type, tc_alpha, tc_loss)

        self._layers = layers
        self._latent_size = latent_size
        self._dropout = [dropout] * len(layers) if isinstance(dropout, float) else dropout

        self.encoder = Encoder(layers, dropout, input_size)
        self.decoder = Decoder(layers[::-1], latent_size, dropout, input_size)

        self.linear_means = nn.Linear(layers[-1], latent_size)
        self.linear_log_var = nn.Linear(layers[-1], latent_size)

    def forward(self, target, background) -> VaeContrastiveOutput:
        """
        Method for performing forward pass
        :param target: target data for contrastive autoencoder
        :param background: background data for contrastive autoencoder
        :return: ContrastiveData with reconstructed background and target
        """
        target = self.encoder(target)
        tg_mean, tg_log_var = self.linear_means(target), self.linear_log_var(target)

        background = self.encoder(background)
        bg_mean, bg_log_var = self.linear_means(background), self.linear_log_var(background)

        tg_latent: Tensor = reparameterize(tg_mean, tg_log_var)
        bg_latent: Tensor = reparameterize(bg_mean, bg_log_var)

        tg_output = self.decoder(tg_latent)
        bg_output = self.decoder(bg_latent)

        return VaeContrastiveOutput(target=tg_output, background=bg_output,
                                    target_latent=tg_latent, target_mean=tg_mean, target_log_var=tg_log_var,
                                    background_latent=bg_latent,
                                    background_mean=bg_mean,
                                    background_log_var=bg_log_var)

    def get_params(self) -> dict:
        """
        Function for returning model layer sizes
        :return: dictionary with model layer sizes
        """
        return {
            'model_layers': self._layers,
            'model_latent': self._latent_size,
            'model_dropouts': self._dropout,
            'model_tc_alpha': self.tc_alpha,
            'model_tc_loss': self.tc_component
        }
