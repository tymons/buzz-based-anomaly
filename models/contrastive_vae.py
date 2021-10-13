import torch
import torch.nn.functional as F
from torch import nn, Tensor

from typing import Union, List

from models.discriminator import Discriminator
from models.contrastive_base_model import ContrastiveBaseModel
from models.ae import Encoder, Decoder
from models.vae import reparameterize, kld_loss

from features.contrastive_feature_dataset import ContrastiveInput, ContrastiveOutput


class ContrastiveVAE(ContrastiveBaseModel):
    def __init__(self, layers: List[int], latent_size: int, input_size: int,
                 dropout: Union[List[float], float] = 0.2, use_discriminator: bool = False):
        super().__init__()

        self._layers = layers
        self._use_discriminator = use_discriminator
        self._latent_size = latent_size
        self._dropout = [dropout] * len(layers) if isinstance(dropout, float) else dropout

        self.s_encoder = Encoder(layers, dropout, input_size)
        self.z_encoder = Encoder(layers, dropout, input_size)
        self.decoder = Decoder(layers[::-1], latent_size, dropout, input_size)

        self.s_linear_means = nn.Linear(layers[-1], latent_size)
        self.s_linear_log_var = nn.Linear(layers[-1], latent_size)

        self.z_linear_means = nn.Linear(layers[-1], latent_size)
        self.z_linear_log_var = nn.Linear(layers[-1], latent_size)

    def loss_fn(self, x: ContrastiveInput, y: ContrastiveOutput, discriminator: Discriminator) -> torch.Tensor:
        """
        Method for calculating loss function for pytorch model
        :param discriminator:
        :param x: data input
        :param y: model output data
        :return: loss
        """
        # reconstruction loss for target and background
        loss = nn.functional.mse_loss(x.target, y.target, reduction='mean')
        loss += nn.functional.mse_loss(x.background, y.background, reduction='mean')
        # KLD losses
        loss += kld_loss(y.target_qs_mean, y.target_qs_log_var)
        loss += kld_loss(y.target_qz_mean, y.target_qz_log_var)
        loss += kld_loss(y.background_qz_mean, y.background_qz_log_var)

        # total correction loss
        with torch.no_grad():
            q = torch.cat((y.target_qs_latent, y.target_qz_latent))
            q_score, _ = discriminator(q, torch.zeros_like(q))
            disc_loss = torch.mean(torch.log(q_score / (1 - q_score)))
            loss += disc_loss

        return loss

    def forward(self, x: ContrastiveInput) -> ContrastiveOutput:
        """
        Method for performing forward pass
        :param x: input data for contrastive autoencoder
        :return: ContrastiveData with reconstructed background and target
        """
        target_s = self.s_encoder(x.target)
        tg_s_mean, tg_s_log_var = self.s_linear_means(target_s), self.s_linear_log_var(target_s)

        target_z = self.z_encoder(x.target)
        tg_z_mean, tg_z_log_var = self.z_linear_means(target_z), self.z_linear_log_var(target_z)

        background_z = self.z_encoder(x.background)
        bg_z_mean, bg_z_log_var = self.z_linear_means(background_z), self.z_linear_log_var(background_z)

        tg_s: Tensor = reparameterize(tg_s_mean, tg_s_log_var)
        tg_z: Tensor = reparameterize(tg_z_mean, tg_z_log_var)
        bg_z: Tensor = reparameterize(bg_z_mean, bg_z_log_var)

        tg_output = self.decoder(torch.cat(tensors=[tg_s, tg_z]))
        bg_output = self.decoder(torch.cat(tensors=[torch.zeros_like(tg_s), bg_z]))

        return ContrastiveOutput(target=tg_output, background=bg_output,
                                 target_qs_mean=tg_s_mean, target_qs_log_var=tg_s_log_var,
                                 target_qz_mean=tg_z_mean, target_qz_log_var=tg_z_log_var,
                                 background_qz_mean=bg_z_mean, background_qz_log_var=bg_z_log_var,
                                 target_qs_latent=tg_s, target_qz_latent=tg_z)

    def get_params(self) -> dict:
        """
        Function for returning model layer sizes
        :return: dictionary with model layer sizes
        """
        return {
            'model_layers': self._layers,
            'model_latent': self._latent_size,
            'model_dropouts': self._dropout
        }


def latent_permutation(latent_batch: Tensor, inplace: bool = False):
    """
    Function for latent permutation.
    :param latent_batch: concatenated batch of z's and s's
    :param inplace: flag for inplace operation
    :return:
    """
    latent_batch = latent_batch.squeeze()

    data = latent_batch.clone().detach() if inplace is False else latent_batch

    rand_indices = torch.randperm(data[:, 0:data.shape[1] // 2].shape[0])
    data[:, 0:data.shape[1] // 2] = data[:, 0:data.shape[1] // 2][rand_indices]
    rand_indices = torch.randperm(data[:, data.shape[1] // 2:].shape[0])
    data[:, data.shape[1] // 2:] = data[:, data.shape[1] // 2:][rand_indices]

    return data
