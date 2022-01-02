from abc import ABC, abstractmethod

import torch
from torch import nn
from features.contrastive_feature_dataset import VanillaContrastiveOutput
from models.discriminator import Discriminator
from torch import functional
from models.model_type import HiveModelType


class ContrastiveBaseModel(ABC, nn.Module):
    encoder: nn.Module
    decoder: nn.Module
    model_type: HiveModelType

    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type

    def loss_fn(self, target, background, model_output: VanillaContrastiveOutput, discriminator: Discriminator):
        """
        Method for contrastive model's loss function
        :param target:
        :param background:
        :param model_output:
        :param discriminator:
        :return:
        """
        target_loss = functional.F.mse_loss(target, model_output.target, reduction='mean')
        background_loss = functional.F.mse_loss(background, model_output.background, reduction='mean')
        recon_loss = target_loss + background_loss
        loss = recon_loss.clone()

        target_latent = model_output.target_latent.squeeze()
        background_latent = model_output.background_latent.squeeze()

        latent_data = torch.vstack((target_latent, background_latent))
        latent_labels = torch.hstack((torch.ones(target_latent.shape[0]),
                                      torch.zeros(background_latent.shape[0]))).reshape(-1, 1)

        probs = discriminator(latent_data)
        disc_loss = 0.1 * discriminator.loss_fn(latent_labels, probs.cpu())

        loss += disc_loss

        return loss, (recon_loss, 0, disc_loss)

    @abstractmethod
    def get_params(self) -> dict:
        pass

    @abstractmethod
    def forward(self, target, background) -> VanillaContrastiveOutput:
        pass

    def get_latent(self, data) -> torch.Tensor:
        latent = self.encoder(data)
        return latent
