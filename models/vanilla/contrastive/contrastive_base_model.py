from abc import ABC, abstractmethod

import torch
from torch import nn
from features.contrastive_feature_dataset import ContrastiveOutput
from models.discriminator import Discriminator
from torch import functional
from models.model_type import HiveModelType


class ContrastiveBaseModel(ABC, nn.Module):
    s_encoder: nn.Module
    z_encoder: nn.Module
    decoder: nn.Module
    model_type: HiveModelType

    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type

    def loss_fn(self, target, background, model_output: ContrastiveOutput, discriminator: Discriminator):
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
        loss = recon_loss

        with torch.no_grad():
            qs_target_latent = model_output.target_qs_latent.squeeze()
            qz_target_latent = model_output.target_qz_latent.squeeze()
            latent_data = torch.vstack((qs_target_latent, qz_target_latent))
            latent_labels = torch.hstack((torch.ones(qs_target_latent.shape[0]),
                                          torch.zeros(qz_target_latent.shape[0]))).reshape(-1, 1)

            tc_loss = -torch.mean(torch.log(torch.exp(torch.logit(discriminator(qs_target_latent)))))

            probs = discriminator(latent_data)
            disc_loss = discriminator.loss_fn(latent_labels, probs.cpu())

            loss += (0.1 * (tc_loss + disc_loss))
            return loss, (recon_loss, tc_loss, disc_loss)

    @abstractmethod
    def get_params(self) -> dict:
        pass

    @abstractmethod
    def forward(self, target, background) -> ContrastiveOutput:
        pass

    def get_latent(self, data) -> torch.Tensor:
        latent = self.s_encoder(data)
        return latent
