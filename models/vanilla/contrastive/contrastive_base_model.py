from abc import ABC, abstractmethod

import torch
from torch import nn
from torch import functional

from models.model_type import HiveModelType
from collections import namedtuple

_vanilla_fields = ['target', 'background', 'target_latent', 'background_latent']
VanillaContrastiveOutput = namedtuple('VanillaContrastiveOutput', _vanilla_fields,
                                      defaults=(None,) * len(_vanilla_fields))


class ContrastiveBaseModel(ABC, nn.Module):
    encoder: nn.Module
    decoder: nn.Module
    model_type: HiveModelType
    tc_alpha: float
    tc_component: bool

    def __init__(self, model_type, tc_alpha, include_tc_loss):
        super().__init__()
        self.model_type = model_type
        self.tc_alpha = tc_alpha
        self.tc_component = include_tc_loss

    def loss_fn(self, target, background, model_output: VanillaContrastiveOutput, discriminator):
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

        target_latent = model_output.target_latent.squeeze(dim=1)
        background_latent = model_output.background_latent.squeeze(dim=1)

        q_score, q_bar_score = discriminator(target_latent, background_latent)
        disc_loss = discriminator.loss_fn(q_score, q_bar_score)
        loss += disc_loss

        tc_loss = -torch.mean(torch.logit(q_score, eps=1e-7))
        if self.tc_component is True:
            loss += self.tc_alpha * tc_loss

        return loss, (recon_loss.item(), disc_loss.item(), tc_loss.item())

    def forward(self, target, background) -> VanillaContrastiveOutput:
        """
        Forward method for NN
        :param target: target sample
        :param background: background smaple
        :return: ContrastiveOutput
        """
        target_latent = self.encoder(target)
        background_latent = self.encoder(background)
        target_output = self.decoder(target_latent)
        background_output = self.decoder(background_latent)

        return VanillaContrastiveOutput(target=target_output, background=background_output,
                                        target_latent=target_latent, background_latent=background_latent)

    @abstractmethod
    def get_params(self) -> dict:
        pass

    def get_latent(self, data) -> torch.Tensor:
        latent = self.encoder(data)
        return latent
