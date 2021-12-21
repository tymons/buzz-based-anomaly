from abc import ABC, abstractmethod

import torch
from torch import nn
from features.contrastive_feature_dataset import ContrastiveOutput
from models.discriminator import Discriminator
from torch import functional
from sklearn.preprocessing import MinMaxScaler
import models.variational.contrastive.contrastive_variational_base_model as cvbm
from models.model_type import HiveModelType


class ContrastiveBaseModel(ABC, nn.Module):
    s_encoder: nn.Module
    z_encoder: nn.Module
    decoder: nn.Module
    model_type: HiveModelType
    mm = MinMaxScaler()

    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type

    def loss_fn(self, target, background, model_output: ContrastiveOutput, discriminator: Discriminator) -> nn.Module:
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
        loss = target_loss + background_loss

        q = torch.cat((model_output.target_qs_latent, model_output.target_qz_latent), dim=-1).squeeze()
        q = (q - q.min(axis=0).values) / (q.max(axis=0).values - q.min(axis=0).values)
        q = torch.nan_to_num(q, nan=0.0)
        q_bar = cvbm.latent_permutation(q)
        q_score, q_bar_score = discriminator(q, q_bar)
        tc_loss = torch.mean(torch.logit(q_score))
        loss += tc_loss

        disc_loss = discriminator.loss_fn(q_score, q_bar_score)
        loss += disc_loss

        return loss

    @abstractmethod
    def get_params(self) -> dict:
        pass

    @abstractmethod
    def forward(self, target, background) -> ContrastiveOutput:
        pass

    def get_latent(self, data) -> torch.Tensor:
        latent = self.s_encoder(data)
        return latent
