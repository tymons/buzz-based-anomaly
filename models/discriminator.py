import torch
import torch.nn.functional as F
from torch import nn

from features.contrastive_feature_dataset import VaeContrastiveOutput, VanillaContrastiveOutput
from typing import Union

from models.variational.contrastive.contrastive_variational_base_model import latent_permutation


class Discriminator(nn.Module):
    def __init__(self, input_size: int):
        super(Discriminator, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loss_fn(self, true_labels: torch.Tensor, probs: torch.Tensor):
        """
        Method for loss calculation in discriminator model
        :param true_labels:
        :param probs: tensor of probability for samples
        :return:
        """
        loss = F.binary_cross_entropy(probs, true_labels, reduction='mean')
        return loss

    def _vanilla_get_latent(self, model_output: VanillaContrastiveOutput):
        """
        Prepare latent data for
        :param model_output:
        :return:
        """
        target_latent = model_output.target_latent.clone().detach().squeeze()
        background_latent = model_output.background_latent.clone().detach().squeeze()

        latent_data = torch.vstack((target_latent, background_latent)).to(self.device)
        latent_labels = torch.hstack((torch.ones(target_latent.shape[0]),
                                      torch.zeros(background_latent.shape[0]))).reshape(-1, 1).to(self.device)

        return latent_data, latent_labels

    def _variational_get_latent(self, model_output: VaeContrastiveOutput, indices=None):
        q = torch.cat((model_output.target_qs_latent.clone().detach().squeeze(dim=1),
                       model_output.target_qz_latent.clone().detach().squeeze(dim=1)), dim=-1)
        q_bar, _ = latent_permutation(q, indices=indices)

        latent_data = torch.vstack((q, q_bar)).to(self.device)
        latent_labels = torch.hstack((torch.ones(q.shape[0]),
                                      torch.zeros(q_bar.shape[0]))).reshape(-1, 1).to(self.device)

        return latent_data, latent_labels

    def forward(self, x):
        """
        Method for forward pass
        :param x:
        :return:
        """
        class_probability = torch.sigmoid(self.linear(x))
        return class_probability

    def forward_with_loss(self, model_output: Union[VaeContrastiveOutput, VanillaContrastiveOutput], indices=None):
        """
        Method for forward pass with loss calculation for discriminator
        :param model_output:
        :return:
        """
        if isinstance(model_output, VanillaContrastiveOutput):
            x, labels = self._vanilla_get_latent(model_output)
        elif isinstance(model_output, VaeContrastiveOutput):
            x, labels = self._variational_get_latent(model_output, indices)
        else:
            raise ValueError('Contrastive output not supported!')

        probs = self(x)
        loss = self.loss_fn(labels, probs)
        return loss
