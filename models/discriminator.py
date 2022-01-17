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

    def loss_fn(self, q_score, q_bar_score):
        """
        Method for loss calculation in discriminator model
        :param q_bar_score:
        :param q_score:
        :return:
        """
        # loss = F.binary_cross_entropy(probs, true_labels, reduction='mean')

        q_score_max = torch.clip(q_score, min=1e-36, max=1e36)
        q_bar_score_max = torch.clip(q_bar_score, min=1e-36, max=1e36)
        loss = - torch.log(q_score_max) - torch.log(1 - q_bar_score_max)
        loss = torch.mean(loss)
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

    def variational_get_latent(self, model_output: VaeContrastiveOutput, indices=None):
        q = torch.cat((model_output.target_qs_latent.clone().detach().squeeze(dim=1),
                       model_output.target_qz_latent.clone().detach().squeeze(dim=1)), dim=-1)
        q_bar, _ = latent_permutation(q, indices=indices)

        # latent_data = torch.vstack((q, q_bar)).to(self.device)
        # latent_labels = torch.hstack((torch.ones(q.shape[0]),
        #                               torch.zeros(q_bar.shape[0]))).reshape(-1, 1).to(self.device)

        return q, q_bar

    def forward(self, p, q):
        """
        Method for forward pass
        :param q:
        :param p:
        :return:
        """
        p_class_probability = torch.sigmoid(self.linear(p))
        q_class_probability = torch.sigmoid(self.linear(q))
        return p_class_probability, q_class_probability

    def forward_with_loss(self, model_output: Union[VaeContrastiveOutput, VanillaContrastiveOutput], indices=None):
        """
        Method for forward pass with loss calculation for discriminator
        :param model_output:
        :return:
        """
        if isinstance(model_output, VanillaContrastiveOutput):
            x, labels = self._vanilla_get_latent(model_output)

            probs = self(x)
            loss = self.loss_fn(labels, probs)
        elif isinstance(model_output, VaeContrastiveOutput):
            q, q_bar = self.variational_get_latent(model_output, indices)
            q_score, q_bar_score = self(q, q_bar)
            loss = self.loss_fn(q_score, q_bar_score)
        else:
            raise ValueError('Contrastive output not supported!')

        return loss
