import torch
import torch.nn.functional as F
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, input_size: int):
        super(Discriminator, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)

    def loss_fn(self, probs, true_labels):
        """
        Method for loss calculation in discriminator model
        :param true_labels:
        :param probs: tensor of probability for samples
        :return:
        """
        loss = F.binary_cross_entropy(probs, true_labels, reduction='mean')
        return loss

    def forward(self, x):
        """
        Method for forward pass
        :param x:
        :return:
        """
        class_probability = torch.sigmoid(self.linear(x))
        return class_probability
