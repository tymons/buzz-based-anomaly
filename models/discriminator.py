import torch
from torch import nn


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
        loss = -torch.mean(torch.log(q_score))
        loss -= torch.mean(torch.log(1 - q_bar_score))
        return loss

    def forward(self, q, q_bar):
        """
        Method for forward pass
        :param q_bar:
        :param q:
        :return:
        """
        q_class_probability = torch.sigmoid(self.linear(q))
        q_bar_class_probability = torch.sigmoid(self.linear(q_bar))
        return q_class_probability, q_bar_class_probability
