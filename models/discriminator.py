import torch
import torch.nn.functional as F
from torch import nn
from typing import List


class Discriminator(nn.Module):
    def __init__(self, layers_sizes: List, input_size: int):
        super(Discriminator, self).__init__()

        self.MLP = nn.Sequential()
        self.MLP.add_module(name=f'FC{0}', module=nn.Linear(input_size, layers_sizes[0]))
        self.MLP.add_module(name=f'A{0}', module=nn.ReLU())

        for i, (input_size, output_size) in enumerate(zip(layers_sizes[:-1], layers_sizes[1:]), 1):
            self.MLP.add_module(name=f'FC{i}', module=nn.Linear(input_size, output_size))
            self.MLP.add_module(name=f'A{i}', module=nn.ReLU())

        self.MLP.add_module(name=f'FC{len(layers_sizes) + 1}', module=nn.Linear(layers_sizes[-1], 1))
        self.MLP.add_module(name=f'A{len(layers_sizes) + 1}', module=nn.Sigmoid())

    def loss_fn(self, p_prob: torch.Tensor, q_prob: torch.Tensor):
        """
        Method for loss calculation in discriminator model
        :param p_prob: tensor of probability for samples p(x)
        :param q_prob: tensor or probability for samples q(x)
        :return:
        """
        p_loss = F.binary_cross_entropy(p_prob, torch.zeros_like(p_prob), reduction='mean')
        q_loss = F.binary_cross_entropy(q_prob, torch.ones_like(q_prob), reduction='mean')
        return p_loss + q_loss

    def forward(self, p, q):
        """
        Method for forward pass
        :param p - samples from p-distribution
        :param q - samples from q-distribution
        :return:
        """
        p_class_probability = self.MLP(p)
        q_class_probability = self.MLP(q)
        return p_class_probability, q_class_probability
