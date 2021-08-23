import torch

from torch import nn


def discriminator_loss(log_ratio_p, log_ratio_q):
    """ Function for discriminaotr loss """
    loss_p = nn.functional.binary_cross_entropy_with_logits(log_ratio_p, torch.ones_like(log_ratio_p), reduction='mean')
    loss_q = nn.functional.binary_cross_entropy_with_logits(log_ratio_q, torch.zeros_like(log_ratio_q),
                                                            reduction='mean')
    return loss_p + loss_q


class Discriminator(nn.Module):
    """ Dummy class which implements MLP as discriminator """

    def __init__(self, layers_sizes, input_size):
        """ Constructor for discriminator class """
        assert type(layers_sizes) == list
        assert type(input_size) == int

        super(Discriminator, self).__init__()

        self.MLP = nn.Sequential()

        self.MLP.add_module(name=f'FC{0}', module=nn.Linear(input_size, layers_sizes[0]))
        self.MLP.add_module(name=f'A{0}', module=nn.ReLU())

        for i, (input_size, output_size) in enumerate(zip(layers_sizes[:-1], layers_sizes[1:]), 1):
            self.MLP.add_module(name=f'FC{i}', module=nn.Linear(input_size, output_size))
            self.MLP.add_module(name=f'A{i}', module=nn.ReLU())

        self.MLP.add_module(name=f'FC{len(layers_sizes) + 1}', module=nn.Linear(layers_sizes[-1], 1))
        self.MLP.add_module(name=f'A{len(layers_sizes) + 1}', module=nn.Sigmoid())

    def forward(self, p, q):
        p_log_probability = self.MLP(p)
        q_log_probability = self.MLP(q)

        return p_log_probability, q_log_probability
