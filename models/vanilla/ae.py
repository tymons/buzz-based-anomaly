import torch

from torch import nn
from models.vanilla.base_model import BaseModel
from models.model_type import HiveModelType
import torch.nn.functional as F

from typing import Union, List


class Autoencoder(BaseModel):
    """ Vanilla autoencoder """

    def __init__(self, model_type: HiveModelType, layers: List[int], latent_size: int, input_size: int,
                 dropout: Union[List[float], float] = 0.2):
        super().__init__(model_type)

        self._layers = layers
        self._latent_size = latent_size
        self._dropout = [dropout] * len(layers) if isinstance(dropout, float) else dropout

        self.encoder = EncoderWithLatent(layers, latent_size, dropout, input_size)
        self.decoder = Decoder(layers[::-1], latent_size, dropout, input_size)

    def loss_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Method for calculating loss function for pytorch model
        :param x: data input
        :param y: model output data
        :return: loss
        """
        mse = F.mse_loss(y, x, reduction='mean')
        return mse

    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        return y

    def get_params(self) -> dict:
        """
        Function for returning model layer sizes
        :return: dictionary with model layer sizes
        """
        return {
            'model_layers': self._layers,
            'model_latent': self._latent_size,
            'model_dropouts': self._dropout
        }


class Encoder(nn.Module):
    """ Class for encoder """

    def __init__(self, layer_sizes: List[int], dropout: List[float], input_size: Union[int, tuple]):
        super().__init__()

        self.MLP = nn.Sequential()

        # input layer
        self.MLP.add_module(name=f"L{0}", module=nn.Linear(input_size, layer_sizes[0]))
        self.MLP.add_module(name=f"D{0}", module=nn.Dropout(p=dropout[0]))
        self.MLP.add_module(name=f"A{0}", module=nn.ReLU())
        # following layers
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]), 1):
            self.MLP.add_module(name=f"L{i}", module=nn.Linear(in_size, out_size))
            if i < len(dropout):
                self.MLP.add_module(name=f"D{i}", module=nn.Dropout(p=dropout[i]))
            self.MLP.add_module(name=f"A{i}", module=nn.ReLU())

    def forward(self, x):
        x = self.MLP(x)
        return x


class Decoder(nn.Module):
    """ Class for decoder """

    def __init__(self, layer_sizes: List[int], latent_size: int, dropout: List[float],
                 output_size: Union[int, tuple]):
        super().__init__()

        self.MLP = nn.Sequential()

        input_size = latent_size
        layers = list(zip([input_size] + layer_sizes[:-1], layer_sizes))
        for i, (in_size, out_size) in enumerate(layers):
            self.MLP.add_module(name=f"L{i}", module=nn.Linear(in_size, out_size))
            if 0 < i < len(dropout):
                self.MLP.add_module(name=f"D{i}", module=nn.Dropout(p=dropout[i]))
            self.MLP.add_module(name=f"A{i}", module=nn.ReLU())

        # output layer
        self.MLP.add_module(name=f"L{len(layers)}", module=nn.Linear(layer_sizes[-1], output_size))
        if len(dropout) >= len(layers):
            self.MLP.add_module(name=f"D{len(layers)}", module=nn.Dropout(p=dropout[len(layers) - 1]))
        self.MLP.add_module(name=f"A{len(layers)}", module=nn.Sigmoid())

    def forward(self, z):
        x = self.MLP(z)
        return x


class EncoderWithLatent(Encoder):
    def __init__(self, layer_sizes: List[int], latent_size: int, dropout: List[float],
                 input_size: Union[int, tuple]):
        super().__init__(layer_sizes, dropout, input_size)

        # add last fc layer to get latent vector
        self.MLP.add_module(name=f'latent_layer_{len(layer_sizes)}', module=nn.Linear(layer_sizes[-1], latent_size))

    def forward(self, x):
        x = self.MLP(x)
        return x
