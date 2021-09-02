import torch

from torch import nn
from models.base_model import BaseModel
import torch.nn.functional as F

from typing import Union, List


class Autoencoder(BaseModel):
    """ Vanilla autoencoder """

    def __init__(self, encoder_layer_sizes: List[int], latent_size: int, decoder_layer_sizes: List[int],
                 input_size: Union[int, tuple], dropout: Union[List[float], float] = 0.2):
        super().__init__()

        assert len(decoder_layer_sizes) == len(encoder_layer_sizes)

        self._encoder_sizes = encoder_layer_sizes
        self._decoder_sizes = decoder_layer_sizes
        self._latent_size = latent_size
        self._dropout = [dropout] * len(encoder_layer_sizes) if isinstance(dropout, float) else dropout

        self.encoder = EncoderWithLatent(encoder_layer_sizes, latent_size, dropout, input_size)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, dropout, input_size)

    def loss_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Method for calculating loss function for pytorch model
        :param x: data input
        :param y: model output data
        :return: loss
        """
        mse = F.mse_loss(x, y, reduction='mean')
        return mse

    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        return y

    def inference(self, x):
        y = self.encoder(x)
        return y

    def get_params(self) -> dict:
        """
        Function for returning model layer sizes
        :return: dictionary with model layer sizes
        """
        return {
            'model_encoder_layers': self._encoder_sizes,
            'model_decoder_layers': self._decoder_sizes,
            'model_latent': self._latent_size
        }


class Encoder(nn.Module):
    """ Class for encoder """

    def __init__(self, layer_sizes: List[int], dropout: List[float], input_size: Union[int, tuple]):
        super().__init__()

        self.MLP = nn.Sequential()

        # input layer
        self.MLP.add_module(name=f"L{0}", module=nn.Linear(input_size, layer_sizes[0]))
        self.MLP.add_module(name=f"A{0}", module=nn.ReLU())
        # following layers
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]), 1):
            self.MLP.add_module(name=f"L{i}", module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name=f"A{i}", module=nn.ReLU())
            self.MLP.add_module(name=f"D{i}", module=nn.Dropout(p=dropout[i]))

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
        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name=f"L{i}", module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name=f"A{i}", module=nn.ReLU())
            self.MLP.add_module(name=f"D{i}", module=nn.Dropout(p=dropout[i]))
        # output layer
        self.MLP.add_module(name=f"L{len(layer_sizes) + 1}", module=nn.Linear(layer_sizes[-1], output_size))
        self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z):
        x = self.MLP(z)
        return x


class EncoderWithLatent(Encoder):
    def __init__(self, layer_sizes: List[int], latent_size: int, dropout: List[float],
                 input_size: Union[int, tuple]):
        super().__init__(layer_sizes, dropout, input_size)
        # add last fc layer to get latent vector
        self.latent_layer = nn.Linear(layer_sizes[-1], latent_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.MLP(x)
        x = self.latent_layer(x)
        x = self.relu(x)
        return x
