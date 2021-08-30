import torch

from torch import nn
from models.base_model import BaseModel
import torch.nn.functional as F


def ae_loss_fun(data_input, model_output_dict):
    """ Function for calculating loss for vanilla autoencoders """
    mse = F.mse_loss(data_input, model_output_dict['target'], reduction='mean')
    return mse


class Autoencoder(BaseModel):
    """ Vanilla autoencoder """

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, input_size):
        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list
        assert type(input_size) == int

        self._encoder_sizes = encoder_layer_sizes
        self._decoder_sizes = decoder_layer_sizes
        self._latent_size = latent_size

        self.encoder = EncoderWithLatent(encoder_layer_sizes, latent_size, input_size)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, input_size)

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

    def __init__(self, layer_sizes, input_size):
        super().__init__()

        self.MLP = nn.Sequential()

        # input layer
        self.MLP.add_module(name=f"L{0}", module=nn.Linear(input_size, layer_sizes[0]))
        self.MLP.add_module(name=f"A{0}", module=nn.ReLU())
        self.MLP.add_module(name=f"D{0}", module=nn.Dropout(p=0.1))
        # following layers
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]), 1):
            self.MLP.add_module(name=f"L{i}", module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name=f"A{i}", module=nn.ReLU())
            self.MLP.add_module(name=f"D{i}", module=nn.Dropout(p=0.1))

    def forward(self, x):
        x = self.MLP(x)
        return x


class Decoder(nn.Module):
    """ Class for decoder """

    def __init__(self, layer_sizes, latent_size, output_size):
        super().__init__()

        self.MLP = nn.Sequential()

        input_size = latent_size
        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name=f"L{i}", module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name=f"A{i}", module=nn.ReLU())
            self.MLP.add_module(name=f"D{i}", module=nn.Dropout(p=0.1))
        # output layer
        self.MLP.add_module(name=f"L{len(layer_sizes) + 1}", module=nn.Linear(layer_sizes[-1], output_size))
        self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z):
        x = self.MLP(z)
        return x


class EncoderWithLatent(Encoder):
    def __init__(self, layer_sizes, latent_size, input_size):
        super().__init__(layer_sizes, input_size)
        # add last fc layer to get latent vector
        self.latent_layer = nn.Linear(layer_sizes[-1], latent_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.MLP(x)
        x = self.latent_layer(x)
        x = self.relu(x)
        return x
