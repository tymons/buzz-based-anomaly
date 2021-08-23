import math
import torch

from torch import nn
from functools import reduce


def conv_mlp_layer_shape(input_shape: tuple, conv_feature_sizes: list, kernel: int,
                         stride: int, padding: int, max_pool: tuple):
    """ Function for calculating input size for mlp in combination with convolutional layers """
    input_shape_list = list(input_shape)
    for layer_number in range(len(conv_feature_sizes)):
        input_shape_list[0] = int((((input_shape_list[0] - kernel + 2 * padding) / stride) + 1) // max_pool[0])
        input_shape_list[1] = int((((input_shape_list[1] - kernel + 2 * padding) / stride) + 1) // max_pool[1])

    input_shape_list.insert(0, int(conv_feature_sizes[-1]))
    return tuple(input_shape_list)


def _conv2d_block(in_f, out_f, *args, **kwargs):
    """ Function for building convolutional block

        Attributes
            in_f - number of input features
            out_f - number of output features
    """
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
        nn.Dropout2d(p=0.2)
    )


def _conv2d_transpose_block(in_f, out_f, *args, **kwargs):
    """ Function for building transpose convolutional block

        Attributes
            in_f - number of input features
            out_f - number of output features
    """
    return nn.Sequential(
        nn.ConvTranspose2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
        nn.Dropout2d(p=0.2)
    )


class View(nn.Module):
    """ Function for nn.Sequentional to reshape data """

    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class ConvolutionalAE(nn.Module):
    """ Class for convolutional autoencoder """

    def __init__(self, encoder_conv_sizes, encoder_mlp_sizes,
                 decoder_conv_sizes, decoder_mlp_sizes, latent_size, input_shape):
        assert type(encoder_conv_sizes) == list
        assert type(encoder_mlp_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_conv_sizes) == list
        assert type(decoder_mlp_sizes) == list
        assert type(input_shape) == tuple

        super().__init__()

        conv_to_mlp_shape = conv_mlp_layer_shape(input_shape, encoder_conv_sizes, kernel=3, stride=1, padding=1,
                                                 max_pool=(2, 2))

        self.encoder = ConvolutionalEncoderWithLatent(encoder_conv_sizes, encoder_mlp_sizes, latent_size,
                                                      connector_shape=conv_to_mlp_shape)
        self.decoder = ConvolutionalDecoder(decoder_conv_sizes, decoder_mlp_sizes, latent_size,
                                            connector_shape=conv_to_mlp_shape)

    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        return {'target': y}


class ConvolutionalEncoder(nn.Module):
    """ Class for conv encoder without last latent layer as this class should be used in both vae and ae """

    def __init__(self, conv_features_sizes, linear_layer_sizes, connector_shape):
        """ constructor for convolutional encoder 
        
        Parameters:
            conv_features_sizes (list): feature sizes for convolutional filters
            linear_layer_sizes (list): mlp layer sizes
            input_shape (tuple): shape which will be used to calculate and add extra layer on mlp
                                if you dont want to add that extra layer just skip input_shape parameter
        """
        super().__init__()

        self.conv = nn.Sequential()
        self.mlp = nn.Sequential()
        self.flat = nn.Flatten()

        self.conv.add_module(name=f"e-fconv{0}",
                             module=_conv2d_block(1, conv_features_sizes[0], kernel_size=3, padding=1))
        self.conv.add_module(name=f"e-max{0}", module=nn.MaxPool2d(2, 2))
        for i, (in_size, out_size) in enumerate(zip(conv_features_sizes[:-1], conv_features_sizes[1:]), 1):
            self.conv.add_module(name=f"e-fconv{i}", module=_conv2d_block(in_size, out_size, kernel_size=3, padding=1))
            self.conv.add_module(name=f"e-max{i}", module=nn.MaxPool2d(2, 2))

        mlp_input_shape = int(reduce((lambda x, y: x * y), connector_shape))
        self.mlp.add_module(name=f"e-linear{0}", module=nn.Linear(mlp_input_shape, linear_layer_sizes[0]))
        self.mlp.add_module(name=f"e-batchnorm{0}", module=nn.BatchNorm1d(linear_layer_sizes[0]))
        self.mlp.add_module(name=f"e-relu{0}", module=nn.ReLU())
        for i, (in_size, out_size) in enumerate(zip(linear_layer_sizes[:-1], linear_layer_sizes[1:]), 1):
            self.mlp.add_module(name=f"e-linear{i}", module=nn.Linear(in_size, out_size))
            self.mlp.add_module(name=f"e-batchnorm{i}", module=nn.BatchNorm1d(out_size))
            self.mlp.add_module(name=f"e-relu{i}", module=nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        x = self.flat(x)
        x = self.mlp(x)
        return x


class ConvolutionalDecoder(nn.Module):
    """ Class for conv decoder """

    def __init__(self, conv_features_sizes, linear_layer_sizes, latent_size, connector_shape):
        super().__init__()

        self.conv = nn.Sequential()
        self.mlp = nn.Sequential()
        self.sigmoid = nn.Sigmoid()

        self.mlp.add_module(name=f"d-linear{0}", module=nn.Linear(latent_size, linear_layer_sizes[0]))
        self.mlp.add_module(name=f"d-batchnorm{0}", module=nn.BatchNorm1d(linear_layer_sizes[0]))
        self.mlp.add_module(name=f"d-relu{0}", module=nn.ReLU())
        for i, (in_size, out_size) in enumerate(zip(linear_layer_sizes[:-1], linear_layer_sizes[1:]), 1):
            self.mlp.add_module(name=f"d-linear{i}", module=nn.Linear(in_size, out_size))
            self.mlp.add_module(name=f"d-batchnorm{i}", module=nn.BatchNorm1d(out_size))
            self.mlp.add_module(name=f"d-relu{i}", module=nn.ReLU())

        mlp_output_shape = int(reduce((lambda x, y: x * y), connector_shape))
        self.mlp.add_module(name=f"d-linear{len(linear_layer_sizes) + 1}",
                            module=nn.Linear(linear_layer_sizes[-1], mlp_output_shape))
        self.mlp.add_module(name=f"d-batchnorm{len(linear_layer_sizes) + 1}", module=nn.BatchNorm1d(mlp_output_shape))
        self.mlp.add_module(name=f"d-relu{len(linear_layer_sizes) + 1}", module=nn.ReLU())
        self.view = View([-1, *connector_shape])
        for i, (in_size, out_size) in enumerate(zip(conv_features_sizes[:-1], conv_features_sizes[1:])):
            self.conv.add_module(name=f"d-fconv{i}",
                                 module=_conv2d_transpose_block(in_size, out_size, kernel_size=2, stride=2))
        self.conv.add_module(name=f"d-conv{i}",
                             module=nn.ConvTranspose2d(conv_features_sizes[-1], 1, kernel_size=2, stride=2))

    def forward(self, x):
        x = self.mlp(x)
        x = self.view(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


class ConvolutionalEncoderWithLatent(ConvolutionalEncoder):
    """ Class for convolutional encoder with last layer for latent vector """

    def __init__(self, conv_features_sizes, linear_layer_sizes, latent_size, connector_shape):
        super().__init__(conv_features_sizes, linear_layer_sizes, connector_shape)
        # add last fc layer to get latent vector
        self.latent_layer = nn.Linear(linear_layer_sizes[-1], latent_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.flat(x)
        x = self.mlp(x)
        x = self.latent_layer(x)
        x = self.relu(x)
        return x
