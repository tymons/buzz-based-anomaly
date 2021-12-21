from models.model_type import HiveModelType
from models.vanilla.base_model import BaseModel
from torch import nn, functional
from typing import List
from models.conv_utils import convolutional_to_mlp


def _conv1d_block(in_f: int, out_f: int, dropout_prob: float, *args, **kwargs):
    """
    Function for 1d conv block
    :param in_f: number of input features
    :param out_f: number of output features
    :param dropout_prob: dropout probability
    :param args: list arguments for nn.Conv1D method
    :param kwargs: dict arguments for nn.Conv1D method
    :return: nn.Sequential object
    """
    return nn.Sequential(
        nn.Conv1d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm1d(out_f),
        nn.ReLU(),
        nn.Dropout(p=dropout_prob)
    )


def _conv1d_transpose_block(in_f: int, out_f: int, dropout_prob: float, *args, **kwargs):
    """
    Function for reverse 1d conv block
    :param out_f: number of output features
    :param dropout_prob: dropout probability
    :param args: list arguments for nn.Conv1D method
    :param kwargs: dict arguments for nn.Conv1D method
    :return: nn.Sequential object
    """
    return nn.Sequential(
        nn.ConvTranspose1d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm1d(out_f),
        nn.ReLU(),
        nn.Dropout(p=dropout_prob)
    )


class Conv1DEncoder(nn.Module):
    def __init__(self, features: List[int], dropout_probs: List[float], kernel_size: int, padding: int, max_pool: int):
        super().__init__()
        self.conv = nn.Sequential()

        self.conv.add_module(name='le-conv-0', module=_conv1d_block(1, features[0], dropout_probs[0],
                                                                    kernel_size=kernel_size,
                                                                    padding=padding))
        self.conv.add_module(name='ae-conv-0', module=nn.MaxPool1d(max_pool))
        for i, (in_size, out_size) in enumerate(zip(features[:-1], features[1:]), 1):
            self.conv.add_module(name=f'le-conv-{i}', module=_conv1d_block(in_size, out_size, dropout_probs[i],
                                                                           kernel_size=kernel_size,
                                                                           padding=padding))
            self.conv.add_module(name=f'ae-conv-{i}', module=nn.MaxPool1d(max_pool))

    def forward(self, x):
        y = self.conv(x)
        return y


class Conv1DEncoderWithLatent(nn.Module):
    def __init__(self, features: List[int], dropout_probs: List[float], kernel_size: int, padding: int, max_pool: int,
                 latent: int, conv_to_mlp_size: int):
        super().__init__()
        self.encoder = Conv1DEncoder(features, dropout_probs, kernel_size, padding, max_pool)
        self.flatten = nn.Flatten()
        self.mlp = nn.Linear(conv_to_mlp_size, latent)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        latent = self.mlp(x)
        return latent


class Conv1DDecoder(nn.Module):
    def __init__(self, features: List[int], dropout_probs: List[float], kernel_size: int, padding: int, latent: int,
                 conv_to_mlp_size: int, forced_conv_shapes: List[int]):
        super().__init__()
        self.conv = nn.Sequential()

        self.mlp = nn.Linear(latent, conv_to_mlp_size)
        self.unflatten = nn.Unflatten(1, (features[0], forced_conv_shapes[0]))
        for i, (in_size, out_size) in enumerate(zip(features[:-1], features[1:]), 1):
            self.conv.add_module(name=f'ld-conv-{i}',
                                 module=_conv1d_transpose_block(in_size, out_size, dropout_probs[i],
                                                                kernel_size=kernel_size,
                                                                padding=padding))
            self.conv.add_module(name=f'ld-upsample-{i}', module=nn.Upsample(size=forced_conv_shapes[i]))
        self.conv.add_module(name=f"ld-conv-{len(features) + 1}", module=nn.ConvTranspose1d(features[-1], 1,
                                                                                            kernel_size=kernel_size,
                                                                                            padding=padding))
        self.conv.add_module(name=f'ld-upsample-{len(features) + 1}', module=nn.Upsample(size=forced_conv_shapes[-1]))
        self.conv.add_module(name=f'activation-conv-{len(features) + 1}', module=nn.Sigmoid())

    def forward(self, latent):
        x = self.mlp(latent)
        x = self.unflatten(x)
        y = self.conv(x)
        return y


class Conv1DAE(BaseModel):
    def __init__(self, model_type: HiveModelType, features: List[int], dropout_probs: List[float], kernel_size: int,
                 padding: int, max_pool: int, latent: int, input_size: int):
        super().__init__(model_type)
        self._feature_map = features
        self._dropout_probs = dropout_probs
        self._kernel_size = kernel_size
        self._padding = padding
        self._latent = latent
        self._max_pool = max_pool

        connector_size, conv_temporal = convolutional_to_mlp(input_size, len(features), kernel_size, padding, max_pool)
        self.encoder = Conv1DEncoderWithLatent(features, dropout_probs, kernel_size, padding, max_pool, latent,
                                               features[-1] * connector_size)
        self.decoder = Conv1DDecoder(features[::-1], dropout_probs[::-1], kernel_size, padding, latent,
                                     features[-1] * connector_size, conv_temporal[::-1])

    def forward(self, x):
        latent = self.encoder(x)
        y = self.decoder(latent)
        return y

    def loss_fn(self, x, y):
        """
        Function for calculating convolutional autoencoder loss
        :param x: input data
        :param y: model output
        :return:
        """
        mse = functional.F.mse_loss(y, x, reduction='mean')
        return mse

    def get_params(self):
        """
        Method for getting params for the model
        :return: dictionary with parameters
        """
        return {
            'model_feature_map': self._feature_map,
            'model_dropouts': self._dropout_probs,
            'model_kernel_size': self._kernel_size,
            'model_padding': self._padding,
            'model_latent': self._latent,
            'model_max_pool': self._max_pool
        }
