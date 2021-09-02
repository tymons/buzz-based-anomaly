from typing import Callable

import torch
import traceback

from models.model_type import HiveModelType
from torchsummary import summary
from models.base_model import BaseModel
from models.ae import Autoencoder


def model_check(model, input_shape):
    """ Function for model check """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        summary(model.to(device), input_shape)
        print(f'model check success! {model}')
        return model
    except Exception as e:
        print('model self-check failure!')
        print(traceback.print_exc())
        print(e)
        return None


class HiveModelFactory:
    """ Factory for ML models """

    @staticmethod
    def _get_autoencoder_model(config: dict, input_shape: int) -> BaseModel:
        """
        Method for building vanilla autoencoder model
        :param config: config for model
        :param input_shape: data input shape
        :return: model, config used
        """
        encoder_layer_sizes = config.get('encoder', {'layers': [256, 32, 16]})
        decoder_layer_sizes = config.get('decoder', {'layers': [16, 32, 256]})
        latent_size = config.get('latent', 2)

        print(f'building ae model with config: encoder_layers({encoder_layer_sizes.get("layers")}),'
              f' decoder_layer_sizes({decoder_layer_sizes.get("layers")}),  latent({latent_size})')
        return Autoencoder(encoder_layer_sizes.get("layers"), latent_size,
                           decoder_layer_sizes.get("layers"), input_shape)


    @staticmethod
    def build_model(model_type: HiveModelType, config: dict, input_shape: int) -> BaseModel:
        """
        Method for building ML models
        :param model_type: model type enum
        :param config: dictionary for model config
        :param input_shape: data input shape
        :return:
        """
        model_func: Callable[[dict, int], (BaseModel, dict)] = \
            getattr(HiveModelFactory, f'_get_{model_type.value.lower()}_model',
                    lambda x, y: print('invalid model type!'))
        model = model_func(config, input_shape)

        return model_check(model, (1, 1, input_shape))
