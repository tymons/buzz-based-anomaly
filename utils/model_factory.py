from typing import Callable

import optuna
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


def build_optuna_ae_config(trial: optuna.Trial, input_size: int) -> dict:
    """
    Function for building optuna trial config for autoencoder
    :param trial: optuna trial object
    :param input_size: data input size
    :return: config dictionary
    """
    n_layers = trial.suggest_int('n_layers', 1, 4)
    latent = trial.suggest_int('latent', 2, 16)

    layers = []
    dividers = []
    dropouts = []
    last_input_size = input_size
    for i in range(n_layers):
        dividers.append(trial.suggest_uniform(f'divider_{i}', 1, 3))
        last_input_size = last_input_size // dividers[i]
        layers.append(trial.suggest_int(f'layer_{i}', 1, max(last_input_size, latent)))
        dropouts.append(trial.suggest_uniform(f'dropout_{i}', 0.1, 0.5))

    return {
        'encoder': {'layers': layers},
        'decoder': {'layers': layers[::-1]},
        'dropout': {'layers': dropouts},
        'latent': latent
    }


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
        dropout_layer_probabilities = config.get('dropout', {'layers': [0.2, 0.2, 0.2]})
        latent_size = config.get('latent', 2)

        print(f'building ae model with config: encoder_layers({encoder_layer_sizes.get("layers")}),'
              f' decoder_layer_sizes({decoder_layer_sizes.get("layers")}), latent({latent_size}),'
              f' dropout({dropout_layer_probabilities.get("layers")})')
        return Autoencoder(encoder_layer_sizes.get("layers"), latent_size,
                           decoder_layer_sizes.get("layers"), input_shape, dropout_layer_probabilities.get('layers'))

    @staticmethod
    def build_model(model_type: HiveModelType, input_shape: int, config: dict) -> BaseModel:
        """
        Method for building ML models
        :param trail: optuna trail object
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
