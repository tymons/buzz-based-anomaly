import logging
import optuna

from models.model_type import HiveModelType
from torchsummary import summary
from models.base_model import BaseModel
from models.ae import Autoencoder
from models.conv1d_ae import Conv1DAE
from typing import Callable


def model_check(model, input_shape, device="cuda"):
    """ Function for model check """
    summary(model, input_shape, device=device)
    logging.debug(f'model check success! {model}')
    return model


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
        layer_size = trial.suggest_int(f'layer_{i}', 1, max(int(last_input_size // dividers[i]), latent))
        last_input_size = layer_size
        layers.append(layer_size)
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
        layers = config.get('layers', [256, 32, 16])
        dropouts = config.get('dropout',[0.2, 0.2, 0.2])
        latent_size = config.get('latent', 2)

        logging.debug(f'building ae model with config: layers({layers}), latent({latent_size}),'
                      f' dropout({dropouts})')
        return Autoencoder(layers, latent_size, input_shape, dropouts)

    @staticmethod
    def _get_conv1d_autoencoder_model(config: dict, input_size: int) -> BaseModel:
        """
        Method for building 1D convolutional Autoencoder
        :param config: model config
        :param input_size: input size
        :return: model
        """
        layers = config.get('layers', [256, 64, 16])
        dropout = config.get('dropout', [0.1, 0.1, 0.1])
        latent_size = config.get('latent', 2)
        kernel = config.get('kernel', 2)
        padding = config.get('padding', 0)
        max_pool = config.get('max_pool', 2)

        logging.debug(f'building conv1d ae model with config: encoder_layers({layers}),'
                      f' dropout({dropout}), latent({latent_size}), kernel({kernel}), padding({padding}),'
                      f' max_pool({max_pool})')
        return Conv1DAE(layers, dropout, kernel_size=kernel, padding=padding, latent=latent_size,
                        input_size=input_size, max_pool=max_pool)

    @staticmethod
    def build_model(model_type: HiveModelType, input_shape: int, config: dict) -> BaseModel:
        """
        Method for building ML models
        :param model_type: model type enum
        :param config: dictionary for model config
        :param input_shape: data input shape
        :return:
        """
        model_func: Callable[[dict, int], (BaseModel, dict)] = \
            getattr(HiveModelFactory, f'_get_{model_type.value.lower()}_model',
                    lambda x, y: logging.error('invalid model type!'))
        model = model_func(config, input_shape)
        return model

    @staticmethod
    def build_model_and_check(model_type: HiveModelType, input_shape: int, config: dict) -> BaseModel:
        """
        Method for building and verifying model
        :param model_type: model type enum
        :param config: dictionary for model config
        :param input_shape: data input shape
        """
        return model_check(HiveModelFactory.build_model(model_type, input_shape, config), (1, input_shape),
                           device='cpu')
