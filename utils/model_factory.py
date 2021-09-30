import logging
import optuna

from models.model_type import HiveModelType
from torchsummary import summary
from models.base_model import BaseModel
from models.ae import Autoencoder
from models.conv1d_ae import Conv1DAE
from models.conv2d_ae import Conv2DAE
from models.vae import VAE
from typing import Callable, Tuple


def model_check(model, input_shape, device="cuda"):
    """ Function for model check """
    summary(model, input_shape, device=device)
    logging.debug(f'model check success! {model}')
    return model


def build_optuna_model_config(model_type: HiveModelType, input_shape: Tuple, trial: optuna.Trial) -> dict:
    """
    Function for building optuna trial config for autoencoder
    :param model_type: model type
    :param trial: optuna trial object
    :param input_shape: data input size
    :return: config dictionary
    """
    n_layers = trial.suggest_int('n_layers', 1, 4)
    latent = trial.suggest_int('latent', 2, 32)

    layers = []
    dividers = []
    dropouts = []

    last_input_size = min(*input_shape) if len(input_shape) > 1 else input_shape[0]
    for i in range(n_layers):
        dividers.append(trial.suggest_uniform(f'divider_{i}', 1, 3))
        layer_size = trial.suggest_int(f'layer_{i}', 1, max(int(last_input_size // dividers[i]), latent))
        last_input_size = layer_size
        layers.append(layer_size)
        dropouts.append(trial.suggest_uniform(f'dropout_{i}', 0.1, 0.5))

    config: dict = {
        'layers': layers,
        'dropout': dropouts,
        'latent': latent
    }

    if model_type.value.startswith('conv'):
        kernel: int = trial.suggest_int('kernel', 2, 8)
        config['padding'] = trial.suggest_int('padding', 0, kernel)
        config['max_pool'] = 2
        config['kernel'] = kernel

    return config


class HiveModelFactory:
    """ Factory for ML models """

    @staticmethod
    def _get_autoencoder_model(config: dict, input_shape: Tuple) -> Autoencoder:
        """
        Method for building vanilla autoencoder model
        :param config: config for model
        :param input_shape: data input shape
        :return: model, config used
        """
        layers = config.get('layers', [256, 32, 16])
        dropouts = config.get('dropout', [0.2, 0.2, 0.2])
        latent_size = config.get('latent', 2)

        logging.debug(f'building ae model with config: layers({layers}), latent({latent_size}),'
                      f' dropout({dropouts})')
        return Autoencoder(layers, latent_size, input_shape[0], dropouts)

    @staticmethod
    def _get_conv1d_autoencoder_model(config: dict, input_size: Tuple) -> Conv1DAE:
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
                        input_size=input_size[0], max_pool=max_pool)

    @staticmethod
    def _get_conv2d_autoencoder_model(config: dict, input_size: Tuple) -> Conv2DAE:
        """
        Method for building 2D convolutional Autoencoder
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

        return Conv2DAE(layers, dropout, kernel_size=kernel, padding=padding, latent=latent_size,
                        input_size=input_size, max_pool=max_pool)

    @staticmethod
    def _get_vae_model(config: dict, input_size: Tuple) -> VAE:
        """
        Method for building 1D Variational Autoencoder
        :param config: model config
        :param input_size: input size
        :return: model
        """
        layers = config.get('layers', [256, 32, 16])
        dropouts = config.get('dropout', [0.2, 0.2, 0.2])
        latent_size = config.get('latent', 2)

        logging.debug(f'building vae model with config: layers({layers}), latent({latent_size}),'
                      f' dropout({dropouts})')
        return VAE(layers, latent_size, input_size[0], dropouts)

    @staticmethod
    def build_model(model_type: HiveModelType, input_shape: Tuple, config: dict) -> BaseModel:
        """
        Method for building ML models
        :param model_type: model type enum
        :param config: dictionary for model config
        :param input_shape: data input shape
        :return:
        """
        model_func: Callable[[dict, tuple], (BaseModel, dict)] = \
            getattr(HiveModelFactory, f'_get_{model_type.value.lower()}_model',
                    lambda x, y: logging.error('invalid model type!'))
        model = model_func(config, input_shape)
        return model

    @staticmethod
    def build_model_and_check(model_type: HiveModelType, input_shape: Tuple, config: dict) -> BaseModel:
        """
        Method for building and verifying model
        :param model_type: model type enum
        :param config: dictionary for model config
        :param input_shape: data input shape
        """
        return model_check(HiveModelFactory.build_model(model_type, input_shape, config), (1, *input_shape),
                           device='cpu')
