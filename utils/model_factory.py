import logging
import optuna

from models.model_type import HiveModelType
from torchsummary import summary
from models.base_model import BaseModel
from models.ae import Autoencoder
from models.conv1d_ae import Conv1DAE
from models.conv2d_ae import Conv2DAE
from models.vae import VAE
from models.conv1d_vae import Conv1DVAE
from models.conv2d_vae import Conv2DVAE
from models.contrastive_vae import ContrastiveVAE
from models.contrastive_base_model import ContrastiveBaseModel
from models.discriminator import Discriminator
from typing import Callable, Tuple, Union


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
        'model': {
            'layers': layers,
            'dropout': dropouts,
            'latent': latent
        }
    }

    if model_type.value.startswith('conv'):
        extend_config_for_convolution(config, trial)

    if model_type.value.startswith('contrastive'):
        extend_config_for_contrastive(config)

    return config


def extend_config_for_contrastive(config: dict, trial: optuna.Trial = None):
    """
    Method for adding discriminator parameters
    :param config: dictionary with existing config
    :param trial: optuna trial
    :return:
    """
    config['discriminator'] = {}
    config['discriminator']['layers'] = [64, 32, 256] if trial is None else \
        [trial.suggest_int(f'discriminator_layer_{i}', 2, 256) for i in range(3)]
    return config


def extend_config_for_convolution(config: dict, trial: optuna.Trial):
    """
    Method for appending config values specific for
    :param config: dictionary
    :param trial: optuna trial
    :return: dict
    """
    kernel: int = trial.suggest_int('kernel', 2, 8)
    config['model']['padding'] = trial.suggest_int('padding', 0, kernel)
    config['model']['max_pool'] = 2
    config['model']['kernel'] = kernel
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

        logging.debug(f'building conv2d ae model with config: encoder_layers({layers}),'
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
    def _get_conv1d_vae_model(config: dict, input_size: Tuple) -> Conv1DVAE:
        """
        Function for getting convolutional 1d VAE model
        :param config: config for conv1d vae model
        :param input_size: input size
        :return:
        """
        layers = config.get('layers', [256, 64, 16])
        dropout = config.get('dropout', [0.1, 0.1, 0.1])
        latent_size = config.get('latent', 2)
        kernel = config.get('kernel', 2)
        padding = config.get('padding', 0)
        max_pool = config.get('max_pool', 2)

        logging.debug(f'building conv1d vae model with config: encoder_layers({layers}),'
                      f' dropout({dropout}), latent({latent_size}), kernel({kernel}), padding({padding}),'
                      f' max_pool({max_pool})')
        return Conv1DVAE(layers, dropout, kernel_size=kernel, padding=padding, latent=latent_size,
                         input_size=input_size[0], max_pool=max_pool)

    @staticmethod
    def _get_conv2d_vae_model(config: dict, input_size: Tuple) -> Conv2DVAE:
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

        logging.debug(f'building conv2d vae model with config: encoder_layers({layers}),'
                      f' dropout({dropout}), latent({latent_size}), kernel({kernel}), padding({padding}),'
                      f' max_pool({max_pool})')

        return Conv2DVAE(layers, dropout, kernel_size=kernel, padding=padding, latent=latent_size,
                         input_size=input_size, max_pool=max_pool)

    @staticmethod
    def _get_contrastive_vae_model(config: dict, input_size: Tuple) -> ContrastiveVAE:
        """
        Method for building 1D Variational Autoencoder
        :param config: model config
        :param input_size: input size
        :return: model
        """
        layers = config.get('layers', [256, 32, 16])
        dropouts = config.get('dropout', [0.2, 0.2, 0.2])
        latent_size = config.get('latent', 2)
        use_discriminator = config.get('use_discriminator', False)

        logging.debug(f'building contrastive vae model with config: layers({layers}), latent({latent_size}),'
                      f' dropout({dropouts}), use_discriminator({use_discriminator})')
        return ContrastiveVAE(layers, latent_size, input_size[0], dropouts, use_discriminator)

    @staticmethod
    def get_discriminator(discriminator_config: dict, autoencoder_latent: int) -> Discriminator:
        """
        Method for building discriminator for contrastive autoencoder
        :param discriminator_config: dictionary with config for discriminator. Only 'layers' key is supported
        :param autoencoder_latent:
        :return:
        """
        layers = discriminator_config.get('layers', [8, 16])
        return Discriminator(layers, 2 * autoencoder_latent)

    @staticmethod
    def build_model(model_type: HiveModelType, input_shape: Tuple, config: dict) \
            -> Union[BaseModel, ContrastiveBaseModel]:
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
    def build_model_and_check(model_type: HiveModelType, input_shape: Tuple, config: dict) -> \
            Union[BaseModel, ContrastiveBaseModel]:
        """
        Method for building and verifying model
        :param model_type: model type enum
        :param config: dictionary for model config
        :param input_shape: data input shape
        """
        return model_check(HiveModelFactory.build_model(model_type, input_shape, config), (1, *input_shape),
                           device='cpu')
