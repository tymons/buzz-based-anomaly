import unittest
import torch
from typing import Tuple

from models.vanilla.contrastive.contrastive_base_model import ContrastiveBaseModel as CBM
from models.model_type import HiveModelType
from utils.model_factory import HiveModelFactory


def get_default_config() -> Tuple[dict, Tuple]:
    """
    Function for getting default config along with default input size
    :return: config, input_size
    """
    return {
               'layers': [8, 4, 2],
               'dropout': [0.3, 0.3, 0.3],
               'latent': 2,
               'kernel': 4,
               'padding': 2,
               'max_pool': 2
           }, (256, 2048)


class TestContrastiveConv2dAEModelMethods(unittest.TestCase):

    def test_contrastive_conv2d_autoencoder_model_is_build_basic_setup(self):
        config, input_size = get_default_config()

        model: CBM = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_conv2d_autoencoder'),
                                                  input_size, config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertListEqual(config['layers'], model.get_params().get('model_feature_map'))
        self.assertListEqual(config['dropout'], model.get_params().get('model_dropouts'))
        self.assertEqual(config['kernel'], model.get_params().get('model_kernel_size'))
        self.assertEqual(config['padding'], model.get_params().get('model_padding'))
        self.assertEqual(config['latent'], model.get_params().get('model_latent'))
        self.assertEqual(config['max_pool'], model.get_params().get('model_max_pool'))

        target, background = torch.empty(size=(32, 1, *input_size)), torch.empty(size=(32, 1, *input_size))
        self.assertEqual(model(target, background).target.shape[-2:], input_size)
        self.assertEqual(model(target, background).background.shape[-2:], input_size)

    def test_contrastive_conv2d_autoencoder_model_is_build_with_input_size_175_4523(self):
        config, _ = get_default_config()
        input_size = (175, 4523)

        model: CBM = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_conv2d_autoencoder'),
                                                  input_size, config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertListEqual(config['layers'], model.get_params().get('model_feature_map'))
        self.assertListEqual(config['dropout'], model.get_params().get('model_dropouts'))
        self.assertEqual(config['kernel'], model.get_params().get('model_kernel_size'))
        self.assertEqual(config['padding'], model.get_params().get('model_padding'))
        self.assertEqual(config['latent'], model.get_params().get('model_latent'))
        self.assertEqual(config['max_pool'], model.get_params().get('model_max_pool'))

        target, background = torch.empty(size=(32, 1, *input_size)), torch.empty(size=(32, 1, *input_size))
        self.assertEqual(model(target, background).target.shape[-2:], input_size)
        self.assertEqual(model(target, background).background.shape[-2:], input_size)

    def test_contrastive_conv2d_autoencoder_model_is_build_with_max_pool_5(self):
        config, input_size = get_default_config()
        config['max_pool'] = 5

        model: CBM = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_conv2d_autoencoder'),
                                                  input_size, config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertListEqual(config['layers'], model.get_params().get('model_feature_map'))
        self.assertListEqual(config['dropout'], model.get_params().get('model_dropouts'))
        self.assertEqual(config['kernel'], model.get_params().get('model_kernel_size'))
        self.assertEqual(config['padding'], model.get_params().get('model_padding'))
        self.assertEqual(config['latent'], model.get_params().get('model_latent'))
        self.assertEqual(config['max_pool'], model.get_params().get('model_max_pool'))

        target, background = torch.empty(size=(32, 1, *input_size)), torch.empty(size=(32, 1, *input_size))
        self.assertEqual(model(target, background).target.shape[-2:], input_size)
        self.assertEqual(model(target, background).background.shape[-2:], input_size)

    def test_contrastive_conv2d_autoencoder_model_is_build_with_padding_4(self):
        config, input_size = get_default_config()
        config['padding'] = 4

        model: CBM = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_conv2d_autoencoder'),
                                                  input_size, config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertListEqual(config['layers'], model.get_params().get('model_feature_map'))
        self.assertListEqual(config['dropout'], model.get_params().get('model_dropouts'))
        self.assertEqual(config['kernel'], model.get_params().get('model_kernel_size'))
        self.assertEqual(config['padding'], model.get_params().get('model_padding'))
        self.assertEqual(config['latent'], model.get_params().get('model_latent'))
        self.assertEqual(config['max_pool'], model.get_params().get('model_max_pool'))

        target, background = torch.empty(size=(32, 1, *input_size)), torch.empty(size=(32, 1, *input_size))
        self.assertEqual(model(target, background).target.shape[-2:], input_size)
        self.assertEqual(model(target, background).background.shape[-2:], input_size)

    def test_contrastive_conv2d_autoencoder_model_is_build_with_kernel_6(self):
        config, input_size = get_default_config()
        config['kernel'] = 6

        model: CBM = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_conv2d_autoencoder'),
                                                  input_size, config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertListEqual(config['layers'], model.get_params().get('model_feature_map'))
        self.assertListEqual(config['dropout'], model.get_params().get('model_dropouts'))
        self.assertEqual(config['kernel'], model.get_params().get('model_kernel_size'))
        self.assertEqual(config['padding'], model.get_params().get('model_padding'))
        self.assertEqual(config['latent'], model.get_params().get('model_latent'))
        self.assertEqual(config['max_pool'], model.get_params().get('model_max_pool'))

        target, background = torch.empty(size=(32, 1, *input_size)), torch.empty(size=(32, 1, *input_size))
        self.assertEqual(model(target, background).target.shape[-2:], input_size)
        self.assertEqual(model(target, background).background.shape[-2:], input_size)

    def test_contrastive_conv2d_autoencoder_model_inference(self):
        config, input_size = get_default_config()
        batch_size = 32
        input_tensor = torch.empty((batch_size, 1, *input_size))
        model: CBM = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_conv2d_autoencoder'),
                                                  input_size, config)
        self.assertTupleEqual(model.get_latent(input_tensor).shape, (batch_size, config['latent']))


if __name__ == '__main__':
    unittest.main()
