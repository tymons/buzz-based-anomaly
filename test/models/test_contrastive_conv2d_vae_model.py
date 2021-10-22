import unittest
import torch
from typing import Tuple

from models.base_model import BaseModel
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


class ContrastiveConv2DVAEModelTest(unittest.TestCase):

    def test_model_is_build_basic_setup(self):
        config, input_size = get_default_config()

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_conv2d_vae'),
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

    def test_model_is_build_different_input_size(self):
        config, _ = get_default_config()
        input_size = (175, 4523)

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_conv2d_vae'),
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

    def test_model_is_build_bigger_max_pool(self):
        config, input_size = get_default_config()
        config['max_pool'] = 5

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_conv2d_vae'),
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

    def test_model_is_build_different_padding(self):
        config, input_size = get_default_config()
        config['padding'] = 4

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_conv2d_vae'),
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

    def test_model_is_build_different_kernel(self):
        config, input_size = get_default_config()
        config['kernel'] = 6

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_conv2d_vae'),
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


if __name__ == '__main__':
    unittest.main()
