import unittest
import torch
from typing import Tuple

from models.base_model import BaseModel
from models.model_type import HiveModelType
from utils.model_factory import HiveModelFactory, model_check


def get_default_config() -> Tuple[dict, Tuple]:
    """
    Function for getting default config along with default input size
    :return: config, input_size
    """
    return {
               'layers': [64, 32, 16],
               'dropout': [0.3, 0.3, 0.3],
               'latent': 8,
               'kernel': 4,
               'padding': 2,
               'max_pool': 2
           }, (256, 2048)


class Conv2DVAEModelTest(unittest.TestCase):

    def test_model_is_build_basic_setup(self):
        config, input_size = get_default_config()

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('conv2d_vae'), input_size,
                                                        config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertIsNotNone(model_check(model, (1, *input_size), device='cpu'), "model verification failed!")
        self.assertListEqual(config['layers'], model.get_params().get('model_feature_map'))
        self.assertListEqual(config['dropout'], model.get_params().get('model_dropouts'))
        self.assertEqual(config['kernel'], model.get_params().get('model_kernel_size'))
        self.assertEqual(config['padding'], model.get_params().get('model_padding'))
        self.assertEqual(config['latent'], model.get_params().get('model_latent'))
        self.assertEqual(config['max_pool'], model.get_params().get('model_max_pool'))
        self.assertEqual(input_size, model(torch.empty(size=(1, 1, *input_size))).output.shape[2:])
        self.assertEqual(config['latent'], model(torch.empty(size=(1, 1, *input_size))).mean.shape[1])
        self.assertEqual(config['latent'], model(torch.empty(size=(1, 1, *input_size))).log_var.shape[1])

    def test_model_is_build_different_input_size(self):
        config, _ = get_default_config()
        input_size = (175, 4523)

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('conv2d_vae'), input_size,
                                                        config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertIsNotNone(model_check(model, (1, *input_size), device='cpu'), "model verification failed!")
        self.assertListEqual(config['layers'], model.get_params().get('model_feature_map'))
        self.assertListEqual(config['dropout'], model.get_params().get('model_dropouts'))
        self.assertEqual(config['kernel'], model.get_params().get('model_kernel_size'))
        self.assertEqual(config['padding'], model.get_params().get('model_padding'))
        self.assertEqual(config['latent'], model.get_params().get('model_latent'))
        self.assertEqual(config['max_pool'], model.get_params().get('model_max_pool'))
        self.assertEqual(input_size, model(torch.empty(size=(1, 1, *input_size))).output.shape[2:])
        self.assertEqual(config['latent'], model(torch.empty(size=(1, 1, *input_size))).mean.shape[1])
        self.assertEqual(config['latent'], model(torch.empty(size=(1, 1, *input_size))).log_var.shape[1])

    def test_model_is_build_bigger_max_pool(self):
        config, input_size = get_default_config()
        config['max_pool'] = 5

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('conv2d_vae'), input_size,
                                                        config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertIsNotNone(model_check(model, (1, *input_size), device='cpu'), "model verification failed!")
        self.assertListEqual(config['layers'], model.get_params().get('model_feature_map'))
        self.assertListEqual(config['dropout'], model.get_params().get('model_dropouts'))
        self.assertEqual(config['kernel'], model.get_params().get('model_kernel_size'))
        self.assertEqual(config['padding'], model.get_params().get('model_padding'))
        self.assertEqual(config['latent'], model.get_params().get('model_latent'))
        self.assertEqual(config['max_pool'], model.get_params().get('model_max_pool'))
        self.assertEqual(input_size, model(torch.empty(size=(1, 1, *input_size))).output.shape[2:])
        self.assertEqual(config['latent'], model(torch.empty(size=(1, 1, *input_size))).mean.shape[1])
        self.assertEqual(config['latent'], model(torch.empty(size=(1, 1, *input_size))).log_var.shape[1])

    def test_model_is_build_different_padding(self):
        config, input_size = get_default_config()
        config['padding'] = 4

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('conv2d_vae'), input_size,
                                                        config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertIsNotNone(model_check(model, (1, *input_size), device='cpu'), "model verification failed!")
        self.assertListEqual(config['layers'], model.get_params().get('model_feature_map'))
        self.assertListEqual(config['dropout'], model.get_params().get('model_dropouts'))
        self.assertEqual(config['kernel'], model.get_params().get('model_kernel_size'))
        self.assertEqual(config['padding'], model.get_params().get('model_padding'))
        self.assertEqual(config['latent'], model.get_params().get('model_latent'))
        self.assertEqual(config['max_pool'], model.get_params().get('model_max_pool'))
        self.assertEqual(input_size, model(torch.empty(size=(1, 1, *input_size))).output.shape[2:])
        self.assertEqual(config['latent'], model(torch.empty(size=(1, 1, *input_size))).mean.shape[1])
        self.assertEqual(config['latent'], model(torch.empty(size=(1, 1, *input_size))).log_var.shape[1])

    def test_model_is_build_different_kernel(self):
        config, input_size = get_default_config()
        config['kernel'] = 6

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('conv2d_vae'), input_size,
                                                        config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertIsNotNone(model_check(model, (1, *input_size), device='cpu'), "model verification failed!")
        self.assertListEqual(config['layers'], model.get_params().get('model_feature_map'))
        self.assertListEqual(config['dropout'], model.get_params().get('model_dropouts'))
        self.assertEqual(config['kernel'], model.get_params().get('model_kernel_size'))
        self.assertEqual(config['padding'], model.get_params().get('model_padding'))
        self.assertEqual(config['latent'], model.get_params().get('model_latent'))
        self.assertEqual(config['max_pool'], model.get_params().get('model_max_pool'))
        self.assertEqual(input_size, model(torch.empty(size=(1, 1, *input_size))).output.shape[2:])
        self.assertEqual(config['latent'], model(torch.empty(size=(1, 1, *input_size))).mean.shape[1])
        self.assertEqual(config['latent'], model(torch.empty(size=(1, 1, *input_size))).log_var.shape[1])


if __name__ == '__main__':
    unittest.main()