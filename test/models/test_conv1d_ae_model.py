import unittest
import torch
from typing import Tuple

from models.vanilla.base_model import BaseModel
from models.model_type import HiveModelType
from utils.model_factory import HiveModelFactory, model_check


def get_default_config() -> Tuple[dict, int]:
    """
    Function for getting default config along with default input size
    :return: config, input_size
    """
    return {
               'layers': [16, 8, 4],
               'dropout': [0.3, 0.3, 0.3],
               'latent': 8,
               'kernel': 4,
               'padding': 2,
               'max_pool': 2
           }, 256


class TestConvolutional1DAutoencoderModelMethods(unittest.TestCase):

    def test_conv1d_autoencoder_model_is_build_with_basic_setup(self):
        config, input_size = get_default_config()

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('conv1d_autoencoder'), (input_size,),
                                                        config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertIsNotNone(model_check(model, (1, input_size), device='cpu'), "model verification failed!")
        self.assertListEqual(model.get_params().get('model_feature_map'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_kernel_size'), config['kernel'])
        self.assertEqual(model.get_params().get('model_padding'), config['padding'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])
        self.assertEqual(model.get_params().get('model_max_pool'), config['max_pool'])
        self.assertEqual(model(torch.empty(size=(1, 1, input_size))).shape[2], input_size)

    def test_conv1d_autoencoder_model_is_build_with_input_size_2053(self):
        config, _ = get_default_config()
        input_size = 2053

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('conv1d_autoencoder'), (input_size,),
                                                        config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertIsNotNone(model_check(model, (1, input_size), device='cpu'), "model verification failed!")
        self.assertListEqual(model.get_params().get('model_feature_map'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_kernel_size'), config['kernel'])
        self.assertEqual(model.get_params().get('model_padding'), config['padding'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])
        self.assertEqual(model.get_params().get('model_max_pool'), config['max_pool'])
        self.assertEqual(model(torch.empty(size=(1, 1, input_size))).shape[2], input_size)

    def test_conv1d_autoencoder_model_is_build_with_max_pool_3(self):
        config, input_size = get_default_config()
        config['max_pool'] = 3

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('conv1d_autoencoder'), (input_size,),
                                                        config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertIsNotNone(model_check(model, (1, input_size), device='cpu'), "model verification failed!")
        self.assertListEqual(model.get_params().get('model_feature_map'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_kernel_size'), config['kernel'])
        self.assertEqual(model.get_params().get('model_padding'), config['padding'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])
        self.assertEqual(model.get_params().get('model_max_pool'), config['max_pool'])
        self.assertEqual(model(torch.empty(size=(1, 1, input_size))).shape[2], input_size)

    def test_conv1d_autoencoder_model_is_build_with_padding_4(self):
        config, input_size = get_default_config()
        config['padding'] = 4

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('conv1d_autoencoder'), (input_size,),
                                                        config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertIsNotNone(model_check(model, (1, input_size), device='cpu'), "model verification failed!")
        self.assertListEqual(model.get_params().get('model_feature_map'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_kernel_size'), config['kernel'])
        self.assertEqual(model.get_params().get('model_padding'), config['padding'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])
        self.assertEqual(model.get_params().get('model_max_pool'), config['max_pool'])
        self.assertEqual(model(torch.empty(size=(1, 1, input_size))).shape[2], input_size)

    def test_conv1d_autoencoder_model_is_build_with_kernel_6(self):
        config, input_size = get_default_config()
        config['kernel'] = 6

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('conv1d_autoencoder'), (input_size,),
                                                        config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertIsNotNone(model_check(model, (1, input_size), device='cpu'), "model verification failed!")
        self.assertListEqual(model.get_params().get('model_feature_map'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_kernel_size'), config['kernel'])
        self.assertEqual(model.get_params().get('model_padding'), config['padding'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])
        self.assertEqual(model.get_params().get('model_max_pool'), config['max_pool'])
        self.assertEqual(model(torch.empty(size=(1, 1, input_size))).shape[2], input_size)

    def test_conv1d_autoencoder_model_inference(self):
        config, input_size = get_default_config()
        batch_size = 32
        input_tensor = torch.empty((batch_size, 1, input_size))
        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('conv1d_autoencoder'),
                                                        (input_size,), config)
        self.assertTupleEqual(model.get_latent(input_tensor).shape, (batch_size, config['latent']))


if __name__ == '__main__':
    unittest.main()
