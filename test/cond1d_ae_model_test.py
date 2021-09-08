import unittest
import torch

from models.base_model import BaseModel
from models.model_type import HiveModelType
from utils.model_factory import HiveModelFactory, model_check


class Conv1DAeModelTest(unittest.TestCase):

    def test_model_is_build_basic_setup(self):
        input_size = 2048
        config = {
            'layers': [64, 32, 16],
            'dropout': [0.3, 0.3, 0.3],
            'latent': 8,
            'kernel': 4,
            'padding': 2,
            'max_pool': 2,
            'stride': 1,
        }
        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('conv1d_autoencoder'), input_size,
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

    def test_model_is_build_different_input_size(self):
        input_size = 4523
        config = {
            'layers': [64, 32, 16],
            'dropout': [0.3, 0.3, 0.3],
            'latent': 8,
            'kernel': 4,
            'padding': 2,
            'max_pool': 2,
            'stride': 1
        }
        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('conv1d_autoencoder'), input_size,
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

    def test_model_is_build_bigger_max_pool(self):
        input_size = 2048
        config = {
            'layers': [64, 32, 16],
            'dropout': [0.3, 0.3, 0.3],
            'latent': 8,
            'kernel': 4,
            'padding': 2,
            'max_pool': 5,
            'stride': 1
        }
        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('conv1d_autoencoder'), input_size,
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

    def test_model_is_build_different_padding(self):
        input_size = 2048
        config = {
            'layers': [64, 32, 16],
            'dropout': [0.3, 0.3, 0.3],
            'latent': 8,
            'kernel': 4,
            'padding': 4,
            'max_pool': 5,
            'stride': 1
        }
        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('conv1d_autoencoder'), input_size,
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

    def test_model_is_build_different_kernel(self):
        input_size = 2048
        config = {
            'layers': [64, 32, 16],
            'dropout': [0.3, 0.3, 0.3],
            'latent': 8,
            'kernel': 6,
            'padding': 4,
            'max_pool': 5,
            'stride': 1
        }
        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('conv1d_autoencoder'), input_size,
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


if __name__ == '__main__':
    unittest.main()
