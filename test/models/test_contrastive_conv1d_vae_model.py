import unittest
import torch
from typing import Tuple

from models.base_model import BaseModel
from models.model_type import HiveModelType
from utils.model_factory import HiveModelFactory


def get_default_config() -> Tuple[dict, int]:
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
           }, 2048


class ContrastiveConv1DVAEModelTest(unittest.TestCase):

    def test_model_is_build_basic_setup(self):
        config, input_size = get_default_config()

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_conv1d_vae'), (input_size,),
                                                        config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertListEqual(model.get_params().get('model_feature_map'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_kernel_size'), config['kernel'])
        self.assertEqual(model.get_params().get('model_padding'), config['padding'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])
        self.assertEqual(model.get_params().get('model_max_pool'), config['max_pool'])

        target, background = torch.empty(size=(32, 1, input_size)), torch.empty(size=(32, 1, input_size))
        self.assertEqual(model(target, background).target.shape[-1], input_size)
        self.assertEqual(model(target, background).background.shape[-1], input_size)
        self.assertEqual(model(target, background).target_qs_latent.shape[-1], config['latent'])

    def test_model_is_build_different_input_size(self):
        config, _ = get_default_config()
        input_size = 4523

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_conv1d_vae'), (input_size,),
                                                        config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertListEqual(model.get_params().get('model_feature_map'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_kernel_size'), config['kernel'])
        self.assertEqual(model.get_params().get('model_padding'), config['padding'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])
        self.assertEqual(model.get_params().get('model_max_pool'), config['max_pool'])

        target, background = torch.empty(size=(32, 1, input_size)), torch.empty(size=(32, 1, input_size))
        self.assertEqual(model(target, background).target.shape[-1], input_size)
        self.assertEqual(model(target, background).background.shape[-1], input_size)
        self.assertEqual(model(target, background).target_qs_latent.shape[-1], config['latent'])

    def test_model_is_build_bigger_max_pool(self):
        config, input_size = get_default_config()
        config['max_pool'] = 5

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_conv1d_vae'), (input_size,),
                                                        config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertListEqual(model.get_params().get('model_feature_map'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_kernel_size'), config['kernel'])
        self.assertEqual(model.get_params().get('model_padding'), config['padding'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])
        self.assertEqual(model.get_params().get('model_max_pool'), config['max_pool'])

        target, background = torch.empty(size=(32, 1, input_size)), torch.empty(size=(32, 1, input_size))
        self.assertEqual(model(target, background).target.shape[-1], input_size)
        self.assertEqual(model(target, background).background.shape[-1], input_size)
        self.assertEqual(model(target, background).target_qs_latent.shape[-1], config['latent'])

    def test_model_is_build_different_padding(self):
        config, input_size = get_default_config()
        config['padding'] = 4

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_conv1d_vae'), (input_size,),
                                                        config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertListEqual(model.get_params().get('model_feature_map'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_kernel_size'), config['kernel'])
        self.assertEqual(model.get_params().get('model_padding'), config['padding'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])
        self.assertEqual(model.get_params().get('model_max_pool'), config['max_pool'])

        target, background = torch.empty(size=(32, 1, input_size)), torch.empty(size=(32, 1, input_size))
        self.assertEqual(model(target, background).target.shape[-1], input_size)
        self.assertEqual(model(target, background).background.shape[-1], input_size)
        self.assertEqual(model(target, background).target_qs_latent.shape[-1], config['latent'])

    def test_model_is_build_different_kernel(self):
        config, input_size = get_default_config()
        config['kernel'] = 6

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_conv1d_vae'), (input_size,),
                                                        config)

        self.assertIsNotNone(model, "model build failed!")
        self.assertListEqual(model.get_params().get('model_feature_map'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_kernel_size'), config['kernel'])
        self.assertEqual(model.get_params().get('model_padding'), config['padding'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])
        self.assertEqual(model.get_params().get('model_max_pool'), config['max_pool'])

        target, background = torch.empty(size=(32, 1, input_size)), torch.empty(size=(32, 1, input_size))
        self.assertEqual(model(target, background).target.shape[-1], input_size)
        self.assertEqual(model(target, background).background.shape[-1], input_size)
        self.assertEqual(model(target, background).target_qs_latent.shape[-1], config['latent'])


if __name__ == '__main__':
    unittest.main()
