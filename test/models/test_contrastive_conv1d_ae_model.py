import unittest
import torch

from utils.model_factory import HiveModelFactory, HiveModelType
from models.base_model import BaseModel


class ContrastiveConv1DAEModelTest(unittest.TestCase):
    def test_model_is_build(self):
        input_size = 512
        config = {
            'layers': [64, 32, 16],
            'dropout': [0.3, 0.3, 0.3],
            'latent': 8,
            'kernel': 4,
            'padding': 2,
            'max_pool': 2
        }

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_conv1d_autoencoder'),
                                                        (input_size,), config)

        self.assertListEqual(model.get_params().get('model_feature_map'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_kernel_size'), config['kernel'])
        self.assertEqual(model.get_params().get('model_padding'), config['padding'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])
        self.assertEqual(model.get_params().get('model_max_pool'), config['max_pool'])

        target, background = torch.empty(size=(32, 1, input_size)), torch.empty(size=(32, 1, input_size))
        self.assertEqual(model(target, background).target.shape[-1], input_size)
        self.assertEqual(model(target, background).background.shape[-1], input_size)


if __name__ == '__main__':
    unittest.main()
