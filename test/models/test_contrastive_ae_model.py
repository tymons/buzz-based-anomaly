import unittest
import torch

from utils.model_factory import HiveModelFactory, HiveModelType
from models.base_model import BaseModel


class ContrastiveAEModelTest(unittest.TestCase):
    def test_model_is_build(self):
        input_size = 512
        config = {
            'layers': [64, 32, 16],
            'dropout': [0.3, 0.3, 0.3],
            'latent': 8
        }

        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_autoencoder'),
                                                        (input_size,), config)

        self.assertListEqual(model.get_params().get('model_layers'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])

        target, background = torch.empty(size=(32, 1, input_size)), torch.empty(size=(32, 1, input_size))
        self.assertEqual(model(target, background).target.shape[-1], input_size)
        self.assertEqual(model(target, background).background.shape[-1], input_size)


if __name__ == '__main__':
    unittest.main()
