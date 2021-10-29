import unittest
import torch

from utils.model_factory import HiveModelFactory, HiveModelType, model_check
from models.base_model import BaseModel


class TestAutoencoderModelMethods(unittest.TestCase):
    def test_autoencoder_model_is_build(self):
        input_size = 512
        config = {
            'layers': [64, 32, 16],
            'dropout': [0.3, 0.3, 0.3],
            'latent': 8
        }
        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('autoencoder'), (input_size,), config)
        self.assertIsNotNone(model_check(model, (1, input_size), device='cpu'), "model verification failed!")
        self.assertListEqual(model.get_params().get('model_layers'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])
        self.assertEqual(model(torch.empty(size=(1, input_size))).shape[1], input_size)

    def test_autoencoder_model_inference(self):
        input_size = 512
        config = {
            'layers': [64, 32, 16],
            'dropout': [0.3, 0.3, 0.3],
            'latent': 8
        }
        batch_size = 32
        input_tensor = torch.empty((batch_size, 512))
        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('autoencoder'), (input_size,), config)
        self.assertTupleEqual(model.inference(input_tensor).shape, (batch_size, config['latent']))


if __name__ == '__main__':
    unittest.main()
