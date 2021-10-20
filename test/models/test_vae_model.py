import unittest
import torch

from utils.model_factory import HiveModelFactory, HiveModelType, model_check
from models.base_model import BaseModel


class TestModelFactoryMethods(unittest.TestCase):
    def test_model_is_build(self):
        input_size = 512
        config = {
            'layers': [64, 32, 16],
            'dropout': [0.3, 0.3, 0.3],
            'latent': 8
        }
        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('vae'), (input_size,), config)
        self.assertIsNotNone(model_check(model, (1, input_size), device='cpu'), "model verification failed!")
        self.assertListEqual(model.get_params().get('model_layers'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])
        self.assertEqual(model(torch.empty(size=(1, input_size))).output.shape[1], input_size)
        self.assertEqual(model(torch.empty(size=(1, input_size))).mean.shape[1], config['latent'])
        self.assertEqual(model(torch.empty(size=(1, input_size))).log_var.shape[1], config['latent'])