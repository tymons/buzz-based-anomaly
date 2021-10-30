import unittest
import torch

from utils.model_factory import HiveModelFactory, HiveModelType, model_check
from models.variational.vae_base_model import VaeBaseModel


def get_default_config():
    input_size = 256
    config = {
        'layers': [16, 8, 4],
        'dropout': [0.3, 0.3, 0.3],
        'latent': 8
    }
    return config, input_size


class TestVAEModelMethods(unittest.TestCase):
    def test_vae_model_is_build(self):
        config, input_size = get_default_config()
        model: VaeBaseModel = HiveModelFactory.build_model(HiveModelType.from_name('vae'), (input_size,), config)
        self.assertIsNotNone(model_check(model, (1, input_size), device='cpu'), "model verification failed!")
        self.assertListEqual(model.get_params().get('model_layers'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])
        self.assertEqual(model(torch.empty(size=(1, input_size))).output.shape[1], input_size)
        self.assertEqual(model(torch.empty(size=(1, input_size))).mean.shape[1], config['latent'])
        self.assertEqual(model(torch.empty(size=(1, input_size))).log_var.shape[1], config['latent'])

    def test_vae_model_inference(self):
        config, input_size = get_default_config()
        batch_size = 32
        input_tensor = torch.empty((batch_size, input_size))
        model: VaeBaseModel = HiveModelFactory.build_model(HiveModelType.from_name('vae'), (input_size,), config)
        self.assertTupleEqual(model.get_latent(input_tensor).shape, (batch_size, config['latent']))
