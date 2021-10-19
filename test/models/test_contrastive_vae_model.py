import unittest
import torch

from utils.model_factory import HiveModelFactory
from models.contrastive_variational_base_model import ContrastiveVariationalBaseModel
from models.model_type import HiveModelType


class ContrastiveAEModelTest(unittest.TestCase):

    def test_model_is_built_basic_setup(self):
        config, input_size = {
                                 'layers': [64, 32, 16],
                                 'dropout': [0.3, 0.3, 0.3],
                                 'latent': 8
                             }, 2048

        model: ContrastiveVariationalBaseModel = HiveModelFactory.build_model(
            HiveModelType.from_name('contrastive_vae'),
            (input_size,), config)
        self.assertListEqual(model.get_params().get('model_layers'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])

        target, background = torch.empty(size=(32, 1, input_size)), torch.empty(size=(32, 1, input_size))
        self.assertEqual(model(target, background).target.shape[-1], input_size)
        self.assertEqual(model(target, background).background.shape[-1], input_size)
        self.assertEqual(model(target, background).target_qs_latent.shape[-1], config['latent'])

    def test_discriminator_is_built_basic_setup(self):
        config = {
            'layers': [8, 32, 64]
        }

        discriminator_model: torch.nn.Module = HiveModelFactory.get_discriminator(config, 2)
        self.assertGreaterEqual(discriminator_model(torch.empty((1, 4)), torch.empty((1, 4)))[0].squeeze().item(), 0.0,
                                "discriminator output should be scaled")
        self.assertGreaterEqual(discriminator_model(torch.empty((1, 4)), torch.empty((1, 4)))[1].squeeze().item(), 0.0,
                                "discriminator output should be scaled")
        self.assertLessEqual(discriminator_model(torch.empty((1, 4)), torch.empty((1, 4)))[0].squeeze().item(), 1.0,
                             "discriminator output should be scaled")
        self.assertLessEqual(discriminator_model(torch.empty((1, 4)), torch.empty((1, 4)))[1].squeeze().item(), 1.0,
                             "discriminator output should be scaled")
