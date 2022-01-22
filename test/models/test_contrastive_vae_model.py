import unittest
import torch

from utils.model_factory import HiveModelFactory
from models.variational.contrastive.contrastive_variational_base_model import ContrastiveVariationalBaseModel as CVBM
from models.model_type import HiveModelType


def get_vae_default_config():
    config, input_size = {
                             'layers': [16, 8, 4],
                             'dropout': [0.3, 0.3, 0.3],
                             'latent': 8
                         }, 256

    return config, input_size


class TestContrastiveVAEModelMethods(unittest.TestCase):

    def test_contrastive_vae_model_is_built_basic_setup(self):
        config, input_size = get_vae_default_config()

        model: CVBM = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_vae'), (input_size,), config)
        self.assertListEqual(model.get_params().get('model_layers'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])

        target, background = torch.empty(size=(32, 1, input_size)), torch.empty(size=(32, 1, input_size))
        self.assertEqual(model(target, background).target.shape[-1], input_size)
        self.assertEqual(model(target, background).background.shape[-1], input_size)
        self.assertEqual(model(target, background).target_latent.shape[-1], config['latent'])

    def test_discriminator_is_built_basic_setup(self):
        discriminator_model: torch.nn.Module = HiveModelFactory.get_discriminator(4)
        self.assertGreaterEqual(discriminator_model(torch.empty((1, 4))).squeeze().item(), 0.0,
                                "discriminator output should be scaled")
        self.assertLessEqual(discriminator_model(torch.empty((1, 4))).squeeze().item(), 1.0,
                             "discriminator output should be scaled")

    def test_contrastive_vae_model_inference(self):
        config, input_size = get_vae_default_config()
        batch_size = 32
        input_tensor = torch.empty((batch_size, input_size))
        model: CVBM = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_vae'), (input_size,),
                                                   config)
        self.assertTupleEqual(model.get_latent(input_tensor).shape, (batch_size, config['latent']))
