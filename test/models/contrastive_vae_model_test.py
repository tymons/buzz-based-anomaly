import unittest
import torch

from utils.model_factory import HiveModelFactory
from models.contrastive_base_model import ContrastiveBaseModel
from models.model_type import HiveModelType
from features.contrastive_feature_dataset import ContrastiveInput


class ContrastiveAEModelTest(unittest.TestCase):

    def test_model_is_build_basic_setup(self):
        config, input_size = {
                                 'layers': [64, 32, 16],
                                 'dropout': [0.3, 0.3, 0.3],
                                 'latent': 8
                             }, 2048

        model: ContrastiveBaseModel = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_vae'),
                                                                   (input_size, ), config)
        self.assertListEqual(model.get_params().get('model_layers'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])

        input_data = ContrastiveInput(torch.empty(size=(1, input_size)), torch.empty(size=(1, input_size)))
        self.assertEqual(model(input_data).target.shape[1], input_size)
        self.assertEqual(model(input_data).background.shape[1], input_size)
        self.assertEqual(model(input_data).target_qs_latent.shape[1], config['latent'])
