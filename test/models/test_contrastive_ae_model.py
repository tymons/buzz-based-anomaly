import unittest
import torch

from utils.model_factory import HiveModelFactory, HiveModelType
from models.vanilla.contrastive.contrastive_base_model import ContrastiveBaseModel


def get_default_config():
    input_size = 256
    config = {
        'layers': [16, 8, 4],
        'dropout': [0.3, 0.3, 0.3],
        'latent': 8
    }
    return config, input_size


class TestContrastiveAutoencoderModelMethods(unittest.TestCase):
    def test_contrastive_autoencoder_model_is_build(self):
        config, input_size = get_default_config()

        model: ContrastiveBaseModel = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_autoencoder'),
                                                                   (input_size,), config)

        self.assertListEqual(model.get_params().get('model_layers'), config['layers'])
        self.assertListEqual(model.get_params().get('model_dropouts'), config['dropout'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])

        target, background = torch.empty(size=(32, 1, input_size)), torch.empty(size=(32, 1, input_size))
        self.assertEqual(model(target, background).target.shape[-1], input_size)
        self.assertEqual(model(target, background).background.shape[-1], input_size)

    def test_contrastive_autoencoder_model_inference(self):
        config, input_size = get_default_config()
        batch_size = 32
        input_tensor = torch.empty((batch_size, input_size))
        model: ContrastiveBaseModel = HiveModelFactory.build_model(HiveModelType.from_name('contrastive_autoencoder'),
                                                                   (input_size,), config)
        self.assertTupleEqual(model.get_latent(input_tensor).shape, (batch_size, config['latent']))


if __name__ == '__main__':
    unittest.main()
