import unittest

from utils.model_factory import HiveModelFactory, HiveModelType, model_check
from models.base_model import BaseModel


class TestModelFactoryMethods(unittest.TestCase):
    def test_model_is_build(self):
        input_size = 512
        config = {
            'encoder': {'layers': [64, 32, 16]},
            'decoder': {'layers': [16, 32, 64]},
            'latent': 8
        }
        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('autoencoder'), input_size, config)
        self.assertIsNotNone(model_check(model, (1, input_size)), "model verification failed!")
        self.assertListEqual(model.get_params().get('model_encoder_layers'), config['encoder']['layers'])
        self.assertListEqual(model.get_params().get('model_decoder_layers'), config['decoder']['layers'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])


if __name__ == '__main__':
    unittest.main()
