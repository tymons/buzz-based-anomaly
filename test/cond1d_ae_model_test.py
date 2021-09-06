import unittest

from models.base_model import BaseModel
from models.model_type import HiveModelType
from utils.model_factory import HiveModelFactory, model_check


class Conv1DAeModelTest(unittest.TestCase):
    def test_model_is_build(self):
        input_size = 2048
        config = {
            'encoder': {'features': [64, 32, 16]},
            'decoder': {'features': [16, 32, 64]},
            'latent': 8
        }
        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('conv1d_autoencoder'), input_size,
                                                        config)
        self.assertIsNotNone(model, "model build failed!")
        self.assertIsNotNone(model_check(model, (1, 1, input_size)), "model verification failed!")
        self.assertListEqual(model.get_params().get('model_encoder_layers'), config['encoder']['layers'])
        self.assertListEqual(model.get_params().get('model_decoder_layers'), config['decoder']['layers'])
        self.assertEqual(model.get_params().get('model_latent'), config['latent'])


if __name__ == '__main__':
    unittest.main()
