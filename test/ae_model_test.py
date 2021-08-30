import unittest

from utils.model_factory import HiveModelFactory, HiveModelType, model_check


class TestModelFactoryMethods(unittest.TestCase):

    def test_model_is_build(self):
        input_size = 512
        config = {
            'encoder': {'layers': [64, 32, 16]},
            'decoder': {'layers': [16, 32, 64]},
            'latent': 8
        }
        model = HiveModelFactory.build_model(HiveModelType.from_name('autoencoder'), config, input_size)
        self.assertIsNotNone(model_check(model, (1, 1, input_size)), "Model verification failed!")


if __name__ == '__main__':
    unittest.main()
