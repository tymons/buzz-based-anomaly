import unittest
import torch

from torch.utils.data import TensorDataset, DataLoader

from utils.model_factory import HiveModelFactory, HiveModelType
from models.vanilla.base_model import BaseModel
from utils.model_runner import ModelRunner


class TestModelRunnerMethods(unittest.TestCase):
    def test_method_inference_for_autoencoder(self):
        input_size = 512
        config = {
            'layers': [64, 32, 16],
            'dropout': [0.3, 0.3, 0.3],
            'latent': 8
        }

        dataset_len = 256
        data = torch.ones((dataset_len, input_size))
        labels = torch.zeros((dataset_len, 1))
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=32)
        model: BaseModel = HiveModelFactory.build_model(HiveModelType.from_name('autoencoder'), (input_size,), config)
        model_runner: ModelRunner = ModelRunner(comet_api_key='dev', torch_device=torch.device('cpu'))

        latent = model_runner.inference_latent(model, dataloader).cpu()
        self.assertIsNotNone(latent, "latent not build")
        self.assertTupleEqual(latent.shape, (dataset_len, config['latent']))
