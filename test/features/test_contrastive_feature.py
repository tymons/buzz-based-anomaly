import unittest

from pathlib import Path

import numpy as np

from utils.feature_factory import SoundFeatureFactory
from features.feature_type import SoundFeatureType
from features.contrastive_feature_dataset import ContrastiveFeatureDataset


class TestDoubleFeatureDataset(unittest.TestCase):
    def setUp(self):
        self.filename_10kHz = Path(Path(__file__).parent.resolve(), '../data/10kHz.wav')
        self.filename_1kHz = Path(Path(__file__).parent.resolve(), '../data/1kHz.wav')

    def test_building_double_feature_dataset(self):
        config = {
            'slice_frequency': {
                'start': 0,
                'stop': 20000
            },
            'nfft': 512,
            'hop_len': 256,
            'window': 'hann',
            'round_power_2': True,
            'nmfccs': 32,
            'nmels': 128
        }

        target_dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('mfcc'),
                                                           [self.filename_10kHz], [1], config)
        background_dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('mfcc'),
                                                               [self.filename_1kHz], [0], config)
        double_feature_dataset = ContrastiveFeatureDataset(target_dataset, background_dataset)

        target, background = double_feature_dataset.__getitem__(0)

        self.assertEqual(type(target), np.ndarray)
        self.assertEqual(type(background), np.ndarray)
        self.assertEqual(target.shape[1], 32)
        self.assertEqual(background.shape[1], 32)
