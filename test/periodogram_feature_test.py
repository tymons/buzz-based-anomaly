import unittest
import numpy as np

from utils.feature_factory import SoundFeatureFactory
from features.feature_type import SoundFeatureType
from pathlib import Path


class TestPeriodogramMethods(unittest.TestCase):
    def setUp(self):
        self.filename_10kHz = Path('data/10kHz.wav')
        self.filename_440Hz = Path('data/440Hz.wav')

    def test_10kHz(self):
        config = {
            'slice_frequency': {
                'start': 0,
                'stop': 20000
            },
            'convert_db': False,
            'normalize': False
        }
        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('periodogram'),
                                                    [self.filename_10kHz], ['DEADBEEF'],
                                                    config)

        (periodogram, frequencies), _ = dataset.get_item(0)
        self.assertEqual(10000, frequencies[np.argmax(periodogram)], "fundamental frequency should be 10kHz")

    def test_440Hz(self):
        config = {
            'slice_frequency': {
                'start': 0,
                'stop': 20000
            },
            'convert_db': False,
            'normalize': False
        }
        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('periodogram'),
                                                    [self.filename_440Hz], ['DEADBEEF'],
                                                    config)

        (periodogram, frequencies), _ = dataset.get_item(0)
        self.assertEqual(440, frequencies[np.argmax(periodogram)], "fundamental frequency should be 440Hz")

    def test_normalization(self):
        config = {
            'slice_frequency': {
                'start': 0,
                'stop': 20000
            },
            'convert_db': False,
            'normalize': True
        }
        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('periodogram'),
                                                    [self.filename_10kHz, self.filename_440Hz], ['1', '1'],
                                                    config)

        (periodogram_10kHz, frequencies_10kHz), _ = dataset.get_item(0)
        self.assertEqual(10000, frequencies_10kHz[np.argmax(periodogram_10kHz)],
                         "fundamental frequency should be 10kHz")
        (periodogram_440Hz, frequencies_440Hz), _ = dataset.get_item(1)
        self.assertEqual(440, frequencies_440Hz[np.argmax(periodogram_440Hz)],
                         "fundamental frequency should be 440Hz")

    def test_converting_to_decibel(self):
        config = {
            'slice_frequency': {
                'start': 0,
                'stop': 20000
            },
            'convert_db': True,
            'normalize': False
        }
        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('periodogram'),
                                                    [self.filename_10kHz, self.filename_440Hz], ['1', '1'],
                                                    config)

        (periodogram_10kHz, frequencies_10kHz), _ = dataset.get_item(0)
        self.assertEqual(10000, frequencies_10kHz[np.argmax(periodogram_10kHz)],
                         "fundamental frequency should be 10kHz")
        (periodogram_440Hz, frequencies_440Hz), _ = dataset.get_item(1)
        self.assertEqual(440, frequencies_440Hz[np.argmax(periodogram_440Hz)],
                         "fundamental frequency should be 440Hz")


if __name__ == '__main__':
    unittest.main()
