import unittest
import numpy as np

from utils.feature_factory import SoundFeatureFactory
from features.feature_type import SoundFeatureType
from pathlib import Path


def _get_default_config() -> dict:
    """
    Function for getting default config
    :return: config, input_size
    """
    return {
        'slice_frequency': {
            'start': 0,
            'stop': 20000
        },
        'convert_db': False,
        'window': 'blackman'
    }


class TestPeriodogramMethods(unittest.TestCase):
    def setUp(self):
        self.filename_10kHz = Path(Path(__file__).parent.resolve(), '../data/10kHz.wav')
        self.filename_1kHz = Path(Path(__file__).parent.resolve(), '../data/1kHz.wav')

    def test_10kHz(self):
        config = _get_default_config()
        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('periodogram'),
                                                    [self.filename_10kHz], [1], config)

        (periodogram, frequencies), _ = dataset.get_item(0)
        self.assertEqual(10000, frequencies[np.argmax(periodogram)], "fundamental frequency should be 10kHz")

    def test_1kHz(self):
        config = _get_default_config()
        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('periodogram'),
                                                    [self.filename_1kHz], [1], config)

        (periodogram, frequencies), _ = dataset.get_item(0)
        self.assertEqual(1000, frequencies[np.argmax(periodogram)], "fundamental frequency should be 1kHz")

    def test_10kHz_1kHz_with_normalization(self):
        config = _get_default_config()
        config['normalize'] = True

        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('periodogram'),
                                                    [self.filename_10kHz, self.filename_1kHz], [1, 1], config)

        (periodogram_10kHz, frequencies_10kHz), _ = dataset.get_item(0)
        self.assertEqual(10000, frequencies_10kHz[np.argmax(periodogram_10kHz)],
                         "fundamental frequency should be 10kHz")
        (periodogram_1kHz, frequencies_1kHz), _ = dataset.get_item(1)
        self.assertEqual(1000, frequencies_1kHz[np.argmax(periodogram_1kHz)],
                         "fundamental frequency should be 1kHz")

    def test_10kHz_1kHz_with_decibel_scale(self):
        config = _get_default_config()
        config['convert_db'] = True

        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('periodogram'),
                                                    [self.filename_10kHz, self.filename_1kHz], [1, 1], config)

        (periodogram_10kHz, frequencies_10kHz), _ = dataset.get_item(0)
        self.assertEqual(10000, frequencies_10kHz[np.argmax(periodogram_10kHz)],
                         "fundamental frequency should be 10kHz")
        (periodogram_1kHz, frequencies_1kHz), _ = dataset.get_item(1)
        self.assertEqual(1000, frequencies_1kHz[np.argmax(periodogram_1kHz)],
                         "fundamental frequency should be 1kHz")


if __name__ == '__main__':
    unittest.main()
