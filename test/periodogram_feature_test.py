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
        'normalize': False
    }


class TestPeriodogramMethods(unittest.TestCase):
    def setUp(self):
        self.filename_10kHz = Path(Path(__file__).parent.resolve(), 'data/10kHz.wav')
        self.filename_440Hz = Path(Path(__file__).parent.resolve(), 'data/440Hz.wav')

    def test_10kHz(self):
        config = _get_default_config()
        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('periodogram'),
                                                    [self.filename_10kHz], [1], config)

        (periodogram, frequencies), _ = dataset.get_item(0)
        self.assertEqual(10000, frequencies[np.argmax(periodogram)], "fundamental frequency should be 10kHz")

    def test_440Hz(self):
        config = _get_default_config()
        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('periodogram'),
                                                    [self.filename_440Hz], [1], config)

        (periodogram, frequencies), _ = dataset.get_item(0)
        self.assertEqual(440, frequencies[np.argmax(periodogram)], "fundamental frequency should be 440Hz")

    def test_10kHz_440Hz_with_normalization(self):
        config = _get_default_config()
        config['normalize'] = True

        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('periodogram'),
                                                    [self.filename_10kHz, self.filename_440Hz], [1, 1], config)

        (periodogram_10kHz, frequencies_10kHz), _ = dataset.get_item(0)
        self.assertEqual(10000, frequencies_10kHz[np.argmax(periodogram_10kHz)],
                         "fundamental frequency should be 10kHz")
        (periodogram_440Hz, frequencies_440Hz), _ = dataset.get_item(1)
        self.assertEqual(440, frequencies_440Hz[np.argmax(periodogram_440Hz)],
                         "fundamental frequency should be 440Hz")

    def test_10kHz_440Hz_with_decibel_scale(self):
        config = _get_default_config()
        config['convert_db'] = True

        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('periodogram'),
                                                    [self.filename_10kHz, self.filename_440Hz], [1, 1], config)

        (periodogram_10kHz, frequencies_10kHz), _ = dataset.get_item(0)
        self.assertEqual(10000, frequencies_10kHz[np.argmax(periodogram_10kHz)],
                         "fundamental frequency should be 10kHz")
        (periodogram_440Hz, frequencies_440Hz), _ = dataset.get_item(1)
        self.assertEqual(440, frequencies_440Hz[np.argmax(periodogram_440Hz)],
                         "fundamental frequency should be 440Hz")


if __name__ == '__main__':
    unittest.main()
