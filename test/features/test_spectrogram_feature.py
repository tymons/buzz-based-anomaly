import unittest
import numpy as np

from utils.feature_factory import SoundFeatureFactory
from features.feature_type import SoundFeatureType
from pathlib import Path
from utils.utils import closest_power_2


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
        'nfft': 512,
        'hop_len': 256,
        'convert_db': False,
        'window': 'hann',
        'round_power_2': True
    }


class TestPeriodogramMethods(unittest.TestCase):
    def setUp(self):
        self.filename_10kHz = Path(Path(__file__).parent.resolve(), '../data/10kHz.wav')
        self.filename_1kHz = Path(Path(__file__).parent.resolve(), '../data/1kHz.wav')

    def test_10kHz_basic(self):
        config = _get_default_config()
        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('spectrogram'),
                                                    [self.filename_10kHz], [1], config)

        (spectrogram, frequencies, times), _ = dataset.get_item(0)

        self.assertAlmostEqual(10000, frequencies[round(np.argmax(spectrogram, axis=0).mean())], delta=10,
                               msg="fundamental frequency should be around 10kHz")

    def test_10kHz_cropping(self):
        config = _get_default_config()
        config['slice_frequency'] = {'start': 200, 'stop': 5000}

        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('spectrogram'),
                                                    [self.filename_10kHz], [1], config)

        (spectrogram, frequencies, times), _ = dataset.get_item(0)

        self.assertNotAlmostEqual(10000, frequencies[round(np.argmax(spectrogram, axis=0).mean())], delta=10,
                                  msg="fundamental frequency should be not equal to 10kHz")

    def test_10kHz_db_scaled(self):
        config = _get_default_config()
        config['convert_db'] = True
        config['round_power_2'] = False

        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('spectrogram'),
                                                    [self.filename_10kHz], [1], config)

        (spectrogram, frequencies, times), _ = dataset.get_item(0)

        self.assertAlmostEqual(10000, frequencies[round(np.argmax(spectrogram, axis=0).mean())], delta=10,
                               msg="fundamental frequency should be around 10kHz")

    def test_10kHz_normalized(self):
        config = _get_default_config()
        config['normalize'] = True

        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('spectrogram'),
                                                    [self.filename_10kHz], [1], config)

        (spectrogram, frequencies, times), _ = dataset.get_item(0)

        self.assertAlmostEqual(10000, frequencies[round(np.argmax(spectrogram, axis=0).mean())], delta=10,
                               msg="fundamental frequency should be around 10kHz")

    def test_10kHz_db_scaled_normalized(self):
        config = _get_default_config()
        config['normalize'] = True
        config['convert_db'] = True
        config['round_power_2'] = False

        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('spectrogram'),
                                                    [self.filename_10kHz], [1], config)

        (spectrogram, frequencies, times), _ = dataset.get_item(0)

        self.assertAlmostEqual(10000, frequencies[round(np.argmax(spectrogram, axis=0).mean())], delta=10,
                               msg="fundamental frequency should be around 10kHz")

    def test_10kHz_do_not_round(self):
        config = _get_default_config()
        config['round_power_2'] = False

        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('spectrogram'),
                                                    [self.filename_10kHz], [1], config)

        (spectrogram, frequencies, times), _ = dataset.get_item(0)

        self.assertNotEqual(spectrogram.shape[0], 2 ** closest_power_2(spectrogram.shape[0]),
                            "shape should not be transformed to closest power of two")
        self.assertNotEqual(spectrogram.shape[1], 2 ** closest_power_2(spectrogram.shape[1]),
                            "shape should not be transformed to closest power of two")


if __name__ == '__main__':
    unittest.main()
