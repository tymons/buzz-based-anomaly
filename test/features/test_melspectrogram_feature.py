import unittest
from pathlib import Path

from utils.feature_factory import SoundFeatureFactory, SoundFeatureType
from utils.utils import closest_power_2


def _get_default_config() -> dict:
    """
    Function for getting default mel spectrogram config
    :return:
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
        'round_power_2': False,
        'nmels': 10
    }


class TestMelSpectrogramDataset(unittest.TestCase):
    def setUp(self):
        self.filename_10kHz = Path(Path(__file__).parent.resolve(), '../data/1kHz.wav')

    def test_1kHz_nmels_rounding_low(self):
        config = _get_default_config()
        config['round_power_2'] = True

        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('melspectrogram'),
                                                    [self.filename_10kHz], [1], config)

        (melspectrogram, frequencies, times), _ = dataset.get_item(0)

        self.assertEqual(8, melspectrogram.shape[0], "nmels should be rounded to 8")

    def test_1kHz_nmels_rounding_up(self):
        config = _get_default_config()
        config['nmels'] = 15
        config['round_power_2'] = True

        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('melspectrogram'),
                                                    [self.filename_10kHz], [1], config)

        (melspectrogram, frequencies, times), _ = dataset.get_item(0)

        self.assertEqual(16, melspectrogram.shape[0], "nmels should be rounded to 16")

    def test_1kHz_nmels(self):
        config = _get_default_config()

        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('melspectrogram'),
                                                    [self.filename_10kHz], [1], config)

        (melspectrogram, frequencies, times), _ = dataset.get_item(0)

        self.assertEqual(10, melspectrogram.shape[0], "nmels should be equal")
        self.assertNotEqual(melspectrogram.shape[1], 2 ** closest_power_2(melspectrogram.shape[1]),
                            "shape should not be transformed to closest power of two")

    def test_melspectrogram_is_scaled(self):
        config = _get_default_config()

        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('melspectrogram'),
                                                    [self.filename_10kHz], [1], config)

        (melspectrogram, frequencies, times), _ = dataset.get_item(0)

        self.assertGreaterEqual(0.0, melspectrogram.min(), "melspectrogram should be scaled")
        self.assertLessEqual(round(melspectrogram.max().item(), 2), 1.0, "melspectrogram should be scaled")
