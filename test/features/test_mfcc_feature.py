import unittest
from pathlib import Path

from utils.feature_factory import SoundFeatureFactory
from features.feature_type import SoundFeatureType


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
        'window': 'hann',
        'round_power_2': True,
        'nmfccs': 32,
        'nmels': 128,
        'log_mel': False
    }


class TestPeriodogramMethods(unittest.TestCase):
    def setUp(self):
        self.filename_10kHz = Path(Path(__file__).parent.resolve(), '../data/10kHz.wav')
        self.filename_1kHz = Path(Path(__file__).parent.resolve(), '../data/1kHz.wav')

    def test_10kHz_mfcc_scale(self):
        config = _get_default_config()

        dataset = SoundFeatureFactory.build_dataset(SoundFeatureType.from_name('mfcc'),
                                                    [self.filename_10kHz], [1], config)

        (mfccs, mels_numbers, times), _ = dataset.get_item(0)

        self.assertGreaterEqual(mfccs.min(), 0.0, "mfccs should be scaled")
        self.assertLessEqual(mfccs.max(), 1.0, "mfccs should be scaled")
