import torch
import numpy as np
import logging

from typing import List
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from scipy import signal

from features.sound_dataset import SoundDataset
from features.slice_frequency_dataclass import SliceFrequency
from utils.utils import adjust_matrix, closest_power_2, adjust_linear_ndarray

log = logging.getLogger("smartula")


torch_windows = {
    'hann': torch.hann_window,
    'hamming': torch.hamming_window,
    'blackman': torch.blackman_window,
    'bartlett': torch.bartlett_window,
    'none': None,
}


def calculate_spectrogram(samples, sampling_rate: int, n_fft: int, hop_len: int, slice_freq: SliceFrequency = None,
                          convert_db=False, window_name: str = 'blackman') -> (np.ndarray, np.ndarray, np.ndarray):
    """ Function for calculating spectrogram
    :param window_name: window name, see scipy.signal.get_window function
    :param samples: audio samples from which spectrogram should be calculated
    :param sampling_rate: sampling rate for audio
    :param n_fft: samples for fft calculation
    :param hop_len: samples for hop (next fft calculation)
    :param slice_freq: min/max frequency for calculated spectrogram to be constrained
    :param convert_db: should magnitude be converted to db
    :return: spectrogram_magnitude: spectrogram
    """
    frequencies, times, spectrogram = signal.spectrogram(samples, sampling_rate, nperseg=n_fft,
                                                         window=window_name, noverlap=hop_len)
    spectrogram_magnitude = np.abs(spectrogram)
    if convert_db:
        spectrogram_magnitude = 20 * np.log10(spectrogram_magnitude)

    if slice_freq:
        freq_slice = np.where(np.logical_and((slice_freq.start < frequencies), (frequencies < slice_freq.stop)))
        frequencies = frequencies[freq_slice]
        spectrogram_magnitude = spectrogram_magnitude[freq_slice]

    initial_shape = spectrogram_magnitude.shape
    spectrogram_magnitude = MinMaxScaler().fit_transform(spectrogram_magnitude.reshape(-1, 1)).reshape(initial_shape)

    spectrogram_magnitude = spectrogram_magnitude.astype(float)
    return spectrogram_magnitude, frequencies, times


class SpectrogramDataset(SoundDataset):
    normalize: bool
    convert_db: bool
    hop_len: int
    n_fft: int
    slice_freq: SliceFrequency

    def __init__(self, filenames: List[Path], labels: List[int], n_fft: int, hop_len: int,
                 convert_db: bool = False, slice_freq: SliceFrequency = None,
                 round_data_shape: bool = True, window: str = 'hann'):
        SoundDataset.__init__(self, filenames, labels)
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.slice_freq = slice_freq
        self.convert_db = convert_db
        self.round_data_shape = round_data_shape
        self.window = window

    def get_params(self) -> dict:
        """
        Method for getting params of dataset
        :return: dictionary with params
        """
        params = vars(self)
        params.pop('filenames')
        params.pop('labels')
        return params

    def get_item(self, idx) -> tuple:
        """ Function for getting item from Spectrogram dataset
        :param:idx: element idx
        :return: ((spectrogram, frequencies, times), label) (tuple)
        """
        sound_samples, sampling_rate, label = SoundDataset.read_sound(self, idx)
        spectrogram, frequencies, times = calculate_spectrogram(sound_samples, sampling_rate, self.n_fft,
                                                                self.hop_len, self.slice_freq,
                                                                self.convert_db, self.window)

        if self.round_data_shape:
            if self.convert_db:
                # for now we only extend linearly spaced values
                log.warning('ONLY LINEAR SPECTROGRAM COULD BE EXTENDED!')
            else:
                spectrogram = adjust_matrix(spectrogram, 2 ** closest_power_2(spectrogram.shape[0]),
                                            2 ** closest_power_2(spectrogram.shape[1]), fill_with=spectrogram.min())

                frequencies = adjust_linear_ndarray(frequencies, 2 ** closest_power_2(frequencies.shape[0]),
                                                    policy='sequence')
                times = adjust_linear_ndarray(times, 2 ** closest_power_2(times.shape[0]), policy='sequence')

        return (spectrogram, frequencies, times), label

    def __getitem__(self, idx):
        """ Wrapper for getting item from Spectrogram dataset """
        (data, _, _), labels = self.get_item(idx)
        data = data.astype(np.float32)
        return data[None, :], labels
