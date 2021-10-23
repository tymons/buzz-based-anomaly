import librosa.feature
import numpy as np

from typing import List
from pathlib import Path

from features.sound_dataset import SoundDataset
from features.slice_frequency_dataclass import SliceFrequency
from utils.utils import adjust_linear_ndarray, adjust_matrix, closest_power_2

from sklearn.preprocessing import MinMaxScaler


class MfccDataset(SoundDataset):
    def __init__(self, filenames: List[Path], labels: List[int], n_fft: int, hop_len: int,
                 slice_freq: SliceFrequency = None, round_data_shape: bool = True, window: str = 'hann',
                 n_mfccs: int = 16, n_mels: int = 64):
        SoundDataset.__init__(self, filenames, labels)
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.slice_freq = slice_freq
        self.round_data_shape = round_data_shape
        self.window = window
        self.n_mfccs = n_mfccs
        self.n_mels = n_mels

    def get_params(self):
        """
        Method for getting params of dataset
        :return: dictionary with params
        """
        params = vars(self)
        params.pop('filenames')
        params.pop('labels')
        return params

    def get_item(self, idx) -> tuple:
        """ Function for getting item from melspectrogram dataset
        :param:idx: element idx
        :return: ((melspectrogram, frequencies, times), label) (tuple)
        """
        sound_samples, sampling_rate, label = SoundDataset.read_sound(self, idx)
        f_max = min(self.slice_freq.stop, sampling_rate // 2)
        f_min = min(self.slice_freq.start, sampling_rate // 2)

        mfccs = librosa.feature.mfcc(sound_samples, sampling_rate, n_fft=self.n_fft,
                                     hop_length=self.hop_len, n_mfcc=self.n_mfccs, n_mels=self.n_mels,
                                     fmax=f_max, fmin=f_min, window=self.window)

        times = np.linspace(0, len(sound_samples) / sampling_rate, mfccs.shape[1])
        mfccs_coefs = np.linspace(0, self.n_mfccs, mfccs.shape[0])

        initial_shape = mfccs.shape
        mfccs = MinMaxScaler().fit_transform(mfccs.reshape(-1, 1)).reshape(initial_shape)

        if self.round_data_shape:
            mfccs = adjust_matrix(mfccs, 2 ** closest_power_2(mfccs.shape[0]),
                                  2 ** closest_power_2(mfccs.shape[1]), fill_with=mfccs.min())

            mfccs_coefs = adjust_linear_ndarray(mfccs_coefs, 2 ** closest_power_2(mfccs_coefs.shape[0]),
                                                policy='sequence')
            times = adjust_linear_ndarray(times, 2 ** closest_power_2(times.shape[0]), policy='sequence')

        return (mfccs, mfccs_coefs, times), label

    def __getitem__(self, idx):
        """ Wrapper for getting item from mfccs dataset """
        (data, _, _), labels = self.get_item(idx)
        data = data.astype(np.float32)
        return data[None, :], labels
