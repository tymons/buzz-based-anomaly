import numpy as np
import torchaudio
import logging
import torch

from typing import List
from pathlib import Path

from features.sound_dataset import SoundDataset
from features.slice_frequency_dataclass import SliceFrequency
from features.spectrogram_dataset import torch_windows
from utils.utils import adjust_linear_ndarray, adjust_matrix, closest_power_2

from sklearn.preprocessing import MinMaxScaler

log = logging.getLogger("smartula")


class MfccDataset(SoundDataset):
    def __init__(self, filenames: List[Path], labels: List[int], sampling_rate: int, n_fft: int, hop_len: int,
                 slice_freq: SliceFrequency = None, round_data_shape: bool = True, window: str = 'hann',
                 n_mfccs: int = 16, n_mels: int = 64, log_mel=False):
        SoundDataset.__init__(self, filenames, labels)
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.slice_freq = slice_freq
        self.window = window
        self.n_mfccs = n_mfccs
        self.round_data_shape = round_data_shape
        self.n_mels = n_mels

        self.f_max = min(self.slice_freq.stop, sampling_rate // 2)
        self.f_min = min(self.slice_freq.start, sampling_rate // 2)
        self.transform = torchaudio.transforms.MFCC(
            sample_rate=sampling_rate,
            n_mfcc=n_mfccs,
            log_mels=log_mel,
            melkwargs={
                "n_fft": self.n_fft,
                "hop_length": self.hop_len,
                "n_mels": self.n_mels,
                "window_fn": torch_windows.get(self.window, None),
                "f_min": self.f_min,
                "f_max": self.f_max
            }
        )

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
        sound_samples, sampling_rate = torchaudio.load(self.filenames[idx])
        if sound_samples.shape[0] > 1:
            if torch.count_nonzero(sound_samples[1]).item() > (len(sound_samples[0]) // 2):
                sound_samples = torch.mean(sound_samples, dim=0, keepdim=True)
            else:
                sound_samples = sound_samples[0]  # get only first channel

        label = self.labels[idx]

        mfccs = self.transform(sound_samples)
        mfccs = mfccs.squeeze()

        initial_shape = mfccs.shape
        mfccs = MinMaxScaler().fit_transform(mfccs.reshape(-1, 1)).reshape(initial_shape)

        coefs_nums = list(range(0, mfccs.shape[0]))
        times = np.linspace(0, len(sound_samples) / sampling_rate, mfccs.shape[1])

        if self.round_data_shape:
            mfccs = adjust_matrix(mfccs, 2 ** closest_power_2(mfccs.shape[0]), 2 ** closest_power_2(mfccs.shape[1]),
                                  fill_with=mfccs.min())

            coefs_nums = adjust_linear_ndarray(mfccs, 2 ** closest_power_2(mfccs.shape[0]), policy='sequence')
            times = adjust_linear_ndarray(times, 2 ** closest_power_2(times.shape[0]), policy='sequence')

        return (mfccs, coefs_nums, times), label

    def __getitem__(self, idx):
        """ Wrapper for getting item from mfccs dataset """
        (data, _, _), labels = self.get_item(idx)
        data = data.astype(np.float32)
        return data[None, :], labels
