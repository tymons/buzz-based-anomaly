import logging
import numpy as np
import librosa
import torch
import torchaudio

from typing import List
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from features.sound_dataset import SoundDataset
from features.slice_frequency_dataclass import SliceFrequency
from utils.utils import adjust_matrix, adjust_linear_ndarray, closest_power_2

log = logging.getLogger("smartula")

torch_windows = {
    'hann': torch.hann_window,
    'hamming': torch.hamming_window,
    'blackman': torch.blackman_window,
    'bartlett': torch.bartlett_window,
    'none': None,
}


class MelSpectrogramDataset(SoundDataset):
    def __init__(self, filenames: List[Path], labels: List[int], sampling_rate: int, n_fft: int, hop_len: int,
                 convert_db: bool = False, slice_freq: SliceFrequency = None,
                 round_data_shape: bool = True, window: str = 'hann', n_mels: int = 32):
        SoundDataset.__init__(self, filenames, labels)
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.slice_freq = slice_freq
        self.convert_db = convert_db
        self.round_data_shape = round_data_shape
        self.window = window
        self.n_mels = n_mels

        self.f_max = min(self.slice_freq.stop, sampling_rate // 2)
        self.f_min = min(self.slice_freq.start, sampling_rate // 2)
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            n_mels=self.n_mels,
            window_fn=torch_windows.get(self.window, None),
            f_min=self.f_min,
            f_max=self.f_max
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
        sound_samples, sampling_rate, label = SoundDataset.read_sound(self, idx)
        f_max = min(self.slice_freq.stop, sampling_rate // 2)
        f_min = min(self.slice_freq.start, sampling_rate // 2)

        mel_spectrogram = librosa.feature.melspectrogram(sound_samples, sampling_rate, n_fft=self.n_fft,
                                                         hop_length=self.hop_len, n_mels=self.n_mels,
                                                         fmax=f_max, fmin=f_min, window=self.window)

        frequencies = librosa.core.mel_frequencies(fmin=f_min, fmax=f_max, n_mels=self.n_mels)
        times = np.linspace(0, len(sound_samples) / sampling_rate, mel_spectrogram.shape[1])

        if self.convert_db:
            mel_spectrogram = librosa.power_to_db(mel_spectrogram)

        initial_shape = mel_spectrogram.shape
        mel_spectrogram = MinMaxScaler().fit_transform(mel_spectrogram.reshape(-1, 1)).reshape(initial_shape)

        if self.round_data_shape:
            if self.convert_db:
                # for now we only extend linearly spaced values
                log.warning('ONLY LINEAR MELSPECTROGRAM COULD BE EXTENDED!')
            else:
                mel_spectrogram = adjust_matrix(mel_spectrogram, 2 ** closest_power_2(mel_spectrogram.shape[0]),
                                                2 ** closest_power_2(mel_spectrogram.shape[1]),
                                                fill_with=mel_spectrogram.min())

                frequencies = adjust_linear_ndarray(frequencies, 2 ** closest_power_2(frequencies.shape[0]),
                                                    policy='sequence')
                times = adjust_linear_ndarray(times, 2 ** closest_power_2(times.shape[0]), policy='sequence')

        return (mel_spectrogram, frequencies, times), label

    def __getitem__(self, idx):
        """ Wrapper for getting item from melpectrogram dataset """
        (data, _, _), labels = self.get_item(idx)
        data = data.astype(np.float32)
        return data[None, :], labels
