import logging
import numpy as np
import librosa
import torch
import torchaudio

from typing import List
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from features.sound_dataset import SoundDataset
from features.spectrogram_dataset import torch_windows
from features.slice_frequency_dataclass import SliceFrequency
from utils.utils import adjust_matrix, adjust_linear_ndarray, closest_power_2

log = logging.getLogger("smartula")


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
        sound_samples, sampling_rate = torchaudio.load(self.filenames[idx])
        if sound_samples.shape[0] > 1:
            if torch.count_nonzero(sound_samples[1]).item() > (len(sound_samples[0]) // 2):
                sound_samples = torch.mean(sound_samples, dim=0, keepdim=True)
            else:
                sound_samples = sound_samples[0]  # get only first channel

        label = self.labels[idx]

        mel_sound_samples = self.transform(sound_samples)
        if self.convert_db:
            mel_sound_samples = torchaudio.transforms.AmplitudeToDB()(mel_sound_samples)
        mel_sound_samples = mel_sound_samples.squeeze()

        initial_shape = mel_sound_samples.shape
        mel_sound_samples = MinMaxScaler().fit_transform(mel_sound_samples.reshape(-1, 1)).reshape(initial_shape)

        frequencies = librosa.core.mel_frequencies(fmin=self.f_min, fmax=self.f_max, n_mels=self.n_mels)
        times = np.linspace(0, len(sound_samples) / sampling_rate, mel_sound_samples.shape[1])

        if self.round_data_shape:
            if self.convert_db:
                # for now we only extend linearly spaced values
                log.warning('ONLY LINEAR MELSPECTROGRAM COULD BE EXTENDED!')
            else:
                mel_sound_samples = adjust_matrix(mel_sound_samples, 2 ** closest_power_2(mel_sound_samples.shape[0]),
                                                  2 ** closest_power_2(mel_sound_samples.shape[1]),
                                                  fill_with=mel_sound_samples.min())

                frequencies = adjust_linear_ndarray(frequencies, 2 ** closest_power_2(frequencies.shape[0]),
                                                    policy='sequence')
                times = adjust_linear_ndarray(times, 2 ** closest_power_2(times.shape[0]), policy='sequence')

        return (mel_sound_samples, frequencies, times), label

    def __getitem__(self, idx):
        """ Wrapper for getting item from melpectrogram dataset """
        (data, _, _), labels = self.get_item(idx)
        data = data.astype(np.float32)
        return data[None, :], labels
