import numpy as np

from pathlib import Path
from datetime import datetime
from scipy.io import wavfile
from abc import ABC, abstractmethod

from torch.utils.data import Dataset


def _pcm2float(sig: np.ndarray, dynamic_type: str = 'float64') -> np.ndarray:
    """
    Converting pcm signal to -1/1 float signal
    :param sig: signal to be converter
    :param dynamic_type: string representing dynamic type
    :return: converted samples
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    d_type = np.dtype(dynamic_type)
    if d_type.kind != 'f':
        raise TypeError("'d_type' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(d_type) - offset) / abs_max


def read_samples(filename: Path, raw: bool = False) -> (np.ndarray, int):
    """
    Function for reading sound samples from wav file
    :param filename: wav file to be read
    :param raw: flag for reading only raw samples - skipping float conversion
    :return: list of samples, sampling rate
    """
    sampling_rate, sound_samples = wavfile.read(str(filename))
    if len(sound_samples.shape) > 1:
        sound_samples = sound_samples.sum(axis=1) // 2

    if not raw and sound_samples.dtype.kind != 'f':
        sound_samples = _pcm2float(sound_samples, dynamic_type='float32')

    return sound_samples, sampling_rate


class SoundDataset(ABC, Dataset):
    def __init__(self, filenames, labels):
        assert len(filenames) == len(labels), 'Filenames do not match labels length!'

        self.filenames = filenames
        self.labels = labels

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.filenames)

    def read_sound(self, idx: int, raw: bool = False) -> (np.ndarray, int):
        """
        Method for reading sound based on index
        :param idx: index of sound which should be read
        :param raw: flag for reading only raw samples - skipping float conversion
        :return: list of samples, sampling rate, label
        """
        filename = self.filenames[idx]
        label = self.labels[idx]
        sound_samples, sampling_rate = read_samples(filename, raw)
        return sound_samples, sampling_rate, label

    def hour_for_fileid(self, idx: int) -> int:
        """ Wrapper for reading filename """
        return self.datetime_for_fileid(idx).hour

    def hivename_for_fileid(self, idx: int) -> str:
        """ Method for extracting hivename based on file id """
        filename = self.filenames[idx]
        return filename.stem.split('-')[0]

    def datetime_for_fileid(self, idx: int) -> datetime:
        """ Wrapper for reading datetime based on fileid """
        filename = self.filenames[idx]
        return datetime.strptime('-'.join(filename.stem.split('.')[0].split('-')[1:]), '%Y-%m-%dT%H-%M-%S')
