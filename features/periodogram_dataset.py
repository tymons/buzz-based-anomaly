import numpy as np

from sklearn.preprocessing import MinMaxScaler

from features.sound_dataset import SoundDataset


class PeriodogramDataset(SoundDataset):
    """ Periodogram dataset """

    def __init__(self, filenames, labels, convert_db=False, normalize=False, slice_freq=None):
        SoundDataset.__init__(self, filenames, labels)
        self.slice_freq = slice_freq
        self.normalize = normalize  # should scale data to be within 0 and 1
        self.convert_db = convert_db  # should scale data to log amplitude

    def get_params(self):
        """ Method for returning feature params """
        params = dict(self.__dict__)
        params.pop('filenames')
        params.pop('labels')
        return params

    def get_item(self, idx):
        """ Function for getting periodogram """
        should_be_integers = self.convert_db
        sound_samples, sampling_rate, labels = SoundDataset.read_sound(self, idx=idx, raw=should_be_integers)
        periodogram = abs(np.fft.rfft(sound_samples, sampling_rate))[1:]
        if self.convert_db:
            periodogram = 20 * np.log10(periodogram / np.iinfo(sound_samples[0]).max)

        frequencies = np.fft.rfftfreq(sampling_rate, d=(1. / sampling_rate))[1:]

        if self.slice_freq:
            periodogram = periodogram[self.slice_freq[0]:self.slice_freq[1]]
            frequencies = frequencies[self.slice_freq[0]:self.slice_freq[1]]

        if self.normalize:
            periodogram = MinMaxScaler().fit_transform(periodogram.reshape(-1, 1)).squeeze()

        periodogram = periodogram.astype(np.float32)
        return (periodogram, frequencies), labels

    def __getitem__(self, idx):
        """ Method for pytorch dataloader """
        (periodogram, _), labels = self.get_item(idx)
        return periodogram[None, :], labels
