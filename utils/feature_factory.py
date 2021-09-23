import logging

from torch.utils.data import DataLoader, Dataset, random_split

from features.feature_type import SoundFeatureType
from features.sound_dataset import SoundDataset
from features.periodogram_dataset import PeriodogramDataset
from features.spectrogram_dataset import SpectrogramDataset
from features.slice_frequency_dataclass import SliceFrequency

from typing import List, Callable
from pathlib import Path


class SoundFeatureFactory:
    """ Factory for data loaders """

    @staticmethod
    def _get_periodogram_dataset(sound_filenames: List[Path], labels: List[int],
                                 features_params_dict: dict) -> PeriodogramDataset:
        """
        Function for building periodogram dataset
        :param sound_filenames: list of full os paths with sounds
        :param labels: list of int labels
        :param features_params_dict: parameters for
        :return:
        """
        slice_freq = SliceFrequency(**features_params_dict.get('slice_frequency'))
        convert_db = features_params_dict.get('convert_db')

        logging.debug(f'building periodogram dataset of length {len(sound_filenames)}'
                      f' with params: db_scale({convert_db}),  slice_freq({slice_freq})')

        return PeriodogramDataset(sound_filenames, labels, convert_db, slice_freq=slice_freq)

    @staticmethod
    def _get_spectrogram_dataset(sound_filenames: List[Path], labels: List[int],
                                 features_params_dict: dict) -> SpectrogramDataset:
        """
        Function for building spectrogram dataset
        :param sound_filenames:
        :param labels:
        :param features_params_dict:
        :return:
        """
        slice_freq = SliceFrequency(**features_params_dict.get('slice_frequency'))
        n_fft: int = features_params_dict.get('nfft')
        hop_len: int = features_params_dict.get('hop_len')
        convert_db: bool = features_params_dict.get('convert_db')
        window: str = features_params_dict.get('window')
        data_round: bool = features_params_dict.get('round_power_2')

        return SpectrogramDataset(sound_filenames, labels, n_fft, hop_len, convert_db, slice_freq,
                                  data_round=data_round, window=window)

    @staticmethod
    def build_dataset(input_type: SoundFeatureType, sound_filenames: List[Path], labels: List[int],
                      features_params_dict: dict) -> (Dataset, dict):
        """
        Function for building dataset object based on given sound list
        :param input_type: type of dataset which should be build
        :param sound_filenames: list of filenames
        :param labels: labels for sound_filenames
        :param features_params_dict: feature parameters dictionary
        :return: SoundDataset, used feature params
        """
        method_name = f'_get_{input_type.value.lower()}_dataset'
        function: Callable[[List[Path], List[int], dict], SoundDataset] \
            = getattr(SoundFeatureFactory, method_name, lambda: 'invalid dataset')
        dataset: SoundDataset = function(sound_filenames, labels, features_params_dict)

        return dataset

    @staticmethod
    def build_train_and_validation_dataloader(dataset: SoundDataset, batch_size: int,
                                              ratio: float = 0.15, num_workers: int = 0, drop_last=False) -> \
            (DataLoader, DataLoader):
        """
        Building train and validation pytorch loader
        :param dataset: Data
        :param batch_size:
        :param ratio: validation/train ratio
        :param num_workers: pytorch number of dataloader workers
        :param drop_last: should drop last from dataset when batching
        :return:
        """
        val_amount = int(len(dataset) * ratio)
        train_dataset, val_dataset = random_split(dataset, [(len(dataset) - val_amount), val_amount])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last,
                                  num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last,
                                num_workers=num_workers)

        return train_loader, val_loader
