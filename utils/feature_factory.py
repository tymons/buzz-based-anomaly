from enum import Enum
from torch.utils.data import DataLoader, Dataset, random_split

from utils.features.sound_dataset import SoundDataset
from utils.features.periodogram_dataset import PeriodogramDataset
from typing import List, Callable
from pathlib import Path


class SoundFeatureType(Enum):
    PERIODOGRAM = 'periodogram'

    @classmethod
    def from_name(cls, name):
        for _, feature in SoundFeatureType.__members__.items():
            if feature.value == name:
                return feature
        raise ValueError(f"{name} is not a valid feature name")


class SoundFeatureFactory:
    """ Factory for data loaders """

    @staticmethod
    def _get_periodogram_dataset(sound_filenames: List[Path], labels: List[str],
                                 features_params_dict: dict) -> SoundDataset:
        """
        Function for building periodogram dataset
        :param sound_filenames:
        :param labels:
        :param features_params_dict:
        :return:
        """
        frequencies = features_params_dict.get('slice_frequency', {'start': 0, 'stop': 2048})
        convert_db = features_params_dict.get('convert_db', False)
        normalize = features_params_dict.get('normalize', True)

        print(f'building periodogram dataset with params: db_scale({convert_db}), min_max_scale({normalize}),'
              f' slice_freq({(frequencies.get("start"), frequencies.get("stop"))})')

        return PeriodogramDataset(sound_filenames, labels, convert_db, normalize,
                                  slice_freq=(frequencies.get('start'), frequencies.get('stop')))

    @staticmethod
    def build_dataset(input_type: SoundFeatureType, sound_filenames: List[Path], labels: List[str],
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
        function: Callable[[List[Path], List[str], dict], SoundDataset] \
            = getattr(SoundFeatureFactory, method_name, lambda: 'invalid dataset')
        dataset: SoundDataset = function(sound_filenames, labels, features_params_dict)
        feature_params_dict = {f"FEATURE_{key}": val for key, val in dataset.get_params().items()}

        return dataset, feature_params_dict

    @staticmethod
    def build_train_and_validation_dataloader(dataset, batch_size, ratio=0.15, num_workers=0):
        val_amount = int(dataset.__len__() * ratio)
        train_set, val_set = random_split(dataset, [(dataset.__len__() - val_amount), val_amount])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

        return train_loader, val_loader
