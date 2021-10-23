import yaml
import argparse
import logging

import utils.utils as utils

from pathlib import Path
from typing import List

from features.feature_type import SoundFeatureType
from utils.feature_factory import SoundFeatureFactory


def main():
    parser = argparse.ArgumentParser(description='Inference and anomaly score for ML.')
    # positional arguments
    parser.add_argument('model_path', default=Path(__file__).absolute().parent / "model.pth", type=Path,
                        help='model path')
    parser.add_argument('feature', metavar='feature', choices=list(SoundFeatureType), type=SoundFeatureType.from_name,
                        help='input feature type')
    parser.add_argument('smartula_hive_data_folder', metavar='data folder', type=Path,
                        help='root data folder for hive data')
    parser.add_argument('anomaly_data_folder', metavar='anomaly folder', type=Path,
                        help='anomaly data folder for hive data')
    # optional arguments
    parser.add_argument('--feature_config', default=Path(__file__).absolute().parent / "feature_config.yml", type=Path)
    parser.add_argument('--log_folder', default=Path(__file__).absolute().parent / "output/", type=Path)

    args = parser.parse_args()

    utils.logger_setup(args.log_folder, f"inference-{args.model_path.stem.split('-')[0]}-{args.feature.value}")

    with args.feature_config.open('r') as fc:
        feature_config = yaml.load(fc, Loader=yaml.FullLoader)

        smartula_hive_sound_list = utils.get_valid_sounds_from_folders([args.smartula_hive_data_folder])
        anomaly_sound_list = args.anomaly_data_folder.glob('*')
        if not all([smartula_hive_sound_list, anomaly_sound_list]):
            logging.error(f'one of smartula or anomaly sound list is empty! smartula:{len(smartula_hive_sound_list)},'
                          f' anomaly: {anomaly_sound_list}')
            raise Exception('sound list empty!')

        if args.filter_dates:
            smartula_hive_sound_list = utils.filter_by_datetime(smartula_hive_sound_list, args.filter_dates[0],
                                                                args.filter_dates[1])

        # prepare sound filenames
        target_labels: List[int] = [1] * len(smartula_hive_sound_list)
        anomaly_labels: List[int] = [0] * len(anomaly_sound_list)

        sound_list = smartula_hive_sound_list + anomaly_sound_list
        labels = target_labels + anomaly_labels

        # build datasets and dataloader
        # TODO: ensure same shape for target and anomaly data
        dataset = SoundFeatureFactory.build_dataset(args.feature, sound_list, labels, feature_config)
        data_shape = dataset[0][0].squeeze().shape
        logging.debug(f'got dataset of shape: {data_shape}')

        train_loader, val_loader = SoundFeatureFactory.build_train_and_validation_dataloader(dataset, 64)


if __name__ == "__main__":
    main()
