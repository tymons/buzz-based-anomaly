import os
import argparse
import logging

import yaml

from pathlib import Path
from utils.model_runner import ModelRunner

from models.model_type import HiveModelType
from features.feature_type import SoundFeatureType

from utils.model_factory import HiveModelFactory
from utils.feature_factory import SoundFeatureFactory
from utils.data_utils import get_valid_sounds_from_folders, filter_string_list, filter_by_datetime
from datetime import datetime


def logger_setup(log_folder: Path, filename_prefix: str) -> None:
    """
    Method for setting up python logger
    :param log_folder: folder where logs will be saved
    :param filename_prefix: log file filename prefix
    """
    log_level = os.environ.get('LOGLEVEL', 'DEBUG').upper()
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_folder / f"{filename_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                                             f"-{log_level}.log"),
            logging.StreamHandler()
        ]
    )


def main():
    parser = argparse.ArgumentParser(description='Process ML model training.')
    # positional arguments
    parser.add_argument('model', metavar='model', choices=list(HiveModelType),
                        type=HiveModelType.from_name, help="Hive Model Type [AE]")
    parser.add_argument('feature', metavar='feature', choices=list(SoundFeatureType),
                        type=SoundFeatureType.from_name, help='Input feature')
    parser.add_argument('root_folder', metavar='root_folder', type=Path, help='Root folder for data')
    # optional arguments
    parser.add_argument('--filter_hives', default=[], nargs='+', help="Hive names to be excluded from dataset")
    parser.add_argument('--filter_dates', nargs=2, type=datetime.fromisoformat,
                        help="Start and end date for sound data with format YYYY-MM-DD", metavar='START_DATE END_DATE')
    parser.add_argument('--model_config', default=Path(__file__).absolute().parent / "model_config.yml", type=Path)
    parser.add_argument('--feature_config', default=Path(__file__).absolute().parent / "feature_config.yml", type=Path)
    parser.add_argument('--learning_config', default=Path(__file__).absolute().parent / "learning_config.yml",
                        type=Path)
    parser.add_argument('--model_output', default=Path(__file__).absolute().parent / "output/model", type=Path)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--comet_config', default=Path(__file__).absolute().parent / ".comet.config", type=Path)
    parser.add_argument('--search_best', type=int, metavar='N', help="how many trials for searching best architecture")
    parser.add_argument('--log_folder', default=Path(__file__).absolute().parent / "output/", type=Path)

    args = parser.parse_args()

    logger_setup(args.log_folder, f"{args.model.value}-{args.feature.value}")

    with args.feature_config.open('r') as fc, args.model_config.open('r') as mc, \
            args.learning_config.open('r') as lc:
        feature_config = yaml.load(fc, Loader=yaml.FullLoader)
        model_config = yaml.load(mc, Loader=yaml.FullLoader)
        learning_config = yaml.load(lc, Loader=yaml.FullLoader)

        # data
        sound_list = get_valid_sounds_from_folders(args.root_folder.glob('*'))
        if args.filter_dates:
            sound_list = filter_by_datetime(sound_list, args.filter_dates[0], args.filter_dates[1])
        sound_list = filter_string_list(sound_list, *args.filter_hives)
        available_labels = list(set([path.stem.split("-")[0] for path in sound_list]))
        sound_labels = [list(available_labels).index(sound_name.stem.split('-')[0]) for sound_name in sound_list]
        dataset = SoundFeatureFactory.build_dataset(args.feature, sound_list, sound_labels, feature_config)
        data_shape = dataset[0][0][0].squeeze().shape[0]
        train_loader, val_loader = SoundFeatureFactory.build_train_and_validation_dataloader(dataset, args.batch_size)

        model_runner = ModelRunner(train_loader, val_loader, args.model_output, feature_config=feature_config,
                                   comet_config_file=args.comet_config, comet_project_name="bee-sound-anomaly")
        if args.search_best is not None:
            model_runner.find_best(args.model, data_shape, learning_config, n_trials=args.search_best,
                                   output_folder=Path('output/model'))
        else:
            model = HiveModelFactory.build_model_and_check(args.model, model_config, data_shape)
            model = model_runner.train(model, learning_config)


if __name__ == "__main__":
    main()
