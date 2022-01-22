import argparse
import logging

import yaml
import random
from utils.model_runner import ModelRunner
from typing import List
from pathlib import Path

from models.model_type import HiveModelType
from features.feature_type import SoundFeatureType

from utils.model_factory import HiveModelFactory
from utils.feature_factory import SoundFeatureFactory
from datetime import datetime

import utils.utils as utils

log = logging.getLogger("smartula")
log.setLevel(logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(description='Process ML model training.')
    # positional arguments
    parser.add_argument('model', metavar='model', choices=list(HiveModelType),
                        type=HiveModelType.from_name, help="Hive Model Type [AE]")
    parser.add_argument('feature', metavar='feature', choices=list(SoundFeatureType),
                        type=SoundFeatureType.from_name, help='Input feature')
    parser.add_argument('smartula_data_folder', metavar='data_folder', type=Path, help='Smartula folder for sound data')
    # optional arguments
    parser.add_argument('--hives', default=[], nargs='+', help="Hive names to be included in main dataset.")
    parser.add_argument('--filter_dates', nargs=2, type=datetime.fromisoformat,
                        help="Start and end date for sound data with format YYYY-MM-DD", metavar='START_DATE END_DATE')
    parser.add_argument('--model_config', default=Path(__file__).absolute().parent / "model_config.yml", type=Path)
    parser.add_argument('--feature_config', default=Path(__file__).absolute().parent / "feature_config.yml", type=Path)
    parser.add_argument('--learning_config', default=Path(__file__).absolute().parent / "learning_config.yml",
                        type=Path)
    parser.add_argument('--output_folder', default=Path(__file__).absolute().parent / "output/", type=Path)
    parser.add_argument('--comet_config', default=Path(__file__).absolute().parent / ".comet.config", type=Path)
    parser.add_argument('--find_best', type=int, metavar='N', help="how many trials for finding best architecture")
    parser.add_argument('--log_folder', default=Path(__file__).absolute().parent / "output/", type=Path)
    parser.add_argument('--fingerprint_main_hive', type=str,
                        help="hive name for which fingerprint should be calculated")
    parser.add_argument('--fingerprint_feature_file', default=Path(__file__).absolute().parent / "feature.csv",
                        type=Path)
    parser.add_argument('--contrastive_data_folders', nargs='+', type=Path)
    parser.add_argument('--use_contrastive_valid_file', dest='use_contrastive_valid_file', action='store_true',
                        help='Flag for using valid_sound.txt file (applicable only for smartula folders)')
    parser.add_argument('--gpu_ids', type=int, nargs='+', help="ids for gpus which should be used")

    parser.set_defaults(feature=False)
    args = parser.parse_args()

    utils.logger_setup(args.log_folder, f"{args.model.model_name}-{args.feature.value}")

    log.info(f'running {args.model.model_name} model with {args.feature.value}...')
    log.info(f'data folder located at: {args.smartula_data_folder}')
    log.info(f'output folder located at: {args.output_folder}')
    log.info(f'output folder for logs located at: {args.log_folder}')

    # TODO: Check features and model type compatibility

    with args.feature_config.open('r') as fc, args.model_config.open('r') as mc, \
            args.learning_config.open('r') as lc:
        feature_config = yaml.load(fc, Loader=yaml.FullLoader)
        model_config = yaml.load(mc, Loader=yaml.FullLoader)
        learning_config = yaml.load(lc, Loader=yaml.FullLoader)

        # data
        sound_list = utils.get_valid_sounds_from_folders(args.smartula_data_folder.glob('*'))
        if not sound_list:
            log.error('sound list empty!')
            raise Exception('sound list empty!')

        sound_list = utils.filter_path_list(sound_list, *args.hives)
        if args.filter_dates:
            sound_list = utils.filter_by_datetime(sound_list, args.filter_dates[0], args.filter_dates[1])

        if args.fingerprint_main_hive is not None:
            sound_list = utils.filter_hive_fingerprint(args.fingerprint_feature_file,
                                                       args.fingerprint_main_hive,
                                                       sound_list)

        available_labels = list(set([path.stem.split("-")[0] for path in sound_list]))
        sound_labels: List[int] = [list(available_labels).index(sound_name.stem.split('-')[0])
                                   for sound_name in sound_list]

        # preparse background filenames if needed
        if args.contrastive_data_folders is not None:
            if not args.use_contrastive_valid_file:
                background_files = [file for folder in args.contrastive_data_folders
                                    for file in list(folder.glob('**/*.wav'))]
            else:
                background_files = utils.get_valid_sounds_from_folders(args.contrastive_data_folders)
                if args.filter_dates:
                    background_files = utils.filter_by_datetime(background_files,
                                                                args.filter_dates[0], args.filter_dates[1])

            random.shuffle(background_files)
            if len(sound_list) > len(background_files):
                log.info(f'truncating target dataset to the length of {len(background_files)}')
                sound_list = sound_list[:len(background_files)]
                sound_labels = sound_labels[:len(background_files)]
            else:
                log.info(f'truncating background dataset to the length of: {len(sound_list)}')
                background_files = background_files[:len(sound_list)]

        # build datasets and dataloader
        dataset = SoundFeatureFactory.build_dataset(args.feature, sound_list, sound_labels, feature_config)
        data_shape = dataset[0][0].squeeze().shape
        log.debug(f'got dataset of shape: {data_shape}')
        if args.contrastive_data_folders is not None:
            background_dataset = SoundFeatureFactory.build_dataset(args.feature, background_files,
                                                                   [0] * len(background_files),
                                                                   feature_config)
            log.debug(f'got background dataset for contrastive learning of shape:'
                      f' {background_dataset[0][0].squeeze().shape}')
            dataset = SoundFeatureFactory.build_contrastive_feature_dataset(dataset, background_dataset)
        train_loader, val_loader = SoundFeatureFactory.build_train_and_validation_dataloader(dataset,
                                                                                             learning_config.get(
                                                                                                 'batch_size', 32),
                                                                                             drop_last=True)

        model_runner = ModelRunner(args.output_folder, comet_config_file=args.comet_config,
                                   comet_project_name="bee-sound-anomaly", gpu_ids=args.gpu_ids)
        if args.find_best is not None:
            model_runner.find_best(args.model, train_loader, learning_config, n_trials=args.find_best,
                                   output_folder=args.output_folder, feature_config=feature_config)
        else:
            if args.model.num >= HiveModelType.CONTRASTIVE_AE.num:
                model = HiveModelFactory.build_model(args.model, data_shape, model_config['model'])
                discriminator = HiveModelFactory.get_discriminator(model_config['model']['latent'])

                model = model_runner.train_contrastive_with_discriminator(model, train_loader, learning_config,
                                                                          discriminator, val_loader, feature_config)
            else:
                model = HiveModelFactory.build_model_and_check(args.model, data_shape, model_config['model'])
                model = model_runner.train(model, train_loader, learning_config, val_loader, feature_config)


if __name__ == "__main__":
    main()
