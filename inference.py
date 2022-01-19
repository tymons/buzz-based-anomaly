import numpy as np
import yaml
import argparse
import logging

import utils.utils as utils

from pathlib import Path
from typing import List
from torch.utils.data import DataLoader
from datetime import datetime

from features.feature_type import SoundFeatureType
from utils.model_runner import model_load
from utils.feature_factory import SoundFeatureFactory
from utils.model_factory import HiveModelFactory, HiveModelType
from utils.model_runner import ModelRunner


def main():
    parser = argparse.ArgumentParser(description='Inference and anomaly score for ML.')
    # positional arguments
    parser.add_argument('model_path', default=Path(__file__).absolute().parent / "model.pth", type=Path,
                        help='model path')
    parser.add_argument('feature', metavar='feature', choices=list(SoundFeatureType), type=SoundFeatureType.from_name,
                        help='input feature type')
    parser.add_argument('data_folder', metavar='data folder', type=Path,
                        help='root data folder for hive data')
    # optional arguments
    parser.add_argument('--feature_config', default=Path(__file__).absolute().parent / "feature_config.yml", type=Path)
    parser.add_argument('--model_config', default=Path(__file__).absolute().parent / "model_config.yml", type=Path)
    parser.add_argument('--log_folder', default=Path(__file__).absolute().parent / "output/", type=Path)
    parser.add_argument('--filter_dates', nargs=2, type=datetime.fromisoformat,
                        help="Start and end date for sound data with format YYYY-MM-DD", metavar='START_DATE END_DATE')
    parser.add_argument('--no_save_inference_data', dest='save_data', action='store_false')
    parser.add_argument('--output_folder', metavar='output folder', type=Path, help='output data folder for latent '
                                                                                    'hive data')
    parser.set_defaults(feature=True)

    args = parser.parse_args()

    utils.logger_setup(args.log_folder, f"inference-{args.model_path.stem.split('-')[0]}-{args.feature.value}")

    with args.feature_config.open('r') as fc, args.model_config.open('r') as mc:
        feature_config = yaml.load(fc, Loader=yaml.FullLoader)
        model_config = yaml.load(mc, Loader=yaml.FullLoader)

        if args.data_folder.parent.stem.endswith('smartula'):
            sound_list = utils.get_valid_sounds_from_folders([args.data_folder])
            if not sound_list:
                logging.error(f'sound list is empty for folder :{args.data_folder},')
                raise Exception('sound list empty!')
            sound_list = utils.filter_by_datetime(sound_list, args.filter_dates[0], args.filter_dates[1])
        else:
            sound_list = list(args.data_folder.glob('*.wav'))

        labels: List[int] = [1] * len(sound_list)

        hive_dataset = SoundFeatureFactory.build_dataset(args.feature, sound_list, labels,
                                                         feature_config)
        if len(hive_dataset) <= 0:
            raise ValueError('Sound dataset is empty')

        hive_data_shape = hive_dataset[0][0].squeeze().shape

        logging.debug(f'got dataset of shape: {hive_data_shape}')
        dataloader = DataLoader(hive_dataset, batch_size=32, shuffle=True, drop_last=False, num_workers=0)

        model_type: HiveModelType = HiveModelType.from_name(args.model_path.stem.split('-')[0])
        model = HiveModelFactory.build_model(model_type, hive_data_shape, model_config['model'])
        model, last_epoch = model_load(args.model_path, model)
        logging.info(f'model {model_type.model_name} has been loaded from epoch {last_epoch}')

        model_runner = ModelRunner(comet_api_key='DEADBEEF')
        latent = model_runner.inference_latent(model, dataloader)

        if args.save_data:
            output_data_folder = Path(args.output_folder)
            output_data_folder.mkdir(parents=True, exist_ok=True)
            data_file = output_data_folder / Path(
                f'{args.data_folder.stem}-{"-".join(args.model_path.stem.split("-")[:3])}-{args.feature.value}.npy')
            np.save(str(data_file), latent.detach().numpy())


if __name__ == "__main__":
    main()
