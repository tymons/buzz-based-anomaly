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
from utils.gmm_anomaly_scorer import GMMAnomalyScorer
from utils.model_runner import model_load
from utils.feature_factory import SoundFeatureFactory
from utils.model_factory import HiveModelFactory, HiveModelType
from utils.model_runner import ModelRunner
from utils.anomaly_scorer_type import AnomalyScorerType


def main():
    parser = argparse.ArgumentParser(description='Inference and anomaly score for ML.')
    # positional arguments
    parser.add_argument('model_path', default=Path(__file__).absolute().parent / "model.pth", type=Path,
                        help='model path')
    parser.add_argument('feature', metavar='feature', choices=list(SoundFeatureType), type=SoundFeatureType.from_name,
                        help='input feature type')
    parser.add_argument('target_data_folder', metavar='data folder', type=Path,
                        help='root data folder for hive data')
    parser.add_argument('anomaly_data_folder', metavar='anomaly folder', type=Path,
                        help='anomaly data folder for hive data')
    # optional arguments
    parser.add_argument('--feature_config', default=Path(__file__).absolute().parent / "feature_config.yml", type=Path)
    parser.add_argument('--model_config', default=Path(__file__).absolute().parent / "model_config.yml", type=Path)
    parser.add_argument('--log_folder', default=Path(__file__).absolute().parent / "output/", type=Path)
    parser.add_argument('--anomaly-model', default=AnomalyScorerType.GMM, type=AnomalyScorerType.from_name)
    parser.add_argument('--filter_dates', nargs=2, type=datetime.fromisoformat,
                        help="Start and end date for sound data with format YYYY-MM-DD", metavar='START_DATE END_DATE')
    parser.add_argument('--no_save_inference_data', dest='save_data', action='store_false')
    parser.set_defaults(feature=True)

    args = parser.parse_args()

    utils.logger_setup(args.log_folder, f"inference-{args.model_path.stem.split('-')[0]}-{args.feature.value}")

    with args.feature_config.open('r') as fc, args.model_config.open('r') as mc:
        feature_config = yaml.load(fc, Loader=yaml.FullLoader)
        model_config = yaml.load(mc, Loader=yaml.FullLoader)

        smartula_hive_sound_list = utils.get_valid_sounds_from_folders([args.target_data_folder])
        anomaly_sound_list = list(args.anomaly_data_folder.glob('*'))
        if not all([smartula_hive_sound_list, anomaly_sound_list]):
            logging.error(f'one of smartula or anomaly sound list is empty! smartula:{len(smartula_hive_sound_list)},'
                          f' anomaly: {anomaly_sound_list}')
            raise Exception('sound list empty!')

        if args.filter_dates:
            smartula_hive_sound_list = utils.filter_by_datetime(smartula_hive_sound_list, args.filter_dates[0],
                                                                args.filter_dates[1])

        target_labels: List[int] = [1] * len(smartula_hive_sound_list)
        anomaly_labels: List[int] = [0] * len(anomaly_sound_list)

        hive_dataset = SoundFeatureFactory.build_dataset(args.feature, smartula_hive_sound_list, target_labels,
                                                         feature_config)
        anomaly_dataset = SoundFeatureFactory.build_dataset(args.feature, anomaly_sound_list, anomaly_labels,
                                                            feature_config)
        hive_data_shape = hive_dataset[0][0].squeeze().shape
        anomaly_data_shape = hive_dataset[0][0].squeeze().shape

        assert hive_data_shape == anomaly_data_shape, "anomaly and hive data are not consistent!"

        logging.debug(f'got dataset of shape: {hive_data_shape}')
        hive_dataloader = DataLoader(hive_dataset, batch_size=64, shuffle=True, drop_last=False, num_workers=0)
        anomaly_dataloader = DataLoader(anomaly_dataset, batch_size=64, shuffle=True, drop_last=False, num_workers=0)

        model_type: HiveModelType = HiveModelType.from_name(args.model_path.stem.split('-')[0])
        model = HiveModelFactory.build_model(model_type, hive_data_shape, model_config['model'])
        model, last_epoch, last_loss = model_load(args.model_path, model)
        logging.info(f'model {model_type.model_name} has been loaded from epoch {last_epoch} with loss: {last_loss}')

        model_runner = ModelRunner(comet_api_key='DEADBEEF')
        hive_latent = model_runner.inference_latent(model, hive_dataloader)
        anomaly_latent = model_runner.inference_latent(model, anomaly_dataloader)

        if args.save_data:
            output_data_folder = Path('output/data/')
            output_data_folder.mkdir(parents=True, exist_ok=True)
            target_data_file = output_data_folder / Path(f'{model_type.model_name}') / Path(
                f'{args.target_data_folder.stem}-{"-".join(args.model_path.stem.split("-")[:3])}-{args.feature.value}'
                f'-target_data.npy')
            anomaly_data_file = output_data_folder / Path(f'{model_type.model_name}') / Path(
                f'{args.anomaly_data_folder.stem}-{"-".join(args.model_path.stem.split("-")[:3])}-{args.feature.value}'
                f'-anomaly_data.npy')
            np.save(str(target_data_file), hive_latent.detach().numpy())
            np.save(str(anomaly_data_file), anomaly_latent.detach().numpy())

        anomaly = GMMAnomalyScorer()
        score = anomaly.fit(hive_latent, anomaly_latent).score()
        print(f'score: {score}')


if __name__ == "__main__":
    main()
