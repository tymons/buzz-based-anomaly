import argparse
import yaml
import optuna

from pathlib import Path
from utils.model_runner import ModelRunner

from models.model_type import HiveModelType
from features.feature_type import SoundFeatureType

from utils.model_factory import HiveModelFactory
from utils.feature_factory import SoundFeatureFactory
from utils.data_utils import get_valid_sounds_from_folders


def main():
    parser = argparse.ArgumentParser(description='Process ML model training.')
    # positional arguments
    parser.add_argument('model', metavar='model', choices=list(HiveModelType),
                        type=HiveModelType.from_name, help="Hive Model Type [AE]")
    parser.add_argument('feature', metavar='feature', choices=list(SoundFeatureType),
                        type=SoundFeatureType.from_name, help='Input feature')
    parser.add_argument('root_folder', metavar='root_folder', type=Path, help='Root folder for data')
    # optional arguments
    parser.add_argument('--model_config', default=Path(__file__).absolute().parent / "model_config.yml", type=Path)
    parser.add_argument('--feature_config', default=Path(__file__).absolute().parent / "feature_config.yml", type=Path)
    parser.add_argument('--learning_config', default=Path(__file__).absolute().parent / "learning_config.yml",
                        type=Path)
    parser.add_argument('--model_output', default=Path(__file__).absolute().parent / "output/model", type=Path)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--comet_config', default=Path(__file__).absolute().parent / ".comet.config", type=Path)
    parser.add_argument('--search_best', dest='search_best', action='store_true',
                        help="should use optuna to tune architecture")

    parser.set_defaults(search_best=False)
    args = parser.parse_args()

    with args.feature_config.open('r') as fc, args.model_config.open('r') as mc, \
            args.learning_config.open('r') as lc:
        feature_config = yaml.load(fc, Loader=yaml.FullLoader)
        model_config = yaml.load(mc, Loader=yaml.FullLoader)
        learning_config = yaml.load(lc, Loader=yaml.FullLoader)

        # data
        sound_list = get_valid_sounds_from_folders(args.root_folder.glob('*'))
        available_labels = list(set([path.stem.split("-")[0] for path in sound_list]))
        sound_labels = [list(available_labels).index(sound_name.stem.split('-')[0]) for sound_name in sound_list]
        dataset = SoundFeatureFactory.build_dataset(args.feature, sound_list, sound_labels, feature_config)
        train_loader, val_loader = SoundFeatureFactory.build_train_and_validation_dataloader(dataset, args.batch_size)

        model = HiveModelFactory.build_model(args.model, model_config, dataset[0][0][0].squeeze().shape[0])
        model_runner = ModelRunner(train_loader, val_loader, args.model_output, feature_config=feature_config,
                                   comet_config_file=args.comet_config, comet_project_name="bee-sound-anomaly")
        model_runner.train(model, learning_config)


if __name__ == "__main__":
    main()
