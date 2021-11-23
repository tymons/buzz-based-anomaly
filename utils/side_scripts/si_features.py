import argparse
import features.sound_indices.indices as indices
import pandas as pd
import pytz
import json

from pathlib import Path
from features.sound_indices.si_feature_type import SoundIndicesFeatureType
from features.sound_dataset import read_samples
from features.spectrogram_dataset import calculate_spectrogram
from features.slice_frequency_dataclass import SliceFrequency
from datetime import datetime
from utils import utils as ut
from multiprocessing.pool import ThreadPool
from tqdm import tqdm


def get_feature_func(feature_name: str):
    sound_feature = SoundIndicesFeatureType.from_name(feature_name)
    if sound_feature == SoundIndicesFeatureType.SPECTRAL_ENTROPY:
        return indices.compute_sh
    elif sound_feature == SoundIndicesFeatureType.ACI:
        return indices.compute_aci
    else:
        raise ValueError(f'{feature_name} not supported!')


parser = argparse.ArgumentParser(description='Convert sound data to specific feature')
parser.add_argument('--smartula-data-folder', metavar='data_folder', type=Path,
                    help='Smartula folder for sound data')
parser.add_argument('--hives', default=[], nargs='+', help="Hive names to be included in dataset")
parser.add_argument('--filter-dates', nargs=2, type=datetime.fromisoformat,
                    help="Start and end date for sound data with format YYYY-MM-DD", metavar='START_DATE END_DATE')
parser.add_argument('--output-file', metavar='csv_file', default='feature.csv', type=Path,
                    help='csv feature output file path')
parser.add_argument('--num-workers', type=int, default=1, help='number of work threads')
parser.add_argument('--feature', metavar='feature', type=get_feature_func, default='entropy', help="Sound Indice type")
parser.add_argument('--timezone', metavar='timezone', type=pytz.timezone, default='UTC',
                    help="timezone to be applied")
parser.add_argument('--nfft', metavar='nfft', type=int, default=512, help="number of fft samples")
parser.add_argument('--hop-len', metavar='hop_len', type=int, default=256, help="hop length for spectrogram")
parser.add_argument('--indice-param', type=json.loads, default=None, help="dictionary for sound indices feature "
                                                                          "calculation")

args = parser.parse_args()


def process(filename):
    hive_name = filename.split('\\')[-2].split('_')[0]
    utc_timezone = pytz.timezone('UTC')
    utc_sound_datetime = utc_timezone.localize(
        datetime.strptime('-'.join(filename.split('\\')[-1].split('.')[0].split('-')[1:]),
                          '%Y-%m-%dT%H-%M-%S'))
    sound_datetime = args.timezone.normalize(utc_sound_datetime)
    samples, sampling_freq = read_samples(filename, raw=True)
    spectrogram, _, _ = calculate_spectrogram(samples, sampling_freq, n_fft=args.nfft, hop_len=args.hop_len,
                                              slice_freq=SliceFrequency(0, 8000), convert_db=False)
    feature_val = args.feature(spectrogram) if args.indice_param is None else args.feature(spectrogram, **args.indice_param)
    if type(feature_val) == tuple:
        feature_val = feature_val[0]
    return sound_datetime, hive_name, feature_val


def main():
    sound_list = ut.get_valid_sounds_from_folders(args.smartula_data_folder.glob('*'))
    if not sound_list:
        raise Exception('sound list empty!')

    if args.filter_dates:
        sound_list = ut.filter_by_datetime(sound_list, args.filter_dates[0], args.filter_dates[1])

    sound_list = ut.filter_path_list(sound_list, *args.hives)
    sound_list = list(map(str, sound_list))
    with ThreadPool(args.num_workers) as pool:
        feature_tuples = list(tqdm(pool.imap(process, sound_list), total=len(sound_list)))
        df = pd.DataFrame(feature_tuples, columns=['datetime', 'hive', 'feature'])
        df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
