import os
import glob
import math
import logging
import collections
import pandas as pd

import torch
import numpy as np

from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from typing import List, Union, Optional, Callable, Dict

from utils.side_scripts.weather_feature_type import WeatherFeatureType


def logger_setup(log_folder: Path, filename_prefix: str) -> None:
    """
    Method for setting up python logger
    :param log_folder: folder where logs will be saved
    :param filename_prefix: log file filename prefix
    """
    log_folder.mkdir(parents=True, exist_ok=True)
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


def flatten(x):
    """
    Flatten array
    :param x:
    :return:
    """
    if isinstance(x, collections.abc.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def create_valid_sounds_datalist(root_folder: Union[str, Path], valid_file_filename='valid_sounds.txt', prefix="",
                                 upper_rms_threshold=0.8, lower_rms_threshold=0.0000001):
    """Scans specified folder for files with prefix
    Parameters:
        valid_file_filename (str): file which will be created
        root_folder (str): root folder where scan will be performed
        prefix (str): optional prefix for folderrs
        upper_rms_threshold (float): rms threshold for sound (reject too loud samples)
        lower_rms_threshold (float): rms threshold for sound (reject empty samples)

    Returns:
        valid_count (dict): dictionary of folders and valid files count
    """
    folders = [folder for folder in glob.glob(f"{root_folder}\\{prefix}*\\")]
    valid_dict_count = {}
    for folder in folders:
        logging.info(f"reading sounds in folder {folder.split(os.sep)[-2]}...", flush=True)
        files = [file for file in glob.glob(f"{folder}*.wav")]
        valid_files = []
        for filename in tqdm(files):
            # check filename and write its filename to list if is valid
            sample_rate, sound_samples = wavfile.read(filename)
            if len(sound_samples.shape) > 1:
                sound_samples = sound_samples.T[0]
            sound_samples = sound_samples / (2.0 ** 31)
            rms = math.sqrt(sum(sound_samples ** 2) / len(sound_samples))
            if upper_rms_threshold > rms > lower_rms_threshold:
                valid_files.append(filename.split(os.sep)[-1])

        with open(f'{folder}{valid_file_filename}', 'w') as f:
            f.write("\n".join(valid_files))

        valid_dict_count[folder.split(os.sep)[-1]] = len(valid_files)

    return valid_dict_count


def get_valid_sounds_from_folders(folder_list: List[Path], valid_file_filename: str = 'valid_sounds.txt') -> List[Path]:
    """
    Reads valid sounds files in specific directories. Note that files should exists,
    see create_valid_sounds_datalist method
    :param folder_list: list of folders which should be scanned
    :param valid_file_filename: filename for 'valid' sound filenames file.
    :return: list of
    """
    sound_filenames = []

    for folder in folder_list:
        summary_file = folder / valid_file_filename
        if summary_file.exists():
            with summary_file.open('r') as f:
                sound_filenames += list(map(lambda x: folder / x, f.read().splitlines()))
        else:
            logging.warning(f'{valid_file_filename} for folder {folder} does not exists! skipping')

    return sound_filenames


def filter_by_datetime(files: List[Path], start: datetime, end: datetime) -> List[Path]:
    """
    Filtering list of Path based on their datetime encoded within filename. Note that filename should be of format:
    "HIVENAME-YYYY-MM-DDTHH-mm-ss" e.g. DEADBEEF94-2020-08-09T22-10-25"
    :param files:   list of sound files paths which should be filtered
    :param start:   start datetime
    :param end:     end datetime
    :return: list of filtered paths
    """

    def _is_within_timerange(elem):
        """ predicate performing filename datetime parsing and checking timerange """
        elem = "-".join(elem.stem.split('-')[1:]).split('.')[0]
        datetime_elem = datetime.strptime(elem, '%Y-%m-%dT%H-%M-%S')
        return start <= datetime_elem <= end

    return list(filter(_is_within_timerange, files))


def filter_string_list(paths: List[Path], *names: str) -> List[Path]:
    """
    Filter sounds based on filenames. Returning only these files which filename contains something from names param
    Note that filename should be of format: "HIVENAME-YYYY-MM-DDTHH-mm-ss" e.g. DEADBEEF94-2020-08-09T22-10-25"
    :param paths: list of files
    :param names: unpacked string list containing names to be checked
    :return: list of filtered paths
    """
    return list(filter(lambda str_elem: (any(x in str_elem.stem for x in [*names])), paths))


def batch_normalize(batch_data):
    """ Function for data normalization accross batch """
    return _batch_perform(batch_data, lambda a: MinMaxScaler().fit_transform(a))


def batch_standarize(batch_data):
    """ Function for data standarization across batch """
    return _batch_perform(batch_data, lambda a: StandardScaler().fit_transform(a))


def batch_add_noise(batch, noise_factor=0.1):
    """ Function for adding noise to batch """
    noised_batch_input = batch + noise_factor * torch.randn(*batch.shape)
    noised_batch_input = np.clip(noised_batch_input, 0., 1.)
    return noised_batch_input


def _batch_perform(batch_data: torch.Tensor, operation: Callable):
    """ Function for data normalization accross batch """
    input_target = batch_data[:, 0, :]
    initial_shape = input_target.shape

    if input_target.ndim > 2:
        input_target = input_target.reshape(initial_shape[0], -1)

    output = torch.Tensor(operation(input_target).astype(float))

    if len(initial_shape) > 2:
        output = output.reshape(initial_shape)

    batch_data[:, 0, :] = output

    return batch_data


def closest_power_2(x):
    """ Function returning nerest power of two """
    possible_results = math.floor(math.log(x, 2)), math.ceil(math.log(x, 2))
    return min(possible_results, key=lambda z: abs(x - 2 ** z))


def adjust_linear_ndarray(input_array: np.ndarray, length: int, policy: str = 'zeros') -> np.ndarray:
    """
    Function for adjusting nd array
    :param input_array: input data
    :param length: length to be expanded or truncated
    :param policy: policy ('zeros' or 'sequence')
    :return: nd array
    """
    diff_len = length - len(input_array)
    if diff_len > 0:
        step = input_array[-1] - input_array[-2]
        if policy == 'sequence':
            start = input_array[-1] + step
            stop = start + (diff_len * step)
            new_values = np.arange(start, stop, step)
        else:
            new_values = np.zeros(length)
        output = np.hstack([input_array, new_values])
    elif diff_len < 0:
        output = input_array[:diff_len]
    else:
        output = input_array
    return output


def adjust_matrix(matrix, *lengths, fill_with: int = 0):
    """ Function for truncating or expanding matrix to lengths
    Parameters:
        :param matrix: matrix to be truncated or expanded
        :param fill_with: fill auxilary data with value
     """
    for i, length in enumerate(lengths):
        shape = matrix.shape
        if length > shape[i]:
            # pad with zeros
            diff = length - shape[i]
            new_shape = list(shape)
            new_shape[i] = diff
            new_shape = tuple(new_shape)
            values = np.empty(new_shape)
            values.fill(fill_with)
            matrix = np.append(matrix, values, axis=i)
        else:
            matrix = np.swapaxes(matrix, 0, i)
            matrix = matrix[:length, ...]
            matrix = np.swapaxes(matrix, i, 0)

    return matrix


def truncate_lists_to_smaller_size(arg1, arg2):
    """ Function for truncating two lists to smaller size.
    Note that there possibly should be better way to do this operation. """
    if len(arg1) > len(arg2):
        arg1 = arg1[:len(arg2)]
    else:
        arg2 = arg2[:len(arg1)]

    return arg1, arg2


def _sort_and_haverage_feature(df: pd.DataFrame, weather_type: WeatherFeatureType) -> pd.DataFrame:
    """
    Function for sorting and hourly averagin
    :param df: dataframe with index of datetime
    :return: sorted and averaged df
    """
    df = df.sort_values('datetime')
    hour_means = df.groupby(pd.Grouper(freq='1H')).mean()
    hour_means[weather_type.value] = hour_means[weather_type.value].round()
    return hour_means


def common_state_per_hour(df: pd.DataFrame) -> Dict:
    """
    Function for calculating most common feature value range along with mean temperature
    for that range
    :return:
    """
    hour_dict = {}
    for hour, df_group in df.groupby(df.index.map(lambda x: x.hour)):
        data = df_group.dropna()
        hist, bins = np.histogram(data['feature'], bins=20)
        bin_intervals = list(zip(bins[:-1], bins[1:]))
        hist_mean_temperatures = [round(data.loc[(data['feature'] > low_lim)
                                                 & (data['feature'] <= up_lim)]['temperature'].mean())
                                  for low_lim, up_lim in bin_intervals]

        arg_max_idx = np.argmax(hist)
        hour_dict[hour] = (bin_intervals[arg_max_idx], hist_mean_temperatures[arg_max_idx])

    return hour_dict


def hive_fingerprint(csv_feature_weather_path: Path,
                     weather_type: WeatherFeatureType,
                     hive_name: Optional[str] = None) -> List[Path]:
    """
    Function for fingerprint filtering method from https://www.sciencedirect.com/science/article/pii/S0168169921005068
    :param hive_name: hive which should be used form csv, if none - csv file should contain only data for one hive
    :param weather_type:
    :param csv_feature_weather_path: Path to csv file with feature (temperature for the basic case)
    :return: filtered sound list
    """
    df = pd.read_csv(csv_feature_weather_path, usecols=['datetime', 'hive', 'feature', weather_type.value])
    if hive_name is not None:
        df = df[df['hive'] == hive_name]
    original_timezones = pd.to_datetime(df['datetime']).map(lambda x: x.astimezone().tzinfo)

    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime')
    df = _sort_and_haverage_feature(df, weather_type)
    most_common_temperatures = {key: value[1] for key, value in common_state_per_hour(df).items()}

    return df
