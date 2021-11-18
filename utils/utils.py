import os
import glob
import math
import logging
import collections

import matplotlib.pyplot as plt
import pandas as pd

import torch
import numpy as np
import utils.plotting as plots

from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from typing import List, Union, Optional, Callable, Dict
from scipy.signal import resample

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


def sort_and_haverage_hive_feature(df: pd.DataFrame, weather_type: WeatherFeatureType) -> pd.DataFrame:
    """
    Function for sorting, hourly averaging feature and rounding hive data
    This function basically prepares one hive data for fingerprint analysis
    :param weather_type: type of weather feature which should be used
    :param df: dataframe with index of datetime for one hive
    :return: sorted and averaged df
    """
    df = df.sort_values('datetime')
    hour_means = df.groupby(pd.Grouper(freq='1H')).mean()
    hour_means[weather_type.value] = hour_means[weather_type.value].round()
    return hour_means


def common_state_per_hour(df: pd.DataFrame, bars_no: int = 30, fig_path: Path = None) -> Dict:
    """
    Function for calculating most common feature value range and temperature for every hour in df.
    :return: dict(tuple, int)
    """
    q1 = df['feature'].quantile(0.05)
    q2 = df['feature'].quantile(0.95)
    feature_bins = np.arange(q1, q2, abs(q1 - q2) / bars_no)

    fig = None
    if fig_path is not None:
        fig = plt.figure(figsize=(20, 26))

    hour_dict = {}
    for idx, (hour, df_hour) in enumerate(df.groupby(df.index.map(lambda x: x.hour))):
        data = df_hour.dropna()
        hist, bins = np.histogram(data['feature'], bins=feature_bins)
        bin_intervals = list(zip(bins[:-1], bins[1:]))
        hist_mean_temperatures = pd.Series([data.loc[(data['feature'] > low_lim)
                                                     & (data['feature'] <= up_lim)]['temperature'].mean()
                                            for low_lim, up_lim in bin_intervals])

        arg_max_idx = np.argmax(hist)
        hour_dict[hour] = bin_intervals[arg_max_idx], round(hist_mean_temperatures.values[arg_max_idx])

        if fig_path is not None:
            ax = fig.add_subplot(8, 3, idx + 1)
            ax.set_title(f'hour: {hour}')
            plots.histogram_feature_weather(ax, hist, (bins[1:]+bins[:-1])/2, hist_mean_temperatures, abs(q1-q2)/bars_no)

    if fig_path is not None:
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)

    return hour_dict


def trend_std(series: pd.Series, reference: float):
    """
    Function for calculating std with respect to given reference instead of mean value
    :param series: pandas series with data
    :param reference: reference value
    :return: reference std value
    """
    return np.sqrt((1 / len(series)) * np.sum((series - reference) ** 2))


def temperature_threshold_per_hour(df: pd.DataFrame, wt: WeatherFeatureType, hour_common_temperature: dict,
                                   polyval_dev: int = 2, fig_path: Path = None) -> Dict:
    """
    Function for calculating start temperatures for every hour within 24-h day cycle where bees
    produces distinctive tones. Start temperatures could be used as thresholds for filtering data to contains sound
    samples recorded during preferable for bees work weather conditions.
    :param fig_path: figure name for visualization
    :param polyval_dev: polynomial degree for trend line
    :param df: dataframe with sound features and weather features
    :param hour_common_temperature: dict with hour and most common temperature
                                    from that hour (calculated by common_state_per_hour function)
    :param wt:
    :return: dictionary with hours as keys and values as lower bound of temperature range
    """
    hour_temperature_threshold = {}

    fig = None
    if fig_path is not None:
        fig = plt.figure(figsize=(20, 26))

    for idx, (hour, df_hour) in enumerate(df.groupby(df.index.map(lambda x: x.hour))):
        data = df_hour.dropna()

        # calculate feature trend across temperatures within given hour
        z = np.polyfit(data[wt.value], data['feature'], polyval_dev)
        data_temperatures = sorted(data[wt.value].unique())
        x_temperatures = np.arange(data_temperatures[0], data_temperatures[-1] + 1, 1)
        trend = np.polyval(z, x_temperatures)
        trend_plot_data = dict(zip(x_temperatures, trend))

        # calculate residuals for every temperature within given hour
        residuals = {}
        for temperature in x_temperatures:
            features = data[data[wt.value] == temperature]['feature']
            residuals[temperature] = trend_std(features, trend_plot_data[temperature]) if not features.empty else 0

        # calculate start temperature for given hour
        residual_threshold = residuals[hour_common_temperature[hour]]
        try:
            start_temperature = next(temperature for temperature, residual in residuals.items() if
                                     residual > residual_threshold and temperature > hour_common_temperature[hour])
        except StopIteration:
            start_temperature = hour_common_temperature[hour]
        hour_temperature_threshold[hour] = start_temperature

        if fig_path is not None:
            ax = fig.add_subplot(8, 3, idx + 1)
            ax.set_title(f'hour: {hour}')
            plots.scatter_feature_temperature_for_temperature_with_std(ax,
                                                                       data[wt.value], data['feature'],
                                                                       x_temperatures, trend,
                                                                       [value for key, value in residuals.items()],
                                                                       start_temperature, hour_common_temperature[hour])

    if fig_path is not None:
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)

    return hour_temperature_threshold


def compare_feature_trends(main_trend, aux_trend):
    pass


def signal_upsample(sig, upsample_ratio=4, x_offset=0) -> List:
    """
    Method for upsampling signal
    :param sig: signal to be upsampled
    :param upsample_ratio: ratio
    :param x_offset: value which should be added to x axis
    :return:
    """
    p_resampled = resample(sig, len(sig) * upsample_ratio, domain='time')
    x = np.linspace(0, len(sig) - 1, len(p_resampled)) + x_offset
    return list(zip(x, p_resampled))


def hour_feature_trend(df: pd.DataFrame, temperature_threshold: dict = None) -> Dict:
    """
    Function for calculating average feature for each day hour.
    :param df:
    :param temperature_threshold:
    """
    hive_trend = {}
    for hour, df_hour in df.groupby(df.index.map(lambda x: x.hour)):
        threshold = temperature_threshold.get(hour, -math.inf) if temperature_threshold is not None else -math.inf
        hive_trend[hour] = df[df[WeatherFeatureType.TEMPERATURE.value] >= threshold]['feature'].mean()

    return hive_trend


def hive_fingerprint(csv_feature_weather_path: Path,
                     weather_type: WeatherFeatureType,
                     hive_name: Optional[str] = None):
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
    # original_timezones = pd.to_datetime(df['datetime']).map(lambda x: x.astimezone().tzinfo)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime')

    df = sort_and_haverage_hive_feature(df, weather_type)
    hour_common_temperature_dict = {key: value[1] for key, value in common_state_per_hour(df).items()}
    hour_start_temperature = temperature_threshold_per_hour(df, weather_type, hour_common_temperature_dict)

    return hour_start_temperature
