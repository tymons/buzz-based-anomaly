import os
import glob
import math
import logging
import collections

import comet_ml
import pytz

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

import torch
import numpy as np
import utils.plotting as plots

from datetime import time
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from typing import List, Union, Tuple, Callable, Dict
from scipy.signal import resample
from multiprocessing.pool import ThreadPool

from utils.side_scripts.weather_feature_type import WeatherFeatureType

log = logging.getLogger("smartula")
matplotlib.use('Agg')


def logger_setup(log_folder: Path, filename_prefix: str) -> None:
    """
    Method for setting up python logger
    :param log_folder: folder where logs will be saved
    :param filename_prefix: log file filename prefix
    """
    log_folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_folder / f"{filename_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
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
        log.info(f"reading sounds in folder {folder.split(os.sep)[-2]}...", flush=True)
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
            log.warning(f'{valid_file_filename} for folder {folder} does not exists!')
            if folder.is_dir():
                log.debug('trying to go one level deeper...')
                sound_filenames = get_valid_sounds_from_folders(list(folder.glob('*')))

    return sound_filenames


def parse_smartula_hivename(filename: Path) -> str:
    """
    Method for parsing sound filename and extracting hive name
    :param filename:
    """
    return filename.stem.split('-')[0]


def parse_smartula_datetime(filename: Path) -> datetime:
    """
    Method for parsing smartula filename to extract datetime object
    :param filename:
    :return:
    """
    elem = "-".join(filename.stem.split('-')[1:]).split('.')[0]
    return datetime.strptime(elem, '%Y-%m-%dT%H-%M-%S')


def filter_by_time(files: List[Path], start_time: datetime.time, end_time: datetime.time) -> List[Path]:
    """
    Method for filtering list smartula sound files paths by hours
    :param files: list of paths objects
    :param start_time: start time
    :param end_time: end time
    :return: filtered list
    """

    def _is_within_timerange(elem):
        """ predicate for datetime filtering within given day"""
        time_elem = parse_smartula_datetime(elem).time()
        return start_time <= time_elem <= end_time

    return list(filter(_is_within_timerange, files))


def filter_by_datetime(files: List[Path], start: datetime, end: datetime) -> List[Path]:
    """
    Filtering list of Path based on their datetime encoded within filename. Note that filename should be of format:
    "HIVENAME-YYYY-MM-DDTHH-mm-ss" e.g. DEADBEEF94-2020-08-09T22-10-25"
    :param files:   list of sound files paths which should be filtered
    :param start:   start datetime (UTC)
    :param end:     end datetime (UTC)
    :return: list of filtered paths
    """

    def _is_within_daytimerange(elem):
        """ predicate performing filename datetime parsing and checking timerange """
        datetime_elem = parse_smartula_datetime(elem)
        return start <= datetime_elem <= end

    return list(filter(_is_within_daytimerange, files))


def filter_path_list(paths: List[Path], *names: str) -> List[Path]:
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
            plots.histogram_feature_weather(ax, hist, (bins[1:] + bins[:-1]) / 2, hist_mean_temperatures,
                                            abs(q1 - q2) / bars_no)

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
                                   polyval_dev: int = 2, std_alpha=1.0, fig_path: Path = None) -> Dict:
    """
    Function for calculating start temperatures for every hour within 24-h day cycle where bees
    produces distinctive tones. Start temperatures could be used as thresholds for filtering data to contains sound
    samples recorded during preferable for bees work weather conditions.
    :param fig_path: figure name for visualization
    :param polyval_dev: polynomial degree for trend line
    :param df: dataframe with sound features and weather features
    :param hour_common_temperature: dict with hour and most common temperature
                                    from that hour (calculated by common_state_per_hour function)
    :param wt: weather type enum
    :param std_alpha: scaler for std threshold value
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
        lowband_residuals = [res for temperature, res in residuals.items()
                             if temperature <= hour_common_temperature[hour]]
        residual_threshold = sum(lowband_residuals) / len(lowband_residuals)
        try:
            start_temperature = next(temperature for temperature, residual in residuals.items() if
                                     residual > std_alpha * residual_threshold and temperature >
                                     hour_common_temperature[hour])
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


def signal_upsample_with_x(sig, upsample_ratio=4, x_offset=0) -> List[Tuple]:
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
        hive_trend[hour] = df_hour[df_hour[WeatherFeatureType.TEMPERATURE.value] >= threshold]['feature'].mean()

    return hive_trend


def temperature_step_filter(df_hive: pd.DataFrame, weather_type: WeatherFeatureType, hour_beeday_start: int,
                            hour_beeday_end: int, upsample: int = 4, std_alpha: float = 1.0, bars_no: int = 30):
    """
    Method for calulating beehive upsampled, temperature-filtered feature trend across all 24 hour within day
    :param df_hive: pandas dataframe with data for particular hive
    :param weather_type: weather type for reading proper column from csv file
    :param hour_beeday_start: assumed beeday hour start for narrowing data
    :param hour_beeday_end: assumed beeday hour end for narrowing data
    :param upsample: upsample ratio for Fourier upsample method
    :param std_alpha: std multiple for thresholding step, we get only those sounds from temperatures where std
                      was bigger than std_alpha*(mean(stds_ for_common_temperature_and_lower))
    :param bars_no: bars no for common temperature histogram calculation
    :return: upsampled filtered feature 24-cycle hive trend, start_temperature dictionary (for every hour there is
             the temperature which maximizes sound entropy)
    """
    hour_common_temperature_dict = {key: value[1] for key, value in
                                    common_state_per_hour(df_hive, bars_no=bars_no).items()}
    hour_start_temperature = temperature_threshold_per_hour(df_hive, weather_type, hour_common_temperature_dict,
                                                            std_alpha=std_alpha)
    main_trend_filtered = hour_feature_trend(df_hive, hour_start_temperature)
    main_trend_filtered_list = [value for key, value in main_trend_filtered.items()
                                if hour_beeday_start <= key <= hour_beeday_end]
    upsampled = signal_upsample_with_x(main_trend_filtered_list, upsample_ratio=upsample, x_offset=hour_beeday_start)

    return upsampled, hour_start_temperature


def hour_step_filter(xy_trend: List[Tuple], aux_trends: Dict[str, List[Tuple]],
                     step_sensitivity: int) -> Tuple[datetime.time, datetime.time]:
    """
    Method for calculating hour fingerprint ranges based on gradient value from mse between trends
    :param xy_trend: main hive 24-cycle feature trend
    :param aux_trends: auxiliary hives feature trends
    :param step_sensitivity: min length of interval between cross-section indexes
    :return: tuple of two times object reflecting most distinctive period of time (start_time, end)
    """
    xy_trend = sorted(xy_trend)
    y_trend = np.array([x[1] for x in xy_trend])
    x_trend = np.array([x[0] for x in xy_trend])
    y_trend = y_trend - y_trend.mean()

    start_daystamps = []
    end_daystamps = []
    for idx, (hive_name, aux_trend) in enumerate(aux_trends.items()):
        aux_y_trend = np.array([x[1] for x in sorted(aux_trend)])
        aux_y_trend = aux_y_trend - aux_y_trend.mean()

        start_time, end_time = most_varied_interval(y_trend, aux_y_trend, x_trend, step_sensitivity)
        log.debug(f'fingerprint base vs aux hive ({hive_name}) start/end: {start_time}/{end_time}')
        start_daystamps.append(start_time.hour * 3600 + start_time.minute + 60 + start_time.second)
        end_daystamps.append(end_time.hour * 3600 + end_time.minute + 60 + end_time.second)

    start_mean = sum(start_daystamps) / len(start_daystamps)
    end_mean = sum(end_daystamps) / len(end_daystamps)

    averaged_start_time = time_from_daystamp(round(start_mean))
    averaged_end_time = time_from_daystamp(round(end_mean))

    return averaged_start_time, averaged_end_time


def time_from_daystamp(daystamp: int):
    """
    Method for transforming daystamp to time object
    :param daystamp: seconds offset within day
    :return: time object
    """
    hours, mod_val = divmod(daystamp, 3600)
    minutes, seconds = divmod(mod_val, 60)
    return time(hour=hours, minute=minutes, second=seconds)


def most_varied_interval(core_trend, aux_trend, x_values, step_sensitivity=1):
    """
    Method for calculating most different interval by calculating gradient from mse between two trends
    :param core_trend: main data array
    :param aux_trend: aux data array
    :param x_values: x axis to perform calculation, should be in decimal
    :param step_sensitivity: step_sensitivity: min length of interval between cross-section indexes
    :return:
    """
    difference = np.sqrt((core_trend - aux_trend) ** 2)
    gradient = np.gradient(difference)

    # we choose spots where gradient change sign from - to +
    crosssec_idxes = np.argwhere(np.diff(np.sign(gradient)) > 0).flatten()
    crosssec_idxes = np.insert(crosssec_idxes, 0, 0)
    crosssec_idxes = np.insert(crosssec_idxes, len(crosssec_idxes), len(difference) - 1)

    # cross sec sensitivity
    areas_idxes = list(zip(crosssec_idxes[:-1], crosssec_idxes[1:]))
    areas_idxes = [(start, end) for start, end in areas_idxes if abs(start - end) > step_sensitivity]
    crosssec_idxes = [start for start, _ in areas_idxes]
    # build new area indexes based on sensitivity output
    areas_idxes = list(zip(crosssec_idxes[:-1], crosssec_idxes[1:]))

    # calculate integrals and get max
    integral_values = [np.trapz(difference[start_idx:stop_idx], x=x_values[start_idx:stop_idx])
                       for area_idx, (start_idx, stop_idx) in enumerate(areas_idxes)]
    area_max_idx = np.argmax(integral_values)

    start_hour, start_minutes = int(divmod(x_values[areas_idxes[area_max_idx][0]], 1)[0]), int(
        60 * divmod(x_values[areas_idxes[area_max_idx][0]], 1)[1])
    end_hour, end_minutes = int(divmod(x_values[areas_idxes[area_max_idx][1]], 1)[0]), int(
        60 * divmod(x_values[areas_idxes[area_max_idx][1]], 1)[1])

    return time(hour=start_hour, minute=start_minutes), time(hour=end_hour, minute=end_minutes)


def hive_fingerprint(df: pd.DataFrame, fingerprint_hive_name: str, weather_type: WeatherFeatureType,
                     hour_beeday_start: int = 4, hour_beeday_end: int = 23):
    """
    Method for extracting bee fingerprint
    :param df: pandas dataframe with all hives, features and outdoor feature (temperature, humidity etc.)
               index should be a timestmap
    :param fingerprint_hive_name: hivename for main fingerprint calculation
    :param weather_type: weather type used in csv file (eg. temperature)
    :param hour_beeday_start: bee day start hour
    :param hour_beeday_end: bee day end hour
    """
    hives_available = df['hive'].unique()
    hive_filtered_trends = {}
    hive_temperature_map = {}
    for idx, hive_name in enumerate(hives_available):
        df_hive = df[df['hive'] == hive_name]
        df_hive = sort_and_haverage_hive_feature(df_hive, weather_type)
        trend, temperatures = temperature_step_filter(df_hive, weather_type, hour_beeday_start=hour_beeday_start,
                                                      hour_beeday_end=hour_beeday_end,
                                                      upsample=4, std_alpha=1.0, bars_no=30)

        hive_filtered_trends[hive_name]: Dict[str, List[Tuple]] = trend
        hive_temperature_map[hive_name]: Dict[str, Dict] = temperatures

    sorted_main_hive_trend: List[Tuple] = hive_filtered_trends.pop(fingerprint_hive_name)
    start_time, end_time = hour_step_filter(sorted_main_hive_trend, hive_filtered_trends, 7)

    return (start_time, end_time), hive_temperature_map[fingerprint_hive_name]


def filter_hive_fingerprint(csv_feature_weather_path: Path,
                            fingerprint_hive_name: str,
                            sound_list: List[Path],
                            weather_type: WeatherFeatureType = WeatherFeatureType.TEMPERATURE,
                            hour_beeday_start: int = 4,
                            hour_beeday_end: int = 23,
                            quiet=False,
                            num_workers=8):
    """
    Function for fingerprint filtering method from https://www.sciencedirect.com/science/article/pii/S0168169921005068
    :param num_workers: number of workers used in fingerpint soundfile filtering
    :param sound_list: sound list to be filtered note that this should be sounds only for fingerprint hive!
    :param hour_beeday_end:
    :param hour_beeday_start:
    :param fingerprint_hive_name: hive name for which fingerprint should be calculated
    :param weather_type:
    :param csv_feature_weather_path: Path to csv file with feature (temperature for the basic case)
    :param quiet: verbose level
    :return: utc most distinctive start time, utc most distinctive end time,
             temperatures threshold for every hour within day
    """
    f_hive_sound_list = filter_path_list(sound_list.copy(), fingerprint_hive_name)

    df = pd.read_csv(csv_feature_weather_path, usecols=['datetime', 'hive', 'feature', weather_type.value])
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime')

    # calculate main fingerprint
    (f_start, f_end), f_temperatures = hive_fingerprint(df, fingerprint_hive_name, weather_type, hour_beeday_start,
                                                        hour_beeday_end)

    # fingerprint filter by temperature
    hive_tfiltered_sound_timestamps = []
    df_fingerprint_hive = df[df['hive'] == fingerprint_hive_name]
    for hour, hgroup in df_fingerprint_hive.groupby(df_fingerprint_hive.index.map(lambda x: x.hour)):
        hive_tfiltered_sound_timestamps.extend(hgroup[hgroup[weather_type.value] >= f_temperatures[hour]].index.values)

    # fingerprint filter by time
    f_hive_sound_datetime = set(map(lambda x: pd.Timestamp(parse_smartula_datetime(x)).tz_localize(pytz.UTC),
                                    f_hive_sound_list))
    temperature_filtered_datetime = set(map(lambda x: pd.Timestamp(x).tz_localize(pytz.UTC),
                                            hive_tfiltered_sound_timestamps))
    fingerprint_datetimes = f_hive_sound_datetime & temperature_filtered_datetime
    fingerprint_datetimes = list(filter(lambda x: f_start <= x.time() <= f_end, fingerprint_datetimes))
    fingerprint_datetimes = list(map(lambda y: y.strftime("%Y-%m-%dT%H-%M-%S"), fingerprint_datetimes))

    def process(datetime_str):
        return filter_path_list(f_hive_sound_list, datetime_str)

    with ThreadPool(num_workers) as pool:
        fingerprint_filenames = list(tqdm(pool.imap(process, fingerprint_datetimes), total=len(fingerprint_datetimes)))
        fingerprint_filenames = [x for sublist in fingerprint_filenames for x in sublist]  # flatten

    if not quiet:
        log.info(f'fingerprint time range: {f_start}/{f_end}')
        log.info(f'fingerprint temperatures: {f_temperatures}')
        log.info(f'after fingerprint filtering got {len(fingerprint_filenames)} recordings '
                 f'which is {len(fingerprint_filenames) / len(f_hive_sound_list) * 100:.2f}% of initial data')

    return fingerprint_filenames


def plot_latent(target: torch.Tensor, folder: Path, epoch=0, background: torch.Tensor = None,
                experiment: comet_ml.Experiment = None):
    """
    Method for plotting latent data
    :param target:
    :param folder:
    :param epoch:
    :param background:
    :param experiment:
    :return:
    """
    plt.ioff()
    fig = plt.figure(figsize=(10, 10))
    filepath = folder / Path(f'{folder.stem}-latent-epoch-{epoch}.png')
    plt.scatter(x=target.T[0], y=target.T[1] if target.shape[1] > 1 else target.T[0], label='target', c='g')
    if background is not None:
        plt.scatter(x=background.T[0], y=background.T[1] if background.shape[1] > 1 else background.T[0],
                    label='auxiliary', c='r')
    plt.xlabel('Latent 1')
    plt.ylabel('Latent 2')
    plt.legend()
    plt.savefig(filepath)
    plt.close(fig)

    if experiment is not None:
        experiment.log_image(str(filepath))

    return
