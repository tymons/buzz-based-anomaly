import os
import glob
import math
import logging
import collections

import torch
import numpy as np

from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Callable
from datetime import datetime
from tqdm import tqdm


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


def create_valid_sounds_datalist(root_folder, valid_file_filename='valid_sounds.txt', prefix="",
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


def get_valid_sounds_datalist(folder_list, valid_file_filename):
    """Reads valid sounds files in specific directories. Note that files should exists, 
    see create_valid_sounds_datalis method
    
    Parameters:
        folder_list (str): list with folder which will be scanned
        valid_file_filename (str): filename which will be read from folder_list
    
    Returns:

    """
    sound_filenames = []

    for folder in folder_list:
        summary_file = os.path.join(folder, valid_file_filename)
        if os.path.isfile(summary_file):
            with open(summary_file, 'r') as f:
                sound_filenames += list(map(lambda x: os.path.join(folder, x), f.read().splitlines()))
        else:
            logging.warning(f'{valid_file_filename} for folder {folder} does not exists! skipping')

    return sound_filenames


def filter_by_datetime(str_list, start, end):
    """ Function for filtering sound samples names by date time """

    def _is_str_indatetimerage(elem):
        """ closure for filter_bydatetiem """
        elem = "-".join(elem.split(os.sep)[-1].split('-')[1:]).split('.')[0]
        datetime_elem = datetime.strptime(elem, '%Y-%m-%dT%H-%M-%S')
        return start <= datetime_elem <= end

    return list(filter(_is_str_indatetimerage, str_list))


def filter_string_list(input_str_list, *names):
    """ Filter sound_filenames as it returns only these files which includes hive_names

    Parameters:
        input_str_list (list): list of strings to be filtered
        names (varg): names to be search for inside input_str_list

    Returns:
    """
    return list(filter(lambda str_elem: (any(x in str_elem for x in [*names])), input_str_list))


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

    output = torch.Tensor(operation(input_target).astype(np.float32))

    if len(initial_shape) > 2:
        output = output.reshape(initial_shape)

    batch_data[:, 0, :] = output

    return batch_data


def closest_power_2(x):
    """ Function returning nerest power of two """
    possible_results = math.floor(math.log(x, 2)), math.ceil(math.log(x, 2))
    return min(possible_results, key=lambda z: abs(x - 2 ** z))


def adjust_matrix(matrix, *lengths):
    """ Function for truncating matrix to lengths
    
    Parameters:
        matrix: matrix to be truncated or expanded
     """
    for i, length in enumerate(lengths):
        shape = matrix.shape
        if length > shape[i]:
            # pad with zeros
            diff = length - shape[i]
            new_shape = list(shape)
            new_shape[i] = diff
            new_shape = tuple(new_shape)
            zeros = np.zeros(new_shape)
            matrix = np.append(matrix, zeros, axis=i)
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


def read_comet_api_key(config_file_fullpath):
    """ Function for reading comet api key form file """
    with open(config_file_fullpath, 'r') as f:
        api_key = [b.split('=')[-1] for b in f.read().splitlines() if b.startswith('api_key')][0]

    return api_key
