# %%
import os
import errno
import logging
import requests
import argparse

from pathlib import Path
from datetime import datetime, date
from pydub import AudioSegment
from zipfile import ZipFile
from tqdm import tqdm
from typing import List

from utils.utils import create_valid_sounds_datalist
from utils.data_prepare_type import DataPrepareType


def generate_wav_from_mp3(mp3_filepath: Path, remove_mp3: bool = False):
    """
    Function that generates wav file form mp3
    :rtype: str
    :param remove_mp3:
    :param mp3_filepath:
    """
    sound = AudioSegment.from_mp3(mp3_filepath)
    new_wav_filename = mp3_filepath.with_suffix('.wav')
    sound.export(new_wav_filename, format="wav")
    if remove_mp3:
        try:
            mp3_filepath.unlink()
        except OSError as e:
            logging.error(f'Error at deleting mp3 file: {mp3_filepath} with {e.strerror}')

    return new_wav_filename


def extract_bees_sounds_from_file(labfile_path: Path, output_folder: Path):
    """
    Function for generating wav files containing only bees sounds
    :param output_folder: where generated file will be stored
    :param labfile_path: labfile path object
    """
    with labfile_path.open() as lab_file:
        file_name = lab_file.readline()
        sound_filenames = list(labfile_path.parent.glob(f'**/{file_name.rstrip()}*.[!lab]*'))
        if not sound_filenames:
            # there is no sound file for given filename
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_name)

        wav_sound_filenames = [file for file in sound_filenames if file.suffix == '.wav']
        try:
            wav_filename = generate_wav_from_mp3(sound_filenames[0]) if not wav_sound_filenames and sound_filenames[
                0].suffix == '.mp3' else wav_sound_filenames[0]
            output_filenames = []
            for idx, line in enumerate(lab_file):
                line_list = line.rstrip().split("\t")
                output_filename = output_folder / f'{wav_filename.stem}-{idx}{wav_filename.suffix}'
                if line_list[-1].rstrip() == 'bee' and not output_filename.exists():
                    new_audio = AudioSegment.from_wav(wav_filename)
                    new_audio = new_audio[float(line_list[0]) * 1000:float(line_list[1]) * 1000]
                    new_audio.export(output_filename, format='wav')
                    output_filenames.append(output_filename)
            return output_filenames
        except IndexError:
            logging.error(f'file {sound_filenames[0]} not supported for wav conversion! only mp3 format supported!')
            return []


def prepare_nuhive_data(path: Path):
    """
    Process *.lab files and mp3/wav files to create new audio files only with bees sounds
    :param path:  where
    """
    files = list(path.glob('**/*.lab'))
    logging.info(f'got  {len(files)} lab files to process')
    for file_path in files:
        try:
            new_filenames = extract_bees_sounds_from_file(Path(file_path), path.parent / 'nu-hive-processed')
            logging.info(f'generated {len(new_filenames)} from {file_path} file!')
        except FileNotFoundError:
            logging.warning(f'missing sound file for {file_path}.')


def _download_data(download_folder: Path, start_utc_timestamp: int, end_utc_timestamp: int, hive_sn: str, api_url: str,
                   token: str):
    """
    Function for performin api call to smartula server
    :param download_folder: path object where downloaded zip should be saved
    :param start_utc_timestamp: start timestamp for sound range
    :param end_utc_timestamp: end timestamp for sound range
    :param hive_sn: sn for hive
    :param token: token for api call
    :return: str: downloaded zip file name
    """
    payload = {'start': start_utc_timestamp, 'end': end_utc_timestamp}
    headers = {'Authorization': f'Bearer {token}'}
    r = requests.get(requests.compat.urljoin(api_url, "/".join([hive_sn, 'sounds'])), headers=headers, params=payload)
    r.raise_for_status()
    filename = download_folder / f'{hive_sn}-{start_utc_timestamp}-{end_utc_timestamp}.zip'
    with filename.open('wb+') as f:
        f.write(r.content)

    return filename


def prepare_smartula_data(dataset_path: Path, start_utc_imestamp: int, end_utc_timestamp: int, hive_list: List[str],
                          api_url: str, token: str = ''):
    """
    Function for downloading and validating data from smartula system
    :param api_url: root api url for smartula
    :param dataset_path: root data folder where data will be saved
    :param start_utc_imestamp:
    :param end_utc_timestamp:
    :param hive_list:
    :param token:
    """
    timestamps = [*list(range(start_utc_imestamp, end_utc_timestamp, 3 * 7 * 24 * 60 * 60)), end_utc_timestamp]
    intervals = list(zip(timestamps[:-1], timestamps[1:]))

    for hive_sn in hive_list:
        logging.debug(f'downloading data for {hive_sn} hive...', flush=True)
        hive_path = dataset_path / hive_sn
        hive_path.mkdir(parents=True, exist_ok=True)
        for start, end in tqdm(intervals):
            file_path = _download_data(hive_path, start, end, hive_sn, api_url, token)
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(hive_path)

            try:
                file_path.unlink()
            except OSError as e:
                logging.error(f'Error: {file_path} : {e.strerror}')

    create_valid_sounds_datalist(dataset_path)


def fragment_sound_data(files: List[Path], duration_ms: int) -> List[str]:
    """
    Method for processing sound files
    :param files:
    :param duration_ms:
    """
    output_filenames = []
    logging.info(f'got  {len(files)} sound files to process')
    for file_path in tqdm(files):
        audio = AudioSegment.from_wav(file_path)
        for elem_idx, element in enumerate(audio[::duration_ms]):
            if element.duration_seconds * 1000 == duration_ms:
                sub_filename = f"{file_path.with_suffix('')}-{elem_idx}.wav"
                with open(sub_filename, "wb") as f:
                    element.export(f, format='wav')
                output_filenames.append(sub_filename)

    return output_filenames


def main():
    parser = argparse.ArgumentParser(description='Process data preparation arguments.')
    parser.add_argument('task', metavar='task', choices=list(DataPrepareType),
                        type=DataPrepareType.from_name, help="Data prepare task type")
    parser.add_argument('--start', '-s', type=date.fromisoformat, metavar='S',
                        help='Start date for Smartula data in format YYYY-MM-DD')
    parser.add_argument('--end', '-e', type=date.fromisoformat, metavar='E',
                        help='End date for Smartula data in format YYYY-MM-DD', default=datetime.now())
    parser.add_argument('--smartula_hives', type=str, nargs='+', metavar='H', help='Smartula hives sns')
    parser.add_argument('--data_folder', default=Path(__file__).absolute().parent / "dataset/", type=Path)
    parser.add_argument('--duration', type=int, help="sound duration for files to be truncated in seconds")

    args = parser.parse_args()

    if args.task == DataPrepareType.GET_NUHIVE_BEES:
        # nu-hive data - only bees
        prepare_nuhive_data(args.data_folder)
    elif args.task == DataPrepareType.SMARTULA:
        # download and validate smartula data
        smartula_api_env = os.getenv('SMARTULA_API')
        smartula_token_env = os.getenv('SMARTULA_TOKEN')
        prepare_smartula_data(args.data_folder,
                              int(datetime(year=args.start.year, month=args.start.month,
                                           day=args.start.day).timestamp()),
                              int(datetime(year=args.end.year, month=args.end.month, day=args.end.day).timestamp()),
                              args.smartula_hives, smartula_api_env, smartula_token_env)
    elif args.task == DataPrepareType.FRAGMENT_HIVE_BEES:
        # slice sound segments
        sound_files = list(args.data_folder.glob('**/*.wav'))
        fragment_sound_data(sound_files, args.duration * 1000)
        # delete original files:
        for file in sound_files:
            file.unlink()


if __name__ == "__main__":
    main()
