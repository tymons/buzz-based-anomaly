import pandas as pd
import argparse
import pytz

from pathlib import Path
from weather_feature_type import WeatherFeatureType
from datetime import datetime


def _get_openweather_feature_name(weather_file_feature_name: WeatherFeatureType):
    """
    Util for mapping feature name with column name for open weather file
    :param weather_file_feature_name:
    :return: column name
    """
    feature_dict = {
        WeatherFeatureType.TEMPERATURE: 'temp'
    }
    return feature_dict.get(weather_file_feature_name)


def main():
    parser = argparse.ArgumentParser(description='Merge csv feature with thermal values')
    parser.add_argument('--feature-file', metavar='feature_file', default='feature.csv',
                        type=Path, help='csv feature file')
    parser.add_argument('--weather-file', metavar='weather_file', default='weather.csv',
                        type=Path, help='csv weather file')
    parser.add_argument('--open-weather-file', metavar='weather_file', type=Path,
                        help='csv supplementary open weather file')
    parser.add_argument('--output-file', metavar='output_file', default='output.csv',
                        type=Path, help='csv output merged file')
    parser.add_argument('--weather-file-feature-name', default=WeatherFeatureType.TEMPERATURE,
                        choices=list(WeatherFeatureType), type=WeatherFeatureType.from_name,
                        help='feature name to be used with weather csv file')
    parser.add_argument('--timezone', metavar='timezone', type=pytz.timezone, default='UTC',
                        help="timezone to be applied to concatenated dataframe")

    args = parser.parse_args()
    f_name = args.weather_file_feature_name.value

    df_weather = pd.read_csv(args.weather_file, usecols=['datetime', f_name])
    df_feature = pd.read_csv(args.feature_file, usecols=['datetime', 'hive', 'feature'])
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'], utc=True)
    df_feature['datetime'] = pd.to_datetime(df_feature['datetime'], utc=True)
    df_feature.sort_values('datetime', inplace=True)
    df_weather.sort_values('datetime', inplace=True)

    df_weather_reindex = df_weather.set_index('datetime').reindex(df_feature.set_index('datetime').index,
                                                                  method='nearest', tolerance='50min').reset_index()
    data = pd.merge(df_feature, df_weather_reindex, on='datetime').drop_duplicates()

    if data[f_name].isna().sum() and args.open_weather_file is not None:
        print(f'got some missing data on {args.weather_file_feature_name} with len '
              f'{data[f_name].isna().sum()}/{data[f_name].count()} updating with open weather file...')
        open_weather_col = _get_openweather_feature_name(args.weather_file_feature_name)
        df_open_weather = pd.read_csv(args.open_weather_file, usecols=['dt_iso', open_weather_col])
        df_open_weather.columns = ['datetime', f_name]
        df_open_weather['datetime'] = df_open_weather['datetime'].apply(
            lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S %z %Z'))
        df_open_weather.sort_values('datetime', inplace=True)
        df_open_weather[f_name] = df_open_weather[f_name].round()
        df_open_weather_reindex = df_open_weather.set_index('datetime').reindex(
            data.set_index('datetime').index, method='nearest', tolerance='65min').reset_index()
        data = data.set_index('datetime').fillna(df_open_weather_reindex.set_index('datetime')).reset_index()

    data['datetime'] = data['datetime'].dt.tz_convert(args.timezone)

    print(f'saving data to file {args.output_file}...')
    data.to_csv(args.output_file, index=False)
    print('success!')


if __name__ == "__main__":
    main()
