import requests
import pandas as pd
import argparse
import pytz

from pathlib import Path
from weather_feature_type import WeatherFeatureType

OPEN_WEATHER_API = 'http://history.openweathermap.org/data/2.5/history/city'


def update_row_with_openweather(x: pd.DataFrame, weather_feature: WeatherFeatureType, api_key: str, lat: float,
                                long: float):
    start_utc = int((x['datetime'].tz_convert('UTC')).timestamp())
    print(f'making api call for: {start_utc}...', end=' ')

    payload = {'lat': str(lat), 'lon': str(long), 'type': 'hour', 'appid': api_key, 'start': str(start_utc), 'cnt': '1'}
    r = requests.get(OPEN_WEATHER_API, params=payload)
    if r.status_code == 200:
        print('success!')
        r_json = r.json()
        if weather_feature == WeatherFeatureType.TEMPERATURE:
            x[weather_feature.value] = round(r_json['list'][0]['main']['temp'] - 273.15, 2)
        return x
    else:
        raise ValueError(f"Error at http call: {r.text}")


def main():
    parser = argparse.ArgumentParser(description='Merge csv feature with thermal values')
    parser.add_argument('--feature-file', metavar='feature_file', default='feature.csv',
                        type=Path, help='csv feature file')
    parser.add_argument('--weather-file', metavar='weather_file', default='weather.csv',
                        type=Path, help='csv weather file')
    parser.add_argument('--output-file', metavar='output_file', default='output.csv',
                        type=Path, help='csv output merged file')
    parser.add_argument('--open-weather-file', metavar='weather_file', default='open-weather.csv',
                        type=Path, help='csv open weather file')
    parser.add_argument('--weather-file-feature-name', default='temperature', type=str,
                        help='feature name to be used with weather csv file')
    parser.add_argument('--open-weather-file-feature-name', default='temperature', type=str,
                        help='feature name to be used with weather csv file')
    parser.add_argument('--timezone', metavar='timezone', type=pytz.timezone, default='UTC',
                        help="timezone to be applied to concatenated dataframe")

    parser.set_defaults(open_weather=False)
    args = parser.parse_args()

    df_weather = pd.read_csv(args.weather_file, usecols=['datetime', args.weather_file_feature_name])
    df_feature = pd.read_csv(args.feature_file, usecols=['datetime', 'hive', 'feature'])
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'], utc=True)
    df_feature['datetime'] = pd.to_datetime(df_feature['datetime'], utc=True)
    df_feature.sort_values('datetime', inplace=True)
    df_weather.sort_values('datetime', inplace=True)

    df_weather_reindex = df_weather.set_index('datetime').reindex(df_feature.set_index('datetime').index,
                                                                  method='nearest', tolerance='50min').reset_index()

    data = pd.merge(df_feature, df_weather_reindex, on='datetime').drop_duplicates()
    data['datetime'] = data['datetime'].dt.tz_convert(args.timezone)

    if data[args.weather_feature_name].isna().sum() > 0:
        print(f'got some missing data on {args.weather_file_feature_name} with len '
              f'{data[args.weather_file_feature_name].isna().sum()}/{data[args.weather_file_feature_name].count()}'
              f' updating with open weather file...')
        # TODO implement merging with supplementary file
        return

    print(f'saving data to file {args.output_file}...')
    data.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
