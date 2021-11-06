import requests
import pandas as pd
import argparse

from pathlib import Path

OPEN_WEATHER_API = 'http://history.openweathermap.org/data/2.5/history/city'


def get_temperature_humidity_openweather(x, api_key, lat, long):
    """ Function for making call to openeather api """
    start_utc = int((x['datetime'] - pd.Timedelta(hours=2)).timestamp())
    print(f'making api call for: {start_utc}...', end=' ')

    payload = {'lat': str(lat), 'lon': str(long), 'type': 'hour', 'appid': api_key, 'start': str(start_utc), 'cnt': '1'}
    r = requests.get(OPEN_WEATHER_API, params=payload)
    if r.status_code == 200:
        print('success!')
        r_json = r.json()
        x['outdoor_temperature'] = round(r_json['list'][0]['main']['temp'] - 273.15, 2)
        x['outdoor_humidity'] = round(r_json['list'][0]['main']['humidity'], 2)
        return x
    else:
        raise ValueError(f"Error at http call: {r.text}")


def main():
    parser = argparse.ArgumentParser(description='Fulfill missing values in csv feature file')
    parser.add_argument('file', metavar='csv_file', default='feature.csv', type=Path, help='csv feature file')
    parser.add_argument('latitude', default='64.308498766', metavar='latitude', type=float, help='location latitude')
    parser.add_argument('longitude', default='28.468998124', metavar='longitude', type=float, help='location latitude')
    parser.add_argument('key', metavar='key', default='deadbeef', type=str, help='open weather api key')
    args = parser.parse_args()


if __name__ == "__main__":
    main()
