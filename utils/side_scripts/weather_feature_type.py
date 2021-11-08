from enum import Enum


class WeatherFeatureType(Enum):
    TEMPERATURE = 'temperature'

    @classmethod
    def from_name(cls, name):
        for _, feature in WeatherFeatureType.__members__.items():
            if feature.value == name:
                return feature
        raise ValueError(f"{name} is not a valid weather feature name")
