from enum import Enum


class SoundFeatureType(Enum):
    PERIODOGRAM = 'periodogram'
    SPECTROGRAM = 'spectrogram'
    MELSPECTROGRAM = 'melspectrogram'
    MFCC = 'mfcc'

    @classmethod
    def from_name(cls, name):
        for _, feature in SoundFeatureType.__members__.items():
            if feature.value == name:
                return feature
        raise ValueError(f"{name} is not a valid feature name")
