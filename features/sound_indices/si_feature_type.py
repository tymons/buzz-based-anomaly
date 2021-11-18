from enum import Enum


class SoundIndicesFeatureType(Enum):
    SPECTRAL_ENTROPY = 'entropy'
    ACI = 'aci'

    @classmethod
    def from_name(cls, name):
        for _, feature in SoundIndicesFeatureType.__members__.items():
            if feature.value == name:
                return feature
        raise ValueError(f"{name} is not a valid sound indices name")
