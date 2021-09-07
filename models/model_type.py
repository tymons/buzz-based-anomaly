from enum import Enum


class HiveModelType(Enum):
    AE = 'autoencoder'
    CONV1D_AE = 'conv1d_autoencoder'

    @classmethod
    def from_name(cls, name):
        for _, feature in HiveModelType.__members__.items():
            if feature.value == name:
                return feature
        raise ValueError(f"{name} is not a valid supported model name")