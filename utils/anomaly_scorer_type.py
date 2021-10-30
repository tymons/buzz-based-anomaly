from enum import Enum


class AnomalyScorerType(Enum):
    GMM = 'gmm'

    @classmethod
    def from_name(cls, name):
        for _, feature in AnomalyScorerType.__members__.items():
            if feature.value == name:
                return feature
        raise ValueError(f"{name} is not a valid anomaly method name")
