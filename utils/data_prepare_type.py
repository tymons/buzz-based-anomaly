from enum import Enum


class DataPrepareType(Enum):
    SMARTULA = 'download-smartula-sounds'
    GET_NUHIVE_BEES = 'extract-nuhive-bees'
    FRAGMENT_HIVE_BEES = 'fragment-sound-bees'

    @classmethod
    def from_name(cls, name):
        for _, feature in DataPrepareType.__members__.items():
            if feature.value == name:
                return feature
        raise ValueError(f"{name} is not a valid data prepare task name")
