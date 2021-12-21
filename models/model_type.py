from enum import Enum
from functools import total_ordering


@total_ordering
class HiveModelType(Enum):
    AE = (1, 'autoencoder')
    VAE = (2, 'vae')

    CONV1D_AE = (3, 'conv1d_autoencoder')
    CONV2D_AE = (4, 'conv2d_autoencoder')
    CONV1D_VAE = (5, 'conv1d_vae')
    CONV2D_VAE = (6, 'conv2d_vae')

    CONTRASTIVE_AE = (7, 'contrastive_autoencoder')
    CONTRASTIVE_CONV1D_AE = (8, 'contrastive_conv1d_autoencoder')
    CONTRASTIVE_CONV2D_AE = (9, 'contrastive_conv2d_autoencoder')

    CONTRASTIVE_VAE = (10, 'contrastive_vae')
    CONTRASTIVE_CONV1D_VAE = (11, 'contrastive_conv1d_vae')
    CONTRASTIVE_CONV2D_VAE = (12, 'contrastive_conv2d_vae')

    @classmethod
    def from_name(cls, name: str):
        for _, feature in HiveModelType.__members__.items():
            if feature.model_name == name:
                return feature
        raise ValueError(f"{name} is not a valid supported model name")

    def __init__(self, num, model_name):
        self.num = num
        self.model_name = model_name

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.num < other.num
        return NotImplemented
