from enum import Enum


class HiveModelType(Enum):
    AE: str = 'autoencoder'
    CONV1D_AE: str = 'conv1d_autoencoder'
    CONV2D_AE: str = 'conv2d_autoencoder'
    VAE: str = 'vae'
    CONTRASTIVE_VAE: str = 'contrastive_vae'
    CONV1D_VAE: str = 'conv1d_vae'
    CONV2D_VAE: str = 'conv2d_vae'

    @classmethod
    def from_name(cls, name):
        for _, feature in HiveModelType.__members__.items():
            if feature.value == name:
                return feature
        raise ValueError(f"{name} is not a valid supported model name")
