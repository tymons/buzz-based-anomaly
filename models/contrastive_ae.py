from torch import nn

from typing import List, Union

from models.contrastive_base_model import ContrastiveBaseModel

from features.contrastive_feature_dataset import ContrastiveOutput


class ContrastiveAE(ContrastiveBaseModel):
    def __init__(self, layers: List[int], latent_size: int, input_size: int,
                 dropout: Union[List[float], float] = 0.2):
        super().__init__()

    def loss_fn(self, target, background, model_output: ContrastiveOutput, discriminator) -> nn.Module:
        pass

    def forward(self, target, background) -> ContrastiveOutput:
        pass

    def get_params(self) -> dict:
        pass
