import torch

from typing import List, Union

from models.vanilla.contrastive.contrastive_base_model import ContrastiveBaseModel
from models.vanilla.ae import EncoderWithLatent, Decoder

from features.contrastive_feature_dataset import ContrastiveOutput


class ContrastiveAE(ContrastiveBaseModel):
    def __init__(self, layers: List[int], latent_size: int, input_size: int,
                 dropout: Union[List[float], float] = 0.2):
        super().__init__()

        self._layers = layers
        self._latent_size = latent_size
        self._dropout = dropout

        self.s_encoder = EncoderWithLatent(self._layers, self._latent_size, self._dropout, input_size)
        self.z_encoder = EncoderWithLatent(self._layers, self._latent_size, self._dropout, input_size)
        self.decoder = Decoder(self._layers[::-1], 2 * self._latent_size, self._dropout, input_size)

    def forward(self, target, background) -> ContrastiveOutput:
        """
        Forward method for NN
        :param target: target sample
        :param background: background smaple
        :return: ContrastiveOutput
        """
        target_s = self.s_encoder(target)
        target_z = self.z_encoder(target)
        background_z = self.z_encoder(background)

        temp = torch.cat(tensors=[target_s, target_z], dim=-1)
        target_output = self.decoder(temp)
        background_output = self.decoder(torch.cat(tensors=[torch.zeros_like(target_s), background_z], dim=-1))

        return ContrastiveOutput(target=target_output, background=background_output, target_qs_latent=target_s,
                                 target_qz_latent=target_z)

    def get_params(self) -> dict:
        """
        Method for getting model params
        :return:
        """
        return {
            'model_layers': self._layers,
            'model_latent': self._latent_size,
            'model_dropouts': self._dropout
        }
