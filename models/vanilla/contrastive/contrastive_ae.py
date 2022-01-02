from typing import List, Union

from models.model_type import HiveModelType
from models.vanilla.contrastive.contrastive_base_model import ContrastiveBaseModel
from models.vanilla.ae import EncoderWithLatent, Decoder

from features.contrastive_feature_dataset import VanillaContrastiveOutput


class ContrastiveAE(ContrastiveBaseModel):
    def __init__(self, model_type: HiveModelType, layers: List[int], latent_size: int, input_size: int,
                 dropout: Union[List[float], float] = 0.2):
        super().__init__(model_type)

        self._layers = layers
        self._latent_size = latent_size
        self._dropout = dropout

        self.encoder = EncoderWithLatent(self._layers, self._latent_size, self._dropout, input_size)
        self.decoder = Decoder(self._layers[::-1], self._latent_size, self._dropout, input_size)

    def forward(self, target, background) -> VanillaContrastiveOutput:
        """
        Forward method for NN
        :param target: target sample
        :param background: background smaple
        :return: ContrastiveOutput
        """
        target_latent = self.encoder(target)
        background_latent = self.encoder(background)
        target_output = self.decoder(target_latent)
        background_output = self.decoder(background_latent)

        return VanillaContrastiveOutput(target=target_output, background=background_output,
                                        target_latent=target_latent, background_latent=background_latent)

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
