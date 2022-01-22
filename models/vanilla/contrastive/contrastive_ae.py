from typing import List, Union

from models.model_type import HiveModelType
from models.vanilla.contrastive.contrastive_base_model import ContrastiveBaseModel
from models.vanilla.ae import EncoderWithLatent, Decoder


class ContrastiveAE(ContrastiveBaseModel):
    def __init__(self, model_type: HiveModelType, layers: List[int], latent_size: int, input_size: int,
                 dropout: Union[List[float], float] = 0.2, tc_alpha: float = 0.1, include_tc_loss: bool = False):
        super().__init__(model_type, tc_alpha, include_tc_loss)

        self._layers = layers
        self._latent_size = latent_size
        self._dropout = dropout

        self.encoder = EncoderWithLatent(self._layers, self._latent_size, self._dropout, input_size)
        self.decoder = Decoder(self._layers[::-1], self._latent_size, self._dropout, input_size)

    def get_params(self) -> dict:
        """
        Method for getting model params
        :return:
        """
        return {
            'model_layers': self._layers,
            'model_latent': self._latent_size,
            'model_dropouts': self._dropout,
            'model_tc_alpha': self.tc_alpha,
            'model_tc_loss': self.tc_component
        }
