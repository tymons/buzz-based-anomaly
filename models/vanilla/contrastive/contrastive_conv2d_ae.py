from typing import List

from models.model_type import HiveModelType
from models.vanilla.contrastive.contrastive_base_model import ContrastiveBaseModel
from models.conv_utils import convolutional_to_mlp
from models.vanilla.conv2d_ae import Conv2DEncoderWithLatent, Conv2DDecoder


class ContrastiveConv2DAE(ContrastiveBaseModel):
    def __init__(self, model_type: HiveModelType, features: List[int], dropout_probs: List[float], kernel_size: int,
                 padding: int, max_pool: int, latent: int, input_size: tuple, tc_alpha: float = 0.1,
                 include_tc_loss: bool = False):
        super().__init__(model_type, tc_alpha, include_tc_loss)
        self._feature_map = features
        self._dropout_probs = dropout_probs
        self._kernel_size = kernel_size
        self._padding = padding
        self._latent = latent
        self._max_pool = max_pool

        connector_size, conv_temporal = convolutional_to_mlp(input_size, len(features), kernel_size, padding, max_pool)
        self.encoder = Conv2DEncoderWithLatent(features, dropout_probs, kernel_size, padding, max_pool, latent,
                                               features[-1] * connector_size)
        self.decoder = Conv2DDecoder(features[::-1], dropout_probs[::-1], kernel_size, padding, latent,
                                     features[-1] * connector_size, conv_temporal[::-1])

    def get_params(self) -> dict:
        """
        Method for getting model params
        :return:
        """
        return {
            'model_feature_map': self._feature_map,
            'model_dropouts': self._dropout_probs,
            'model_kernel_size': self._kernel_size,
            'model_padding': self._padding,
            'model_latent': self._latent,
            'model_max_pool': self._max_pool,
            'model_tc_alpha': self.tc_alpha,
            'model_tc_loss': self.tc_component
        }
