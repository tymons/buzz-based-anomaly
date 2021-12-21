import torch

from typing import List

from models.model_type import HiveModelType
from models.vanilla.contrastive.contrastive_base_model import ContrastiveBaseModel
from models.conv_utils import convolutional_to_mlp
from models.vanilla.conv1d_ae import Conv1DEncoderWithLatent, Conv1DDecoder

from features.contrastive_feature_dataset import ContrastiveOutput


class ContrastiveConv1DAE(ContrastiveBaseModel):
    def __init__(self, model_type: HiveModelType, features: List[int], dropout_probs: List[float], kernel_size: int, padding: int, max_pool: int,
                 latent: int, input_size: int):
        super().__init__(model_type)
        self._feature_map = features
        self._dropout_probs = dropout_probs
        self._kernel_size = kernel_size
        self._padding = padding
        self._latent = latent
        self._max_pool = max_pool

        connector_size, conv_temporal = convolutional_to_mlp(input_size, len(features), kernel_size, padding, max_pool)
        self.s_encoder = Conv1DEncoderWithLatent(features, dropout_probs, kernel_size, padding, max_pool, latent,
                                                 features[-1] * connector_size)
        self.z_encoder = Conv1DEncoderWithLatent(features, dropout_probs, kernel_size, padding, max_pool, latent,
                                                 features[-1] * connector_size)
        self.decoder = Conv1DDecoder(features[::-1], dropout_probs[::-1], kernel_size, padding, 2 * latent,
                                     features[-1] * connector_size, conv_temporal[::-1])

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
                                 target_qz_latent=target_z, target_qz_mean=None, target_qs_mean=None,
                                 background_qz_mean=None, background_qz_log_var=None, target_qs_log_var=None,
                                 target_qz_log_var=None)

    def get_params(self) -> dict:
        """
        Method for getting model params
        :return: dcitionary with parameters
        """
        return {
            'model_feature_map': self._feature_map,
            'model_dropouts': self._dropout_probs,
            'model_kernel_size': self._kernel_size,
            'model_padding': self._padding,
            'model_latent': self._latent,
            'model_max_pool': self._max_pool
        }
