import logging
from pathlib import Path

import comet_ml
import torch
import utils.utils as util

from typing import Union, Type
from models.variational.contrastive.contrastive_variational_base_model import ContrastiveVariationalBaseModel
from models.variational.contrastive.contrastive_variational_base_model import VaeContrastiveOutput
from models.vanilla.contrastive.contrastive_base_model import ContrastiveBaseModel, VanillaContrastiveOutput
from features.sound_dataset import SoundDataset
from torch.utils.data import Dataset


log = logging.getLogger("smartula")


class ContrastiveFeatureDataset(Dataset):
    def __init__(self, target: SoundDataset, background: SoundDataset):
        assert len(target) == len(background)
        self.target = target
        self.background = background

    def get_params(self):
        return self.target.get_params()

    def __getitem__(self, idx):
        target_sample, _ = self.target.__getitem__(idx)
        background_sample, _ = self.background.__getitem__(idx)
        return target_sample, background_sample

    def __len__(self):
        return len(self.target)


class ContrastiveFeatureLogger:
    def __init__(self, folder: Path,
                 model_type: Type[Union[ContrastiveBaseModel, ContrastiveVariationalBaseModel]] = None,
                 experiment: comet_ml.Experiment = None):
        self.model_type = model_type
        self.folder = folder
        self.experiment = experiment
        # vanilla contrastive output buffers
        self.vanilla_target_data = torch.Tensor()
        self.vanilla_background_data = torch.Tensor()
        # vae contrastive output buffers
        self.vae_target_qs_data = torch.Tensor()
        self.vae_target_qz_data = torch.Tensor()
        self.vae_background_qz_data = torch.Tensor()
        # aux data
        self.aux_data_p = torch.Tensor()
        self.aux_data_q = torch.Tensor()

    def batch_collect_aux(self, aux_data_p_batch, aux_data_q_batch):
        self.aux_data_p = torch.cat((self.aux_data_p, aux_data_p_batch), dim=0)
        self.aux_data_q = torch.cat((self.aux_data_q, aux_data_q_batch), dim=0)

    def aux_data_flush(self, epoch: int):
        folder = self.folder / Path('p-q')
        folder.mkdir(exist_ok=True, parents=True)
        util.plot_latent(self.aux_data_p, folder, epoch, background=self.aux_data_q, experiment=self.experiment)
        self.aux_data_p = torch.Tensor()
        self.aux_data_q = torch.Tensor()

    def batch_collect(self, model_output: Union[VaeContrastiveOutput, VanillaContrastiveOutput]):
        if isinstance(model_output, VanillaContrastiveOutput):
            self.vanilla_target_data = torch.cat((self.vanilla_target_data, model_output.target_latent.cpu()), dim=0)
            self.vanilla_background_data = torch.cat((self.vanilla_background_data,
                                                      model_output.background_latent.cpu()), dim=0)
        elif isinstance(model_output, VaeContrastiveOutput):
            self.vae_target_qs_data = torch.cat((self.vae_target_qs_data,
                                                 model_output.target_qs_mean.cpu().squeeze(dim=1)), dim=0)
            self.vae_target_qz_data = torch.cat((self.vae_target_qz_data,
                                                 model_output.target_qz_mean.cpu().squeeze(dim=1)), dim=0)
            self.vae_background_qz_data = torch.cat((self.vae_background_qz_data,
                                                     model_output.background_qz_mean.cpu().squeeze(dim=1)), dim=0)

    def clear_buffers(self):
        self.vae_target_qz_data = torch.Tensor()
        self.vae_target_qs_data = torch.Tensor()
        self.vae_background_qz_data = torch.Tensor()
        self.vanilla_background_data = torch.Tensor()
        self.vanilla_target_data = torch.Tensor()

    def data_flush(self, epoch: int):
        if ContrastiveBaseModel in self.model_type.__bases__:
            folder = self.folder / Path('target-background')
            folder.mkdir(exist_ok=True, parents=True)
            util.plot_latent(self.vanilla_target_data, folder, epoch, background=self.vanilla_background_data,
                             experiment=self.experiment)
        elif ContrastiveVariationalBaseModel in self.model_type.__bases__:
            folder_qs_qz = self.folder / Path('target-qs-qz')
            folder_qs_qz_bg = self.folder / Path('target-qs-background-qz')
            folder_qs_qz.mkdir(exist_ok=True, parents=True)
            folder_qs_qz_bg.mkdir(exist_ok=True, parents=True)
            util.plot_latent(self.vae_target_qs_data, folder_qs_qz, epoch, background=self.vae_target_qz_data,
                             experiment=self.experiment)
            util.plot_latent(self.vae_target_qs_data, folder_qs_qz_bg, epoch, background=self.vae_background_qz_data,
                             experiment=self.experiment)
        else:
            log.warning(f'Model {self.model_type} empty or not supported, latent data will not be saved!')

        self.clear_buffers()
