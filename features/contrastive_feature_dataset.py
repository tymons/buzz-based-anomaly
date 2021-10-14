from features.sound_dataset import SoundDataset
from torch.utils.data import Dataset
from dataclasses import dataclass
from torch import Tensor


@dataclass(frozen=True)
class ContrastiveOutput:
    target: Tensor
    background: Tensor
    target_qs_mean: float
    target_qs_log_var: float
    target_qz_mean: float
    target_qz_log_var: float
    background_qz_mean: float
    background_qz_log_var: float
    target_qs_latent: Tensor
    target_qz_latent: Tensor


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
