from features.sound_dataset import SoundDataset
from torch.utils.data import Dataset
from collections import namedtuple

_fields = ['target', 'background', 'target_qs_mean', 'target_qs_log_var', 'target_qz_mean', 'target_qz_log_var',
           'background_qz_mean', 'background_qz_log_var', 'target_qs_latent', 'target_qz_latent', 'background_qs_latent']
ContrastiveOutput = namedtuple('ContrastiveOutput', _fields, defaults=(None,) * len(_fields))


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
