from features.sound_dataset import SoundDataset
from torch.utils.data import Dataset


class DoubleFeatureDataset(Dataset):
    """ Wrapper class for contrastive neural networks """
    def __init__(self, target: SoundDataset, background: SoundDataset):
        assert len(target) == len(background)
        self.target = target
        self.background = background

    def get_params(self):
        """ Function for returning params """
        return self.target.get_params()

    def __getitem__(self, idx):
        target_sample, label = self.target.__getitem__(idx)
        background_sample, _ = self.background.__getitem__(idx)
        return [target_sample, background_sample], label

    def __len__(self):
        return len(self.target)
