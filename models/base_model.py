from abc import ABC, abstractmethod

from torch import nn


class BaseModel(ABC, nn.Module):
    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass
