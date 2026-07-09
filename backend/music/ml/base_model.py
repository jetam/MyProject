from abc import ABC, abstractmethod
import torch.nn as nn


class BaseMusicModel(nn.Module, ABC):
  @abstractmethod
  def fineTune(self, song): ...


  @abstractmethod
  def generate(self, seedSong, length=200): ...

