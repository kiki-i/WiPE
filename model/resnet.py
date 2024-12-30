from torch import Tensor

import torch.nn as nn


def convLayer(nChannels, kernelSize) -> nn.Module:
  return nn.Conv2d(nChannels, nChannels, kernelSize, padding="same", bias=False)


class BasicBlock(nn.Module):
  def __init__(
    self,
    nChannels: int,
    kernelSize: int = 3,
    activation=nn.ReLU(inplace=True),
  ) -> None:
    super().__init__()

    self.conv1 = nn.Sequential(
      convLayer(nChannels, kernelSize),
      nn.BatchNorm2d(nChannels),
    )
    self.activation = activation
    self.conv2 = nn.Sequential(
      convLayer(nChannels, kernelSize),
      nn.BatchNorm2d(nChannels),
    )

  def forward(self, x: Tensor) -> Tensor:
    residual = x

    out = self.conv1(x)
    out = self.activation(out)
    out = self.conv2(out)

    out += residual
    out = self.activation(out)

    return out
