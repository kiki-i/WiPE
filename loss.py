import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLoss(nn.Module):
  def __init__(
    self,
    device: torch.device,
    mWeightJhmsParam: tuple[float, float],
    maskWeight: float,
    jhmsRegularNrom: int,
    jhmsRegularWeight: float,
  ):
    super().__init__()
    self.device = device

    self.jhmMweightK = mWeightJhmsParam[0]
    self.jhmMweightB = mWeightJhmsParam[1]
    self.jhmsRegularNrom = jhmsRegularNrom
    self.jhmsRegularWeight = jhmsRegularWeight

    self.maskWeight = maskWeight

  def forward(
    self,
    jhms: torch.Tensor,
    jhmsTarget: torch.Tensor,
    mask: torch.Tensor,
    maskTarget: torch.Tensor,
  ) -> tuple[torch.Tensor, tuple]:
    jhmsLoss = F.smooth_l1_loss(jhms, jhmsTarget, reduction="none")
    if self.jhmMweightK and self.jhmMweightB:
      thr = 0.5
      jhmsII = (
        torch.zeros(jhmsTarget.shape, device=self.device)
        .masked_fill(jhmsTarget >= thr, 1)
        .masked_fill(jhmsTarget < thr, 0)
      )
      mWeightJhm = self.jhmMweightK * jhmsTarget + self.jhmMweightB * jhmsII
      jhmsLoss = mWeightJhm * F.smooth_l1_loss(jhms, jhmsTarget, reduction="none")

    jhmsLoss = jhmsLoss.mean()

    if self.jhmsRegularWeight:
      jhmsRegularDim = tuple(range(1, jhms.dim()))
      jhmsRegular = jhms.norm(self.jhmsRegularNrom, jhmsRegularDim)
      jhmsLoss = jhmsLoss + self.jhmsRegularWeight * jhmsRegular.mean()

    if self.maskWeight:
      maskLoss = F.binary_cross_entropy(mask, maskTarget)
    else:
      maskLoss = torch.zeros(1, device=self.device)

    totalLoss = jhmsLoss + self.maskWeight * maskLoss
    lossVal = (totalLoss.item(), jhmsLoss.item(), maskLoss.item())
    return totalLoss, lossVal
