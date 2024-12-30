from model.model import Wipe
import loss as loss_

from scipy.stats import multivariate_normal
from sklearn.preprocessing import minmax_scale
import numpy as np

from multiprocessing import cpu_count
import json
import pickle
import torch

nSubThreads = cpu_count() // 2
torch.backends.cudnn.benchmark = True


def keypoints2Jhms(
  keypoints: np.ndarray,
  gaussian: float,
) -> np.ndarray:
  outRes: tuple[int, int] = (56, 100)
  originRes: tuple[int, int] = (1080, 1920)
  scaleFactor: tuple[float, float] = (
    outRes[0] / originRes[0],
    outRes[1] / originRes[1],
  )

  heatmapList = []
  for point in keypoints:
    if point[0] > 0:
      x = point[1] * scaleFactor[1]
      y = point[2] * scaleFactor[0]
      heatmap = xy2Heatmap((x, y), outRes, gaussian)
      shape = heatmap.shape
      heatmap = minmax_scale(heatmap.reshape((1, -1)), axis=1).reshape(shape)
    else:
      heatmap = np.zeros(outRes)
    heatmapList.append(heatmap)

  return np.array(heatmapList)


def xy2Heatmap(
  xy: tuple[float, float],
  shape: tuple[int, int],
  gaussian: float,
) -> np.ndarray:
  if gaussian:
    cov = np.eye(2) * gaussian
    k = multivariate_normal(mean=xy, cov=cov)

    x = np.arange(0, shape[1])
    y = np.arange(0, shape[0])
    xx, yy = np.meshgrid(x, y)

    xxyy = np.stack([xx.ravel(), yy.ravel()]).T
    heatmap = k.pdf(xxyy).reshape(shape)

  else:
    heatmap = np.zeros(shape)
    heatmap[xy[1], xy[0]] = 1

  return heatmap


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  nInChannel = 114
  nOutChannel = 17 + 1

  model = Wipe(nInChannel, nOutChannel).to(device)
  optimizer = torch.optim.Adam(params=model.parameters())

  mag = torch.tensor(np.load("sample/mag.npy").mean(0, keepdims=True))

  with open("sample/mmpose.json", "rt") as f:
    mmpose = json.load(f)[0]
  xyNp = np.array(mmpose["keypoints"])
  confidenceNp = np.array(mmpose["keypoint_scores"])
  keypoints = np.concatenate((confidenceNp[:, np.newaxis], xyNp), axis=1)
  jhms = torch.tensor(keypoints2Jhms(keypoints, 0.7))
  jhms = jhms.type(torch.float32).unsqueeze(0).to(device)

  with open("sample/detectron2.pkl", "rb") as f:
    detectron2: dict[str, np.ndarray] = pickle.load(f)
  bbox = detectron2["zoomedBbox"]
  mask = torch.tensor(detectron2["zoomedMask"])
  mask = mask.type(torch.float32).unsqueeze(0).to(device)

  criterion = loss_.MyLoss(device, (1, 1), 2e-4, 1, 1e-7)

  y = model(mag.type(torch.float32).to(device))
  maskY = y[:, 0, :, :]
  jhmsY = y[:, 1:, :, :]

  loss, lossVal = criterion(jhmsY, jhms, maskY, mask)

  print(f"Loss: {lossVal}")
