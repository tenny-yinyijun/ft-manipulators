import torch
from torch import nn

device = ("cuda" if torch.cuda.is_available() else "cpu")

class FTWSNetwork2D(nn.Module):
    def __init__(self, res=40):
      super(FTWSNetwork2D,self).__init__()
      self.flatten = nn.Flatten()
      self.combine_net = nn.Sequential(
           nn.Linear(4, 128),
          nn.ReLU(),
          nn.Linear(128, 512),
          nn.ReLU(),
          nn.Linear(512, res * res)
      )

    def forward(self, x):
        result = self.combine_net(x)
        min_val = result.min()
        max_val = result.max()

        # Normalize the tensor
        result = (result - min_val) / (max_val - min_val)
        return result

class FTWSNetwork3D(nn.Module):
    def __init__(self, res=10):
      super(FTWSNetwork3D,self).__init__()
      self.flatten = nn.Flatten()
      self.combine_net = nn.Sequential(
           nn.Linear(7, 128),
          nn.ReLU(),
          nn.Linear(128, 512),
          nn.ReLU(),
          nn.Linear(512, res * res * res)
      )

    def forward(self, x):
        result = self.combine_net(x)
        min_val = result.min()
        max_val = result.max()

        # Normalize the tensor
        result = (result - min_val) / (max_val - min_val)
        return result