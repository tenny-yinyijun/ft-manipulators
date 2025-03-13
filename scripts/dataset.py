import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
    
class FTWSDataset(Dataset):
    """FTWS Calculation Dataset"""

    def __init__(self, phys_configs, ftws_maps):
        self.phys_configs = phys_configs
        self.ftws_maps = ftws_maps

    def __len__(self):
        return len(self.phys_configs)

    def __getitem__(self, index):
        cfg_sample = self.phys_configs[index]
        target = self.ftws_maps[index]
        return cfg_sample, target