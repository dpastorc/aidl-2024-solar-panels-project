import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SolarPanelDataset(Dataset):
    #Custom dataset class for solar panel images and masks.

    def __init__(self, images_paths, idx, transform=None, target_transform=None):
        self.images_paths = images_paths
        self.idx = idx
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        oneimg_path = self.images_paths[self.idx[index]]
        onemask_path = self.images_paths[self.idx[index] + 1]
        image = plt.imread(oneimg_path)
        mask = np.expand_dims(plt.imread(onemask_path), axis=(-1))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask
