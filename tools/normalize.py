#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import cv2
import glob
import time

import torch
import torchvision.datasets as datasets
from torchvision import transforms as T
from torch.utils.data import Dataset

from PIL import Image
from typing import Tuple
from tqdm import tqdm

def format_time(elapsed_time) -> str:
    days = 0
    if elapsed_time >= 86400:
        days = int(elapsed_time / 86400)
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    return str(days) + ":" + elapsed_str

# Define the custom dataset
class SolarPanelDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

def compute_v0(source_dir) -> Tuple[float, float]:
    ext = ['bmp'] #, 'png', 'jpg', 'gif']    # Add image formats here

    files = []
    [files.extend(glob.glob(os.path.join(source_dir, '*.' + e))) for e in ext]
    rgb_values = np.concatenate(
        [Image.open(img).getdata() for img in files], 
        axis=0
    ) / 255.

    # rgb_values.shape == (n, 3), 
    # where n is the total number of pixels in all images, 
    # and 3 are the 3 channels: R, G, B.

    # Each value is in the interval [0; 1]
    mu_rgb = np.mean(rgb_values, axis=0)  # mu_rgb.shape == (3,)
    std_rgb = np.std(rgb_values, axis=0)  # std_rgb.shape == (3,)
    return mu_rgb, std_rgb


def compute_v1(source_dir) -> Tuple[float, float]:
    transform = T.Compose([
        #transforms.Resize((256, 256)),
        T.ToTensor(),
    ])

    device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
    if sys.platform == "darwin" and not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")

    #if not torch.backends.mps.is_available():
    #    if not torch.backends.mps.is_built():
    #        print("MPS not available because the current PyTorch install was not "
    #            "built with MPS enabled.")
    #    else:
    #        print("MPS not available because the current MacOS version is not 12.3+ "
    #            "and/or you do not have an MPS-enabled device on this machine.")
    #
    #else:
    #    device = torch.device("mps")

    dataset = SolarPanelDataset(image_dir=source_dir, transform=transform)

    print(f"Start computing mean and std of {len(dataset)} images...")

    # Based on https://datascience.stackexchange.com/questions/77084/how-imagenet-mean-and-std-derived
    total_computed = 0
    last_log_percentage = 0.0
    means = []
    stds = []
    for img in dataset:
        img.to(device)
        means.append(torch.mean(img))
        stds.append(torch.std(img))
        total_computed = total_computed + 1
        total_percentage = 100 * total_computed / len(dataset)
        if (total_percentage - last_log_percentage > 5.0):
            last_log_percentage = total_percentage = 100 * total_computed / len(dataset)
            print(f"Computing {total_percentage:.2f}%", end='\r', flush=True)

    print("Computed 100% - Done!")

    mean = torch.mean(torch.tensor(means))
    std = torch.mean(torch.tensor(stds))
    return mean, std

def compute_v1_1(source_dir) -> Tuple[float, float]:
    transform = T.Compose([
        #transforms.Resize((256, 256)),
        T.ToTensor(),
    ])

    device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
    if sys.platform == "darwin" and not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")

    dataset = SolarPanelDataset(image_dir=source_dir, transform=transform)

    print(f"Start computing mean and std of {len(dataset)} images...")

    total_computed = 0
    last_log_percentage = 0.0
    means = []
    stds = []
    for img in dataset:
        img.to(device)
        means.append(torch.mean(img, dim=[1, 2]))
        stds.append(torch.std(img, dim=[1, 2]))
        total_computed = total_computed + 1
        total_percentage = 100 * total_computed / len(dataset)
        if (total_percentage - last_log_percentage > 5.0):
            last_log_percentage = total_percentage = 100 * total_computed / len(dataset)
            print(f"Computing {total_percentage:.2f}%", end='\r', flush=True)

    print("Computed 100% - Done!")

    mean = torch.mean(torch.stack(means, dim=0))
    std = torch.mean(torch.stack(stds, dim=0))
    return mean, std

def compute_v1_2(source_dir) -> Tuple[torch.Tensor, torch.Tensor]:
    transform = T.Compose([
        #transforms.Resize((256, 256)),
        T.ToTensor(),
    ])

    device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
    if sys.platform == "darwin" and not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")

    dataset = SolarPanelDataset(image_dir=source_dir, transform=transform)

    print(f"Start computing mean and std of {len(dataset)} images...")

    total_computed = 0
    last_log_percentage = 0.0
    means = []
    stds = []
    for img in dataset:
        img.to(device)
        means.append(torch.mean(img, dim=[1, 2]))
        stds.append(torch.std(img, dim=[1, 2]))
        total_computed = total_computed + 1
        total_percentage = 100 * total_computed / len(dataset)
        if (total_percentage - last_log_percentage > 5.0):
            last_log_percentage = total_percentage = 100 * total_computed / len(dataset)
            print(f"Computing {total_percentage:.2f}%", end='\r', flush=True)

    print("Computed 100% - Done!")
  
    dataset_means = torch.stack(means, dim=0).mean(dim=0)
    dataset_std = torch.stack(stds, dim=0).mean(dim=0)

    return dataset_means, dataset_std
    #mean_dataset[0] = torch.mean(torch.stack(means, dim=0))
    ##a = torch.mean(dim=0))
    #a = torch.stack(means, dim=0)
    #b = torch.stack(means, dim=1)
    #mean_dataset[1] = torch.mean(torch.stack(means, dim=1))
    #mean_dataset[2] = torch.mean(torch.stack(means, dim=2))
    #type(mean_dataset)
#
    #std_dataset = np.zeros(3)
    #std_dataset[0] = torch.mean(torch.stack(stds, dim=0))
    #std_dataset[1] = torch.mean(torch.stack(stds, dim=1))
    #std_dataset[2] = torch.mean(torch.stack(stds, dim=2))
    #type(std_dataset)
#
    #mean = torch.mean(torch.stack(means, dim=0))
    #std = torch.mean(torch.stack(stds, dim=0))
    #return mean, std

import matplotlib.pyplot as plt 

#
# Manel version
#
def compute_v2(source_dir) -> Tuple[float, float]:
    # Let's find characteristics of the images (mean,std) for normalizing all the images in
    # the transform process.
    mean_acum=np.zeros(3)
    std_acum=np.zeros(3)
    total_files = 0
    for filename in os.listdir(source_dir):
        img_path = os.path.join(source_dir, filename)
        image = plt.imread(img_path)
        mean_acum= mean_acum + np.mean(image,axis=(0, 1))
        std_acum = std_acum + np.var(image,axis=(0, 1))
        #print ("media aucumulada", mean_acum)
        total_files = total_files + 1
    
    print("ficheros mirados", total_files)
    mean_dataset= mean_acum / total_files # shouldn't use total_files, should totoal ammount of pixels!
    std_dataset= std_acum / total_files
    return mean_dataset, std_dataset

from torchvision import transforms
def compute_v3(source_dir):
    transform = transforms.ToTensor()

    # Initialize variables to store the sum and sum of squares
    sum_rgb = torch.zeros(3)
    sum_sq_rgb = torch.zeros(3)
    num_pixels = 0
    source_files = os.listdir(source_dir)
    total_files = len(source_files)
    computed_files = 0
    # Loop through images and compute the sums
    for filename in source_files:
        img_path = os.path.join(source_dir, filename)
        img = Image.open(img_path)
        img_tensor = transform(img)

        # Add to the sums
        sum_rgb += img_tensor.sum(dim=(1, 2))
        sum_sq_rgb += (img_tensor ** 2).sum(dim=(1, 2))
        num_pixels += img_tensor.shape[1] * img_tensor.shape[2]

        computed_files = computed_files + 1
        total_percentage = 100 * computed_files / total_files
        print(f"Computing {computed_files}/{total_files} {total_percentage:.2f}%", end='\r', flush=True)

    # Calculate mean and variance
    mean_rgb = sum_rgb / num_pixels
    var_rgb = (sum_sq_rgb / num_pixels) - (mean_rgb ** 2)

    return mean_rgb, var_rgb

def compute_v33(source_dir):
    device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
    if sys.platform == "darwin" and not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")

    #data_transforms = transforms.ToTensor()
    image_resize = 256
    print(f"Warning: resizing input dataset to {image_resize}x{image_resize}")
    data_transforms = transforms.Compose([
        transforms.Resize((image_resize, image_resize)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(source_dir, transform=data_transforms)
    batch_size = 64
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize variables to store the sum and sum of squares
    sum_rgb = torch.zeros(3)
    sum_sq_rgb = torch.zeros(3)
    num_pixels = 0

    for images, _ in tqdm(loader):
        images.to(device)
        batch_size, num_channels, height, width = images.shape

        # Add to the sums
        sum_rgb += images.sum(axis=(0, 2, 3)) #dim=(1, 2))
        sum_sq_rgb += (images ** 2).sum(axis=(0, 2, 3)) #dim=(1, 2))
        #num_pixels += batch_size * images.shape[1] * images.shape[2]
        num_pixels += batch_size * images.shape[2] * images.shape[3]

    # Calculate mean and variance
    mean_rgb = sum_rgb / num_pixels
    var_rgb = (sum_sq_rgb / num_pixels) - (mean_rgb ** 2)

    return mean_rgb, var_rgb

#
# https://saturncloud.io/blog/how-to-normalize-image-dataset-using-pytorch/
#
# PV_ALL: Mean=tensor(1.9415e-07), std=(1.0597e-07)
# Mean=1.9412355811709858e-07 Std=1.0593041110951162e-07
#
def get_mean_std_v0(loader, device):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        images.to(device)
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()

    mean /= num_pixels
    std /= num_pixels

    return mean, std

#
# PV-ALL: Mean=tensor([6.4775e-08, 6.5530e-08, 6.3815e-08]) Std=tensor([3.7019e-08, 3.3223e-08, 3.5673e-08])
#
def get_mean_std_v1(loader, device):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for images, _ in tqdm(loader):
        images.to(device)
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3))
        std += images.std(axis=(0, 2, 3))

    mean /= num_pixels
    std /= num_pixels

    return mean, std

def compute_v4(source_dir):
    device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
    if sys.platform == "darwin" and not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")

    data_transforms = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(source_dir, transform=data_transforms)

    batch_size = 64
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #mean, std = get_mean_std_v0(loader, device)
    mean, std = get_mean_std_v1(loader, device)
   
    return mean, std

#
# https://kozodoi.me/blog/20210308/compute-image-stats
#
# Mean=tensor([0.2713, 0.2744, 0.2673]) Std=tensor([0.1558, 0.1399, 0.1502])
#
def get_mean_std_v2(loader, device):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    psum = torch.zeros(3)
    psum_sq = torch.zeros(3)
    for images, _ in tqdm(loader):
        images.to(device)
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        psum += images.sum(axis=[0, 2, 3])
        psum_sq += (images**2).sum(axis=[0, 2, 3])

    total_mean = psum / num_pixels
    total_var = (psum_sq / num_pixels) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    return total_mean, total_std

def compute_v5(source_dir, transform_resize_dim:int = None):
    device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
    if sys.platform == "darwin" and not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")

    if transform_resize_dim is not None:
        print(f"Warning: resizing input dataset to {transform_resize_dim}x{transform_resize_dim}")
        data_transforms = transforms.Compose([
            transforms.Resize((transform_resize_dim, transform_resize_dim)),
            transforms.ToTensor(),
        ])
    else:
        data_transforms = transforms.ToTensor()

    dataset = datasets.ImageFolder(source_dir, transform=data_transforms)

    batch_size = 64
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    mean, std = get_mean_std_v2(loader, device)
   
    return mean, std

#
# BAD?
#
def get_mean_std_v3(loader, device):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for images, _ in tqdm(loader):
        images.to(device)
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3))
        std += images.std(axis=(0, 2, 3))

    #mean /= num_pixels
    #std /= num_pixels

    return mean, std

def compute_v6(source_dir):
    device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
    if sys.platform == "darwin" and not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")

    data_transforms = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(source_dir, transform=data_transforms)

    batch_size = 64
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    mean, std = get_mean_std_v3(loader, device)
   
    return mean, std

def main() -> int:
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-i", "--input", required=True, help="input folder")
    args = vars(ap.parse_args())

    images_fullpath = os.path.realpath(args['input'])
    
    
    #start_time = time.perf_counter()
    #mu_rgb, std_rgb = compute_v4(images_fullpath)
    #elapsed = time.perf_counter() - start_time
    #print(f"V4 Took: {format_time(elapsed)}")
    #print(f"V4 -> Mean={mu_rgb} Std={std_rgb}\n")
    #
    ## ZEONODO-split (train only): V4 -> Mean=tensor([3.4744e-08, 3.5110e-08, 3.0485e-08]) Std=tensor([2.0893e-08, 1.9309e-08, 1.8655e-08])
    ## ZEONODO-split (train only with 256 resize): v4 ->
    
    ####################################################################
    
    start_time = time.perf_counter()
    mu_rgb, std_rgb = compute_v5(images_fullpath, transform_resize_dim=256)
    elapsed = time.perf_counter() - start_time
    print(f"V5 Took: {format_time(elapsed)}")
    print(f"V5 -> Mean={mu_rgb} Std={std_rgb}\n")
    
    # ZEONODO-split (train only): V5 -> Mean=tensor([0.3549, 0.3587, 0.3114], dtype=torch.float64) Std=tensor([0.2137, 0.1975, 0.1908], dtype=torch.float64)
    # ZEONODO-split (train only with 256 resize): V5 -> Mean=tensor([0.3549, 0.3587, 0.3114], dtype=torch.float64) Std=tensor([0.2060, 0.1894, 0.1828], dtype=torch.float64)
    
    ####################################################################
    
    start_time = time.perf_counter()
    mu_rgb, std_rgb = compute_v33(images_fullpath) # Manel
    elapsed = time.perf_counter() - start_time
    print(f"V33 Took: {format_time(elapsed)}")
    print(f"V33 -> Mean={mu_rgb} Std={std_rgb}\n")
    
    # ZEONODO-split (train only): V33 -> Mean=tensor([0.3549, 0.3587, 0.3114]) Std=tensor([0.0457, 0.0390, 0.0364])
    # ZEONODO-split (train only with 256 resize): V33 -> Mean=tensor([0.3549, 0.3587, 0.3114]) Std=tensor([0.0425, 0.0359, 0.0334])
    
    return
    
    ##########################################
    
    start_time = time.perf_counter()
    #mu_rgb, std_rgb = compute_v3(os.path.join(images_fullpath, "solar_panels")) # Manel
    mu_rgb, std_rgb = compute_v3(os.path.join(images_fullpath, "control")) # Manel
    elapsed = time.perf_counter() - start_time
    print(f"\nV3 Took: {format_time(elapsed)}")
    print(f"Mean={mu_rgb} Std={std_rgb}\n")

    start_time = time.perf_counter()
    mu_rgb, std_rgb = compute_v33(images_fullpath) # Manel
    elapsed = time.perf_counter() - start_time
    print(f"V33 Took: {format_time(elapsed)}")
    print(f"Mean={mu_rgb} Std={std_rgb}\n")

    #mu_rgb, std_rgb = compute_v1_2(images_fullpath) # David (bad)

    start_time = time.perf_counter()
    mu_rgb, std_rgb = compute_v4(images_fullpath)
    elapsed = time.perf_counter() - start_time
    print(f"V4 Took: {format_time(elapsed)}")
    print(f"Mean={mu_rgb} Std={std_rgb}\n")

    start_time = time.perf_counter()
    mu_rgb, std_rgb = compute_v5(images_fullpath)
    elapsed = time.perf_counter() - start_time
    print(f"V5 Took: {format_time(elapsed)}")
    print(f"Mean={mu_rgb} Std={std_rgb}\n")

    start_time = time.perf_counter()
    mu_rgb, std_rgb = compute_v6(images_fullpath) # Bad?
    elapsed = time.perf_counter() - start_time
    print(f"V6 Took: {format_time(elapsed)}")
    print(f"Mean={mu_rgb} Std={std_rgb}\n")

    return 0

import torch
from torchvision import transforms
from PIL import Image 

if __name__ == '__main__':
    
    H, W = 32, 32
    img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)
    
    img = Image.open(r"/Users/davidpastor/workspace/UPC/DeepLearning/aidl-2024-solar-panels-project/datasets_old/Zenodo-Multi-resolution dataset for photovoltaic panel segmentation from satellite and aerial imagery/PV01/PV01_Rooftop_Brick/PV01_324942_1203839.bmp")
    imgA = img

    train_mean=[0.485, 0.456, 0.406]
    train_std=[0.229, 0.224, 0.225]
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std) # Mean and std resizing train dataset to 256x256
    ])
    #imgB = transformations(img.to(torch.float))
    imgB = transformations(img)
    
    denorm_mean=tuple(-m / s for m, s in zip(train_mean, train_std))
    denorm_std=tuple(1.0 / s for s in train_std)
    inverse_normalization_transform = transforms.Compose([
        transforms.Normalize(
            mean=denorm_mean,
            std=denorm_std
        ),
        lambda x: x*255  # https://stackoverflow.com/questions/65469814/convert-image-to-tensor-with-range-0-255-instead-of-0-1
    ])
    imgC = inverse_normalization_transform(imgB)
    
    # FromFloat(max_value=255, dtype="uint8"), -> https://github.com/albumentations-team/albumentations/blob/main/albumentations/augmentations/functional.py#L861
    #imgC = (imgC * 255.0)

    fig, axes = plt.subplots(nrows=1, ncols=3)
    #axes[0].imshow(imgA.permute(1, 2, 0).numpy())  # Original image
    axes[0].imshow(imgA)  # Original image
    axes[1].imshow(imgB.permute(1, 2, 0).to(torch.uint8).numpy())
    axes[2].imshow(imgC.permute(1, 2, 0).to(torch.uint8).numpy())
    plt.tight_layout()
    plt.show()
    sys.exit(0)
    sys.exit(main())  # next section explains the use of sys.exit
