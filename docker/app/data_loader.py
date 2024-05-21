import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from class_solar_panel_dataset import SolarPanelDataset

#Function to get data loaders for training and testing datasets.
def get_data_loaders(root_path, batch_size, test_batch_size):

    # Get category paths
    categories_paths = os.listdir(root_path)
    print('DATA_LOADER-1:', categories_paths);
    categories_paths = [os.path.join(root_path, cat_path) for cat_path in categories_paths]
    print('DATA_LOADER-2: Path ', categories_paths);

    # Print number of files in each category
    for cat_path in categories_paths:
        for _, _, files in os.walk(cat_path):
            print("DATA_LOADER-3: {}: {}".format(cat_path, len(files)))

    # Get all image paths
    images_paths = []
    for cat_path in categories_paths:
        for root, _, files in os.walk(cat_path):
            cd_images = [os.path.join(root, file) for file in files]
            [images_paths.append(img) for img in cd_images]
    images_paths = sorted(images_paths)

    n_images = len(images_paths)
    print('DATA_LOADER-4: Number Images total', n_images);
    images_idx = range(0, n_images, 2)
    print('DATA_LOADER-4: Number Images distinct', len(images_idx));

    # Split indices into training and testing sets
    train_idx, test_idx = train_test_split(images_idx, test_size=0.15)
    print("DATA_LOADER-5: Number Images train {}, Number images test {} ".format(len(train_idx), len(test_idx)));

    image_new_size = (256, 256)

    # Define transformations for training and testing sets
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_new_size)
    ])

    mask_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_new_size)
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_new_size)
    ])

    
    # Create datasets and data loaders
    trainset = SolarPanelDataset(images_paths, train_idx, transform=train_transforms, target_transform=mask_transforms)
    testset = SolarPanelDataset(images_paths, test_idx, transform=test_transforms, target_transform=mask_transforms)

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset=testset,
        batch_size=test_batch_size,
        shuffle=False,
    )

    return train_loader, test_loader
