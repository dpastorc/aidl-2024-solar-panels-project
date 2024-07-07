# -*- coding: utf-8 -*-
"""AIDL2024_Solar_Panel_Detector_Dataset.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sy26gaC8ESiTat8QP7opg6DwUYG4ilfx

# AIDL 2024 - Solar Panel Detector Project: **Dataset** (step 1/4)

Notebook created for the Postgraduate course in artificial intelligence with deep learning in UPC School (2024).
Project members: Manel Toril, Raphael Gagliardi, David Pastor and Daniel Domingo. Supervisor: Amanda Duarte.

The purpose of this notebook is to download and pre-process datasets that contain satellite or aerial images of installed solar panels and corresponding masks (ground truth). We have identified 3 differents datasets named PV01, PV03 and Google.

## How to run this code:

1. Download dataset source files from the following locations:
*   **PV01**: https://zenodo.org/records/5171712
*   **PV03**: https://www.kaggle.com/datasets/salimhammadi07/solar-panel-detection-and-identification?resource=download
*   **Google**: https://zenodo.org/records/7358126


2. Place the zip files under the root folder
*   **PV01.zip**
*   **PV03.zip**
*   **GOOGLE.zip**

3. Under **Quick Configuration**, set the *Train*, *Validation* and *Test* split percentages

4. Execute all cells in the notebook from top, in sequential order

5. Download the resulting zip files, each containing *images* and *masks* subfolders under *test*, *train* and *val* folders:

*   **PV01-split.zip**: PV01 dataset split in test, train and validation sets
*   **PV03-CROP-split.zip**: PV03 dataset cropped to 256px patches, and split in test, train and validation sets
*   **PV-ALL-split.zip**: PV01 and PV03 (cropped to 256px patches) datasets split in test, train and validation sets
*   **GOOGLE-split.zip**: GOOGLE dataset split in test, train and validation sets

# Quick configuration
"""

# Dataset split ratios
train_ratio = 0.6           # Train set (0.6 = 60%)
val_ratio = 0.2             # Validation set (0.2 = 20%)
test_ratio = 0.2            # Test set (0.2 = 20%)

"""# Parameters"""

# Paths
local_dir = '/content/'                               # Root folder
pv01_zip_file_path = local_dir + "PV01.zip"           # PV01 zip file path
pv03_zip_file_path = local_dir + "PV03.zip"           # PV03 zip file path
google_zip_file_path = local_dir + "GOOGLE.zip"       # GOOGLE zip file path

"""# Libraries"""

import os
import zipfile
import shutil
import random
import re
from PIL import Image
import math

"""# Supporting functions

**Crop functions**
"""

# Genarate the filename
def generate_output_filename(fullpath_filename:str, tile_number:int, crop_region) -> str:
  path, filename = os.path.split(fullpath_filename)
  base_filename, extension = os.path.splitext(filename)
  #return (base_filename
  #    + '-{}x{}x{}x{}'.format(crop_region[0],crop_region[1],crop_region[2],crop_region[3])
  #    + extension)
  t = re.subn('_label$', f'_{tile_number}_label', base_filename)
  base_filename = t[0]
  if t[1] == 0:
      base_filename += f'_{tile_number}'
  return base_filename + extension

# Function to crop
def crop(input_folder : str, output_folder : str, num_tiles : int, verbose = False) -> list:
  tiles = list()
  num_rows = int(math.sqrt(num_tiles))
  num_cols = int(math.sqrt(num_tiles))
  os.makedirs(output_folder, exist_ok=True)
  dirs = os.listdir(input_folder)
  for item in dirs:
    input_file = os.path.join(input_folder, item)
    if os.path.isfile(input_file):
      im = Image.open(input_file)
      w = im.size[0]
      h = im.size[1]
      tile_number = 1
      tiles_images = list()
      for j in range(num_cols):
        for i in range(num_rows):
          crop_left = i * int(w / num_rows)
          crop_top = j * int(h / num_cols)
          crop_right = crop_left + int(w / num_rows)
          crop_bottom = crop_top + int(h / num_cols)
          crop_region = (crop_left, crop_top, crop_right, crop_bottom)
          imCrop = im.crop(crop_region)
          output_filename = os.path.join(output_folder, generate_output_filename(input_file, tile_number, crop_region))
          if (verbose):
              print(f"Filename: {output_filename} -> {crop_region}")
          imCrop.save(output_filename, 'JPEG', quality=90)
          tiles_images.append(imCrop)
          tile_number = tile_number + 1
          tile_number = tile_number % num_tiles
          if tiles_images:
            tiles.append(tiles_images)
  return tiles

# Function to traverse all files in the subdirectories and separate images and masks
def move_files_label(root_dir, images_dir, masks_dir):
    # Check if the destination directory exists, if not, create it
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)

    # Traverse all files in the subdirectories
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.bmp'):
                file_path = os.path.join(subdir, file)
                if '_label' in file:
                    # Move files with '_label' in the name to the masks folder
                    shutil.move(file_path, os.path.join(masks_dir, file))
                else:
                    # Move other .bmp files to the images folder
                    shutil.move(file_path, os.path.join(images_dir, file))

# Function to move/copy all files from a source directory to a destination directory
def move_files(source_directory, destination_directory, copy=True):
    # Check if the destination directory exists, if not, create it
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # List all files in the source directory
    for file_name in os.listdir(source_directory):
        source = os.path.join(source_directory, file_name)
        destination = os.path.join(destination_directory, file_name)

        # Move each file
        if copy:
            shutil.copy2(source, destination)
        else:
            shutil.move(source, destination)

"""**Split functions**"""

# Function to split the dataset
def split_dataset(images_path, masks_path):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Train, validation, and test ratios must sum to 1"

    # List all images and masks
    all_images = os.listdir(images_path)
    all_masks = os.listdir(masks_path)

    # Ensure the images and masks are sorted to match correctly
    all_images.sort()
    all_masks.sort()

    # Pair images and masks to keep them together
    data = list(zip(all_images, all_masks))

    # Shuffle the data
    random.seed(42)
    random.shuffle(data)

    # Calculate split sizes
    total_size = len(data)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Split the data
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # Separate the pairs into individual lists (unzip the data)
    train_images, train_masks = zip(*train_data)
    val_images, val_masks = zip(*val_data)
    test_images, test_masks = zip(*test_data)

    return (train_images, train_masks), (val_images, val_masks), (test_images, test_masks)

# Function to move files to their corresponding directories
def move_files_list(file_list, source_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for file_name in file_list:
        shutil.move(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))

# Function to clean google diretoctory
def clean_images_without_masks(image_directory, mask_directory):
    # List all files in both directories
    image_files = set(os.listdir(image_directory))
    mask_files = set(os.listdir(mask_directory))

    # Iterate over image files
    for image_file in image_files:
        image_name, image_extension = os.path.splitext(image_file)

        # Check if there is a corresponding mask
        corresponding_found = False
        for mask_file in mask_files:
            mask_name, mask_extension = os.path.splitext(mask_file)
            if image_name == mask_name:
                corresponding_found = True
                break

        # If no corresponding mask is found, remove the image
        if not corresponding_found:
            image_path = os.path.join(image_directory, image_file)
            os.remove(image_path)
            print(f'Removed: {image_path}')

"""**Zip function**"""

def zip_directories(output_zip, *directories):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for directory in directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.join(directory, '..'))
                    zipf.write(file_path, arcname)

"""# Execution"""

#  Phase 1: Unzip and organize directories

print('Phase 1: Unzip and organizing the directories')

#Unzip the files
with zipfile.ZipFile(pv01_zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(local_dir)
with zipfile.ZipFile(pv03_zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(local_dir)
with zipfile.ZipFile(google_zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(local_dir)

#Create destination folder
os.makedirs(local_dir + "PV01-split")
os.makedirs(local_dir + "PV03-CROP-split")
os.makedirs(local_dir + "PV-ALL-split")
os.makedirs(local_dir + "GOOGLE-split")
os.makedirs(local_dir + "result")

#PV01 - Move files to destination path
move_files_label(local_dir + "PV01", local_dir + "PV01/img", local_dir + "PV01/mask")

#PV03 - CROP
dirs = os.listdir(local_dir + "PV03")
for item in dirs:
  input_file = os.path.join(local_dir + "PV03", item)
  crop(input_file, local_dir + "PV03-CROP", 16)

#PV03 - Move files to destination path
move_files_label(local_dir + "PV03-CROP", local_dir + "PV03-CROP/img", local_dir + "PV03-CROP/mask")

#  Phase 2: Split files

print('Phase 2: Splitting files')

# PV01 - Split the dataset
(train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = split_dataset(local_dir + "PV01/img", local_dir + "PV01/mask")

# PV01 - Move files to their respective directories
move_files_list(train_images, local_dir + "PV01/img", local_dir + "PV01-split/train/images")
move_files_list(train_masks, local_dir + "PV01/mask", local_dir + "PV01-split/train/masks")
move_files_list(val_images, local_dir + "PV01/img", local_dir + "PV01-split/val/images")
move_files_list(val_masks, local_dir + "PV01/mask", local_dir + "PV01-split/val/masks")
move_files_list(test_images, local_dir + "PV01/img", local_dir + "PV01-split/test/images")
move_files_list(test_masks, local_dir + "PV01/mask", local_dir + "PV01-split/test/masks")

# PV03 - Split the dataset
(train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = split_dataset(local_dir + "PV03-CROP/img", local_dir + "PV03-CROP/mask")

# PV03 - Move files to their respective directories
move_files_list(train_images, local_dir + "PV03-CROP/img", local_dir + "PV03-CROP-split/train/images")
move_files_list(train_masks, local_dir + "PV03-CROP/mask", local_dir + "PV03-CROP-split/train/masks")
move_files_list(val_images, local_dir + "PV03-CROP/img", local_dir + "PV03-CROP-split/val/images")
move_files_list(val_masks, local_dir + "PV03-CROP/mask", local_dir + "PV03-CROP-split/val/masks")
move_files_list(test_images, local_dir + "PV03-CROP/img", local_dir + "PV03-CROP-split/test/images")
move_files_list(test_masks, local_dir + "PV03-CROP/mask", local_dir + "PV03-CROP-split/test/masks")

# PVALL - COPY FILES FROM PV01 AND PV03
move_files(local_dir + "PV01-split/train/images", local_dir + "PV-ALL-split/train/images", copy=True)
move_files(local_dir + "PV03-CROP-split/train/images", local_dir + "PV-ALL-split/train/images", copy=True)
move_files(local_dir + "PV01-split/train/masks", local_dir + "PV-ALL-split/train/masks", copy=True)
move_files(local_dir + "PV03-CROP-split/train/masks", local_dir + "PV-ALL-split/train/masks", copy=True)
move_files(local_dir + "PV01-split/val/images", local_dir + "PV-ALL-split/val/images", copy=True)
move_files(local_dir + "PV03-CROP-split/val/images", local_dir + "PV-ALL-split/val/images", copy=True)
move_files(local_dir + "PV01-split/val/masks", local_dir + "PV-ALL-split/val/masks", copy=True)
move_files(local_dir + "PV03-CROP-split/val/masks", local_dir + "PV-ALL-split/val/masks", copy=True)
move_files(local_dir + "PV01-split/test/images", local_dir + "PV-ALL-split/test/images", copy=True)
move_files(local_dir + "PV03-CROP-split/test/images", local_dir + "PV-ALL-split/test/images", copy=True)
move_files(local_dir + "PV01-split/test/masks", local_dir + "PV-ALL-split/test/masks", copy=True)
move_files(local_dir + "PV03-CROP-split/test/masks", local_dir + "PV-ALL-split/test/masks", copy=True)

#GOOGlE - Clean directory
clean_images_without_masks(local_dir + "bdappv/google/img", local_dir + "bdappv/google/mask")

#GOOGlE - Split the dataset
(train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = split_dataset(local_dir + "bdappv/google/img", local_dir + "bdappv/google/mask")

#GOOGlE - Move files to their respective directories
move_files_list(train_images, local_dir + "bdappv/google/img", local_dir + "GOOGLE-split/train/images")
move_files_list(train_masks, local_dir + "bdappv/google/mask", local_dir + "GOOGLE-split/train/masks")
move_files_list(val_images, local_dir + "bdappv/google/img", local_dir + "GOOGLE-split/val/images")
move_files_list(val_masks, local_dir + "bdappv/google/mask", local_dir + "GOOGLE-split/val/masks")
move_files_list(test_images, local_dir + "bdappv/google/img", local_dir + "GOOGLE-split/test/images")
move_files_list(test_masks, local_dir + "bdappv/google/mask", local_dir + "GOOGLE-split/test/masks")

#  Phase 3: Zip resulting folders

print('Phase 3: Zipping the folders')

zip_directories(local_dir + "result/PV01-split.zip",
                local_dir + "PV01-split/train",
                local_dir + "PV01-split/val",
                local_dir + "PV01-split/test")

zip_directories(local_dir + "result/PV03-CROP-split.zip",
                local_dir + "PV03-CROP-split/train",
                local_dir + "PV03-CROP-split/val",
                local_dir + "PV03-CROP-split/test")

zip_directories(local_dir + "result/PV-ALL-split.zip",
                local_dir + "PV-ALL-split/train",
                local_dir + "PV-ALL-split/val",
                local_dir + "PV-ALL-split/test")

zip_directories(local_dir + "result/GOOGLE-split.zip",
                local_dir + "GOOGLE-split/train",
                local_dir + "GOOGLE-split/val",
                local_dir + "GOOGLE-split/test")