#!/usr/bin/env python

import os
import sys
import argparse
import zipfile
import shutil
import random
import re
from PIL import Image
import math

#crop
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

# Traverse all files in the subdirectories and separate images and masks
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

#Function that move/copy all files from a source directory to a destination directory.
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

# Function to split the dataset
def split_dataset(images_path, masks_path, train_ratio, val_ratio, test_ratio):
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

#Function to clean google diretoctory
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

#Function to clena google diretoctory
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

def zip_directories(output_zip, *directories):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for directory in directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.join(directory, '..'))
                    zipf.write(file_path, arcname)

def main() -> int:
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-r", "--root_dir", required=True, default=".../dataset/", help="Root path of the project (required to download dataset and preprocess to required layout)")
    ap.add_argument("--skip-pv01-extract", required=False, action='store_true', default=False, help="Skip PV1 dataset zip extract")
    ap.add_argument("--skip-pv03-extract", required=False, action='store_true', default=False, help="Skip PV3 dataset zip extract")
    ap.add_argument("--skip-google-extract", required=False, action='store_true', default=False, help="Skip PV Google dataset zip extract")
    args, unknown = ap.parse_known_args()
    args = vars(args)

    #1º - Download of the files: PV01, PV03 and google
    #PV01:https://zenodo.org/records/5171712
    #PV03:https://www.kaggle.com/datasets/salimhammadi07/solar-panel-detection-and-identification?resource=download
    #Google: https://zenodo.org/records/7358126

    #2º - put the zip with the names: PV01.zip, PV03.zip and GOOGLE.zip in local_dir

    #3º - Define Local directory where the zip was placed and the size of split
    local_dir = args['root_dir']
    train_ratio=0.6
    val_ratio=0.2
    test_ratio=0.2

    #4º - The result will be 4 zip files in result directory:
    #  PV01-split.zip > PV1 splited in test, train and validation test images masks train images masks val images masks
    #  PV03-CROP-split.zip > PV3 cropped and splited in test, train and validation test images masks train images masks val images masks
    #  PV-ALL-split.zip > PV1 and PV3 cropped and splited in test, train and validation test images masks train images masks val images masks
    # GOOGLE-split.zip > GOOGLE splited in test, train and validation test images masks train images masks val images masks

    #PV01
    pv01_zip_file_path = os.path.join(local_dir, "PV01.zip")
    pv03_zip_file_path = os.path.join(local_dir, "PV03.zip")
    google_zip_file_path = os.path.join(local_dir, "GOOGLE.zip")

    #
    #  Phase 1: Unzip and organizing the directories
    #
    print('Phase 1: Unzip and organizing the directories')

    #Unzip the files
    if args['skip_pv01_extract'] == False:
        print("  Unzip PV01 dataset...")
        with zipfile.ZipFile(pv01_zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(local_dir)

    if args['skip_pv03_extract'] == False:
        print("  Unzip PV03 dataset...")
        with zipfile.ZipFile(pv03_zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(local_dir)

    if args['skip_google_extract'] == False:
        print("  Unzip Google dataset...")
        with zipfile.ZipFile(google_zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(local_dir)

    #Create destination folder
    os.makedirs(os.path.join(local_dir, "PV01-split"), exist_ok=True)
    os.makedirs(os.path.join(local_dir, "PV03-CROP-split"), exist_ok=True)
    os.makedirs(os.path.join(local_dir, "PV-ALL-split"), exist_ok=True)
    os.makedirs(os.path.join(local_dir, "GOOGLE-split"), exist_ok=True)
    os.makedirs(os.path.join(local_dir, "result"), exist_ok=True)

    #PV01 - Moving files to destination path
    move_files_label(os.path.join(local_dir, "PV01"),
                     os.path.join(local_dir, "PV01/img"),
                     os.path.join(local_dir, "PV01/mask"))

    #PV03 - CROP
    dirs = os.listdir(os.path.join(local_dir, "PV03"))
    for item in dirs:
        input_file = os.path.join(local_dir, "PV03", item)
        crop(input_file, os.path.join(local_dir, "PV03-CROP"), num_tiles=16)

    #PV03 - Moving files to destination path
    move_files_label(os.path.join(local_dir, "PV03-CROP"),
                     os.path.join(local_dir, "PV03-CROP/img"),
                     os.path.join(local_dir, "PV03-CROP/mask"))

    #
    #  Phase 2: Spliting files 
    #
    print('Phase 2: Spliting files')

    # PV01 - Split the dataset
    (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = split_dataset(
        os.path.join(local_dir, "PV01/img"),
        os.path.join(local_dir, "PV01/mask"),
        train_ratio,
        val_ratio,
        test_ratio)

    # PV01 - Move files to their respective directories
    move_files_list(train_images, os.path.join(local_dir, "PV01/img"), os.path.join(local_dir, "PV01-split/train/images"))
    move_files_list(train_masks, os.path.join(local_dir, "PV01/mask"), os.path.join(local_dir, "PV01-split/train/masks"))
    move_files_list(val_images, os.path.join(local_dir, "PV01/img"), os.path.join(local_dir, "PV01-split/val/images"))
    move_files_list(val_masks, os.path.join(local_dir, "PV01/mask"), os.path.join(local_dir, "PV01-split/val/masks"))
    move_files_list(test_images, os.path.join(local_dir, "PV01/img"), os.path.join(local_dir, "PV01-split/test/images"))
    move_files_list(test_masks, os.path.join(local_dir, "PV01/mask"), os.path.join(local_dir, "PV01-split/test/masks"))

    # PV03 - Split the dataset
    (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = split_dataset(
        os.path.join(local_dir, "PV03-CROP/img"),
        os.path.join(local_dir, "PV03-CROP/mask"),
        train_ratio,
        val_ratio,
        test_ratio)

    # PV03 - Move files to their respective directories
    move_files_list(train_images, os.path.join(local_dir, "PV03-CROP/img"), os.path.join(local_dir, "PV03-CROP-split/train/images"))
    move_files_list(train_masks, os.path.join(local_dir, "PV03-CROP/mask"), os.path.join(local_dir, "PV03-CROP-split/train/masks"))
    move_files_list(val_images, os.path.join(local_dir, "PV03-CROP/img"), os.path.join(local_dir, "PV03-CROP-split/val/images"))
    move_files_list(val_masks, os.path.join(local_dir, "PV03-CROP/mask"), os.path.join(local_dir, "PV03-CROP-split/val/masks"))
    move_files_list(test_images, os.path.join(local_dir, "PV03-CROP/img"), os.path.join(local_dir, "PV03-CROP-split/test/images"))
    move_files_list(test_masks, os.path.join(local_dir, "PV03-CROP/mask"), os.path.join(local_dir, "PV03-CROP-split/test/masks"))

    #PVALL - COPY FILES FROM PV01 AND PV03
    move_files(os.path.join(local_dir, "PV01-split/train/images"), os.path.join(local_dir, "PV-ALL-split/train/images"), copy=True)
    move_files(os.path.join(local_dir, "PV03-CROP-split/train/images"), os.path.join(local_dir, "PV-ALL-split/train/images"), copy=True)
    move_files(os.path.join(local_dir, "PV01-split/train/masks"), os.path.join(local_dir, "PV-ALL-split/train/masks"), copy=True)
    move_files(os.path.join(local_dir, "PV03-CROP-split/train/masks"), os.path.join(local_dir, "PV-ALL-split/train/masks"), copy=True)
    move_files(os.path.join(local_dir, "PV01-split/val/images"), os.path.join(local_dir, "PV-ALL-split/val/images"), copy=True)
    move_files(os.path.join(local_dir, "PV03-CROP-split/val/images"), os.path.join(local_dir, "PV-ALL-split/val/images"), copy=True)
    move_files(os.path.join(local_dir, "PV01-split/val/masks"), os.path.join(local_dir, "PV-ALL-split/val/masks"), copy=True)
    move_files(os.path.join(local_dir, "PV03-CROP-split/val/masks"), os.path.join(local_dir, "PV-ALL-split/val/masks"), copy=True)
    move_files(os.path.join(local_dir, "PV01-split/test/images"), os.path.join(local_dir, "PV-ALL-split/test/images"), copy=True)
    move_files(os.path.join(local_dir, "PV03-CROP-split/test/images"), os.path.join(local_dir, "PV-ALL-split/test/images"), copy=True)
    move_files(os.path.join(local_dir, "PV01-split/test/masks"), os.path.join(local_dir, "PV-ALL-split/test/masks"), copy=True)
    move_files(os.path.join(local_dir, "PV03-CROP-split/test/masks"), os.path.join(local_dir, "PV-ALL-split/test/masks"), copy=True)

    #GOOGlE - Clean directory
    clean_images_without_masks(os.path.join(local_dir, "bdappv/google/img"),
                            os.path.join(local_dir, "bdappv/google/mask"))

    #GOOGlE - Split the dataset
    (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = split_dataset(
        os.path.join(local_dir, "bdappv/google/img"),
        os.path.join(local_dir, "bdappv/google/mask"),
        train_ratio,
        val_ratio,
        test_ratio)

    #GOOGlE - Move files to their respective directories
    move_files_list(train_images, os.path.join(local_dir, "bdappv/google/img"), os.path.join(local_dir, "GOOGLE-split/train/images"))
    move_files_list(train_masks, os.path.join(local_dir, "bdappv/google/mask"), os.path.join(local_dir, "GOOGLE-split/train/masks"))
    move_files_list(val_images, os.path.join(local_dir, "bdappv/google/img"), os.path.join(local_dir, "GOOGLE-split/val/images"))
    move_files_list(val_masks, os.path.join(local_dir, "bdappv/google/mask"), os.path.join(local_dir, "GOOGLE-split/val/masks"))
    move_files_list(test_images, os.path.join(local_dir, "bdappv/google/img"), os.path.join(local_dir, "GOOGLE-split/test/images"))
    move_files_list(test_masks, os.path.join(local_dir, "bdappv/google/mask"), os.path.join(local_dir, "GOOGLE-split/test/masks"))


    #
    #  Phase 3: Zipping the folders
    #
    print('Phase 3: Zipping the folders')
    
    zip_directories(os.path.join(local_dir, "result/PV01-split.zip"),
                    os.path.join(local_dir, "PV01-split/train"),
                    os.path.join(local_dir, "PV01-split/val"),
                    os.path.join(local_dir, "PV01-split/test"))

    zip_directories(os.path.join(local_dir, "result/PV03-CROP-split.zip"),
                    os.path.join(local_dir, "PV03-CROP-split/train"),
                    os.path.join(local_dir, "PV03-CROP-split/val"),
                    os.path.join(local_dir, "PV03-CROP-split/test"))

    zip_directories(os.path.join(local_dir, "result/PV-ALL-split.zip"),
                    os.path.join(local_dir, "PV-ALL-split/train"),
                    os.path.join(local_dir, "PV-ALL-split/val"),
                    os.path.join(local_dir, "PV-ALL-split/test"))

    zip_directories(os.path.join(local_dir, "result/GOOGLE-split.zip"),
                    os.path.join(local_dir, "GOOGLE-split/train"),
                    os.path.join(local_dir, "GOOGLE-split/val"),
                    os.path.join(local_dir, "GOOGLE-split/test"))

    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
