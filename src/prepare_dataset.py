#!/usr/bin/env python

import os
import sys
import argparse
import zipfile
import random

from packages import crop, files, ziptools

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

def main() -> int:
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-r", "--root_dir", required=True, default="./dataset/", help="Root path of the project (required to download dataset and preprocess to required layout)")
    ap.add_argument("--skip-pv01-extract", required=False, action='store_true', default=False, help="Skip PV1 dataset zip extract")
    ap.add_argument("--skip-pv03-extract", required=False, action='store_true', default=False, help="Skip PV3 dataset zip extract")
    ap.add_argument("--skip-google-extract", required=False, action='store_true', default=False, help="Skip PV Google dataset zip extract")
    args, unknown = ap.parse_known_args()
    args = vars(args)

    #1º - Download of the files: PV01, PV03 and google
    # PV01: https://zenodo.org/records/5171712
    #   Bash command: wget https://zenodo.org/records/5171712/files/PV01.zip?download=1 -O dataset_prepare/PV01.zip
    # PV03: https://www.kaggle.com/datasets/salimhammadi07/solar-panel-detection-and-identification?resource=download
    #   You have to donwload from browser due you need some credentials to download from kaggle.
    #   Please donload using the black button on top-right, other links will donwload a different zip file.
    # Google: https://zenodo.org/records/7358126
    #   Bash command: wget https://zenodo.org/records/7358126/files/bdappv.zip?download=1 -O dataset_prepare/GOOGLE.zip

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
    #  GOOGLE-split.zip > GOOGLE splited in test, train and validation test images masks train images masks val images masks

    pv01_zip_file_path = os.path.join(local_dir, "PV01.zip")
    pv03_zip_file_path = os.path.join(local_dir, "PV03.zip")
    google_zip_file_path = os.path.join(local_dir, "GOOGLE.zip")


    """# Execution"""

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
    files.move_files_label(os.path.join(local_dir, "PV01"),
                           os.path.join(local_dir, "PV01/img"),
                           os.path.join(local_dir, "PV01/mask"))

    #PV03 - CROP
    dirs = os.listdir(os.path.join(local_dir, "PV03"))
    for item in dirs:
        input_file = os.path.join(local_dir, "PV03", item)
        crop.crop(input_file, os.path.join(local_dir, "PV03-CROP"), num_tiles=16)

    #PV03 - Moving files to destination path
    files.move_files_label(os.path.join(local_dir, "PV03-CROP"),
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
    files.move_files_list(train_images, os.path.join(local_dir, "PV01/img"), os.path.join(local_dir, "PV01-split/train/images"))
    files.move_files_list(train_masks, os.path.join(local_dir, "PV01/mask"), os.path.join(local_dir, "PV01-split/train/masks"))
    files.move_files_list(val_images, os.path.join(local_dir, "PV01/img"), os.path.join(local_dir, "PV01-split/val/images"))
    files.move_files_list(val_masks, os.path.join(local_dir, "PV01/mask"), os.path.join(local_dir, "PV01-split/val/masks"))
    files.move_files_list(test_images, os.path.join(local_dir, "PV01/img"), os.path.join(local_dir, "PV01-split/test/images"))
    files.move_files_list(test_masks, os.path.join(local_dir, "PV01/mask"), os.path.join(local_dir, "PV01-split/test/masks"))

    # PV03 - Split the dataset
    (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = split_dataset(
        os.path.join(local_dir, "PV03-CROP/img"),
        os.path.join(local_dir, "PV03-CROP/mask"),
        train_ratio,
        val_ratio,
        test_ratio)

    # PV03 - Move files to their respective directories
    files.move_files_list(train_images, os.path.join(local_dir, "PV03-CROP/img"), os.path.join(local_dir, "PV03-CROP-split/train/images"))
    files.move_files_list(train_masks, os.path.join(local_dir, "PV03-CROP/mask"), os.path.join(local_dir, "PV03-CROP-split/train/masks"))
    files.move_files_list(val_images, os.path.join(local_dir, "PV03-CROP/img"), os.path.join(local_dir, "PV03-CROP-split/val/images"))
    files.move_files_list(val_masks, os.path.join(local_dir, "PV03-CROP/mask"), os.path.join(local_dir, "PV03-CROP-split/val/masks"))
    files.move_files_list(test_images, os.path.join(local_dir, "PV03-CROP/img"), os.path.join(local_dir, "PV03-CROP-split/test/images"))
    files.move_files_list(test_masks, os.path.join(local_dir, "PV03-CROP/mask"), os.path.join(local_dir, "PV03-CROP-split/test/masks"))

    # PVALL - COPY FILES FROM PV01 AND PV03
    files.move_files(os.path.join(local_dir, "PV01-split/train/images"), os.path.join(local_dir, "PV-ALL-split/train/images"), copy=True)
    files.move_files(os.path.join(local_dir, "PV03-CROP-split/train/images"), os.path.join(local_dir, "PV-ALL-split/train/images"), copy=True)
    files.move_files(os.path.join(local_dir, "PV01-split/train/masks"), os.path.join(local_dir, "PV-ALL-split/train/masks"), copy=True)
    files.move_files(os.path.join(local_dir, "PV03-CROP-split/train/masks"), os.path.join(local_dir, "PV-ALL-split/train/masks"), copy=True)
    files.move_files(os.path.join(local_dir, "PV01-split/val/images"), os.path.join(local_dir, "PV-ALL-split/val/images"), copy=True)
    files.move_files(os.path.join(local_dir, "PV03-CROP-split/val/images"), os.path.join(local_dir, "PV-ALL-split/val/images"), copy=True)
    files.move_files(os.path.join(local_dir, "PV01-split/val/masks"), os.path.join(local_dir, "PV-ALL-split/val/masks"), copy=True)
    files.move_files(os.path.join(local_dir, "PV03-CROP-split/val/masks"), os.path.join(local_dir, "PV-ALL-split/val/masks"), copy=True)
    files.move_files(os.path.join(local_dir, "PV01-split/test/images"), os.path.join(local_dir, "PV-ALL-split/test/images"), copy=True)
    files.move_files(os.path.join(local_dir, "PV03-CROP-split/test/images"), os.path.join(local_dir, "PV-ALL-split/test/images"), copy=True)
    files.move_files(os.path.join(local_dir, "PV01-split/test/masks"), os.path.join(local_dir, "PV-ALL-split/test/masks"), copy=True)
    files.move_files(os.path.join(local_dir, "PV03-CROP-split/test/masks"), os.path.join(local_dir, "PV-ALL-split/test/masks"), copy=True)

    # GOOGlE - Clean directory
    clean_images_without_masks(os.path.join(local_dir, "bdappv/google/img"),
                            os.path.join(local_dir, "bdappv/google/mask"))

    # GOOGlE - Split the dataset
    (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = split_dataset(
        os.path.join(local_dir, "bdappv/google/img"),
        os.path.join(local_dir, "bdappv/google/mask"),
        train_ratio,
        val_ratio,
        test_ratio)

    # GOOGlE - Move files to their respective directories
    files.move_files_list(train_images, os.path.join(local_dir, "bdappv/google/img"), os.path.join(local_dir, "GOOGLE-split/train/images"))
    files.move_files_list(train_masks, os.path.join(local_dir, "bdappv/google/mask"), os.path.join(local_dir, "GOOGLE-split/train/masks"))
    files.move_files_list(val_images, os.path.join(local_dir, "bdappv/google/img"), os.path.join(local_dir, "GOOGLE-split/val/images"))
    files.move_files_list(val_masks, os.path.join(local_dir, "bdappv/google/mask"), os.path.join(local_dir, "GOOGLE-split/val/masks"))
    files.move_files_list(test_images, os.path.join(local_dir, "bdappv/google/img"), os.path.join(local_dir, "GOOGLE-split/test/images"))
    files.move_files_list(test_masks, os.path.join(local_dir, "bdappv/google/mask"), os.path.join(local_dir, "GOOGLE-split/test/masks"))

    #
    #  Phase 3: Zip resulting folders
    #
    print('Phase 3: Zipping the folders')
    
    ziptools.zip_directories(os.path.join(local_dir, "result/PV01-split.zip"),
                             os.path.join(local_dir, "PV01-split/train"),
                             os.path.join(local_dir, "PV01-split/val"),
                             os.path.join(local_dir, "PV01-split/test"))

    ziptools.zip_directories(os.path.join(local_dir, "result/PV03-CROP-split.zip"),
                             os.path.join(local_dir, "PV03-CROP-split/train"),
                             os.path.join(local_dir, "PV03-CROP-split/val"),
                             os.path.join(local_dir, "PV03-CROP-split/test"))

    ziptools.zip_directories(os.path.join(local_dir, "result/PV-ALL-split.zip"),
                             os.path.join(local_dir, "PV-ALL-split/train"),
                             os.path.join(local_dir, "PV-ALL-split/val"),
                             os.path.join(local_dir, "PV-ALL-split/test"))

    ziptools.zip_directories(os.path.join(local_dir, "result/GOOGLE-split.zip"),
                             os.path.join(local_dir, "GOOGLE-split/train"),
                             os.path.join(local_dir, "GOOGLE-split/val"),
                             os.path.join(local_dir, "GOOGLE-split/test"))

    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
