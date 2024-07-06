#!/usr/bin/env python

import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
#import torchvision.transforms.functional as TF
import zipfile
import requests
import imageio
import math

from io import BytesIO
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, jaccard_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
from IPython.display import display, Image as IPImage

from packages import time_fmt
from models import unet

"""# Parameters"""

root_dir = '/content/'                                                          # Root directory in Colab

# Model parameters
dict_location = 'https://temp-posgraduation.s3.amazonaws.com/'                  # Base public URL where pretrained model dictionaries have been placed for download
# Segformer model parameters
seg_pretrained_model_name = "nvidia/segformer-b0-finetuned-ade-512-512"         # Pretrained Segformer
seg_pretrained_model_dict = 'segformer_solar_panel_detector.pth'                # Dictionary of the selected fine-tuned Segformer model for solar panel detection
id2label = {0: 'background', 1: 'solar_panel'}                                  # dictionary of solar paner labels
label2id = {label: id for id, label in id2label.items()}                        # dictionary to load the segformer
num_labels = len(id2label)                                                      # parameter to load the segformer
# UNet model parameters
unet_pretrained_model_dict = 'unet_solar_panel_detector.pth'                    # Dictionary of the selected fine-tuned Segformer model for solar panel detection

# Inference parameters
batch_size = 8                                                                  # 8 on T4 GPU
merged_filename = "merged_image"                                                # Base filename for merged images
merged_fileformat = "JPEG"                                                      # File format for merged images
overlay_rgb_color=(255, 0, 255)

# Parameters for power calculation
#spatial_resolution = 0.15625                                                    # spatial resolution of the predicted mask in meters by pixel, calculated from patch_size, resolution and image_size
avg_W = 210                                                                     # Average power per square meter (https://autosolar.es)

# Animation GIFs parameters
transition_time = 1000                                                          # Transition time in mseconds for the animated GIF
target_size = (1024, 768)                                                       # Animated GIF dimensions
crop_params = None                                                              # Crop area for the target animated gif in percentages (xmin, ymin, xmax, ymax), i.e. (0.3, 0.55, 0.5, 0.7)

# Zip parameters
zip_filename = 'Solar_Panel_Generalization.zip'                                 # Name of the zip file to save the experiment outputs
exclude_folders = ['sample_data', 'raw', 'dataset', 'gen/predimgs', '.config', zip_filename]  # Paths to exclude in zip file

"""# Supporting Functions"""

"""**Functions to manage files and folders**"""

# Function to list all files by subfolder (year)

def list_files_by_subfolder(base_dir):
    # Iterate over subfolders
    for year_folder in os.listdir(base_dir):
        year_folder_path = os.path.join(base_dir, year_folder)
        if os.path.isdir(year_folder_path):
            file_count = len(os.listdir(year_folder_path))
            print(f"  Year {year_folder}: {file_count} files")

"""**Function to retrieve and generate a dataset from ICGC for a given year**

"""

# Function to fetch, convert, merge and save images from ICGC for a given year

def fetch_and_save_images(year, bounding_box, icgc_params, merged_filename="merged_image", merged_fileformat="JPEG"):
    image_base_dir = icgc_params['rawimage_dir']
    image_target_dir = icgc_params['image_dir']
    merged_target_dir = icgc_params['merge_dir']
    input_file_extension = icgc_params['input_file_extension']
    output_file_extension = icgc_params['output_file_extension']
    patch_size = icgc_params['patch_size']
    image_size = icgc_params['image_size']
    resolution = icgc_params['resolution']

    images_dir = os.path.join(image_base_dir, str(year))
    os.makedirs(images_dir, exist_ok=True)
    images_target_dir = os.path.join(image_target_dir, str(year))
    os.makedirs(images_target_dir, exist_ok=True)
    merged_target_dir = os.path.join(merged_target_dir, str(year))
    os.makedirs(merged_target_dir, exist_ok=True)

    # Map image format to MIME type for URL
    format_mime_map = {
        "JPEG": "image/jpeg",
        "TIF": "image/tiff"
    }

    # Convert resolution in meters to appropriate string
    if resolution < 1:
        resolution_str = f"{int(resolution * 100)}cm"
    else:
        resolution_str = f"{resolution:.1f}m"

    # Calculate width and height of each patch in meters
    patch_size_meters = patch_size * resolution

    # Calculate number of patches required in x and y directions
    x_min, y_min, x_max, y_max = bounding_box
    x_range = x_max - x_min
    y_range = y_max - y_min
    num_patches_x = math.ceil(x_range / patch_size_meters)
    num_patches_y = math.ceil(y_range / patch_size_meters)
    total_patches = num_patches_x * num_patches_y
    print(f"Processing year: {year}. Patch size in meters: {patch_size_meters}. Total patches: {total_patches} (x:{num_patches_x}, y:{num_patches_y})")

    with tqdm(total=total_patches, desc=f"  Downloading ICGC images for {year}") as pbar:
        for i in range(num_patches_x):
            for j in range(num_patches_y):
                patch_x_min = x_min + i * patch_size_meters
                patch_y_min = y_min + j * patch_size_meters
                patch_x_max = min(patch_x_min + patch_size_meters, x_max)
                patch_y_max = min(patch_y_min + patch_size_meters, y_max)

                # Calculate the pixel dimensions for the request
                patch_width_pixels = math.ceil((patch_x_max - patch_x_min) / resolution)
                patch_height_pixels = math.ceil((patch_y_max - patch_y_min) / resolution)

                bbox = f"{patch_x_min},{patch_y_min},{patch_x_max},{patch_y_max}"
                url = (f"https://geoserveis.icgc.cat/servei/catalunya/orto-territorial/wms?"
                       f"REQUEST=GetMap&VERSION=1.3.0&SERVICE=WMS&CRS=EPSG:25831&BBOX={bbox}&"
                       f"WIDTH={patch_width_pixels}&HEIGHT={patch_height_pixels}&LAYERS=ortofoto_{resolution_str}_color_{year}&"
                       f"STYLES=&FORMAT={format_mime_map[input_file_extension]}&BGCOLOR=0xFFFFFF&TRANSPARENT=TRUE&EXCEPTION=INIMAGE")

                response = requests.get(url)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    # Add padding if necessary
                    pad_x = patch_size - patch_width_pixels if patch_width_pixels < patch_size else 0
                    pad_y = patch_size - patch_height_pixels if patch_height_pixels < patch_size else 0
                    if pad_x > 0 or pad_y > 0:
                        image = ImageOps.expand(image, (0, pad_y, pad_x, 0), fill='black')
                    image_path = os.path.join(images_dir, f"patch_{i}_{j}.{input_file_extension.lower()}")
                    image.save(image_path, format=input_file_extension if input_file_extension != "TIF" else "TIFF")
                else:
                    print(f"Failed to retrieve image for bbox {bbox}")

                pbar.update(1)

    # Convert and upscale images

    # Create directories recursively
    for root, dirs, _ in os.walk(image_base_dir):
        for dir_name in dirs:
            output_subdir = os.path.join(image_target_dir, os.path.relpath(os.path.join(root, dir_name), image_base_dir))
            os.makedirs(output_subdir, exist_ok=True)

    # Loop through all files and subdirectories in the input directory
    total_files = sum([len(files) for _, _, files in os.walk(image_base_dir)])
    with tqdm(total=total_files, desc=f"  Converting and resizing images for {year}") as pbar:
        for root, _, files in os.walk(image_base_dir):
            for filename in files:
                # Construct full file paths
                input_file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(root, image_base_dir)
                output_subdir = os.path.join(image_target_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_file_path = os.path.join(output_subdir, os.path.splitext(filename)[0] + '.png')

                # Check if the file is an image (add more extensions if needed)
                if filename.lower().endswith(('.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff', '.png')):
                    # Open the image
                    with Image.open(input_file_path) as img:
                        # Convert to RGB if necessary
                        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                            img = img.convert('RGBA')
                            background = Image.new('RGBA', img.size, (255, 255, 255))
                            img = Image.alpha_composite(background, img).convert('RGB')
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')

                        # Resize the image to target size
                        img_resized = img.resize((image_size, image_size), Image.LANCZOS)

                        # Save the image in target format to the output directory
                        img_resized.save(output_file_path, output_file_extension)

                pbar.update(1)

    # Merge resized images
    # Calculate merged image dimensions based on patch size and number of patches
    merged_image_width = num_patches_x * image_size
    merged_image_height = num_patches_y * image_size
    merged_image = Image.new('RGB', (merged_image_width, merged_image_height))

    for j in range(num_patches_y):
        for i in range(num_patches_x):
            image_path = os.path.join(images_target_dir, f"patch_{i}_{num_patches_y - 1 - j}.{output_file_extension.lower()}")
            if os.path.exists(image_path):
                image = Image.open(image_path)
                image = image.convert('RGB')  # Ensure image is in RGB format
                merged_image.paste(image, (i * image_size, j * image_size))

    merged_image_path = os.path.join(merged_target_dir, f"{merged_filename}_{year}.{merged_fileformat.lower()}")  # Save as JPEG with year in the filename
    merged_image.save(merged_image_path, format=merged_fileformat.upper())
    print(f"  Created merged image for {year}")

"""**Functions to generate and save images**"""

# Function to save a grid of random images

def save_grid(images, predictions, output_dir="samples", num_samples=8):
    os.makedirs(output_dir, exist_ok=True)

    num_samples = min(num_samples, len(images), len(predictions))

    if num_samples == 0:
        print("No samples to display")
        return

    fig, axes = plt.subplots(nrows=num_samples, ncols=3, figsize=(16, 4 * num_samples))

    indices = random.sample(range(len(images)), num_samples)

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, idx in enumerate(indices):
        if images[idx] is not None and images[idx].ndim == 3:
            img = images[idx].cpu().permute(1, 2, 0).numpy()
            img = img.clip(0, 1)
            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Image")
        else:
            print(f"Warning: Image at index {idx} is not valid for visualization.")
            axes[i, 0].axis('off')

        if predictions[idx] is not None and predictions[idx].ndim == 2:
            pred = predictions[idx].cpu().numpy()
            axes[i, 1].imshow(pred, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title("Prediction")

            if images[idx] is not None and images[idx].ndim == 3:
                image_with_masks = overlay_image(img, pred, color=overlay_rgb_color, alpha=0.3)
                axes[i, 2].imshow(image_with_masks)
                axes[i, 2].set_title('Overlap')
            else:
                axes[i, 2].axis('off')
        else:
            print(f"Warning: Prediction at index {idx} is not valid for visualization.")
            axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"grid_samples.png"))  # Save directly under output_dir
    plt.close()

# Function to merge predicted mask patches

def merge_predicted_masks(pred_dir_base, pred_merge_dir_base, year, patch_size, image_format, merged_filename="merged_image", merged_fileformat="JPEG"):

    if year is not None:
        pred_dir = os.path.join(pred_dir_base, str(year))
        pred_merge_dir = os.path.join(pred_merge_dir_base, str(year))
    else:
        pred_dir = pred_dir_base
        pred_merge_dir = pred_merge_dir_base

    os.makedirs(pred_merge_dir, exist_ok=True)

    # Get all predicted mask files in the prediction directory
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith(f".{image_format.lower()}")]

    # Initialize variables to find the maximum x and y
    max_x = -1
    max_y = -1

    # Extract num_patches_x and num_patches_y from filenames
    for pred_file in pred_files:
        try:
            parts = pred_file.rstrip(f'.{image_format.lower()}').split('_')
            if len(parts) >= 3:
                x = int(parts[-2])
                y = int(parts[-1])
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y
        except ValueError:
            print(f"Skipping invalid filename format: {pred_file}")

    if max_x == -1 or max_y == -1:
        print(f"No valid predicted mask files found in {pred_dir}")
        return

    num_patches_x = max_x + 1
    num_patches_y = max_y + 1

    # Calculate merged image dimensions based on patch size and number of patches
    merged_image_width = num_patches_x * patch_size
    merged_image_height = num_patches_y * patch_size
    merged_image = Image.new('RGB', (merged_image_width, merged_image_height))

    # Paste predicted masks into the merged image
    for pred_file in pred_files:
        try:
            parts = pred_file.rstrip(f'.{image_format.lower()}').split('_')
            if len(parts) >= 3:
                x = int(parts[-2])
                y = int(parts[-1])
                pred_mask_path = os.path.join(pred_dir, pred_file)
                if os.path.exists(pred_mask_path):
                    pred_mask = Image.open(pred_mask_path)
                    pred_mask = pred_mask.convert('RGB')  # Ensure mask is in RGB format
                    merged_image.paste(pred_mask, (x * patch_size, (num_patches_y - 1 - y) * patch_size))
        except ValueError:
            print(f"Skipping invalid filename format: {pred_file}")

    # Save the merged image
    merged_image_path = os.path.join(pred_merge_dir, f"{merged_filename}.{merged_fileformat.lower()}")
    merged_image.save(merged_image_path, format=merged_fileformat.upper())
    print(f"Merged predicted mask saved to {merged_image_path}")

# Function to create an overlay of image and mask

def overlay_image(image, mask, color, alpha, resize=None):
    im_copy = (image * 255).astype(np.uint8)
    im_copy = Image.fromarray(im_copy, "RGB")

    if resize:
        im_copy = im_copy.resize(resize)
        mask = Image.fromarray((mask * 128).astype(np.uint8)).resize(resize)
    else:
        mask = Image.fromarray((mask * 128).astype(np.uint8))

    full_color = Image.new("RGB", im_copy.size, color)
    im_copy = Image.composite(full_color, im_copy, mask)
    return np.array(im_copy)

# Function to create an overlay of images and masks

def create_overlay(merge_dir, pred_merge_dir, overlay_image_dir, year, merged_filename, merged_fileformat, color=(0, 255, 0), alpha=0.3, resize=None):
    if year is not None:
        merge_dir = os.path.join(merge_dir, str(year))
        pred_merge_dir = os.path.join(pred_merge_dir, str(year))
        overlay_image_dir = os.path.join(overlay_image_dir, str(year))

    os.makedirs(overlay_image_dir, exist_ok=True)

    # Ensure merged_fileformat is lowercase
    merged_fileformat = merged_fileformat.lower()

    # Construct file paths with proper file extension
    merged_image_path = os.path.join(merge_dir, f"{merged_filename}_{year}.{merged_fileformat}")
    merged_predicted_path = os.path.join(pred_merge_dir, f"{merged_filename}.{merged_fileformat}")

    # Check if files exist
    if not (os.path.exists(merged_image_path) and os.path.exists(merged_predicted_path)):
        raise FileNotFoundError(f"Either {merged_image_path} or {merged_predicted_path} does not exist.")

    # Load images
    source_merged_image = np.array(Image.open(merged_image_path).convert('RGB')) / 255.0
    predicted_mask = np.array(Image.open(merged_predicted_path).convert('L')) / 255.0

    # Create overlay image
    im_copy = (source_merged_image * 255).astype(np.uint8)
    im_copy = Image.fromarray(im_copy, "RGB")

    if resize:
        im_copy = im_copy.resize(resize)
        predicted_mask = Image.fromarray((predicted_mask * 128).astype(np.uint8)).resize(resize)
    else:
        predicted_mask = Image.fromarray((predicted_mask * 128).astype(np.uint8))

    predicted_mask = predicted_mask.convert("L")

    full_color = Image.new("RGB", im_copy.size, color)
    im_copy = Image.composite(full_color, im_copy, predicted_mask)
    overlay_img = np.array(im_copy)

    # Convert to Image for display and save
    overlay_img_pil = Image.fromarray(overlay_img)
    overlay_output_path = os.path.join(overlay_image_dir, f"{merged_filename}_overlay.{merged_fileformat}")
    overlay_img_pil.save(overlay_output_path)
    print(f"Overlay image saved to {overlay_output_path}")

# Function to create animated gifs of a cropped area

def crop_and_resize_image(image_path, crop_params, target_size):
    # Open the image
    image = Image.open(image_path)

    # Calculate the cropped region based on crop_params
    xmin, ymin, xmax, ymax = crop_params
    left = int(xmin * image.width)
    upper = int(ymin * image.height)
    right = int(xmax * image.width)
    lower = int(ymax * image.height)

    # Crop the image
    cropped_image = image.crop((left, upper, right, lower))

    # Resize the cropped image while maintaining aspect ratio
    cropped_image.thumbnail(target_size, Image.LANCZOS)

    # Create a new image of target size and paste the cropped image on it
    resized_image = Image.new('RGB', target_size)
    offset = ((target_size[0] - cropped_image.width) // 2, (target_size[1] - cropped_image.height) // 2)
    resized_image.paste(cropped_image, offset)

    return resized_image

def create_animated_gif(images_dir_base, years, animation_type, animated_merge_dir, region, target_size, transition_time=1, loop=True, crop_params=None):
    # Create output directory if it doesn't exist
    os.makedirs(animated_merge_dir, exist_ok=True)

    # Initialize list to store image paths
    image_paths = []

    # Determine filename pattern based on animation type
    if animation_type == 'images':
        filename_pattern = f"merged_image_%Y.jpeg"
    elif animation_type == 'overlays':
        filename_pattern = "merged_image_overlay.jpeg"
    else:
        filename_pattern = "merged_image.jpeg"

    # Iterate through years and collect image paths
    for year in years:
        # Construct image file path
        if animation_type == 'images':
            image_file = f"merged_image_{year}.jpeg"
        elif animation_type == 'overlays':
            image_file = "merged_image_overlay.jpeg"
        else:
            image_file = "merged_image.jpeg"

        image_path = os.path.join(images_dir_base, str(year), image_file)

        # Check if image file exists before adding to list
        if os.path.exists(image_path):
            image_paths.append(image_path)
        else:
            print(f"Warning: Image file not found for year {year} and animation type {animation_type}")

    # Ensure there are images to create the animated GIF
    if not image_paths:
        print(f"No images found for animation type {animation_type}")
        return

    # Initialize list to store resized images
    resized_images = []

    # Loop through each image path, crop, resize, add text, and append to resized_images list
    for i, image_path in enumerate(image_paths):
        # Crop and resize the image
        if crop_params:
            resized_image = crop_and_resize_image(image_path, crop_params, target_size)
        else:
            # If no crop_params provided, just resize the original image
            image = Image.open(image_path)
            resized_image = image.resize(target_size, resample=Image.LANCZOS)

        # Add text at the bottom (region and year) with black background
        draw = ImageDraw.Draw(resized_image)

        # Add region text at the top
        region_text = region
        font = ImageFont.load_default()
        region_text_bbox = draw.textbbox((0, 0), region_text, font)
        year_text = f"Year: {years[i]}"
        year_text_bbox = draw.textbbox((0, 0), year_text, font)
        max_text_width = max(region_text_bbox[2], year_text_bbox[2])
        total_text_height = region_text_bbox[3] + year_text_bbox[3] + 10  # Adding 10 for padding

        # Add black background rectangle for text
        draw.rectangle([(0, resized_image.height - total_text_height - 20),
                        (resized_image.width, resized_image.height)], fill="black")

        # Add region text
        draw.text(((resized_image.width - region_text_bbox[2]) // 2, resized_image.height - total_text_height - 10),
                  region_text, fill='white', font=font)

        # Add year text
        draw.text(((resized_image.width - year_text_bbox[2]) // 2, resized_image.height - year_text_bbox[3] - 10),
                  year_text, fill='white', font=font)

        resized_images.append(np.array(resized_image))

    # Save as animated GIF
    gif_file_path = os.path.join(animated_merge_dir, f"animation_{animation_type}.gif")
    with imageio.get_writer(gif_file_path, mode='I', duration=transition_time) as writer:
        for image in resized_images:
            writer.append_data(image)

    if loop:
        # Add loop parameter to control looping
        with imageio.get_writer(gif_file_path, mode='I', duration=transition_time, loop=0 if loop else -1) as writer:
            for image in resized_images:
                writer.append_data(image)

    print(f"Animated GIF saved for {animation_type} at: {gif_file_path}")

# Function to combine various animated gifs into one for comparison

def combine_gifs(gif_paths, output_path, loop=True):
    # Load the GIFs
    gifs = [Image.open(gif_path) for gif_path in gif_paths]

    # Check that all gifs have the same number of frames
    num_frames = gifs[0].n_frames
    for gif in gifs:
        if gif.n_frames != num_frames:
            raise ValueError("All GIFs must have the same number of frames")

    # Initialize a list to store the combined frames
    combined_frames = []

    # Loop through each frame index
    for frame_idx in range(num_frames):
        frames = []

        # Extract and resize each frame
        for gif in gifs:
            gif.seek(frame_idx)
            frame = gif.copy()
            frame = frame.convert("RGBA")  # Ensure the frame is in RGBA mode
            frames.append(frame)

        # Get the maximum height among frames to resize to
        max_height = max(frame.height for frame in frames)

        # Resize frames to the same height
        resized_frames = []
        for frame in frames:
            aspect_ratio = frame.width / frame.height
            new_width = int(aspect_ratio * max_height)
            resized_frame = frame.resize((new_width, max_height))
            resized_frames.append(resized_frame)

        # Concatenate frames horizontally
        total_width = sum(frame.width for frame in resized_frames)
        combined_frame = Image.new("RGBA", (total_width, max_height))

        current_width = 0
        for frame in resized_frames:
            combined_frame.paste(frame, (current_width, 0))
            current_width += frame.width

        combined_frames.append(np.array(combined_frame))

    # Save as animated GIF
    duration = gifs[0].info['duration']  # Assuming all GIFs have the same duration
    with imageio.get_writer(output_path, mode='I', duration=duration, loop=0 if loop else -1) as writer:
        for frame in combined_frames:
            writer.append_data(frame)

    print(f"Combined GIF saved at: {output_path}")

    # Display the resulting GIF
    with open(output_path, 'rb') as file:
        display(IPImage(file.read()))

"""**Functions to calculate and plot PV power**"""

# Area and power calculation

def calculate_area_and_power(mask_image_dir_base, year, merged_filename, merged_fileformat, patch_size, resolution, image_size, average_power):

    # Construct the path to the mask image
    mask_image_path = os.path.join(mask_image_dir_base, str(year), f"{merged_filename}.{merged_fileformat.lower()}")

    # Load the mask image
    mask_image = Image.open(mask_image_path).convert('L')  # Convert to grayscale
    mask = np.array(mask_image)

    # Threshold the mask image to convert to binary (0 and 1)
    thresholded_mask = np.where(mask > 80, 1, 0).astype(np.uint8)

    # Convert the numpy array to a torch tensor
    mask_tensor = torch.tensor(thresholded_mask, dtype=torch.float32)

    # Calculate the total number of pixels in the mask
    total_pixels = mask_tensor.numel()

    # Calculate the number of pixels in the mask that are identified as PV
    pixels_PV = torch.sum(mask_tensor)  # Sum of all pixels identified as PV

    # Print the number of PV pixels and total pixels
    print(f"Number of PV pixels: {pixels_PV}")
    print(f"Total number of pixels: {total_pixels}")

    # Calculate spatial_resolution of the resized images
    physical_size = patch_size * resolution
    spatial_resolution = physical_size / image_size

    # Calculate the PV area
    PV_Area = spatial_resolution * pixels_PV.item()
    print("PV Area: ", PV_Area, "Square meters")

    # Calculate the installed power
    installed_power = average_power * PV_Area
    print("Installed Power: ", installed_power / 1e6, " MegaWatts")

    return PV_Area, installed_power / 1e6

# Function to plot PV Area and Installed Power over the years

def plot_pv_area_and_installed_power(years, PV_Areas, installed_powers, plot_dir):
    # Create the plot directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot PV Area
    ax1.plot(years, PV_Areas, marker='o', linestyle='-', color='b')
    ax1.set_title('PV Area Over Years')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('PV Area (square kilometers)')
    ax1.grid(True)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot Installed Power
    ax2.plot(years, installed_powers, marker='o', linestyle='-', color='g')
    ax2.set_title('Installed Power Over Years')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Installed Power (MW)')
    ax2.grid(True)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Adjust layout to prevent overlap of labels
    plt.tight_layout()

    # Save the combined plot under plot_dir
    plot_path = os.path.join(plot_dir, 'pv_area_installed_power_plots.png')
    plt.savefig(plot_path)

    # Display the combined plot
    plt.show()

"""# Model, dataset class and generalization functions"""

"""**Model dictionary download function**"""

# Function to download the pretrained model dictionary
def download_dict(base_url, model_dir, filename):
    url = os.path.join(base_url, filename)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    file_path = os.path.join(model_dir, filename)
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        with open(file_path, 'wb') as f:
            f.write(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dictionary: {e}")

"""**Dataset class**"""

class SolarPanelGenDataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image, img_name  # Return image and its filename for saving predictions

"""**Generalization function for a single year**"""

def gen_model(device, model_sel, model, image_dir, image_extension, image_size=256, pred_image_dir='predicted_masks', grid_image_dir='grid_images', year=None):
    if year is not None:
        pred_image_dir = os.path.join(pred_image_dir, str(year))
        grid_image_dir = os.path.join(grid_image_dir, str(year))
        image_dir = os.path.join(image_dir, str(year))

    os.makedirs(pred_image_dir, exist_ok=True)
    os.makedirs(grid_image_dir, exist_ok=True)

    # Transformations for the dataset
    transform_images = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Create dataset
    gen_dataset = SolarPanelGenDataset(image_dir=image_dir, transforms=transform_images)
    gen_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Run the model
    model.eval()

    images_to_save = []
    predictions_to_save = []

    with torch.no_grad():
        for images, img_names in tqdm(gen_loader, desc="Generalization"):
            images = images.to(device)

            if model_sel == "Segformer":
                outputs = model(pixel_values=images).logits
                outputs = F.interpolate(outputs, size=(image_size, image_size), mode='bilinear', align_corners=False)
                _, predicted = torch.max(outputs, 1)
            elif model_sel == "UNet":
                outputs = model(images)
                outputs = F.interpolate(outputs, size=(image_size, image_size), mode='bilinear', align_corners=False)
                outputs = outputs.squeeze(1)
                preds = torch.sigmoid(outputs)
                predicted = (preds > 0.5).float()
            else:
                print("Error: Invalid model selection.")

            for i in range(len(images)):
                # Save predicted masks
                pred_mask = predicted[i].cpu().numpy()
                img_name = img_names[i]
                pred_mask_path = os.path.join(pred_image_dir, img_name.replace(image_extension, 'png'))
                plt.imsave(pred_mask_path, pred_mask, cmap='gray')

                # Prepare for grid saving
                images_to_save.append(images[i])
                predictions_to_save.append(predicted[i])

    # Save grid of random images
    save_grid(images_to_save, predictions_to_save, output_dir=grid_image_dir)

def check_config(cfg_filename):
    with open(cfg_filename) as json_file:
        config = json.load(json_file)
        return config

def main() -> int:
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-r", "--root_dir", required=True, help="Root path of the project (required to load dataset and models)")
    ap.add_argument("-c", "--config", required=True, help="Configuration json file (generalization parameters)")
    ap.add_argument("--no-pretrain-download", required=False, action='store_true', default=False, help="Skip pretrain model download")
    ap.add_argument("--no-icgc-download", required=False, action='store_true', default=False, help="Skip ICGC download images")
    args = vars(ap.parse_args())

    json_cfg_file = args['config']
    try:
        settings = check_config(json_cfg_file)
    except ValueError as error:
        print(f"Error: Invalid configuration file ({error})")
        return 1

    print(f"Settings:\n{settings}")

    root_dir = args['root_dir']
    bounding_box = settings['bounding_box'] # [xmin, ymin, xmax, ymax] - Coordinate reference system: ETRS89 UTM fus 31 Nord (EPSG:25831) measured in meters
    years = settings['years']
    region = settings['region']
    model_sel = settings['model']

    # ICGC download and conversion parameters
    igcc_params = {
        'patch_size': 160,                                      # squared patch size in pixels - allowed values up to 4096 max (4096x4096px, 1km2 area with 0,25m/pixel resolution)
        'resolution': 0.25,                                     # raw meters per pixel - allowed values: 0.1, 0.15, 0.25, 0.50, 2.5 or 10 depending on availability
        'input_file_extension': "TIF",                          # allowed values: "JPEG" or "TIF"
        'rawimage_dir': os.path.join(root_dir, 'raw/images/'),  # path to save downloaded patches (patch_size)
        'image_size': 256,                                      # Target resized squared patch size in pixels
        'output_file_extension': "PNG",                         # Target converted patch format
        'image_dir': os.path.join(root_dir, 'dataset/image/'),  # Path to save resized and converted images (image_size)
        'merge_dir': os.path.join(root_dir, 'dataset/merge/')   # Path to save merged image from resized and converted images
    }

    # Model parameters
    model_dir = os.path.join(root_dir, 'pretrained/')           # Path to save pretrained model dictionary

    # Inference parameters
    pred_image_dir = os.path.join(root_dir, 'gen/predimgs/')    # Path to save predicted masks
    pred_merge_dir = os.path.join(root_dir, 'gen/predmerge/')   # Path to save merged mask
    grid_image_dir = os.path.join(root_dir, 'gen/gridimgs/')    # Path to save a grid of sample images and masks
    overlay_image_dir = os.path.join(root_dir, 'gen/overlay/')  # Path to save overlay image from merged image and mask

    # Parameters for power calculation
    plot_dir = os.path.join(root_dir, 'gen/plot/')               # Directory to save PV Area and Installed power plot

    # Animation GIFs parameters
    animated_merge_dir = os.path.join(root_dir, 'gen/animation/')                      # Directory to save animated merged images
    gif_images = os.path.join(root_dir, 'gen/animation/animation_images.gif')          # Path to the animaged gif from original images
    gif_masks = os.path.join(root_dir, 'gen/animation/animation_masks.gif')            # Path to the animaged gif from predicted masks
    gif_overlays = os.path.join(root_dir, 'gen/animation/animation_overlays.gif')      # Path to the animaged gif from overlayed images and masks
    combined_gif_path = os.path.join(root_dir, 'gen/animation/combined_animation.gif') # Output path for the combined GIF
    gif_paths_to_include = [gif_masks, gif_overlays]                                # GIFs to include, 2 or 3 from [gif_images, gif_masks, gif_overlays]

    """# Execution
    **Download and preprocess dataset from ICGC**
    """
    if args['no_icgc_download'] == False:
        # Fetch and merge images for each year
        start_time = time.perf_counter()
        for year in years:
            fetch_and_save_images(year, bounding_box, igcc_params, merged_filename=merged_filename, merged_fileformat=merged_fileformat)
        elapsed_time = time.perf_counter() - start_time
        print(f"Download and coversion took: {time_fmt.format_time(elapsed_time)}\n")

    """**Load the model**"""

    device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
    if sys.platform == "darwin" and not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")

    # Load selected model
    print(f"Loading {model_sel}...")
    if model_sel == "Segformer":
        # Load the pretrained Segformer model
        model = SegformerForSemanticSegmentation.from_pretrained(
            seg_pretrained_model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        # Download and load the state dictionary from a pretrained model
        #download_dict(dict_location, model_dir, seg_pretrained_model_dict)
        seg_pretrained_model_path = os.path.join(model_dir, seg_pretrained_model_dict)
        try:
            model.load_state_dict(torch.load(seg_pretrained_model_path, map_location=device))
            print(f"Successfully loaded the {model_sel} model values from {seg_pretrained_model_path}")
        except Exception as e:
            print(f"Error loading the model from {seg_pretrained_model_path}: {e}")
            return 3
    elif model_sel == "UNet":
        # Load the pretrained UNet model
        model = unet.UNet(in_channels=3, out_channels=1)
        if args['no_pretrain_download'] == False:
            # Download and load the state dictionary from a pretrained model
            download_dict(dict_location, model_dir, unet_pretrained_model_dict)
        unet_pretrained_model_path = os.path.join(model_dir, unet_pretrained_model_dict)
        try:
            model.load_state_dict(torch.load(unet_pretrained_model_path, map_location=device))
            print(f"Successfully loaded the {model_sel} model values from {unet_pretrained_model_path}")
        except Exception as e:
            print(f"Error loading the dictionary from {unet_pretrained_model_path}: {e}")
            return 3
    else:
        print("Error: Invalid model selection")
        return 2

    # Move model to GPU if available
    model.to(device)

    """**Run generalization for multiple years**"""

    # Run generalization for multiple years

    # Initialize lists to store PV Area and Installed Power data
    PV_Areas = []
    installed_powers = []

    # Initialize total elapsed time
    total_elapsed_time = 0

    # Iterate over each year
    for year in years:
        print(f"Running generalization for year: {year}")
        start_time = time.perf_counter()

        # Call gen_model
        gen_model(device, 
                  model_sel, 
                  model, 
                  igcc_params['image_dir'], 
                  image_extension=igcc_params['output_file_extension'],
                  image_size=igcc_params['image_size'], 
                  pred_image_dir=pred_image_dir, 
                  grid_image_dir=grid_image_dir, year=year)

        # Call merge_predicted_masks
        merge_predicted_masks(pred_image_dir, 
                              pred_merge_dir, 
                              year, 
                              patch_size=igcc_params['image_size'], 
                              image_format='png', 
                              merged_filename=merged_filename, 
                              merged_fileformat=merged_fileformat)

        # Call create_overlay
        create_overlay(igcc_params['merge_dir'], pred_merge_dir, overlay_image_dir, year, merged_filename, merged_fileformat, color=overlay_rgb_color)

        # Call calculate_area_and_power
        PV_Area, installed_power = calculate_area_and_power(pred_merge_dir, 
                                                            year, 
                                                            merged_filename, 
                                                            merged_fileformat, 
                                                            igcc_params['patch_size'], 
                                                            igcc_params['resolution'], 
                                                            igcc_params['image_size'], 
                                                            avg_W)

        # Store PV Area and Installed Power
        PV_Areas.append(PV_Area)
        installed_powers.append(installed_power)

        # Calculate elapsed time for current year
        elapsed_time = time.perf_counter() - start_time
        total_elapsed_time += elapsed_time

        # Print the time taken for current year
        print(f"Generalization for year {year} took: {time_fmt.format_time(elapsed_time)}\n")

    # Print total elapsed time
    print(f"Total time for all years: {time_fmt.format_time(total_elapsed_time)}")

    """**Generate plots and animated images**"""

    # Generate plots and animations
    create_animated_gif(igcc_params['merge_dir'], years, 'images', animated_merge_dir, region, target_size, transition_time, crop_params=crop_params)
    create_animated_gif(pred_merge_dir, years, 'masks', animated_merge_dir, region, target_size, transition_time, crop_params=crop_params)
    create_animated_gif(overlay_image_dir, years, 'overlays', animated_merge_dir, region, target_size, transition_time, crop_params=crop_params)
    combine_gifs(gif_paths_to_include, combined_gif_path)

    # Plot PV Area and Installed Powers over the years
    plot_pv_area_and_installed_power(years, PV_Areas, installed_powers, plot_dir)

    """**Zip results**"""
    
    # Create the zip file
    #zip_dir(root_dir, zip_filename, exclude_folders)
    #zip_file_path = os.path.join(root_dir, zip_filename)
    #list_folders_and_files_in_zip(zip_file_path)

    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
