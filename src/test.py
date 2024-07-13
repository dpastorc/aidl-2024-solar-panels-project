#!/usr/bin/env python

import os
import sys
import argparse
import json

"""# Libraries"""

# Import libraries

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
import zipfile
import requests

from PIL import Image
from sklearn.metrics import accuracy_score, jaccard_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm

from packages import time_fmt, plot, ziptools
from models import unet

# Quick configuration
dataset_name = 'GOOGLE-split'                                                   # Datasets available: PV01-split, PV-ALL-split, PV03-CROP-split, GOOGLE-split

"""# Parameters"""

overlay_rgb_color=(255, 0, 255)

# Dataset parameters
pv_file_format = 'bmp'                                                          # File format within the PVXX dataset
google_file_format = 'png'                                                      # File format within the Google dataset
dataset_url = 'https://temp-posgraduation.s3.amazonaws.com/' + dataset_name + '.zip' # Location of the preprocessed and split dataset
root_dir = '/content/'                                                          # Root directory

# Inference parameters
image_size = 256                                                                # Image size for inference

# Model parameters
dict_location = 'https://temp-posgraduation.s3.amazonaws.com/'                  # Base public URL where pretrained model dictionaries have been placed for download

# Segformer model parameters
seg_pretrained_model_name = "nvidia/segformer-b0-finetuned-ade-512-512"         # Pretrained Segformer

"""# Supporting Functions"""

"""**Functions to manage files and folders**"""

# Delete desired folders
paths_to_remove = "/content/results /content/dataset"   # "/content/samples /content/plots /content/sample_data /content/model"
# !rm -rf {paths_to_remove}

"""**Functions to generate and save images**"""

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

# Function to save samples
def save_samples(images, masks, predictions, sample_type, output_dir="results", num_samples=8):
    os.makedirs(output_dir, exist_ok=True)

    # Ensure num_samples is not larger than the length of images, masks, or predictions
    num_samples = min(num_samples, len(images), len(masks), len(predictions))

    if num_samples == 0:
        print("No samples to display")
        return

    fig, axes = plt.subplots(nrows=num_samples, ncols=4, figsize=(16, 4 * num_samples))

    # Generate random indices to select random samples
    indices = random.sample(range(len(images)), num_samples)

    # Ensure axes is a 2D array even if num_samples is 1
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, idx in enumerate(indices):
        if images[idx] is not None and images[idx].ndim == 3:
            img = images[idx].cpu().permute(1, 2, 0).numpy()
            img = img.clip(0, 1)  # Clip values to the range [0, 1]
            axes[i, 0].imshow(img)  # Original image
            axes[i, 0].set_title("Image")
        else:
            print(f"Warning: Image at index {idx} is not valid for visualization.")
            axes[i, 0].axis('off')

        if masks[idx] is not None and masks[idx].ndim == 2:
            mask = masks[idx].cpu().numpy()
            axes[i, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)  # Ground truth mask
            axes[i, 1].set_title("Mask")
        else:
            print(f"Warning: Mask at index {idx} is not valid for visualization.")
            axes[i, 1].axis('off')

        if predictions[idx] is not None and predictions[idx].ndim == 2:
            pred = predictions[idx].cpu().numpy()
            axes[i, 2].imshow(pred, cmap='gray', vmin=0, vmax=1)  # Predicted mask
            axes[i, 2].set_title("Prediction")

            # Overlapped predicted mask and original
            if images[idx] is not None and images[idx].ndim == 3:
                image_with_masks = overlay_image(img, pred, color=overlay_rgb_color, alpha=0.3)
                axes[i, 3].imshow(image_with_masks)
                axes[i, 3].set_title('Overlap')
            else:
                axes[i, 3].axis('off')
        else:
            print(f"Warning: Prediction at index {idx} is not valid for visualization.")
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{sample_type}_samples.png"))
    plt.close()

"""# Dataset download function"""

# Function to download and extract the dataset
def download_and_extract(url, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    dataset_zip = os.path.join(dest_path, 'dataset.zip')

    if not os.path.exists(dataset_zip):
        print("Downloading dataset...")
        r = requests.get(url, stream=True)
        with open(dataset_zip, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

    if dataset_zip.endswith('.zip'):
        print("Extracting dataset...")
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(dest_path)
        print("Extraction complete.")

        # Delete the zip file after extraction
        os.remove(dataset_zip)
        print(f"Deleted {dataset_zip} after extraction.")

"""# Model, dataset class and test functions"""

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

class SolarPanelTestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_format, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.file_format = file_format
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Adjust mask file name to consider the specified format
        mask_name = img_name.replace(f'.{self.file_format}', f'_label.{self.file_format}')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = mask.point(lambda p: p > 80 and 255)

        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)

        mask = torch.tensor(np.array(mask, dtype=np.uint8), dtype=torch.long)
        mask = mask.squeeze()

        return image, mask

"""**Test function**"""

def test_model(device,
               settings,
               model,
               image_dir,
               mask_dir,
               output_dir,
               results_path,
               dataset_name,
               image_size):

    model_sel = settings['model'].lower()
    dict_sel = settings.get('pre_trained')
    batch_size = settings['batch_size']

    # Transformations for the dataset
    transform_images = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Set file_format
    if dataset_name.startswith("PV"):
        file_format = pv_file_format
    else:
        file_format = google_file_format

    # Create dataset
    test_dataset = SolarPanelTestDataset(image_dir=image_dir, mask_dir=mask_dir, file_format=file_format, transforms=transform_images)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Set criterion
    if model_sel == "segformer":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Test
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    all_dice_scores = []
    all_jaccard_indices = []

    with torch.no_grad():
        test_loader_tqdm = tqdm(test_loader, desc="Test")

        for batch_idx, (images, masks) in enumerate(test_loader_tqdm):
            images = images.to(device)
            masks = masks.to(device)

            if model_sel == "segformer":
                  outputs = model(pixel_values=images).logits
                  outputs = F.interpolate(outputs, size=(image_size, image_size), mode='bilinear', align_corners=False)
                  outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.size(1)) # (N, C, H, W) -> (N*H*W, C)
                  masks = masks.reshape(-1) # (N, H, W) -> (N*H*W)
                  _, predicted = torch.max(outputs, 1)
                  loss = criterion(outputs, masks.long())
                  test_loss += loss.item()
                  # Calculate metrics
                  accuracy = accuracy_score(masks.cpu().numpy(), predicted.cpu().numpy())
                  test_accuracy += accuracy
                  dice_score = f1_score(masks.cpu().numpy(), predicted.cpu().numpy(), average='binary', zero_division=0)
                  jaccard_index = jaccard_score(masks.cpu().numpy(), predicted.cpu().numpy(), average='binary', zero_division=0)
            elif model_sel == "unet":
                  outputs = model(images)
                  outputs = F.interpolate(outputs, size=(image_size, image_size), mode='bilinear', align_corners=False)
                  outputs = outputs.squeeze(1)
                  preds = torch.sigmoid(outputs)
                  predicted = (preds > 0.5).float()
                  if predicted.shape != masks.shape:
                      predicted = predicted.view(masks.shape[0], masks.shape[1], masks.shape[2])
                  loss = criterion(outputs, masks.float())  # BCEWithLogitsLoss requires float
                  test_loss += loss.item()
                  # Calculate metrics
                  accuracy = accuracy_score(masks.view(-1).cpu().numpy(), predicted.view(-1).cpu().numpy())
                  test_accuracy += accuracy
                  dice_score = f1_score(masks.view(-1).cpu().numpy(), predicted.view(-1).cpu().numpy(), average='binary', zero_division=0)
                  jaccard_index = jaccard_score(masks.view(-1).cpu().numpy(), predicted.view(-1).cpu().numpy(), average='binary', zero_division=0)
            else:
                print("Error: Invalid model selection.")

            # Append metrics
            all_dice_scores.append(dice_score)
            all_jaccard_indices.append(jaccard_index)

            test_loader_tqdm.set_postfix({"loss": loss.item(), "accuracy": accuracy, "f1-score": dice_score, "jaccard-index": jaccard_index})

    # Save test samples
    save_samples(images, masks.reshape(images.shape[0], image_size, image_size), predicted.reshape(images.shape[0], image_size, image_size), sample_type="test", output_dir=output_dir)
    print(f"Saved test samples")

    avg_test_loss = test_loss / len(test_loader)
    avg_test_accuracy = test_accuracy / len(test_loader)
    avg_dice_score = sum(all_dice_scores) / len(all_dice_scores)
    avg_jaccard_index = sum(all_jaccard_indices) / len(all_jaccard_indices)

    print(f"Test: Avg Test Loss: {avg_test_loss:.4f}")
    print(f"Test: Avg Test Accuracy: {avg_test_accuracy:.4f}")
    print(f"Test: Avg Dice Score: {avg_dice_score:.4f}, Avg Jaccard Index: {avg_jaccard_index:.4f}")

    # Write results to text file
    with open(results_path, 'w') as file:
        file.write(f'Model: {model_sel} with {dict_sel}\n')
        file.write(f'Average Test Loss: {avg_test_loss}\n')
        file.write(f'Average Test Accuracy: {avg_test_accuracy}\n')
        file.write(f'Average Dice Score: {avg_dice_score}\n')
        file.write(f'Average Jaccard Index: {avg_jaccard_index}\n')
        print(f"Results saved to {results_path}")
    return avg_test_loss, avg_test_accuracy, avg_dice_score, avg_jaccard_index

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
    ap.add_argument("--no-dataset-download", required=False, action='store_true', default=False, help="Skip dataset download")
    ap.add_argument("--no-pretrain-download", required=False, action='store_true', default=False, help="Skip pretrain model download")
    args, unknown = ap.parse_known_args()
    args = vars(args)

    json_cfg_file = args['config']
    try:
        settings = check_config(json_cfg_file)
    except ValueError as error:
        print(f"Error: Invalid configuration file ({error})")
        return 1

    print(f"Settings:\n{settings}")

    root_dir = args['root_dir']
    dict_sel = settings.get('pre_trained')

    # Dataset parameters
    dataset_path = os.path.join(root_dir,  'dataset/') # Path to the dataset
    test_image_dir = os.path.join(dataset_path, 'test/images') # Test dataset path - images
    test_mask_dir = os.path.join(dataset_path, 'test/masks')   # Test dataset path - masks

    # test parameters
    experiment_name_folder = os.path.splitext(os.path.basename(json_cfg_file))[0].replace(" ", "_")
    output_dir = os.path.join(root_dir, 'results', experiment_name_folder)                                              # Path to the output results directory
    results_path = os.path.join(output_dir, 'test_results.txt')                                  # Path to the results file

    # Model parameters
    pretrained_model_dir = os.path.join(root_dir, 'pretrained/')                                 # Path to the pretrained model dictionary folder

    """# Execution"""

    # Download and extract the dataset
    start_time = time.perf_counter()
    if args['no_dataset_download'] == False:
        download_and_extract(dataset_url, dataset_path)
    test_images_count = len(os.listdir(test_image_dir))                             # Count the number of images under test folder
    print(f"Number of images in the test set: {test_images_count}")
    elapsed_time = time.perf_counter() - start_time
    print(f"Download & extract took: {time_fmt.format_time(elapsed_time)}")

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
    if settings['model'].lower() == "segformer":
        print("Loading Segformer...")
        # Segformer model parameters
        id2label = {0: 'background', 1: 'solar_panel'}           # dictionary of solar panel labels
        label2id = {label: id for id, label in id2label.items()} # dictionary of ids associated to labels
        # Load the pretrained Segformer model
        model = SegformerForSemanticSegmentation.from_pretrained(
            seg_pretrained_model_name,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    elif settings['model'].lower() == "unet":
        print("Loading UNet...")
        # Load the pretrained UNet model
        model = unet.UNet(in_channels=3, out_channels=1)
    else:
        print("Error: Invalid model selection")
        return 1

    # Download and load the state dictionary from a pretrained model
    if dict_sel is not None:
        if args['no_pretrain_download'] == False:
            download_dict(dict_location, pretrained_model_dir, dict_sel)
        pretrained_model_path = os.path.join(pretrained_model_dir, dict_sel)
        try:
            model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
            print(f"Successfully loaded the model from {pretrained_model_path}")
        except Exception as e:
            print(f"Error loading the model from {pretrained_model_path}: {e}")

    # Move model to GPU if available
    model.to(device)

    """**Run test**"""

    # Run test
    start_time = time.perf_counter()
    print(f"Running test")
    test_model(device,
               settings,
               model,
               image_dir=test_image_dir,
               mask_dir=test_mask_dir,
               output_dir=output_dir,
               results_path=results_path,
               dataset_name=dataset_name,
               image_size=image_size)
    elapsed_time = time.perf_counter() - start_time
    print(f"Test took: {time_fmt.format_time(elapsed_time)}\n")

    """**Zip results**"""

    # Create the zip file
    zip_filename = 'Solar_Panel_Detector_Test_' + experiment_name_folder + '.zip'   # Name of the zip file to save the experiment outputs
    exclude_folders = ['sample_data', 'dataset', 'trash', '.config', zip_filename]  # Paths to exclude in zip file
    ziptools.zip_dir(root_dir, zip_filename, exclude_folders)
    zip_file_path = os.path.join(root_dir, zip_filename)
    ziptools.list_folders_and_files_in_zip(zip_file_path)

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit