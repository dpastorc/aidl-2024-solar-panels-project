import os
import sys
import argparse
import json

"""# Libraries"""

# Import libraries - TBD

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

from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from sklearn.metrics import accuracy_score, jaccard_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from packages import time_fmt, plot, zip
from models import unet

# Quick configuration and Hyperparameters
dataset_name = 'ZENODO-split'                                                   # Datasets available: PV01-split, PV-ALL-split, PV03-CROP-split, ZENODO-split
seg_pretrained_model_name = "nvidia/segformer-b0-finetuned-ade-512-512"         # Pretrained Segformer model range from b0 to b5. Ignore for UNet selection
dict_sel = None                                                                 # Dictionary of the selected fine-tuned model from previous iterations; Use None to train from scratch.
image_size = 256                                                                # Image size for training
val_interval = 5                                                                # Epochs interval to execute validation cycles

# Learning Rate Hyperparameter Optimization
early_stop = False                                                              # Early stopping: True or False
early_stop_patience = 10                                                        # Number of epochs to wait after the last improvement before stopping the training

"""# Other parameters and paths"""

overlay_rgb_color=(255, 0, 255)

# Dataset parameters
pv_file_format = 'bmp'                                                          # File format within the PVXX dataset
google_file_format = 'png'                                                      # File format within the Google dataset
dataset_url = 'https://temp-posgraduation.s3.amazonaws.com/' + dataset_name + '.zip' # Location of the preprocessed and split dataset
root_dir = '/content/'                                                          # Root directory

# Training parameters
num_samples = 10                                                                # Number of train and validation samples to save

# Model parameters
dict_location = 'https://temp-posgraduation.s3.amazonaws.com/'                  # Base public URL where pretrained model dictionaries have been placed for download
model_name = 'solar_panel_detector_train.pth'                                   # Name of the trained model dictionary to save

# Zip parameters
zip_filename = 'Solar_Panel_Detector_Train.zip'                                  # Name of the zip file to save the experiment outputs
exclude_folders = ['sample_data', 'dataset', 'model', '.config', zip_filename]  # Paths to exclude in zip file


"""# Supporting Functions"""

"""**Functions to manage files and folders**"""

# Delete desired folders
paths_to_remove = "/content/results /content/samples /content/plots"   # "/content/samples /content/plots /content/sample_data /content/model"
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
def save_samples(images, masks, predictions, epoch, sample_type, output_dir="results", num_samples=8):
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
    plt.savefig(os.path.join(output_dir, f"epoch_{epoch+1}_{sample_type}_samples.png"))
    plt.close()

# Function to calculate epochs where samples will be saved
def calculate_save_epochs(num_epochs, val_interval, num_samples):

    # Ensure num_epochs and val_interval are valid
    if num_epochs <= 0:
        raise ValueError("num_epochs must be a positive integer")
    if val_interval <= 0:
        raise ValueError("val_interval must be a positive integer")

    # Calculate the epochs where validation occurs based on the validation interval
    validation_epochs = list(range(val_interval - 1, num_epochs, val_interval))

    # Ensure the first and last epochs are included in the save_epochs
    if 0 not in validation_epochs:
        validation_epochs.insert(0, 0)
    if num_epochs - 1 not in validation_epochs:
        validation_epochs.append(num_epochs - 1)

    # If we have more than the desired number of samples, downsample to the desired number
    if len(validation_epochs) > num_samples:
        save_epochs = np.linspace(0, len(validation_epochs) - 1, num=num_samples, dtype=int)
        save_epochs = [validation_epochs[i] for i in save_epochs]
    else:
        save_epochs = validation_epochs

    return save_epochs

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

"""**Data augmentation function**"""

# Function to apply data augmentation

def apply_data_augmentation(config, image, mask):

    # Convert PyTorch Tensor to PIL Image
    image = TF.to_pil_image(image)
    mask = TF.to_pil_image(mask)

    if config['flip']:
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

    if config['rotation']:
        # Random rotation
        angle = random.choice([0, 90, 180, 270])
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

    if config['brightness']:
        # Random brightness adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, factor)

    if config['contrast']:
        # Random contrast adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = TF.adjust_contrast(image, factor)

    if config['saturation']:
        # Random saturation adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = TF.adjust_saturation(image, factor)

    if config['hue']:
        # Random hue adjustment
        if random.random() > 0.5:
            factor = random.uniform(-0.1, 0.1)  # Hue factor is between -0.5 and 0.5
            image = TF.adjust_hue(image, factor)

    if config['blur']:
        # Random blur
        if random.random() > 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2)))

    if config['sharpen']:
        # Random sharpening
        if random.random() > 0.5:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(random.uniform(1.0, 2.0))

    if config['gaussian_noise']:
        # Add Gaussian noise
        image = np.array(image) / 255.0
        noise = np.random.normal(0, 0.1, image.shape)
        image = np.clip(image + noise, 0, 1)
        image = Image.fromarray((image * 255).astype(np.uint8))

    if config['random_padding']:
        # Randomly replace part of the image with white or black padding
        if random.random() > 0.5:
            padding_color = random.choice([0, 255])  # Black or white padding
            draw = ImageDraw.Draw(image)
            draw_mask = ImageDraw.Draw(mask)
            width, height = image.size
            pad_width = random.randint(int(width * 0.1), int(width * 0.5))
            pad_height = random.randint(int(height * 0.1), int(height * 0.5))
            x0 = random.randint(0, width - pad_width)
            y0 = random.randint(0, height - pad_height)
            draw.rectangle([x0, y0, x0 + pad_width, y0 + pad_height], fill=padding_color)
            draw_mask.rectangle([x0, y0, x0 + pad_width, y0 + pad_height], fill=0)

    if config['random_polygons']:
        # Include small white or black polygons randomly within the image
        if random.random() > 0.5:
            polygon_color = random.choice([0, 255])  # Black or white polygons
            draw = ImageDraw.Draw(image)
            draw_mask = ImageDraw.Draw(mask)
            width, height = image.size
            num_polygons = random.randint(1, 5)
            for _ in range(num_polygons):
                shape = random.choice(['square', 'rectangle', 'L'])
                if shape == 'square':
                    size = random.randint(int(min(width, height) * 0.05), int(min(width, height) * 0.2))
                    x0 = random.randint(0, width - size)
                    y0 = random.randint(0, height - size)
                    draw.rectangle([x0, y0, x0 + size, y0 + size], fill=polygon_color)
                    draw_mask.rectangle([x0, y0, x0 + size, y0 + size], fill=0)
                elif shape == 'rectangle':
                    w = random.randint(int(width * 0.05), int(width * 0.2))
                    h = random.randint(int(height * 0.05), int(height * 0.2))
                    x0 = random.randint(0, width - w)
                    y0 = random.randint(0, height - h)
                    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=polygon_color)
                    draw_mask.rectangle([x0, y0, x0 + w, y0 + h], fill=0)
                elif shape == 'L':
                    w = random.randint(int(width * 0.05), int(width * 0.15))
                    h = random.randint(int(height * 0.05), int(height * 0.15))
                    x0 = random.randint(0, width - w * 2)
                    y0 = random.randint(0, height - h * 2)
                    draw.rectangle([x0, y0, x0 + w, y0 + h * 2], fill=polygon_color)
                    draw.rectangle([x0, y0, x0 + w * 2, y0 + h], fill=polygon_color)
                    draw_mask.rectangle([x0, y0, x0 + w, y0 + h * 2], fill=0)
                    draw_mask.rectangle([x0, y0, x0 + w * 2, y0 + h], fill=0)

    # Convert back to PyTorch Tensor
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)

    return image, mask

"""**Dataset class**"""

# Define the custom dataset
class SolarPanelTrainDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_format='bmp', transforms=None, augmentation_cfg=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_format = file_format  # Specify the file format for both images and masks
        self.transforms = transforms
        self.images = os.listdir(image_dir)
        self.augmentation_cfg = augmentation_cfg

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

            if self.augmentation_cfg != None:
                # Apply augmentation to images with solar panels only or all
                apply_augmentation = (not self.augmentation_cfg['contain_solar_panel'] or torch.any(mask > 0)) # Check if the mask contains any pixel value other than 0
                if apply_augmentation:
                    image, mask = apply_data_augmentation(self.augmentation_cfg, image, mask)

        mask = torch.tensor(np.array(mask, dtype=np.uint8), dtype=torch.long)

        # Ensure the mask tensor has the shape (H, W)
        mask = mask.squeeze()

        return image, mask

"""**Train function**"""

def train_model(device, 
                settings, 
                model, 
                early_stop, 
                train_image_dir, train_mask_dir, 
                val_image_dir, val_mask_dir, 
                output_dir,
                results_path, 
                dataset_name,  
                val_interval, 
                num_samples, 
                image_size, 
                early_stop_patience):

    model_sel = settings['model'].lower()
    batch_size = settings['batch_size']
    num_epochs = settings['epochs']
    lr = settings['lr']

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

    # Create datasets
    train_dataset = SolarPanelTrainDataset(image_dir=train_image_dir,
                                           mask_dir=train_mask_dir,
                                           file_format=file_format,
                                           transforms=transform_images,
                                           augmentation_cfg=settings['augmentation'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    val_dataset = SolarPanelTrainDataset(image_dir=val_image_dir,
                                         mask_dir=val_mask_dir,
                                         file_format=file_format,
                                         transforms=transform_images,
                                         augmentation_cfg=settings['augmentation'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Set criterion and optimizer
    if model_sel == "segformer":
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train and Validate
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    ious = []
    f1s = []

    # Calculate epochs to save samples
    save_epochs = calculate_save_epochs(num_epochs, val_interval, num_samples)

    # Early stop variables
    if early_stop:
        lr_patience = (early_stop_patience - 1) // 2                            # 2:1 ratio for lr patience from early stopping patience parameter
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=lr_patience)
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        optimal_lr = optimizer.param_groups[0]['lr']

    for epoch in range(num_epochs):

        # Training

        model.train()
        running_loss = 0.0
        running_accuracy = 0.0

        # Use tqdm to add a progress bar
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (images, masks) in enumerate(train_loader_tqdm):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()

            if model_sel == "segformer":
                  outputs = model(pixel_values=images).logits
                  outputs = F.interpolate(outputs, size=(image_size, image_size), mode='bilinear', align_corners=False) # Upsample the outputs to match the mask size
                  outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.size(1)) # ¡Ensure the outputs and masks have the correct shape (N, C, H, W) -> (N*H*W, C)
                  masks = masks.reshape(-1) # (N, H, W) -> (N*H*W)
                  _, predicted = torch.max(outputs, 1) # Convert outputs to class predictions
                  loss = criterion(outputs, masks.long()) # Ensure target tensor is of type torch.long
                  loss.backward()
                  optimizer.step()
                  running_loss += loss.item()
                  accuracy = accuracy_score(masks.cpu().numpy(), predicted.cpu().numpy())
                  running_accuracy += accuracy
            elif model_sel == "unet":
                  outputs = model(images)
                  outputs = F.interpolate(outputs, size=(image_size, image_size), mode='bilinear', align_corners=False)
                  outputs = outputs.squeeze(1)
                  preds = torch.sigmoid(outputs)
                  predicted = (preds > 0.5).float()
                  loss = criterion(outputs, masks.float())  # BCEWithLogitsLoss requires float
                  loss.backward()
                  optimizer.step()
                  running_loss += loss.item()
                  # Ensure binary targets for accuracy calculation
                  accuracy = accuracy_score(masks.view(-1).cpu().detach().numpy().astype(int), predicted.view(-1).cpu().detach().numpy().astype(int))
                  running_accuracy += accuracy
            else:
                print("Error: Invalid model selection.")

            train_loader_tqdm.set_postfix({"loss":loss.item(), "accuracy":accuracy})

        # Save training samples if the current epoch is in the save_epochs list
        if epoch in save_epochs:
            save_samples(images, masks.reshape(images.shape[0], image_size, image_size), predicted.reshape(images.shape[0], image_size, image_size), epoch, sample_type="train", output_dir=output_dir, num_samples=num_samples)
            print(f"Epoch {epoch+1}/{num_epochs}, Saved training samples")

        # Append metrics
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(running_accuracy / len(train_loader))
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Train Loss: {train_losses[-1]}, Avg Train Accuracy: {train_accuracies[-1]}")

        #
        # Validation
        #
        if (epoch + 1) % val_interval == 0: # Execute based on validation interval
            model.eval()
            val_loss = 0.0
            val_accuracy = 0.0
            all_dice_scores = []
            all_jaccard_indices = []

            with torch.no_grad():

                val_loader_tqdm = tqdm(val_loader, desc="Validation")

                for batch_idx, (images, masks) in enumerate(val_loader_tqdm):
                    images = images.to(device)
                    masks = masks.to(device)

                    if model_sel == "segformer":
                          outputs = model(pixel_values=images).logits
                          outputs = F.interpolate(outputs, size=(image_size, image_size), mode='bilinear', align_corners=False) # Upsample the outputs to match the mask size
                          outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.size(1)) # ¡Ensure the outputs and masks have the correct shape (N, C, H, W) -> (N*H*W, C)
                          masks = masks.reshape(-1) # (N, H, W) -> (N*H*W)
                          _, predicted = torch.max(outputs, 1) # Convert outputs to class predictions
                          loss = criterion(outputs, masks.long()) # Ensure target tensor is of type torch.long
                          val_loss += loss.item()
                          # Calculate metrics for the current batch
                          accuracy = accuracy_score(masks.cpu().numpy(), predicted.cpu().numpy())
                          val_accuracy += accuracy
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
                          val_loss += loss.item()
                          # Calculate metrics for the current batch
                          accuracy = accuracy_score(masks.view(-1).cpu().numpy(), predicted.view(-1).cpu().numpy())
                          val_accuracy += accuracy
                          dice_score = f1_score(masks.view(-1).cpu().numpy(), predicted.view(-1).cpu().numpy(), average='binary', zero_division=0)
                          jaccard_index = jaccard_score(masks.view(-1).cpu().numpy(), predicted.view(-1).cpu().numpy(), average='binary', zero_division=0)
                    else:
                        print("Error: Invalid model selection.")

                    # Append metrics
                    all_dice_scores.append(dice_score)
                    all_jaccard_indices.append(jaccard_index)

                    val_loader_tqdm.set_postfix({"loss": loss.item(), "accuracy": accuracy, "f1-score": dice_score, "jaccard-index": jaccard_index})

            # Save validation samples if the current epoch is in the save_epochs list
            if epoch in save_epochs:
                save_samples(images, masks.reshape(images.shape[0], image_size, image_size), predicted.reshape(images.shape[0], image_size, image_size), epoch, sample_type="val", output_dir=output_dir, num_samples=num_samples)
                print(f"Epoch {epoch+1}/{num_epochs}, Saved validation samples")

            # Append metrics
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_accuracy / len(val_loader))
            avg_dice_score = sum(all_dice_scores) / len(all_dice_scores)
            avg_jaccard_index = sum(all_jaccard_indices) / len(all_jaccard_indices)

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Validation: Avg Train Loss: {train_losses[-1]:.4f}, Avg Val Loss: {val_losses[-1]:.4f}")
            print(f"Validation: Avg Train Accuracy: {train_accuracies[-1]:.4f}, Avg Val Accuracy: {val_accuracies[-1]:.4f}")
            print(f"Validation: Avg Dice Score: {avg_dice_score:.4f}, Avg Jaccard Index: {avg_jaccard_index:.4f}")

            ious.append(avg_jaccard_index)
            f1s.append(avg_dice_score)

            # Early Stop Check
            if early_stop:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"Best validation loss so far: {best_val_loss:.4f}")
                    epochs_without_improvement = 0
                    optimal_lr = optimizer.param_groups[0]['lr']
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= early_stop_patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        print(f"Optimal learning rate identified: {optimal_lr:.8f}")
                        break

                # Step the scheduler based on the validation loss
                if len(val_losses) > 0:
                    scheduler.step(val_losses[-1])

                    # Print scheduler's learning rate adjustment
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Scheduler step: Learning rate is now {current_lr:.8f}")

    # Save and show the final plots at the end of training
    plot.plot_charts(train_losses, train_accuracies, val_losses, val_accuracies, ious, f1s, val_interval)
    plot.save_plots(train_losses, train_accuracies, val_losses, val_accuracies, ious, f1s, val_interval, epoch=None, output_dir=output_dir)
    print(f"Saved final plots")

    return {'train_losses':train_losses, 'train_accuracies':train_accuracies,
            'val_losses':val_losses, 'val_accuracies':val_accuracies,
            'ious':ious, 'f1s':f1s}

# Save the model
def save_model(model, model_dir, model_name):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model at {model_path}")

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
    args = vars(ap.parse_args())

    json_cfg_file = args['config']
    try:
        settings = check_config(json_cfg_file)
    except ValueError as error:
        print(f"Error: Invalid configuration file ({error})")
        return 1

    print(f"Settings:\n{settings}")

    root_dir = args['root_dir']

    # Dataset parameters
    dataset_path = os.path.join(root_dir, 'dataset/')                                            # Path to the dataset
    train_image_dir = os.path.join(dataset_path, 'train/images')                                 # Train dataset path - images
    train_mask_dir = os.path.join(dataset_path, 'train/masks')                                   # Train dataset path - masks
    val_image_dir = os.path.join(dataset_path, 'val/images')                                     # Validation dataset path - images
    val_mask_dir = os.path.join(dataset_path, 'val/masks')                                       # Validation dataset path - masks

    # Training parameters
    experiment_name_folder = os.path.splitext(os.path.basename(json_cfg_file))[0].replace(" ", "_")
    output_dir = os.path.join(root_dir, 'results', experiment_name_folder)                                              # Path to the output results directory
    results_path = os.path.join(output_dir, 'test_results.txt')                                  # Path to the results file

    # Model parameters
    model_dir = os.path.join(root_dir, 'model/')                                                 # Path to the pretrained model dictionary folder

    ########################
    
    """# Execution """

    # Download and extract the dataset
    start_time = time.perf_counter()
    if args['no_pretrain_download'] == False:
        download_and_extract(dataset_url, dataset_path)
    train_images_count = len(os.listdir(train_image_dir))                           # Count the number of images under train folder
    val_images_count = len(os.listdir(val_image_dir))                               # Count the number of images under validation folder
    print(f"Number of images in the train set: {train_images_count}")
    print(f"Number of images in the validation set: {val_images_count}")
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

    # Download and load the state dictionary from a pretrained model
    if dict_sel is not None:
        download_dict(dict_location, model_dir, dict_sel)
        pretrained_model_path = os.path.join(model_dir, dict_sel)
        try:
            model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
            print(f"Successfully loaded the model from {pretrained_model_path}")
        except Exception as e:
            print(f"Error loading the model from {pretrained_model_path}: {e}")

    # Move model to GPU if available
    model.to(device)

    """**Run training and validation**"""

    # Run training and validation
    start_time = time.perf_counter()
    print(f"Running test")
    model_train_log = train_model(device,
                                  settings,
                                  model, 
                                  early_stop,
                                  train_image_dir, train_mask_dir,
                                  val_image_dir, val_mask_dir,
                                  output_dir, results_path,
                                  dataset_name, 
                                  val_interval, 
                                  num_samples, 
                                  image_size,
                                  early_stop_patience)
    save_model(model, output_dir, model_name)
    elapsed_time = time.perf_counter() - start_time
    print(f"Training and validation took: {time_fmt.format_time(elapsed_time)}\n")

    """**Zip results**"""

    # Create the zip file
    #zip.zip_dir(root_dir, zip_filename, exclude_folders)
    #zip_file_path = os.path.join(root_dir, zip_filename)
    #zip.list_folders_and_files_in_zip(zip_file_path)

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit