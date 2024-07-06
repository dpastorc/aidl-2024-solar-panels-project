import os
import shutil

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
            
# Function to move list of files from source folder to destination folder
def move_files_list(file_list, source_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for file_name in file_list:
        shutil.move(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))