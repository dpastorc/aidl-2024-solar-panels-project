import math
import os
import re

from PIL import Image

# Genarate the filename
def __generate_output_filename(fullpath_filename:str, tile_number:int) -> str:
    _, filename = os.path.split(fullpath_filename)
    base_filename, extension = os.path.splitext(filename)
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
                    output_filename = os.path.join(output_folder, __generate_output_filename(input_file, tile_number, crop_region))
                    if (verbose):
                        print(f"Filename: {output_filename} -> {crop_region}")
                    imCrop.save(output_filename, 'JPEG', quality=90)
                    tiles_images.append(imCrop)
                    tile_number = tile_number + 1
                    tile_number = tile_number % num_tiles
                    if tiles_images:
                        tiles.append(tiles_images)
    return tiles
