#!/usr/bin/env python
import os
import sys
import argparse
import re
import math
import matplotlib.pyplot as plt
from PIL import Image

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

closed = False

def handle_close(evt):
    global closed
    closed = True

def waitforbuttonpress():
    while plt.waitforbuttonpress(0.2) is None:
        if closed:
            return False
    return True

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
    
def exponent_power2(n):
    ones_positions = [p for p,v in enumerate(bin(n)[:1:-1]) if int(v)]
    if not len(ones_positions) == 1:
        raise ValueError(f"{n} is not 2**n number where n is even") 
    return ones_positions[0]

def check_valid_crop_value(num_tiles):
    exponent = exponent_power2(num_tiles)
    if exponent > 1 and not (exponent % 2) == 0:
        raise ValueError(f"{num_tiles} is not 2**n number where n is even") 

def main() -> int:
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-i", "--input", required=True, help="input folder")
    ap.add_argument("-o", "--output", required=True, help="output folder")
    ap.add_argument("-c", "--crop", required=True, help="number of squared tiles (should be 2**n where n is even)")
    args = vars(ap.parse_args())

    num_tiles = int(args['crop'])
    try:
        check_valid_crop_value(num_tiles)
    except ValueError as error:
        print(f"Error: {error}")
        return 1

    croped_files = crop(os.path.realpath(args['input']), os.path.realpath(args['output']), num_tiles)
    
    if len(croped_files) > 0:
        num_rows = int(math.sqrt(num_tiles))
        num_cols = int(math.sqrt(num_tiles))
        for crop_file in croped_files:
            fig = plt.figure(figsize=(10, 7))
            fig.canvas.mpl_connect('close_event', handle_close)
            #fig.subplots_adjust(wspace=0.01, hspace=-0.18)
            index = 1
            for tile in crop_file:
                fig.add_subplot(num_rows, num_cols, index) 
                plt.imshow(tile) 
                plt.axis('off')
                index = index + 1
        waitforbuttonpress()

    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
