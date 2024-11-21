#! /usr/bin/env python3
import pathlib
from PIL import Image

def main():

    root_filepath = '/home/zain/Autoware/Privately_Owned_Vehicles/training_data/SuperDepth/MUAD/'

    # Paths to read ground truth depth and input images from training data
    depth_filepath = root_filepath + 'depth/'
    images_filepath = root_filepath + '/rgb'

    # Reading dataset labels and images and sorting returned list in alphabetical order
    depth_maps = sorted([f for f in pathlib.Path(depth_filepath).glob("*.exr")])
    images = sorted([f for f in pathlib.Path(images_filepath).glob("*.png")])

    num_depth_maps = len(depth_maps)
    num_images = len(images)

    is_depth_path_valid = False
    is_image_path_valid = False
    is_data_valid = False

    # Checking if ground truth labels were read and logging error if missing
    if (num_depth_maps > 0):
        print(f'Found {num_depth_maps} ground truth masks')
        is_depth_path_valid = True
    else:
        raise ValueError('No ground truth .exr depth maps found - check your depth filepath:')

    # Checking if input images were read and logging error if missing
    if (num_images > 0):
        print(f'Found {num_images} input images')
        is_image_path_valid = True
    else:
        raise ValueError('No input png images found - check your images filepath')

    # Checking if number of ground truth labels matches number of input images
    # and logging error if mismatched
    if (num_depth_maps != num_images):
        raise ValueError('Number of ground truth depth maps does not match number of input images:')
    else:
        is_data_valid = True

if __name__ == '__main__':
    main()