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

    print('Found', num_depth_maps, 'depth maps and ', num_images, ' images')

if __name__ == '__main__':
    main()