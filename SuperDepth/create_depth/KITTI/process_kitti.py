#! /usr/bin/env python3
#%%
import pathlib
import numpy as np
from PIL import Image

def createDepthMap(depth_data):

    assert(np.max(depth_data) > 255)

    depth_map = depth_data.astype(np.float) / 256.
    depth_map[depth_data == 0] = -1.
    return depth_map

def removeExtraSamples(image_folders):
    
    filtered_images = []

    for i in range(0, len(image_folders)):
        images = sorted([f for f in pathlib.Path(str(image_folders[i])).glob("*image_02/*data/*.png")])
       
        for j in range(5, len(images) - 5):
            filtered_images.append(images[j])

    return filtered_images

def main():

    # Filepaths for data loading and savind
    root_data_path = '/mnt/media/KITTI/'
    root_save_path = '/mnt/media/SuperDepth/UrbanSyn'

    # Paths to read ground truth depth and input images from training data
    depth_filepath = root_data_path + 'train/'
    images_filepath = root_data_path + 'data/'

    # Reading dataset labels and images and sorting returned list in alphabetical order
    depth_maps = sorted([f for f in pathlib.Path(depth_filepath).glob("*/proj_depth/*groundtruth/*image_02/*.png")])
    image_folders = sorted([f for f in pathlib.Path(images_filepath).glob("*")])

    # Remove extra samples
    images = removeExtraSamples(image_folders)

    # If all data checks have been passed
    num_depth_maps = len(depth_maps)
    num_images = len(images)
    print(num_images, num_depth_maps)
    

if __name__ == '__main__':
    main()
#%%