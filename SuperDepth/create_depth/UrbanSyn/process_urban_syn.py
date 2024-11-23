import pathlib
import cv2
from PIL import Image
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def removeExtraSamples(depth_filepath, depth_maps, images_filepath, images):
    
    depth_path = depth_filepath + 'depth_'
    depth_map_indices = []

    for i in range(0, len(depth_maps)):
        depth_map_file = str(depth_maps[i])
        depth_map_index = depth_map_file[:-4]
        depth_map_index = depth_map_index.replace(depth_path,'')
        depth_map_indices.append(depth_map_index)

    image_path = images_filepath + 'rgb_'
    filtered_images = []

    for i in range(0, len(images)):
        image_file = str(images[i])
        image_index = image_file[:-4]
        image_index = image_index.replace(image_path,'')
        
        if(image_index in depth_map_indices):
            filtered_images.append(images[i])

    return filtered_images


def checkData(num_depth_maps, num_images):
    
    is_depth_path_valid = False
    is_image_path_valid = False
    is_data_valid = False
    check_passed = False

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
    if (num_depth_maps != num_images):
        raise ValueError('Number of ground truth depth maps does not match number of input images:')
    else:
        is_data_valid = True
    
    # Final check
    if(is_depth_path_valid and is_image_path_valid and is_data_valid):
        check_passed = True
    
    return check_passed

def main():

    # Filepaths for data loading and savind
    root_data_path = '/home/zain/Autoware/Privately_Owned_Vehicles/training_data/SuperDepth/UrbanSyn/'
    root_save_path = '/mnt/media/SuperDepth/UrbanSyn'

    # Paths to read ground truth depth and input images from training data
    depth_filepath = root_data_path + 'depth/'
    images_filepath = root_data_path + 'rgb/'

    # Reading dataset labels and images and sorting returned list in alphabetical order
    depth_maps = sorted([f for f in pathlib.Path(depth_filepath).glob("*.exr")])
    images = sorted([f for f in pathlib.Path(images_filepath).glob("*.png")])

    # Remove extra samples
    images = removeExtraSamples(depth_filepath, depth_maps, images_filepath, images)
 
    # If all data checks have been passed
    num_depth_maps = len(depth_maps)
    num_images = len(images)

    check_passed = checkData(num_depth_maps, num_images)

    if(check_passed):

        print('Beginning processing of data')
        


if __name__ == '__main__':
    main()
#%%