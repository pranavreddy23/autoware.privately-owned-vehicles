#! /usr/bin/env python3
#%%
# Comment above is for Jupyter execution in VSCode
import pathlib
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def createHeightMap(depth_map):

    # Getting size of depth map
    size = depth_map.shape
    height = size[0]
    width = size[1]

    # Projection centre for Y-axis
    cy = round(height/2)

    # Initializing height-map
    height_map = np.zeros_like(depth_map)

    # Height of camera above ground plane
    camera_height = 1.35 
    
    for i in range(0, height):
        for j in range(0, width):
            depth_val = depth_map[i, j]
            H = (cy-i)*(depth_val)/1024
            height_map[i,j] = H + camera_height
    
    # Clipping height values for dataset
    height_map = height_map.clip(min = -0.5, max = 7)
    
    return height_map

def createDepthMap(depth_data):

    # Getting size of depth data
    size = depth_data.shape
    height = size[0]
    width = size[1]

    # Converting depth data to metric depth values
    depth_map = Image.fromarray(depth_data)
    depth_map = np.asarray(depth_map, dtype=np.float32)
    depth_map = 400 * (1 - depth_map) 

    # Removing erroneous depth data
    max_depth = np.max(depth_map)
    for i in range(0, height):
        for j in range(0, width):
            if (depth_map[i,j] <= 0):
                depth_map[i,j] = max_depth

    return depth_map

def findDepthBoundaries(depth_map):

    # Getting size of depth map
    size = depth_map.shape
    height = size[0]
    width = size[1]

    # Initializing depth boundary mask
    depth_boundaries = np.zeros_like(depth_map)

    # Fiding depth boundaries
    for i in range(1, height-1):
        for j in range(1, width-1):

            # Finding derivative
            x_grad = depth_map[i-1,j] - depth_map[i+1, j]
            y_grad = depth_map[i,j-1] - depth_map[i, j+1]
            grad = abs(x_grad) + abs(y_grad)
            
            # Derivative threshold
            if(grad > 10):
                depth_boundaries[i,j] = 1

    return depth_boundaries

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
    
    # If all data checks have been passed
    if(is_depth_path_valid and is_image_path_valid and is_data_valid):

        print('Beginning processing of data')
        
        # Looping through data
        for index in range(0, 1):
            
            # Open images and pre-existing masks
            image = Image.open(str(images[index]))
            depth_data = cv2.imread(str(depth_maps[index]), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            
            # Create metric depth map and height map
            depth_map = createDepthMap(depth_data)
            depth_boundaries = findDepthBoundaries(depth_map)
            height_map = createHeightMap(depth_map)
            

            plt.figure()
            plt.title('Image')
            plt.imshow(image)
            plt.figure()
            plt.title('Depth')
            plt.imshow(np.log(depth_map), cmap='inferno_r')
            plt.figure()
            plt.title('Depth Boundaries')
            plt.imshow(depth_boundaries)
            plt.figure()
            plt.title('Height')
            plt.imshow(height_map, cmap='inferno_r')
            


if __name__ == '__main__':
    main()
#%%