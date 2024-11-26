#! /usr/bin/env python3
#%%
import pathlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append('../../../')
from Models.data_utils.check_data import CheckData

def removeExtraSamples(image_folders):
    
    filtered_images = []

    for i in range(0, len(image_folders)):
        images = sorted([f for f in pathlib.Path(str(image_folders[i])).glob("*image_02/*data/*.png")])
       
        for j in range(5, len(images) - 5):
            filtered_images.append(images[j])

    return filtered_images

def createDepthMap(depth_data):

    assert(np.max(depth_data) > 255)

    depth_map = depth_data.astype('float32') / 256.

    depth_map = filterDepthMap(depth_map)
    return depth_map

def filterDepthMap(depth_map):

    erosion_kernel = np.ones((3,3), np.uint8) 
    depth_map = cv2.dilate(depth_map, erosion_kernel, iterations=4) 

    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19,19))
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, closing_kernel)
    
    # Getting size of depth map
    size = depth_map.shape
    height = size[0]
    width = size[1]

    # Interpolate along height
    for j in range(0, width):
        interp_depth = 0
        for i in range(height-1, 1, -1):

            lower_depth = depth_map[i,j]
            upper_depth = depth_map[i-1, j]

            if(lower_depth != 0 and upper_depth == 0):
                interp_depth = lower_depth
            
            if(interp_depth != 0 and upper_depth == 0):
                depth_map[i-1, j] = interp_depth
    
    depth_map = cv2.medianBlur(depth_map, 5)
    return depth_map

def cropData(image, depth_map):

    # Getting size of depth map
    size = depth_map.shape
    height = size[0]
    width = size[1]

    # Cropping out those parts of data for which depth is unavailable
    depth_map = depth_map[:, 100 : width - 100]
    image = image.crop((100, 0, width - 100, height-1))

    return image, depth_map

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

    check_passed = CheckData(num_images, num_depth_maps)

    if(check_passed):

        print('Beginning processing of data')
        # Looping through data
        for index in range(0, 1):

            print(f'Processing image {index} of {num_images-1}')
            
            # Open images and pre-existing masks
            image = Image.open(str(images[index]))
            depth_data = np.array(Image.open(str(depth_maps[index])), dtype=int)
            depth_map = createDepthMap(depth_data)
            image, depth_map = cropData(image, depth_map)

            plt.figure()
            plt.imshow(image)
            plt.title('Image')
            plt.figure()
            plt.imshow(depth_map, cmap='inferno')
            plt.title('Depth')
    

if __name__ == '__main__':
    main()
#%%