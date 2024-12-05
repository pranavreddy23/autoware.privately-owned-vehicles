#! /usr/bin/env python3
#%%
import pathlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import sys
sys.path.append('../../../')
from Models.data_utils.check_data import CheckData
from SuperDepth.create_depth.common.lidar_depth_fill import LidarDepthFill
from SuperDepth.create_depth.common.stereo_sparse_supervision import StereoSparseSupervision
from SuperDepth.create_depth.common.height_map import HeightMap

def parseCalib(calib_files):
    
    calib_logs = []
    focal_lengths = []
    centre_y_vals = []

    for i in range (0, len(calib_files)):
        
        # Get filepath to calibration file
        calib_filepath = str(calib_files[i])
        
        # Storing log associated with calibration file
        calib_log = calib_filepath[56:-37]
        calib_logs.append(calib_log)

        # Reading calibration file JSON
        with open(calib_filepath, 'r') as file:
            data = json.load(file)

            for p in range(0, len(data['camera_data_'])):
                camera = (data['camera_data_'][p]['key'])

                # Read focal length and principal point y value
                if(camera == 'image_raw_stereo_front_left_rect'):
                    focal_length = (data['camera_data_'][p]['value']['focal_length_y_px_'])
                    cy = (data['camera_data_'][p]['value']['focal_center_y_px_'])

                    focal_lengths.append(focal_length)
                    centre_y_vals.append(cy)

    return calib_logs, focal_lengths, centre_y_vals

def createDepthMap(depth_data, focal_length, baseline):

    assert(np.max(depth_data) > 255)
    depth_data = depth_data.astype('float32') / 256.

    valid_pixels = depth_data > 0

    # Using the stereo relationship, recover the depth map by:
    depth_map = np.float32((focal_length * baseline) / (depth_data + (1.0 - valid_pixels)))

    depth_map[depth_map > 200] = 0
    #for i in range(0, depth_map.shape[0]):
    #    for j in range(0, depth_map.shape[1]):
            
    #        if(depth_map[i,j] < 200.0):
    #            print(depth_map[i,j])
                
    #depth_map[depth_map > 0] = 200

    return depth_map       

def main():
    
    # Filepaths for data loading and savind
    root_data_path = '/mnt/media/Argoverse/'

    # Paths to read ground truth depth and input images from training data
    depth_filepath = root_data_path + 'disparity_maps_v1.1/'
    images_filepath = root_data_path + 'rectified_stereo_images_v1.1/train/'

    # Reading dataset labels and images and sorting returned list in alphabetical order
    depth_maps = sorted([f for f in pathlib.Path(depth_filepath).glob("*/stereo_front_left_rect_disparity/*.png")])
    images_left = sorted([f for f in pathlib.Path(images_filepath).glob("*/stereo_front_left_rect/*.jpg")])
    images_right = sorted([f for f in pathlib.Path(images_filepath).glob("*/stereo_front_right_rect/*.jpg")])
    calib_files = sorted([f for f in pathlib.Path(images_filepath).glob("*/*.json")])

    # If all data checks have been passed
    num_depth_maps = len(depth_maps)
    num_images_left = len(images_left)
    num_images_right = len(images_right)

    check_data_left = CheckData(num_images_left, num_depth_maps)
    check_data_right = CheckData(num_images_right, num_depth_maps)

    check_passed_left = check_data_left.getCheck()
    check_passed_right = check_data_right.getCheck()

    if(check_passed_left and check_passed_right):

        print('Beginning processing of data')

        calib_logs, focal_lengths, centre_y_vals = parseCalib(calib_files)
        
        # Stereo camera baseline distance
        baseline = 0.2986  

        # Camera height above road surface
        camera_height = 1.67

        # Height map limits
        max_height = 7
        min_height = -7

        for index in range(10, 11):

            print(f'Processing image {index} of {num_depth_maps-1}')

            # Open images and pre-existing masks
            image_left = Image.open(str(images_left[index]))
            image_right = Image.open(str(images_right[index]))
            depth_data = np.array(Image.open(str(depth_maps[index])), dtype=int)

            # Getting index associated with log
            data_log = str(str(images_left[index])[56:-69])
            log_index = calib_logs.index(data_log)
            
            # Reading focal length and principal point y-offset based on log index
            focal_length = focal_lengths[log_index]
            cy = centre_y_vals[log_index]

            # Create depth map
            sparse_depth_map = createDepthMap(depth_data, focal_length, baseline)

            # Fill in sparse depth map
            lidar_depth_fill = LidarDepthFill(sparse_depth_map)
            depth_map = lidar_depth_fill.getDepthMap()

            # Height map
            heightMap = HeightMap(depth_map, max_height, min_height, 
                 camera_height, focal_length, cy)
            height_map = heightMap.getHeightMap()


            plt.figure()
            plt.imshow(image_left)
            plt.figure()
            plt.imshow(depth_map, cmap='inferno_r')
            plt.figure()
            plt.imshow(height_map, cmap='inferno')
            
         
            

if __name__ == '__main__':
    main()
#%%