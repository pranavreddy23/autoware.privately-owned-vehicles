#! /usr/bin/env python3
#%%
import pathlib
import numpy as np
from PIL import Image
import sys
sys.path.append('../../../')
from Models.data_utils.check_data import CheckData
from SuperDepth.create_depth.common.lidar_depth_fill import LidarDepthFill
from SuperDepth.create_depth.common.stereo_sparse_supervision import StereoSparseSupervision
from SuperDepth.create_depth.common.height_map import HeightMap

def main():

    # Filepaths for data loading and savind
    root_data_path = '/mnt/media/Driving_Stereo/'
    root_save_path = '/mnt/media/SuperDepth/KITTI'

    # Paths to read ground truth depth and input images from training data
    depth_filepath = root_data_path + 'depth/'
    images_left_filepath = root_data_path + 'left_rgb/'
    images_right_filepath = root_data_path + 'right_rgb/'

    # Reading dataset labels and images and sorting returned list in alphabetical order
    depth_maps = sorted([f for f in pathlib.Path(depth_filepath).glob("*.png")])
    images_left = sorted([f for f in pathlib.Path(images_left_filepath).glob("*.jpg")])
    images_right = sorted([f for f in pathlib.Path(images_right_filepath).glob("*.jpg")])

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

        # Focal length of camera
        focal_length = 1002.53
        # Projection centre for Y-axis
        cy = 195.96
        # Camera mounting height above ground
        camera_height = 0
        # Stereo camera baseline distance
        baseline = 0.54 

        # Height map limits
        max_height = 7
        min_height = -0.5

if __name__ == '__main__':
    main()
#%%