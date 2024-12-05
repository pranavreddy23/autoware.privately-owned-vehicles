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
    root_data_path = '/mnt/media/Argoverse/'

    # Paths to read ground truth depth and input images from training data
    depth_filepath = root_data_path + 'disparity_maps_v1.1/'
    images_filepath = root_data_path + 'rectified_stereo_images_v1.1/train/'

    # Reading dataset labels and images and sorting returned list in alphabetical order
    depth_maps = sorted([f for f in pathlib.Path(depth_filepath).glob("*/stereo_front_left_rect_disparity/*.png")])
    images_left = sorted([f for f in pathlib.Path(images_filepath).glob("*/stereo_front_left_rect/*.jpg")])
    images_right = sorted([f for f in pathlib.Path(images_filepath).glob("*/stereo_front_right_rect/*.jpg")])

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

if __name__ == '__main__':
    main()
#%%