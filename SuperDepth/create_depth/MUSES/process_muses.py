#! /usr/bin/env python3
#%%
import pathlib
import numpy as np
from PIL import Image
import json
import sys
sys.path.append('../../../')
from Models.data_utils.check_data import CheckData
from SuperDepth.create_depth.common.lidar_depth_fill import LidarDepthFill
from SuperDepth.create_depth.common.stereo_sparse_supervision import StereoSparseSupervision
from SuperDepth.create_depth.common.height_map import HeightMap

def main():
    
    # Filepaths for data loading and saving
    root_data_path = '/mnt/media/MUSES/'

    # Paths to read ground truth depth and input images from training data
    depth_filepath = root_data_path + 'lidar/'
    images_filepath = root_data_path + 'frame_camera/'
    calib_path =  root_data_path + 'calib.json'

    # Reading dataset labels and images and sorting returned list in alphabetical order
    depth_maps = sorted([f for f in pathlib.Path(depth_filepath).glob("*.bin")])
    images = sorted([f for f in pathlib.Path(images_filepath).glob("*.png")])

    # If all data checks have been passed
    num_depth_maps = len(depth_maps)
    num_images = len(images)
   
    checkData = CheckData(num_images, num_depth_maps)
    check_passed = checkData.getCheck()

    if(check_passed):
        print('Beginning processing of data')

if __name__ == '__main__':
    main()
#%%