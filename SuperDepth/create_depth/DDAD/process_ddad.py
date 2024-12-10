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
from SuperDepth.create_depth.common.height_map import HeightMap

def main():
    
    # Filepaths for data loading and saving
    root_data_path = '/mnt/media/ddad_train_val/'
    root_save_path = '/mnt/media/SuperDepth/DDAD/'

    # Reading dataset labels and images and sorting returned list in alphabetical order
    depth_maps = sorted([f for f in pathlib.Path(root_data_path).glob("*/point_cloud/LIDAR/*.npz")])
    images = sorted([f for f in pathlib.Path(root_data_path).glob("*/rgb/CAMERA_01/*.png")])
    calibs = sorted([f for f in pathlib.Path(root_data_path).glob("*/calibration/*.json")])

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