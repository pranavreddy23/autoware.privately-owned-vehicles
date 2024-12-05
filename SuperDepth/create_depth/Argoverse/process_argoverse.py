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

if __name__ == '__main__':
    main()
#%%