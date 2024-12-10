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

def quaternion_rotation_matrix(q0, q1, q2, q3):

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = [[r00, r01, r02],
                  [r10, r11, r12],
                  [r20, r21, r22]]
                            
    return rot_matrix

def parseCalib(calib_files):
    
    calib_logs = []
    world_to_image_transforms = []
    focal_lengths = []
    cy_vals = []


    for i in range (0, len(calib_files)):
        
        # Get filepath to calibration file
        calib_filepath = str(calib_files[i])
        
        # Storing log associated with calibration file
        calib_log = calib_filepath[26:-58]
        calib_logs.append(calib_log)

        # Reading calibration file JSON
        with open(calib_filepath, 'r') as file:
            data = json.load(file)
            camera_rotation = data['extrinsics'][1]['rotation']
            camera_translation = data['extrinsics'][1]['translation']
            camera_intrinsic = data['intrinsics'][1]

            # Get world to camera projection matrix
            q0 = camera_rotation['qx']
            q1 = camera_rotation['qy']
            q2 = camera_rotation['qz']
            q3 = camera_rotation['qw']

            # Convert from quaternion to rotation matrix
            camera_rotation_matrix = quaternion_rotation_matrix(q0, q1, q2, q3)
            
            # Include translation compontent to get final 3x4 matrix
            world_to_camera_projection_matrix = camera_rotation_matrix
            world_to_camera_projection_matrix[0].append(camera_translation['x'])
            world_to_camera_projection_matrix[1].append(camera_translation['y'])
            world_to_camera_projection_matrix[2].append(camera_translation['z'])

            # Get camera to image projection matrix
            camera_to_image_projection_matrix = []
            camera_intrinsic_row = []

            # Contstruct matrix
            camera_intrinsic_row.append(camera_intrinsic['fx'])
            camera_intrinsic_row.append(camera_intrinsic['skew'])
            camera_intrinsic_row.append(camera_intrinsic['cx'])
            camera_to_image_projection_matrix.append(camera_intrinsic_row.copy())
            camera_intrinsic_row.clear()

            camera_intrinsic_row.append(0.0)
            camera_intrinsic_row.append(camera_intrinsic['fy'])
            camera_intrinsic_row.append(camera_intrinsic['cy'])
            camera_to_image_projection_matrix.append(camera_intrinsic_row.copy())
            camera_intrinsic_row.clear()

            camera_intrinsic_row.append(0.0)
            camera_intrinsic_row.append(0.0)
            camera_intrinsic_row.append(1.0)
            camera_to_image_projection_matrix.append(camera_intrinsic_row.copy())

            # Get world to image matrix
            world_to_cam_np = np.array(world_to_camera_projection_matrix)
            cam_to_image_np = np.array(camera_to_image_projection_matrix)
            world_to_image_transform = np.matmul(cam_to_image_np, world_to_cam_np)

            # Append to list
            world_to_image_transforms.append(world_to_image_transform)
            focal_lengths.append(camera_intrinsic['fy'])
            cy_vals.append(camera_intrinsic['cy'])

    return calib_logs, world_to_image_transforms, focal_lengths, cy_vals

def main():
    
    # Filepaths for data loading and saving
    root_data_path = '/mnt/media/ddad_train_val/'
    root_save_path = '/mnt/media/SuperDepth/DDAD/'

    # Reading dataset labels and images and sorting returned list in alphabetical order
    depth_maps = sorted([f for f in pathlib.Path(root_data_path).glob("*/point_cloud/LIDAR/*.npz")])
    images = sorted([f for f in pathlib.Path(root_data_path).glob("*/rgb/CAMERA_01/*.png")])
    calib_files = sorted([f for f in pathlib.Path(root_data_path).glob("*/calibration/*.json")])

    # If all data checks have been passed
    num_depth_maps = len(depth_maps)
    num_images = len(images)
   
    checkData = CheckData(num_images, num_depth_maps)
    check_passed = checkData.getCheck()

    if(check_passed):
        print('Beginning processing of data')

        calib_logs, world_to_image_transforms, focal_lengths, cy_vals = \
            parseCalib(calib_files)
        
        print(calib_logs[0], world_to_image_transforms[0], focal_lengths[0], cy_vals[0])

if __name__ == '__main__':
    main()
#%%