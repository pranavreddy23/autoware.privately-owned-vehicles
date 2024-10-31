#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import pathlib
import sys
sys.tracebacklimit = 0
from argparse import ArgumentParser
from PIL import Image


def removeUnlabelledImages(labels_filepath, images_filepath, labels, images):

    label_ids = []
    image_ids = []
    labelled_images = []

    for i in range(0, len(labels)):
        label_string = str(labels[i]).replace(labels_filepath, '')[:-16]
        label_ids.append(label_string)
       
    for i in range(0, len(images)):
        image_string = str(images[i]).replace(images_filepath, '')[:-4]
        image_ids.append(image_string)
     
    for i in range(0, len(image_ids)):
        image_id = image_ids[i]
        if(image_id in label_ids):
            labelled_images.append(images[i])

    return labelled_images

# Create coarse semantic segmentation mask
# of combined classes
def createMask(colorMap):

    # Initializing and loading pixel data
    row, col = colorMap.size
    px = colorMap.load()
    coarseSegColorMap = Image.new(mode="RGB", size=(row, col))
    cx = coarseSegColorMap.load()

    # Colourmaps for classes
    roadwork_objects_colour = (255, 125, 0)
    background_objects_colour = (61, 93, 255)
    road_colour = (0, 255, 220)

    # Extracting classes and assigning to colourmap
    for x in range(row):
        for y in range(col):

            # Barrier
            if px[x, y] == (246, 116, 185):
                cx[x,y] = roadwork_objects_colour

            # Barricade    
            elif px[x,y] == (248, 135, 182):
                cx[x,y] = roadwork_objects_colour
            
            # Police Vehicle    
            elif px[x,y] == (255, 68, 51):
                cx[x,y] = roadwork_objects_colour

            # Work Vehicle    
            elif px[x,y] == (255,104, 66):
                cx[x,y] = roadwork_objects_colour

            # Police Officer    
            elif px[x,y] == (184, 107, 35):
                cx[x,y] = roadwork_objects_colour

            # Worker    
            elif px[x,y] == (205, 135, 29):
                cx[x,y] = roadwork_objects_colour

            # Worker    
            elif px[x,y] == (30, 119, 179):
                cx[x,y] = roadwork_objects_colour
            
            # Drum    
            elif px[x,y] == (44, 79, 206):
                cx[x,y] = roadwork_objects_colour

            # Vertical Panel    
            elif px[x,y] == (102, 81, 210):
                cx[x,y] = roadwork_objects_colour

            # Tubular Marker    
            elif px[x,y] == (170, 118, 213):
                cx[x,y] = roadwork_objects_colour

            # Work Equipment 
            elif px[x,y] == (214, 154, 219):
                cx[x,y] = roadwork_objects_colour

            # Arrow Board 
            elif px[x,y] == (241, 71, 14):
                cx[x,y] = roadwork_objects_colour

            # TTC Sign 
            elif px[x,y] == (254, 139, 32):
                cx[x,y] = roadwork_objects_colour

            # Road
            elif px[x,y] == (70, 70, 70):
                cx[x,y] = road_colour

            # Background
            else:
                cx[x,y] = background_objects_colour

    return coarseSegColorMap


def main():

    # Paths to read input images and ground truth label masks from training data
    labels_filepath = '/home/zain/Autoware/AutoSeg/training_data/Drive_Seg/ROADWork/gtFine/'
    images_filepath = '/home/zain/Autoware/AutoSeg/training_data/Drive_Seg/ROADWork/all_images/'

    # Paths to save training data with new coarse segmentation masks
    labels_save_path = '/home/zain/Autoware/AutoSeg/training_data/Drive_Seg/ROADWork/gt_masks/'
    images_save_path = '/home/zain/Autoware/AutoSeg/training_data/Drive_Seg/ROADWork/images/'

    # Reading dataset labels and images and sorting returned list in alphabetical order
    labels = sorted([f for f in pathlib.Path(labels_filepath).glob("*labelColors.png")])
    images = sorted([f for f in pathlib.Path(images_filepath).glob("*")])

    # Remove images for which ground truth data is not provided
    labelled_images = removeUnlabelledImages(labels_filepath, images_filepath, labels, images)

    # Checking validity
    is_label_path_valid = False
    is_image_path_valid = False
    is_data_valid = False

    # Getting number of labels and images
    num_labels = len(labels)
    num_images = len(labelled_images)

    # Checking if ground truth labels were read and logging error if missing
    if (num_labels > 0):
        print(f'Found {num_labels} ground truth masks')
        is_label_path_valid = True
    else:
        raise ValueError(f'No ground truth jpg masks found - check your labels filepath: {labels_filepath}')

    # Checking if input images were read and logging error if missing
    if (num_images > 0):
        print(f'Found {num_images} input images')
        is_image_path_valid = True
    else:
        raise ValueError(f'No input png images found - check your images filepath: {images_filepath}')

    # Checking if number of ground truth labels matches number of input images
    # and logging error if mismatched
    if (num_labels != num_images):
        raise ValueError(f'Number of ground truth masks: {num_labels} - does not match number of input images: {num_images}')
    else:
        is_data_valid = True

    # If all data checks have been passed
    if(is_label_path_valid and is_image_path_valid and is_data_valid):

        print('Beginning processing of data')
        
        # Looping through data
        for index in range(0, num_images):
            # Open images and pre-existing masks
            image = Image.open(str(labelled_images[index]))
            label = Image.open(str(labels[index]))
            label = label.convert('RGB')

            row, col = image.size
            half_res = (int(row/2), int(col/2))

            if (row > 2000):
                image = image.resize(half_res)
                label = label.resize(half_res)

            # Create new Coarse Segmentation mask
            driveSegColorMap = createMask(label)

            # Save images
            image.save(images_save_path + str(index) + ".jpg","JPEG")
            driveSegColorMap.save(labels_save_path + str(index) + ".png","PNG")

            print(f'Processing image {index} of {num_images-1}')
    
        print('----- Processing complete -----')

if __name__ == '__main__':
    main()
#%%