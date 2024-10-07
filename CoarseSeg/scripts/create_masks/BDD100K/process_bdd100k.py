#! /usr/bin/env python3
import pathlib
import logging
import sys
sys.tracebacklimit = 0
from argparse import ArgumentParser
from PIL import Image

logging.basicConfig(
    level = logging.DEBUG,
    format = '%(levelname)s - %(message)s'
)

# Create coarse semantic segmentation mask
# of combined classes
def createMask(colorMap):

    # Initializing and loading pixel data
    row, col = colorMap.size
    px = colorMap.load()
    coarseSegColorMap = Image.new(mode="RGB", size=(row, col))
    cx = coarseSegColorMap.load()

    # Colourmaps for classes
    sky_colour = (61, 184, 255)
    background_objects_colour = (61, 93, 255)
    vulnerable_living_colour = (255, 61, 61)
    small_mobile_vehicle_colour = (255, 190, 61)
    large_mobile_vehicle_colour = (255, 116, 61)
    road_edge_delimiter_colour = (216, 255, 61)
    road_colour = (0, 255, 220)

    # Extracting classes and assigning to colourmap
    for x in range(row):
        for y in range(col):

            # SKY
            if px[x, y] == 10:
                cx[x,y] = sky_colour
        
            # BACKGROUND OBJECTS
            # Building    
            elif px[x,y] == 2:
                cx[x,y] = background_objects_colour
            # Pole  
            elif px[x,y] == 5:
                cx[x,y] = background_objects_colour
            # Traffic Light
            elif px[x,y] == 6:
                cx[x,y] = background_objects_colour 
            # Traffic Sign
            elif px[x,y] == 7:
                cx[x,y] = background_objects_colour 
            # Vegetation
            elif px[x,y] == 8:
                cx[x,y] = background_objects_colour
            # Terrain
            elif px[x,y] == 9:
                cx[x,y] = background_objects_colour

            # VULNERABLE LIVING
            # Person
            elif px[x,y] == 11:
                cx[x,y] = vulnerable_living_colour

            # SMALL MOBILE VEHICLE
            # Rider
            elif px[x,y] == 12:
                cx[x,y] = small_mobile_vehicle_colour
            # Motorcylce
            elif px[x,y] == 17:
                cx[x,y] = small_mobile_vehicle_colour
            # Bicycle
            elif px[x,y] == 18:
                cx[x,y] = small_mobile_vehicle_colour

            # LARGE MOBILE VEHICLE
            # Car
            elif px[x,y] == 13:
                cx[x,y] = large_mobile_vehicle_colour
            # Truck
            elif px[x,y] == 14:
                cx[x,y] = large_mobile_vehicle_colour
            # Bus
            elif px[x,y] == 15:
                cx[x,y] = large_mobile_vehicle_colour
            # Train
            elif px[x,y] == 16:
                cx[x,y] = large_mobile_vehicle_colour
            
            # ROAD EDGE DELIMITER
            # Wall
            elif px[x,y] == 3:
                cx[x,y] = road_edge_delimiter_colour
            # Fence
            elif px[x,y] == 4:
                cx[x,y] = road_edge_delimiter_colour

            # ROAD
            elif px[x,y] == 0:
                cx[x,y] = road_colour

    return coarseSegColorMap

def main():

    parser = ArgumentParser()
    parser.add_argument("-l", "--labels", dest="labels_filepath", help="path to folder with input ground truth grayscale label ids")
    parser.add_argument("-i", "--images", dest="images_filepath", help="path to folder with input images")
    parser.add_argument("-ls", "--labels-save", dest="labels_save_path", help="path to folder where processed labels will be saved")
    parser.add_argument("-is", "--images-save", dest="images_save_path", help="path to folder where corresponding images will be saved")
    args = parser.parse_args()

    # Paths to read input images and ground truth label masks from training data
    labels_filepath = args.labels_filepath
    images_filepath = args.images_filepath

    # Paths to save training data with new coarse segmentation masks
    labels_save_path = args.labels_save_path
    images_save_path = args.images_save_path

    # Reading dataset labels and images and sorting returned list in alphabetical order
    labels = sorted([f for f in pathlib.Path(labels_filepath).glob("*.png")])
    images = sorted([f for f in pathlib.Path(images_filepath).glob("*.jpg")])
    
    # Checking validity
    is_label_path_valid = False
    is_image_path_valid = False
    is_data_valid = False

    # Getting number of labels and images
    num_labels = len(labels)
    num_images = len(images)

    # Checking if ground truth labels were read and logging error if missing
    if (num_labels > 0):
        logging.info(f'Found {num_labels} ground truth masks')
        is_label_path_valid = True
    else:
        logging.error(f'No ground truth png masks found - check your labels filepath: {labels_filepath}')
        raise ValueError('Input data is incorrect')

    # Checking if input images were read and logging error if missing
    if (num_images > 0):
        logging.info(f'Found {num_images} input images')
        is_image_path_valid = True
    else:
        logging.error(f'No input jpg images found - check your images filepath: {images_filepath}')
        raise ValueError('Input data is incorrect')

    # Checking if number of ground truth labels matches number of input images
    # and logging error if mismatched
    if (num_labels != num_images):
        logging.error(f'Number of ground truth masks: {num_labels} - does not match number of input images: {num_images}')
        raise ValueError('Input data is incorrect')
    else:
        is_data_valid = True

    # If all data checks have been passed
    if(is_label_path_valid and is_image_path_valid and is_data_valid):

        logging.info('Beginning processing of data')

        # Looping through data
        for index in range(0, num_images):
            
            # Open images and pre-existing masks
            image = Image.open(str(images[index]))
            label = Image.open(str(labels[index]))

            # Create new Coarse Segmentation mask
            coarseSegColorMap = createMask(label) 

            # Save images
            image.save(images_save_path + str(index) + ".png","PNG")
            coarseSegColorMap.save(labels_save_path + str(index) + ".png","PNG")

            logging.info(f'Processing image {index} of {num_images-1}')
    
        logging.info('----- Processing complete -----') 

if __name__ == '__main__':
    main()      