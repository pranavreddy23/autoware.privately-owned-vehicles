#! /usr/bin/env python3
import pathlib
from PIL import Image

# Create coarse semantic segmentation mask
# of combined classes
def createMask(colorMap, skyMap):

    # Initializing and loading pixel data
    row, col = colorMap.size
    px = colorMap.load()

    # Loading and resizing sky pixel mask
    skyMap = skyMap.resize((row,col))
    ps = skyMap.load()

    coarseSegColorMap = Image.new(mode="RGB", size=(row, col))
    cx = coarseSegColorMap.load()

    # Colourmaps for classes
    background_objects_colour = (61, 93, 255)
    foreground_objects_colour = (255, 28, 145)
    road_colour = (0, 255, 220)
    sky_colour = (61, 184, 255)


    # Extracting classes and assigning to colourmap
    for x in range(row):
        for y in range(col):

            if type(px[x,y]) == int:
                # BACKGROUND OBJECTS
                if px[x,y] == 2:
                    cx[x,y] = background_objects_colour

                # EGO VEHICLE
                elif px[x,y] == 4:
                    cx[x,y] = background_objects_colour

                # MOVABLE OBJECTS
                elif px[x,y] == 1:
                    cx[x,y] = foreground_objects_colour

                # LANE MARKINGS
                elif px[x,y] == 0:
                    cx[x,y] = road_colour

                # ROAD
                elif px[x,y] == 3:
                    cx[x,y] = road_colour

                # SKY
                if ps[x,y] == sky_colour:
                    cx[x,y] = sky_colour

            else:
                # BACKGROUND OBJECTS   
                if px[x,y][0:3] == (128, 128, 96):
                    cx[x,y] = background_objects_colour

                # EGO VEHICLE
                elif px[x,y][0:3] == (204, 0, 255):
                    cx[x,y] = background_objects_colour

                # MOVABLE OBJECTS
                elif px[x,y][0:3] == (0, 255, 102):
                    cx[x,y] = foreground_objects_colour

                # LANE MARKINGS
                elif px[x,y][0:3] == (255, 0, 0):
                    cx[x,y] = road_colour

                # ROAD
                elif px[x,y][0:3] == (64, 32, 32):
                    cx[x,y] = road_colour

                # SKY
                if ps[x,y] == sky_colour:
                    cx[x,y] = sky_colour

    return coarseSegColorMap

# Reading dataset labels and images and sorting returned list in alphabetical order
labels = sorted([f for f in pathlib.Path('/home/zain/Autoware/semantic_segmentation/training_data/comma10k/masks').glob("*.png")])
images = sorted([f for f in pathlib.Path('/home/zain/Autoware/semantic_segmentation/training_data/comma10k/images/').glob("*.png")])
sky_masks = sorted([f for f in pathlib.Path('/home/zain/Autoware/semantic_segmentation/training_data/comma10k/sky_masks').glob("*.png")])

# Paths to save training data with new coarse segmentation masks
labels_save_path = '/home/zain/Autoware/semantic_segmentation/training_data/Coarse_Seg/comma10k/gt_masks/'
images_save_path = '/home/zain/Autoware/semantic_segmentation/training_data/Coarse_Seg/comma10k/images/'

# Looping through data
for index in range(0, len(labels)):

    # Open images and pre-existing masks
    image = Image.open(str(images[index])).convert('RGB')
    label = Image.open(str(labels[index]))
    sky_mask = Image.open(str(sky_masks[index]))

    # Create new Coarse Segmentation mask
    coarseSegColorMap = createMask(label, sky_mask)
  
    # Save images
    image.save(images_save_path + str(index) + ".png","PNG")
    coarseSegColorMap.save(labels_save_path + str(index) + ".png","PNG")

    print(index)

print('--- COMPLETE ---')