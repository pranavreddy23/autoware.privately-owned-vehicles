#! /usr/bin/env python3
import pathlib
from typing import Literal
from PIL import Image

class LoadData():
    def __init__(self, labels_filepath, images_filepath, \
        dataset: Literal['ACDC', 'BDD100K', 'IDDAW', 'MUSES', 'MAPILLARY', 'COMMA10K']):

        self.dataset = dataset

        if(self.dataset != 'ACDC' and self.dataset != 'BDD100K' and self.dataset != 'MUSES' \
           and self.dataset != 'MAPILLARY' and self.dataset != 'COMMA10K'):
            raise ValueError('Dataset type is not correctly specified')
        
        self.labels = sorted([f for f in pathlib.Path(labels_filepath).glob("*.png")])
        self.images = sorted([f for f in pathlib.Path(images_filepath).glob("*.png")])

        self.num_images = len(self.images)
        self.num_labels = len(self.labels)

        if(self.num_images != self.num_labels):
            raise ValueError('Number of images and ground truth labels are mismatched')
        
        if (self.num_images == 0):
            raise ValueError('No images found - check the root path')
       
        if (self.num_labels == 0):
            raise ValueError('No ground truth masks found - check the root path')
        
        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []
        
        self.num_train_samples = 0
        self.num_val_samples = 0

        for count in range (0, self.num_images):

            if(count% 9 == 0):
                self.val_images.append(str(self.images[count]))
                self.val_labels.append(str(self.labels[count]))
                self.num_val_samples += 1 
            else:
                self.train_images.append(str(self.images[count]))
                self.train_labels.append(str(self.labels[count]))
                self.num_train_samples += 1

    def getItemCount(self):
        return self.num_train_samples, self.num_val_samples
    
    def getItemTrain(self, index):
        self.train_image = Image.open(str(self.train_images[index]))
        self.train_label = Image.open(str(self.train_labels[index]))

        if(self.dataset == 'ACDC'):
            self.train_image = self.train_image.crop((0, 0, 1919, 990))
            self.train_label = self.train_label.crop((0, 0, 1919, 990))

        return self.train_image, self.train_label
    
    def getItemVal(self, index):
        self.val_image = Image.open(str(self.val_images[index]))
        self.val_label = Image.open(str(self.val_labels[index]))

        if(self.dataset == 'ACDC'):
            self.val_image = self.val_image.crop((0, 0, 1919, 990))
            self.val_label = self.val_label.crop((0, 0, 1919, 990))
            
        return self.val_image, self.val_label
