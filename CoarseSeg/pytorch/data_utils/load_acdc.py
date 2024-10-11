#! /usr/bin/env python3
import pathlib
from PIL import Image
from torch.utils.data import Dataset

class ACDC_Dataset(Dataset):
    def __init__(self, labels_filepath, images_filepath):
        self.labels = sorted([f for f in pathlib.Path(labels_filepath).glob("*.png")])
        self.images = sorted([f for f in pathlib.Path(images_filepath).glob("*.png")])

    def getitem(self, index):
        self.image = Image.open(str(self.images[index]))
        self.image = self.image.crop((0, 0, 1919, 990))
        self.label = Image.open(str(self.labels[index]))
        self.label = self.label.crop((0, 0, 1919, 990))
        return self.image, self.label

    def getlen(self):
        num_images = len(self.images)
        num_labels = len(self.labels)

        if(num_images != num_labels):
            raise ValueError('Number of images and ground truth labels are mismatched')
        
        if (num_images == 0):
            raise ValueError('No images found - check the root path')
       
        if (num_labels == 0):
            raise ValueError('No ground truth masks found - check the root path')
        
        if(num_images == num_labels and num_images != 0):
            return num_images