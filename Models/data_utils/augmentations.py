import albumentations as A
import numpy as np
import random

class Augmentations():
    def __init__(self, input_image, input_mask):
        self.image = np.array(input_image)
        self.mask = np.array(input_mask)
        
        transform_shape = A.Compose(
            [
                A.Resize(width = 640, height = 320), 
                A.HorizontalFlip(p = 0.5),           
            ]
        )

        transform_noise = A.Compose(
            [
                A.MultiplicativeNoise(multiplier=(0.5, 1.5), per_channel=False, p=0.5),
                A.PixelDropout(dropout_prob=0.025, per_channel=True, p=0.25),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.75),
                A.GaussNoise(var_limit=(50.0, 100.0), mean=0, noise_scale_factor=0.2, p=0.5),
                A.GaussNoise(var_limit=(250.0, 250.0), mean=0, noise_scale_factor=1, p=0.5),
                A.ISONoise(color_shift=(0.1, 0.5), intensity=(0.5, 0.5), p=0.5),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.4, p=0.1),
                A.RandomGridShuffle(grid=(1,3), p=0.75),
                A.RandomRain(p=0.1),
                A.Spatter(mean=(0.65, 0.65), std=(0.3, 0.3), gauss_sigma=(2, 2), \
                    cutout_threshold=(0.68, 0.68), intensity=(0.6, 0.6), mode='rain', \
                    p=0.1),
                A.Spatter(mean=(0.65, 0.65), std=(0.3, 0.3), gauss_sigma=(2, 2), \
                    cutout_threshold=(0.68, 0.68), intensity=(0.6, 0.6), mode='mud', \
                     p=0.1),
                A.ToGray(num_output_channels=3, method='weighted_average', p=0.1)                
            ]
        )

        self.adjust_shape = transform_shape(image=self.image, mask=self.mask)
        self.augmented_image = self.adjust_shape["image"]
        self.augmented_mask = self.adjust_shape["mask"]

        if (random.random() >= 0.25):
            
            self.add_noise = transform_noise(image=self.augmented_image, \
                                mask=self.augmented_mask)
            
            self.augmented_image = self.add_noise["image"]
            self.augmented_mask = self.add_noise["mask"]

        self.getAugmentedData()

    def getAugmentedData(self):
        return self.augmented_image, self.augmented_mask