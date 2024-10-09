import albumentations as A
import numpy as np

class Augmentations():
    def __init__(self, input_image, input_mask):
        self.image = np.array(input_image)
        self.mask = np.array(input_mask)
        
        transform = A.Compose(
            [
                A.Resize(width = 640, height = 320),
                A.HorizontalFlip(p = 0.5),
            ]
        )
        self.augmentations = transform(image=self.image, mask=self.mask)
        self.augmented_image = self.augmentations["image"]
        self.augmented_mask = self.augmentations["mask"]

        self.getAugmentedData()

    def getAugmentedData(self):
        return self.augmented_image, self.augmented_mask