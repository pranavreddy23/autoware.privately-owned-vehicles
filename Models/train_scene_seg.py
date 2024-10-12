
#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from model_components.scene_seg import SceneSeg
from data_utils.load_data import LoadData
from data_utils.augmentations import Augmentations

# Checking devices (GPU vs CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} for inference')

# Instantiate model
model = SceneSeg().eval().to(device)

# Load Image
def load_image(image):
    image = loader(image)
    image = image.unsqueeze(0)
    return image.to(device)


loader = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)


# Root path
root = '/home/zain/Autoware/AutoSeg/training_data/Scene_Seg/'

# Data paths
# ACDC
acdc_labels_filepath= root + 'ACDC/gt_masks/'
acdc_images_filepath = root + 'ACDC/images/'

# BDD100K
bdd100k_labels_fileapath = root + 'BDD100K/gt_masks/'
bdd100k_images_fileapath = root + 'BDD100K/images/'

# IDDAW
iddaw_labels_fileapath = root + 'IDDAW/gt_masks/'
iddaw_images_fileapath = root + 'IDDAW/images/'

# MUSES
muses_labels_fileapath = root + 'MUSES/gt_masks/'
muses_images_fileapath = root + 'MUSES/images/'

# MAPILLARY
mapillary_labels_fileapath = root + 'Mapillary_Vistas/gt_masks/'
mapillary_images_fileapath = root + 'Mapillary_Vistas/images/'

# COMMA10K
comma10k_labels_fileapath = root + 'comma10k/gt_masks/'
comma10k_images_fileapath = root + 'comma10k/images/'


# ACDC - Data Loading
acdc_Dataset = LoadData(acdc_labels_filepath, acdc_images_filepath, 'ACDC')
acdc_num_train_samples, acdc_num_val_samples = acdc_Dataset.getItemCount()
print(acdc_num_train_samples, acdc_num_val_samples)

# BDD100K - Data Loading
bdd100k_Dataset = LoadData(bdd100k_labels_fileapath, bdd100k_images_fileapath, 'BDD100K')
bdd100k_num_train_samples, bdd100k_num_val_samples = bdd100k_Dataset.getItemCount()
print(bdd100k_num_train_samples, bdd100k_num_val_samples)

# IDDAW - Data Loading
iddaw_Dataset = LoadData(iddaw_labels_fileapath, iddaw_images_fileapath, 'IDDAW')
iddaw_num_train_samples, iddaw_num_val_samples = iddaw_Dataset.getItemCount()
print(iddaw_num_train_samples, iddaw_num_val_samples)

# MUSES - Data Loading
muses_Dataset = LoadData(muses_labels_fileapath, muses_images_fileapath, 'MUSES')
muses_num_train_samples, muses_num_val_samples = muses_Dataset.getItemCount()
print(muses_num_train_samples, muses_num_val_samples)

# Mapillary - Data Loading
mapillary_Dataset = LoadData(mapillary_labels_fileapath, mapillary_images_fileapath, 'MAPILLARY')
mapillary_num_train_samples, mapillary_num_val_samples = mapillary_Dataset.getItemCount()
print(mapillary_num_train_samples, mapillary_num_val_samples)

# comma10k - Data Loading
comma10k_Dataset = LoadData(comma10k_labels_fileapath, comma10k_images_fileapath, 'COMMA10K')
comma10k_num_train_samples, comma10k_num_val_samples = comma10k_Dataset.getItemCount()
print(comma10k_num_train_samples, comma10k_num_val_samples)
image, label = comma10k_Dataset.getItemVal(10)

# Image augmentation
augmentations = Augmentations(image, label)
image, label = augmentations.getAugmentedData()

# Loading to tensor format
image_tensor = load_image(image)

# Inference
torch.cuda.reset_peak_memory_stats(device=None)
prediction = model(image_tensor)
print(f"gpu used {torch.cuda.max_memory_allocated(device=None)} memory")
param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))
print('params', param_size)

# Get Predictions
prediction = prediction.squeeze(0).cpu().detach()
prediction = prediction.permute(1, 2, 0)
prediction_0 = prediction [:,:,0]
prediction_1 = prediction [:,:,1]
prediction_2 = prediction [:,:,2]
prediction_3 = prediction [:,:,3]

# Visualise
fig = plt.figure()
plt.imshow(image)

fig = plt.figure()
plt.imshow(label)

fig1, axs = plt.subplots(2,2)
axs[0,0].imshow(prediction_0)
axs[0,1].imshow(prediction_1)
axs[1,0].imshow(prediction_2)
axs[1,1].imshow(prediction_3)


# %%
