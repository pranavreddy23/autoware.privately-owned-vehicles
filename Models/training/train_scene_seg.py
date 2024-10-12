
#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork
from data_utils.load_data import LoadData
from data_utils.augmentations import Augmentations

# Checking devices (GPU vs CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} for inference')

# Instantiate model
model = SceneSegNetwork().eval().to(device)

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

# BDD100K - Data Loading
bdd100k_Dataset = LoadData(bdd100k_labels_fileapath, bdd100k_images_fileapath, 'BDD100K')
bdd100k_num_train_samples, bdd100k_num_val_samples = bdd100k_Dataset.getItemCount()

# IDDAW - Data Loading
iddaw_Dataset = LoadData(iddaw_labels_fileapath, iddaw_images_fileapath, 'IDDAW')
iddaw_num_train_samples, iddaw_num_val_samples = iddaw_Dataset.getItemCount()

# MUSES - Data Loading
muses_Dataset = LoadData(muses_labels_fileapath, muses_images_fileapath, 'MUSES')
muses_num_train_samples, muses_num_val_samples = muses_Dataset.getItemCount()

# Mapillary - Data Loading
mapillary_Dataset = LoadData(mapillary_labels_fileapath, mapillary_images_fileapath, 'MAPILLARY')
mapillary_num_train_samples, mapillary_num_val_samples = mapillary_Dataset.getItemCount()

# comma10k - Data Loading
comma10k_Dataset = LoadData(comma10k_labels_fileapath, comma10k_images_fileapath, 'COMMA10K')
comma10k_num_train_samples, comma10k_num_val_samples = comma10k_Dataset.getItemCount()

# Iterators for datasets with fewer number of samples than 
# Mapillary Vistas, which has the greatest number of training samples
acdc_count = 0
bdd100k_count = 0
iddaw_count = 0
muses_count = 0
comma10k_count = 0
mapillary_count = 0

data_list = []
data_list.append('ACDC')
data_list.append('BDD100K')
data_list.append('IDDAW')
data_list.append('MUSES')
data_list.append('MAPILLARY')
data_list.append('COMMA10K')
data_list_count = 0

total_train_samples = acdc_num_train_samples + bdd100k_num_train_samples \
+ iddaw_num_train_samples + muses_num_train_samples \
+ mapillary_num_train_samples + comma10k_num_train_samples

print(total_train_samples, ': total training samples')

# Read images and apply image augmentations
for count in range(0, total_train_samples):
    
    if(acdc_count == acdc_num_train_samples):
        acdc_count = 0
        
    if(bdd100k_count == bdd100k_num_train_samples):
        bdd100k_count = 0
    
    if(iddaw_count == iddaw_num_train_samples):
        iddaw_count = 0
    
    if(muses_count == muses_num_train_samples):
        muses_count = 0
    
    if(mapillary_count == mapillary_num_train_samples):
        muses_count = 0

    if(comma10k_count == comma10k_num_train_samples):
        comma10k_count = 0

    if(data_list_count == len(data_list)):
        data_list_count = 0

    if(data_list[data_list_count] == 'ACDC'):
        image_acdc, label_acdc = acdc_Dataset.getItemTrain(acdc_count)
        image_acdc, label_acdc = \
        Augmentations(image_acdc, label_acdc).getAugmentedData()
        acdc_count += 1
    
    if(data_list[data_list_count] == 'BDD100K'):
        image_bdd100k, label_bdd100k = bdd100k_Dataset.getItemTrain(bdd100k_count)
        image_bdd100k, label_bdd100k = \
        Augmentations(image_bdd100k, label_bdd100k).getAugmentedData()
        bdd100k_count += 1

    if(data_list[data_list_count] == 'IDDAW'):
        image_iddaw, label_iddaw = iddaw_Dataset.getItemTrain(iddaw_count)
        image_iddaw, label_iddaw = \
        Augmentations(image_iddaw, label_iddaw).getAugmentedData()
        iddaw_count += 1

    if(data_list[data_list_count] == 'MUSES'):
        image_muses, label_muses = muses_Dataset.getItemTrain(muses_count)
        image_muses, label_muses = \
        Augmentations(image_muses, label_muses).getAugmentedData()
        muses_count += 1
    
    if(data_list[data_list_count] == 'MAPILLARY'):
        image_mapillary, label_mapillary = mapillary_Dataset.getItemTrain(mapillary_count)
        image_mapillary, label_mapillary = \
        Augmentations(image_mapillary, label_mapillary).getAugmentedData()
        mapillary_count +=1
    
    if(data_list[data_list_count] == 'COMMA10K'):
        image_comma10k, label_comma10k = comma10k_Dataset.getItemTrain(comma10k_count)
        image_comma10k, label_comma10k = \
        Augmentations(image_comma10k, label_comma10k).getAugmentedData()
        comma10k_count += 1
    
    print('ITERATION:', count, ' ACDC:', acdc_count, ' BDD100K:', bdd100k_count, \
          ' IDDAW:', iddaw_count, ' MUSES:', muses_count, \
          'MAPILLARY:', mapillary_count, ' COMMA10K:', comma10k_count)
 
    data_list_count += 1

# Image augmentation
image, label = comma10k_Dataset.getItemTrain(10)
image, label = Augmentations(image, label).getAugmentedData()

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
