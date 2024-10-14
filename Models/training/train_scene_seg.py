
#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
from torchvision import transforms
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork
from data_utils.load_data import LoadData
from data_utils.augmentations import Augmentations
from PIL import Image

# Checking devices (GPU vs CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} for inference')

# Instantiate model
model = SceneSegNetwork().to(device)

# Load Image as Tensor
def load_image_tensor(image):
    image = image_loader(image)
    image = image.unsqueeze(0)
    return image.to(device)

image_loader = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

# Load Ground Truth as Tensor
def load_gt_tensor(gt):
    gt = gt_loader(gt)
    gt = gt.unsqueeze(0)
    return gt.to(device)

gt_loader = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

def visualize_result(prediction):
    shape = prediction.shape
    _, output = torch.max(prediction, dim=2)

    row = shape[0]
    col = shape[1]
    vis_predict = Image.new(mode="RGB", size=(col, row))
 
    vx = vis_predict.load()

    sky_colour = (61, 184, 255)
    background_objects_colour = (61, 93, 255)
    foreground_objects_colour = (255, 28, 145)
    road_colour = (0, 255, 220)

    # Extracting predicted classes and assigning to colourmap
    for x in range(row):
        for y in range(col):
            if(output[x,y].item() == 0):
                vx[y,x] = sky_colour
            elif(output[x,y].item() == 1):
                 vx[y,x] = background_objects_colour
            elif(output[x,y].item() == 2):
                 vx[y,x] = foreground_objects_colour
            elif(output[x,y].item() == 3):
                 vx[y,x] = road_colour
    
    return vis_predict

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

# Iterators for datasets
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

# Total number of training samples
total_train_samples = acdc_num_train_samples + bdd100k_num_train_samples \
+ iddaw_num_train_samples + muses_num_train_samples \
+ mapillary_num_train_samples + comma10k_num_train_samples

print(total_train_samples, ': total training samples')

# Loss, learning rate, optimizer
learning_rate = 0.0001
optimizer = optim.AdamW(model.parameters(), learning_rate)
optimizer.zero_grad()

# Loop through data
for count in range(0, 31):
    
    # Reset iterators
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

    # Read images, apply augmentation, run prediction, calculate
    # loss for iterated image from each dataset, and increment
    # dataset iterators

    # Memory Profiling
    torch.cuda.reset_peak_memory_stats(device=None)

    if(data_list[data_list_count] == 'ACDC'):
        image, gt, class_weights = \
                acdc_Dataset.getItemTrain(acdc_count)
        print(f"ACDC gpu used {torch.cuda.max_memory_allocated(device=None)} memory")
        acdc_count += 1
    
    if(data_list[data_list_count] == 'BDD100K'):
        image, gt, class_weights = \
            bdd100k_Dataset.getItemTrain(bdd100k_count)
        print(f"BDD100K gpu used {torch.cuda.max_memory_allocated(device=None)} memory")
        bdd100k_count += 1

    if(data_list[data_list_count] == 'IDDAW'):
        image, gt, class_weights = \
            iddaw_Dataset.getItemTrain(iddaw_count)      
        print(f"IDDAW gpu used {torch.cuda.max_memory_allocated(device=None)} memory")
        iddaw_count += 1

    if(data_list[data_list_count] == 'MUSES'):
        image, gt, class_weights = \
            muses_Dataset.getItemTrain(muses_count)
        print(f"MUSES gpu used {torch.cuda.max_memory_allocated(device=None)} memory")
        muses_count += 1
    
    if(data_list[data_list_count] == 'MAPILLARY'):
        image, gt, class_weights = \
            mapillary_Dataset.getItemTrain(mapillary_count)
        print(f"MAPILLARY gpu used {torch.cuda.max_memory_allocated(device=None)} memory")
        mapillary_count +=1
    
    if(data_list[data_list_count] == 'COMMA10K'):
        image, gt, class_weights = \
            comma10k_Dataset.getItemTrain(comma10k_count)
        print(f"COMMA10K gpu used {torch.cuda.max_memory_allocated(device=None)} memory")
        comma10k_count += 1
    
    image, augmented = \
        Augmentations(image, gt).getAugmentedData()
        
    # Label for visualization
    label = augmented[0]

    # Ground Truth with probabiliites for each class in separate channels
    gt_fused = np.stack((augmented[1], augmented[2], \
            augmented[3], augmented[4]), axis=2)
        
    # Converting to tensor and loading
    image_tensor = load_image_tensor(image)
    gt_tensor = load_gt_tensor(gt_fused)
    class_weights_tensor = torch.tensor(class_weights).to(device)

    loss = nn.CrossEntropyLoss(weight=class_weights_tensor)
    prediction = model(image_tensor)
    calc_loss = loss(prediction, gt_tensor)
    
    # Gradient accumulation
    calc_loss.backward()
    print('Loss: ', calc_loss)

    prediction = prediction.squeeze(0).cpu().detach()
    prediction = prediction.permute(1, 2, 0)
    
    # Simulating batch size of 3
    # Batch size of 3 gives good results in testing
    if((count+1) % 3 == 0):
        print('OPTIMIZING')
        optimizer.step()
        optimizer.zero_grad()

        # Visualize Results
        vis_predict = visualize_result(prediction)
        fig1, axs = plt.subplots(1,2)
        axs[0].imshow(label)
        axs[1].imshow(vis_predict)

    data_list_count += 1

# %%
