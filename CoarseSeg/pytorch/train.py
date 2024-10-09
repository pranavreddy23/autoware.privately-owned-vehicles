
#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from coarse_seg import CoarseSeg
from data_utils.load_acdc import ACDC_Dataset

# Checking devices (GPU vs CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} for inference')

# Instantiate model
model = CoarseSeg().eval().to(device)

# Load Image
def load_image(image):
    image = loader(image)
    image = image.unsqueeze(0)
    return image.to(device)

# Input Image Size
row = 320
col = 640

loader = transforms.Compose(
    [
        transforms.Resize((row,col)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)


# Path to image
acdc_labels_filepath= '/home/zain/Autoware/AutoSeg/training_data/Coarse_Seg/ACDC/gt_masks/'
acdc_images_filepath = '/home/zain/Autoware/AutoSeg/training_data/Coarse_Seg/ACDC/images/'
acdc_Dataset = ACDC_Dataset(acdc_labels_filepath, acdc_images_filepath)
acdc_dataset_items = acdc_Dataset.getlen()
image, label = acdc_Dataset.getitem(200)
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

# Visualise
prediction = prediction.squeeze(0).cpu().detach()
prediction = prediction.permute(1, 2, 0)
prediction_0 = prediction [:,:,0]
prediction_1 = prediction [:,:,1]
prediction_2 = prediction [:,:,2]
prediction_3 = prediction [:,:,3]


fig = plt.figure()
plt.imshow(image)
fig1, axs = plt.subplots(2,2)
axs[0,0].imshow(prediction_0)
axs[0,1].imshow(prediction_1)
axs[1,0].imshow(prediction_2)
axs[1,1].imshow(prediction_3)


# %%
