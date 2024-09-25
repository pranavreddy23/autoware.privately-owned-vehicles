
#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Encoder Backbone
        self.encoder = models.efficientnet_b0(weights = 'EfficientNet_B0_Weights.IMAGENET1K_V1').features

    def forward(self, image):
        # Sequential layers of efficient net encoder
        l0 = self.encoder[0](image)
        l1 = self.encoder[1](l0)
        l2 = self.encoder[2](l1)
        l3 = self.encoder[3](l2)
        l4 = self.encoder[4](l3)
        l5 = self.encoder[5](l4)
        l6 = self.encoder[6](l5)
        l7 = self.encoder[7](l6)
        l8 = self.encoder[8](l7)
        return [l0, l2, l3, l4, l8]         

class Context(nn.Module):
    def __init__(self):
        super(Context, self).__init__()
        # Standard
        self.GeLU = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        # Context - MLP Layers
        self.context_layer_0 = nn.Linear(1280, 800)
        self.context_layer_1 = nn.Linear(800, 800)
        self.context_layer_2 = nn.Linear(800, 1280)

    def forward(self, features):
        # Pooling and averaging channel layers to get a single vector
        reduced_features = self.pool(features)
        feature_vector = torch.mean(reduced_features, dim = [2,3])

        # MLP
        c0 = self.context_layer_0(feature_vector)
        c0 = self.GeLU(c0)
        c1 = self.context_layer_1(c0)
        c1 = self.GeLU(c1)
        c2 = self.context_layer_2(c1)
        attention = self.sigmoid(c2)
        attention = attention.unsqueeze(-1)
        attention = attention.unsqueeze(-1)

        # Multiplying attention from MLP with features and skip connection
        context = reduced_features*attention  + reduced_features      
        return context   

class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()
        # Standard
        self.GeLU = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        # Decoder - Neck Layers 
        self.upsample_layer_0 = nn.ConvTranspose2d(1280, 1280, 2, 2)
        self.decode_layer_0 = nn.Conv2d(1280, 1024, 3, 1, 1)
        self.decode_layer_1 = nn.Conv2d(1024, 1024, 3, 1, 1)

        self.upsample_layer_1 = nn.ConvTranspose2d(1024, 1024, 2, 2)
        self.skip_link_layer_0 = nn.Conv2d(80, 1024, 3, 1, 1)
        self.decode_layer_2 = nn.Conv2d(1024, 512, 3, 1, 1)
        self.decode_layer_3 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, context, features):

        # Decoder upsample block 1
        # Upsample
        d0 = self.upsample_layer_0(context)
        # Add layer from Encoder
        d0 = d0 + features[4]
        # Double Convolution
        d1 = self.decode_layer_0 (d0)
        d1 = self.GeLU(d1)
        d2 = self.decode_layer_1(d1)
        d2 = self.sigmoid(d2)

        # Decoder upsample block 2
        # Upsample
        d3 = self.upsample_layer_1(d2)
        # Expand and add layer from Encoder
        d3 = d3 + self.skip_link_layer_0(features[3])
        # Double convolution
        d3 = self.decode_layer_2(d3)
        d3 = self.GeLU(d3)
        d4 = self.decode_layer_3(d3)
        neck = self.sigmoid(d4)

        return neck

class CoarseSegHead(nn.Module):
    def __init__(self):
        super(CoarseSegHead, self).__init__()
        # Standard
        self.GeLU = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        # Coarse Segmentation Head - Output Layers
        self.upsample_layer_2 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.skip_link_layer_1 = nn.Conv2d(40, 512, 3, 1, 1)
        self.decode_layer_4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.decode_layer_5 = nn.Conv2d(512, 256, 3, 1, 1)

        self.upsample_layer_3 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.skip_link_layer_2 = nn.Conv2d(24, 256, 3, 1, 1)
        self.decode_layer_6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.decode_layer_7 = nn.Conv2d(256, 128, 3, 1, 1)

        self.upsample_layer_4 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.skip_link_layer_3 = nn.Conv2d(32, 128, 3, 1, 1)
        self.decode_layer_8 = nn.Conv2d(128, 128, 3, 1, 1)
        self.decode_layer_9 = nn.Conv2d(128, 64, 3, 1, 1)

        self.upsample_layer_5 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.decode_layer_10 = nn.Conv2d(64, 64, 3, 1, 1)
        self.decode_layer_11 = nn.Conv2d(64, 4, 3, 1, 1)

    def forward(self, neck, features):

        # Decoder upsample block 3
        # Upsample
        d5 = self.upsample_layer_2(neck)
         # Expand and add layer from Encoder
        d5 = d5 + self.skip_link_layer_1(features[2])
        # Double convolution
        d5 = self.decode_layer_4(d5)
        d5 = self.GeLU(d5)
        d6 = self.decode_layer_5(d5)
        d6 = self.sigmoid(d6)

        # Decoder upsample block 4
        # Upsample
        d7 = self.upsample_layer_3(d6)
         # Expand and add layer from Encoder
        d7 = d7 + self.skip_link_layer_2(features[1])
        # Double convolution
        d7 = self.decode_layer_6(d7)
        d7 = self.GeLU(d7)
        d8 = self.decode_layer_7(d7)
        d8 = self.sigmoid(d8)

        # Decoder upsample block 5
        # Upsample
        d8 = self.upsample_layer_4(d8)
         # Expand and add layer from Encoder
        d8 = d8 + self.skip_link_layer_3(features[0])
        # Double convolution
        d8 = self.decode_layer_8(d8)
        d8 = self.GeLU(d8)
        d9 = self.decode_layer_9(d8)
        d10 = self.sigmoid(d9)

        # Decoder upsample block 6
        # Upsample
        d10 = self.upsample_layer_5(d10)
        # Double convolution
        d10 = self.decode_layer_10(d10)
        d10 = self.GeLU(d10)
        d11 = self.decode_layer_11(d10)
        output = self.sigmoid(d11)

        return output

class AutoSeg(nn.Module):
    def __init__(self):
        super(AutoSeg, self).__init__()
        
        # Encoder
        self.Encoder = Encoder()

        # Context
        self.Context = Context()

        # Neck
        self.Neck = Neck()

        # Head
        self.CoarseSegHead = CoarseSegHead()
    

    def forward(self,image):
        features = self.Encoder(image)
        deep_features = features[4]
        context = self.Context(deep_features)
        neck = self.Neck(context, features)
        output = self.CoarseSegHead(neck, features)
        return output

# Checking devices (GPU vs CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} for inference')

# Instantiate model
model = AutoSeg().eval().to(device)

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
image = Image.open('/home/zain/Autoware/semantic_segmentation/training_data/Coarse_Seg/ACDC/images/156.png')
image_tensor = load_image(image)

# Inference
prediction = model(image_tensor)
print(model)

# Visualise
prediction = prediction.squeeze(0).cpu().detach()
prediction = prediction.permute(1, 2, 0)
prediction_0 = prediction [:,:,0]
prediction_1 = prediction [:,:,1]
prediction_2 = prediction [:,:,2]
prediction_3 = prediction [:,:,3]
print(prediction.size())

fig = plt.figure()
plt.imshow(image)
fig1, axs = plt.subplots(2,2)
axs[0,0].imshow(prediction_0)
axs[0,1].imshow(prediction_1)
axs[1,0].imshow(prediction_2)
axs[1,1].imshow(prediction_3)


# %%
