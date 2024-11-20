#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
from model_components.super_depth_network import SuperDepthNetwork


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference')
        
    model = SuperDepthNetwork()
    model = model.to(device)

    test_data_image = torch.rand((1, 3, 320, 640)).to(device)

    test_data_pyramid_features = []
    feature_0 = torch.rand((1, 1, 320, 640)).to(device)
    feature_1 = torch.rand((1, 1, 160, 320)).to(device)
    feature_2 = torch.rand((1, 1, 80, 160)).to(device)
    feature_3 = torch.rand((1, 1, 40, 80)).to(device)
    feature_4 = torch.rand((1, 1, 20, 40)).to(device)
    feature_5 = torch.rand((1, 1, 10, 20)).to(device)

    test_data_pyramid_features.append(feature_0)
    test_data_pyramid_features.append(feature_1)
    test_data_pyramid_features.append(feature_2)
    test_data_pyramid_features.append(feature_3)
    test_data_pyramid_features.append(feature_4)
    test_data_pyramid_features.append(feature_5)

    prediction, boundary = model(test_data_image, test_data_pyramid_features)

    prediction = prediction.squeeze(0).cpu().detach()
    prediction = prediction.permute(1, 2, 0)
    prediction = prediction.numpy()
    prediction = prediction + abs(np.min(prediction))

    boundary = boundary.squeeze(0).cpu().detach()
    boundary = boundary.permute(1, 2, 0)

    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(prediction)
    axarr[1].imshow(boundary)
    
if __name__ == '__main__':
    main()
# %%
