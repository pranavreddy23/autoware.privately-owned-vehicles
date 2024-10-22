#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
from pytorch_model_summary import summary
import torch
import sys
sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork
from model_components.scene_seg_network_lite import SceneSegNetworkLite
from model_components.scene_seg_network_tiny import SceneSegNetworkTiny


# Instantiate model 
print(summary(SceneSegNetwork(), torch.zeros((1, 3, 320, 640)), show_input=True))
print(summary(SceneSegNetworkLite(), torch.zeros((1, 3, 320, 640)), show_input=True))
print(summary(SceneSegNetworkTiny(), torch.zeros((1, 3, 320, 640)), show_input=True))