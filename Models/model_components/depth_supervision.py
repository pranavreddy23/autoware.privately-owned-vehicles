#! /usr/bin/env python3
import torch.nn as nn

class DepthSupervision(nn.Module):
    def __init__(self):
        super(DepthSupervision, self).__init__()
        # Standard
        self.GeLU = nn.GELU()

        # Depth Supervision - Feature Extraction Layers
        self.depth_feature_layer_0 = nn.Conv2d(1, 64, 3, 1, 1)
        self.depth_feature_layer_1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.depth_feature_layer_2 = nn.Conv2d(1, 64, 3, 1, 1)
        self.depth_feature_layer_3 = nn.Conv2d(1, 64, 3, 1, 1)
        self.depth_feature_layer_4 = nn.Conv2d(1, 64, 3, 1, 1)
        self.depth_feature_layer_5 = nn.Conv2d(1, 64, 3, 1, 1)

        # Depth Supervision - 1D Conv Layers
        self.depth_super_layer_0 = nn.Conv2d(64, 128, 1)
        self.depth_super_layer_1 = nn.Conv2d(64, 256, 1)
        self.depth_super_layer_2 = nn.Conv2d(64, 512, 1)
        self.depth_super_layer_3 = nn.Conv2d(64, 768, 1)
        self.depth_super_layer_4 = nn.Conv2d(64, 1280, 1)
        self.depth_super_layer_5 = nn.Conv2d(64, 1280, 1)
       
    def forward(self, depth_pyramid_features):

        # Depth Supervision Pyramid Features
        # Layer 0
        d0 = self.depth_feature_layer_0(depth_pyramid_features[0])
        d0 = self.depth_super_layer_0(d0)
        d0 = self.GeLU(d0)
        depth_pyramid_features[0] = d0

        # Layer 1
        d1 = self.depth_feature_layer_1(depth_pyramid_features[1])
        d1 = self.depth_super_layer_1(d1)
        d1 = self.GeLU(d1)
        depth_pyramid_features[1] = d1

        # Layer 2
        d2 = self.depth_feature_layer_2(depth_pyramid_features[2])
        d2 = self.depth_super_layer_2(d2)
        d2 = self.GeLU(d2)
        depth_pyramid_features[2] = d2

        # Layer 3
        d3 = self.depth_feature_layer_3(depth_pyramid_features[3])
        d3 = self.depth_super_layer_3(d3)
        d3 = self.GeLU(d3)
        depth_pyramid_features[3] = d3

        # Layer 4
        d4 = self.depth_feature_layer_4(depth_pyramid_features[4])
        d4 = self.depth_super_layer_4(d4)
        d4 = self.GeLU(d4)
        depth_pyramid_features[4] = d4

        # Layer 5
        d5 = self.depth_feature_layer_5(depth_pyramid_features[5])
        d5 = self.depth_super_layer_5(d5)
        d5 = self.GeLU(d5)
        depth_pyramid_features[5] = d5

        return depth_pyramid_features   