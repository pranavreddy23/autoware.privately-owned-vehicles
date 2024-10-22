#! /usr/bin/env python3
import torch.nn as nn

class SceneNeckTiny(nn.Module):
    def __init__(self):
        super(SceneNeckTiny, self).__init__()
        # SceneNeckTiny
        self.GeLU = nn.GELU()
        self.upsample = nn.Upsample(scale_factor=2)

        # Decoder - Neck Layers 
        self.skip_link_layer_0 = nn.Conv2d(80, 1280, 1)

        self.decode_layer_0_pointwise = nn.Conv2d(1280, 512, 1)
        self.decode_layer_0_depthwise = nn.Conv2d(512, 512, 3, padding=1, groups=512)
        self.decode_layer_1 = nn.Conv2d(512, 512, 3, 1, 1)

        self.skip_link_layer_1 = nn.Conv2d(40, 512, 1)
        self.decode_layer_2_pointwise = nn.Conv2d(512, 512, 1)
        self.decode_layer_2_depthwise = nn.Conv2d(512, 512, 3, padding=1, groups=512)
        self.decode_layer_3 = nn.Conv2d(512, 256, 3, 1, 1)

        self.skip_link_layer_2 = nn.Conv2d(24, 256, 1)
        self.decode_layer_4_pointwise = nn.Conv2d(256, 256, 1)
        self.decode_layer_4_depthwise = nn.Conv2d(256, 256, 3, padding=1, groups=256)
        self.decode_layer_5 = nn.Conv2d(256, 256, 3, 1, 1)

    def forward(self, context, features):

        # Decoder upsample block 1
        # Upsample
        d0 = self.upsample(context)
        # Add layer from Encoder
        d0 = d0 + self.skip_link_layer_0(features[3])
        # Double Convolution
        d1 = self.decode_layer_0_pointwise(d0)
        d1 = self.decode_layer_0_depthwise(d1)
        d1 = self.GeLU(d1)
        d2 = self.decode_layer_1(d1)
        d2 = self.GeLU(d2)

        # Decoder upsample block 2
        # Upsample
        d3 = self.upsample(d2)
        # Expand and add layer from Encoder
        d3 = d3 + self.skip_link_layer_1(features[2])
        # Double convolution
        d3 = self.decode_layer_2_pointwise(d3)
        d3 = self.decode_layer_2_depthwise(d3)
        d3 = self.GeLU(d3)
        d4 = self.decode_layer_3(d3)
        d5 = self.GeLU(d4)

        # Decoder upsample block 3
        # Upsample
        d5 = self.upsample(d5)
         # Expand and add layer from Encoder
        d5 = d5 + self.skip_link_layer_2(features[1])
        # Double convolution
        d5 = self.decode_layer_4_pointwise(d5)
        d5 = self.decode_layer_4_depthwise(d5)
        d5 = self.GeLU(d5)
        d6 = self.decode_layer_5(d5)
        neck = self.GeLU(d6)

        return neck