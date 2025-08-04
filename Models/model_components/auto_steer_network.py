from .backbone import Backbone
from .bev_path_context import BEVPathContext
from .ego_lanes_head import EgoLanesHead
from .ego_path_head import EgoPathHead

import torch.nn as nn

class AutoSteerNetwork(nn.Module):
    def __init__(self):
        super(AutoSteerNetwork, self).__init__()

        # Upstream blocks
        self.BEVBackbone = Backbone()

        # Path Context
        self.PathContext = BEVPathContext()

        # EgoLanes Head
        self.EgoLanesHead = EgoLanesHead()

        # EgoPath Head
        self.EgoPathHead = EgoPathHead()
    

    def forward(self, image):
        features = self.BEVBackbone(image)
        deep_features = features[4]
        context = self.PathContext(deep_features)
        ego_path = self.EgoPathHead(context)
        ego_left_lane, ego_right_lane = self.EgoLanesHead(context)
        return ego_path, ego_left_lane, ego_right_lane