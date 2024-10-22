from .backbone import Backbone
from .scene_context_tiny import SceneContextTiny
from .scene_neck_tiny import SceneNeckTiny
from .scene_seg_head import SceneSegHead
import torch.nn as nn

class SceneSegNetworkTiny(nn.Module):
    def __init__(self):
        super(SceneSegNetworkTiny, self).__init__()
        
        # Encoder
        self.Backbone = Backbone()

        # Context
        self.SceneContextTiny = SceneContextTiny()

        # Neck
        self.SceneNeckTiny = SceneNeckTiny()

        # Head
        self.SceneSegHead = SceneSegHead()
    

    def forward(self,image):
        features = self.Backbone(image)
        deep_features = features[4]
        context = self.SceneContextTiny(deep_features)
        neck = self.SceneNeckTiny(context, features)
        output = self.SceneSegHead(neck, features)
        return output