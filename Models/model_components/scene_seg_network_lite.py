from .backbone import Backbone
from .scene_context_lite import SceneContextLite
from .scene_neck_lite import SceneNeckLite
from .scene_seg_head import SceneSegHead
import torch.nn as nn

class SceneSegNetworkLite(nn.Module):
    def __init__(self):
        super(SceneSegNetworkLite, self).__init__()
        
        # Encoder
        self.Backbone = Backbone()

        # Context
        self.SceneContextLite = SceneContextLite()

        # Neck
        self.SceneNeckLite = SceneNeckLite()

        # Head
        self.SceneSegHead = SceneSegHead()
    

    def forward(self,image):
        features = self.Backbone(image)
        deep_features = features[4]
        context = self.SceneContextLite(deep_features)
        neck = self.SceneNeckLite(context, features)
        output = self.SceneSegHead(neck, features)
        return output