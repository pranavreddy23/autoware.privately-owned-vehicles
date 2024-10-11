from .backbone import Backbone
from .context import Context
from .neck import Neck
from .seg_head import SegHead
import torch.nn as nn

class CoarseSeg(nn.Module):
    def __init__(self):
        super(CoarseSeg, self).__init__()
        
        # Encoder
        self.Backbone = Backbone()

        # Context
        self.Context = Context()

        # Neck
        self.Neck = Neck()

        # Head
        self.CoarseSegHead = SegHead()
    

    def forward(self,image):
        features = self.Backbone(image)
        deep_features = features[4]
        context = self.Context(deep_features)
        neck = self.Neck(context, features)
        output = self.CoarseSegHead(neck, features)
        return output