import torch
import torch.nn as nn
from basicsr.archs.arch_util import ConvBlock, CrossModalAttention  # Import utilities if available

class EnhancedEFNet(nn.Module):  
    def __init__(self, in_channels=3, event_channels=5, out_channels=3):  
        super(EnhancedEFNet, self).__init__()  
        self.image_conv1 = ConvBlock(in_channels, 128)  
        self.event_conv1 = ConvBlock(event_channels, 128)  
        self.attention1 = CrossModalAttention(128)

    def forward(self, blurry_image, event_data):  
        img1 = self.image_conv1(blurry_image)  
        evt1 = self.event_conv1(event_data)  
        fused1 = self.attention1(img1, evt1)  
        return fused1
