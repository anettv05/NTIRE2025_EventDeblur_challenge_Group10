import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic Convolutional Block with BatchNorm and ReLU."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CrossModalAttention(nn.Module):
    """Cross-modal attention for fusing event data and image features."""
    def __init__(self, channels):
        super(CrossModalAttention, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, image_features, event_features):
        batch, channels, height, width = image_features.size()
        query = self.query_conv(image_features).view(batch, -1, height * width)
        key = self.key_conv(event_features).view(batch, -1, height * width)
        value = self.value_conv(event_features).view(batch, -1, height * width)

        attention = self.softmax(torch.bmm(query.permute(0, 2, 1), key))
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch, channels, height, width)
        return out + image_features


class EnhancedEFNet(nn.Module):
    """Enhanced Event Fusion Network for Deblurring."""
    def __init__(self, in_channels=3, event_channels=6, out_channels=3):
        super(EnhancedEFNet, self).__init__()

        # Encoder
        self.image_conv1 = ConvBlock(in_channels, 64)
        self.event_conv1 = ConvBlock(event_channels, 64)
        self.attention1 = CrossModalAttention(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.image_conv2 = ConvBlock(64, 128)
        self.event_conv2 = ConvBlock(64, 128)
        self.attention2 = CrossModalAttention(128)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128)
        )

        # Decoder
        self.upconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.image_conv3 = ConvBlock(64 + 64, 64)  # Skip connection
        self.attention3 = CrossModalAttention(64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, blurry_image, event_data):
        # Encoder
        img1 = self.image_conv1(blurry_image)
        evt1 = self.event_conv1(event_data)
        fused1 = self.attention1(img1, evt1)

        img2 = self.pool(fused1)
        evt2 = self.pool(evt1)
        img2 = self.image_conv2(img2)
        evt2 = self.event_conv2(evt2)
        fused2 = self.attention2(img2, evt2)

        # Bottleneck
        bottleneck = self.bottleneck(fused2)

        # Decoder
        up = self.upconv(bottleneck)
        up = torch.cat([up, fused1], dim=1)  # Skip connection
        up = self.image_conv3(up)
        up = self.attention3(up, evt1)
        deblurred_image = self.final_conv(up)

        return deblurred_image


def define_network(opt):
    """Wrapper function to initialize EnhancedEFNet from BasicSR config."""
    return EnhancedEFNet(in_channels=3, event_channels=6, out_channels=3)
