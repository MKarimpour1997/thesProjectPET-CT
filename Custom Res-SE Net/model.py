import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import torch
import torch.nn as nn

class ResidualSEBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation attention"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels//16, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels//16, out_channels, 1),
            nn.Sigmoid()
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        se_weight = self.se(out)
        out = out * se_weight
        out += residual
        return nn.LeakyReLU(0.1)(out)

class NormalConvBlock(nn.Module):
    """Standard Convolutional Block without Residual Connections"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, x):
        return self.conv(x)

class CustomConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Alternating residual and normal blocks
        self.layer1 = self._make_layer(3, 16, 2, use_residual=True)    # Residual
        self.layer2 = self._make_layer(16, 32, 1, use_residual=False)  # Normal
        self.layer3 = self._make_layer(32, 64, 1, use_residual=True)   # Residual
        self.layer4 = self._make_layer(64, 128, 1, use_residual=False) # Normal
        self.layer5 = self._make_layer(128, 256, 1, use_residual=True) # Residual
        self.layer6 = self._make_layer(256, 512, 1, use_residual=False)# Normal
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.global_avg_pool(x)
        return x.view(x.size(0), -1)  # Shape: [batch_size, 512]
    
    
    def _make_layer(self, in_channels, out_channels, stride, use_residual):
        """Factory method to create different block types"""
        if use_residual:
            block = ResidualSEBlock(in_channels, out_channels, stride)
        else:
            block = NormalConvBlock(in_channels, out_channels, stride)
            
        return nn.Sequential(
            block,
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4 if out_channels > 64 else 0.3)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        return self.classifier(out)
