import torch
import torch.nn as nn
from torchvision import models

class ResNet18Custom(nn.Module):
    """
    Modified ResNet-18 for small images.
    """
    def __init__(self, in_channels=3, input_size=32, num_classes=10):
        super(ResNet18Custom, self).__init__()
        self.resnet = models.resnet18(weights=None)
        
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return torch.log_softmax(self.resnet(x), dim=1)

class VGG11Custom(nn.Module):
    """
    Modified VGG-11 for small images.
    """
    def __init__(self, in_channels=3, input_size=32, num_classes=10):
        super(VGG11Custom, self).__init__()
        self.vgg = models.vgg11(weights=None)
        self.vgg.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return torch.log_softmax(self.vgg(x), dim=1)

class MobileNetV2Custom(nn.Module):
    """
    Modified MobileNetV2 for small images.
    """
    def __init__(self, in_channels=3, input_size=32, num_classes=10):
        super(MobileNetV2Custom, self).__init__()
        self.mobilenet = models.mobilenet_v2(weights=None)
        self.mobilenet.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.last_channel, num_classes)

    def forward(self, x):
        return torch.log_softmax(self.mobilenet(x), dim=1)

