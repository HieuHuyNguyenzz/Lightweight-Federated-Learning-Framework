import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward_features(self, x):
        # ResNet architecture: forward pass until the avgpool layer
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

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

    def forward_features(self, x):
        x = self.vgg.features(x)
        x = torch.flatten(x, 1)
        x = self.vgg.avgpool(x)
        x = torch.flatten(x, 1)
        # VGG classifier has 3 linear layers, we take the output of the second one
        x = self.vgg.classifier[0](x)
        x = F.relu(x)
        x = self.vgg.classifier[2](x)
        return F.relu(x)

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

    def forward_features(self, x):
        x = self.mobilenet.features(x)
        x = self.mobilenet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

