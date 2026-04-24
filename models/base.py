import torch
import torch.nn as nn
import torch.nn.functional as F

class GenericCNN(nn.Module):
    """
    A flexible CNN that adapts to different input channels and image sizes.
    """
    def __init__(self, in_channels=1, input_size=28, num_classes=10):
        super(GenericCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.4)
        
        final_size = (input_size - 4) // 2
        self.fc1 = nn.Linear(64 * final_size * final_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        features = self.dropout2(x)
        x = self.fc2(features)
        return F.log_softmax(x, dim=1)

    def forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.dropout2(x)

class LeNet5(nn.Module):
    """
    Classic LeNet-5 architecture for grayscale images.
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def forward_features(self, x):
        x = self.feature_extractor(x)
        return torch.flatten(x, 1)
