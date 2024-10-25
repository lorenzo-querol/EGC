import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = None  # Will be initialized dynamically based on the input image size
        self.fc2 = nn.Linear(128, num_classes)  # This will remain constant

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)  # Downsample by factor of 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)  # Downsample by factor of 2 again

        # Flatten the feature maps
        x = x.view(x.size(0), -1)

        # Dynamically initialize the first fully connected layer based on the input size
        if self.fc1 is None:
            num_features = x.size(1)  # Get the number of features after flattening
            self.fc1 = nn.Linear(num_features, 128).to(x.device)  # Initialize fc1 dynamically on the correct device

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
