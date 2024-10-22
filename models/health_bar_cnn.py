import torch.nn as nn
import torch

# CNN Model for Health Bar Detection
class HealthBarCNN(nn.Module):
    def __init__(self):
        super(HealthBarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Adjust dimensions based on input image size
        self.fc2 = nn.Linear(128, 1)  # Output is the health percentage

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)  # Flatten image tensor
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid to output health percentage (0-1)
        return x

