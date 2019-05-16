import torch
import torch.nn.functional as F
import torch.nn as nn

################################################################################
# Convolutional network to be used as a block in a Residual net

class ResBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size, padding = (kernel_size-1)//2)
        self.bn1 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size, padding = (kernel_size-1)//2)
        self.bn2 = nn.BatchNorm2d(nb_channels)

    def forward(self, x):
        y = self.bn1(self.conv1(x))
        y = F.relu(y)
        y = self.bn2(self.conv2(y))
        y += x
        y = F.relu(y)
        return y

################################################################################
# residual net

class ResNet(nn.Module):
    def __init__(self, nb_channels, kernel_size, nb_blocks, in_channels = 2, out_channels =2):
        super(ResNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, nb_channels, kernel_size = 1)
        self.resblocks = nn.Sequential(*(ResBlock(nb_channels, kernel_size) for _ in range(nb_blocks)))
        self.avg = nn.AvgPool2d(kernel_size = 14)
        self.fc = nn.Linear(nb_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = self.resblocks(x)
        x = F.relu(self.avg(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
