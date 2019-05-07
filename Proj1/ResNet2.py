import torch
import torch.nn.functional as F
import torch.nn as nn

class ResBlock1(nn.Module):
    def __init__(self, nb_channels, kernel_size):
        super(ResBlock1, self).__init__()
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

class SubResBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size):
        super(SubResBlock, self).__init__()
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

class ResBlock2(nn.Module):
    def __init__(self, nb_channels, kernel_size):
        super(ResBlock2, self).__init__()
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

class ResNet(nn.Module):
    def __init__(self, nb_channels_1, nb_channels_2, nb_channels_sub, kernel_size, nb_blocks_1, nb_blocks_2, nb_blocks_sub):
        super(ResNet, self).__init__()
        self.conv0 = nn.Conv2d(2, nb_channels_1, kernel_size = 1)
        self.resblocks1 = nn.Sequential(*(ResBlock1(nb_channels_1, kernel_size) for _ in range(nb_blocks_1)))
        self.avg = nn.AvgPool2d(kernel_size = 21)
        self.fc = nn.Linear(nb_channels, 2)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = self.resblocks(x)
        x = F.relu(self.avg(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
