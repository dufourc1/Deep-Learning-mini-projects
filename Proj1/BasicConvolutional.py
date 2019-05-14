import torch
import torch.nn.functional as F
import torch.nn as nn

class BasicConvolutional(nn.Module):
    def __init__(self, nb_channels_list, kernel_size_list, activation_fc, linear_channels):
        super(BasicConvolutional, self).__init__()

        if len(nb_channels_list) < 4 or len(kernel_size_list) < 3:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(in_channels=nb_channels_list[0], out_channels=nb_channels_list[1], kernel_size=kernel_size_list[0],
         stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=nb_channels_list[1], out_channels=nb_channels_list[2], kernel_size=kernel_size_list[1],
         stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=nb_channels_list[2], out_channels=nb_channels_list[3], kernel_size=kernel_size_list[2],
        stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.activation = activation_fc
        self.linear = nn.Linear(linear_channels, 2)

    def forward(self, x):
        y = self.activation(self.conv1(x))
        y = self.activation(self.conv2(y))
        y = self.activation(self.conv3(y))
        y = y.view(y.size(0), -1)
        return self.activation(self.linear(y))

class BasicConvolutionalBN(nn.Module):
    def __init__(self, nb_channels_list, kernel_size_list, activation_fc, linear_channels):
        super(BasicConvolutionalBN, self).__init__()

        if len(nb_channels_list) < 4 or len(kernel_size_list) < 3:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(in_channels=nb_channels_list[0], out_channels=nb_channels_list[1], kernel_size=kernel_size_list[0],
         stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.bn1 = nn.BatchNorm2d(nb_channels_list[1])
        self.conv2 = nn.Conv2d(in_channels=nb_channels_list[1], out_channels=nb_channels_list[2], kernel_size=kernel_size_list[1],
         stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.bn2 = nn.BatchNorm2d(nb_channels_list[2])
        self.conv3 = nn.Conv2d(in_channels=nb_channels_list[2], out_channels=nb_channels_list[3], kernel_size=kernel_size_list[2],
        stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.bn3 = nn.BatchNorm2d(nb_channels_list[3])
        self.activation = activation_fc
        self.linear = nn.Linear(linear_channels, 2)

    def forward(self, x):
        y = self.bn1(self.activation(self.conv1(x)))
        y = self.bn2(self.activation(self.conv2(y)))
        y = self.bn3(self.activation(self.conv3(y)))
        y = y.view(y.size(0), -1)
        return self.activation(self.linear(y))

class BasicFullyConvolutional(nn.Module):
    def __init__(self, nb_channels_list, kernel_size_list, activation_fc):
        super(BasicFullyConvolutional, self).__init__()

        if len(nb_channels_list) < 4 or len(kernel_size_list) < 4:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(in_channels=nb_channels_list[0], out_channels=nb_channels_list[1], kernel_size=kernel_size_list[0],
         stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=nb_channels_list[1], out_channels=nb_channels_list[2], kernel_size=kernel_size_list[1],
         stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=nb_channels_list[2], out_channels=nb_channels_list[3], kernel_size=kernel_size_list[2],
        stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv_final = nn.Conv2d(in_channels=nb_channels_list[3], out_channels=2, kernel_size=kernel_size_list[3],
        stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.activation = activation_fc

    def forward(self, x):
        y = self.activation(self.conv1(x))
        y = self.activation(self.conv2(y))
        y = self.activation(self.conv3(y))
        y = self.activation(self.conv_final(y))
        return y.view(y.size(0), -1)

class BasicFullyConvolutionalBN(nn.Module):
    def __init__(self, nb_channels_list, kernel_size_list, activation_fc):
        super(BasicFullyConvolutionalBN, self).__init__()

        if len(nb_channels_list) < 4 or len(kernel_size_list) < 4:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(in_channels=nb_channels_list[0], out_channels=nb_channels_list[1], kernel_size=kernel_size_list[0],
         stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.bn1 = nn.BatchNorm2d(nb_channels_list[1])
        self.conv2 = nn.Conv2d(in_channels=nb_channels_list[1], out_channels=nb_channels_list[2], kernel_size=kernel_size_list[1],
         stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.bn2 = nn.BatchNorm2d(nb_channels_list[2])
        self.conv3 = nn.Conv2d(in_channels=nb_channels_list[2], out_channels=nb_channels_list[3], kernel_size=kernel_size_list[2],
        stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.bn3 = nn.BatchNorm2d(nb_channels_list[3])
        self.conv_final = nn.Conv2d(in_channels=nb_channels_list[3], out_channels=2, kernel_size=kernel_size_list[3],
        stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.activation = activation_fc

    def forward(self, x):
        y = self.bn1(self.activation(self.conv1(x)))
        y = self.bn2(self.activation(self.conv2(y)))
        y = self.bn3(self.activation(self.conv3(y)))
        y = self.activation(self.conv_final(y))
        return y.view(y.size(0), -1)
