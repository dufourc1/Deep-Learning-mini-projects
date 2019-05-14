import torch
import torch.nn.functional as F
import torch.nn as nn

################################################################################
# Single fully connected layer with actiation function

class basic_layer(nn.Module):
    def __init__(self, n_in, n_out, activation_fc=F.relu):
        super(basic_layer, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        self.activation = activation_fc

    def forward(self, x):
        return self.activation(self.fc1(x))

################################################################################
# Fully connected Network

class FullyConnected(nn.Module):
    def __init__(self, nodes_in, nodes_hidden, nodes_out, n_hidden, activation_fc=F.relu):
        super(FullyConnected, self).__init__()
        self.activation = nn.Tanh if activation_fc == F.tanh else\
                            nn.LeakyReLU if activation_fc == F.leaky_relu else\
                            nn.ReLU
        self.net = nn.Sequential(nn.Linear(nodes_in, nodes_hidden),
                                 self.activation(),
                                 *(basic_layer(nodes_hidden, nodes_hidden, activation_fc) for _ in  range(n_hidden-1)),
                                 nn.Linear(nodes_hidden, nodes_out))
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

################################################################################
# Single fully connected layer with dropout and activation function

class layer(nn.Module):
    def __init__(self, n_in, n_out, drop):
        super(layer, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return x

################################################################################
# Fully connected layer with dropout and batch normalization and activation function

class layer_with_bn(nn.Module):
    def __init__(self, n_in, n_out, drop):
        super(layer_with_bn, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))
        x = self.dropout(x)
        return x

# no dropout if drop = 0.0, a good value to try can be 0.5, 0.25

################################################################################
# Fully connected network with (optional) dropout and optional batch normalization

class DropoutFullyConnected(nn.Module):
    def __init__(self, nodes_in, nodes_hidden, nodes_out, n_hidden, drop = 0.0, with_batchnorm = False):
        super(Net2, self).__init__()
        if with_batchnorm:
            self.net = nn.Sequential(nn.Linear(nodes_in, nodes_hidden),
                                     nn.ReLU(),
                                     *(layer_with_bn(nodes_hidden, nodes_hidden, drop) for _ in  range(n_hidden-1)),
                                     nn.Linear(nodes_hidden, nodes_out))
        else:
            self.net = nn.Sequential(nn.Linear(nodes_in, nodes_hidden),
                                     nn.ReLU(),
                                     *(layer(nodes_hidden, nodes_hidden, drop) for _ in  range(n_hidden-1)),
                                     nn.Linear(nodes_hidden, nodes_out))
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

Net2 = DropoutFullyConnected
