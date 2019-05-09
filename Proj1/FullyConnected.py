import torch
import torch.nn.functional as F
import torch.nn as nn

class basic_layer(nn.Module):
    def __init__(self, n_in, n_out):
        super(layer, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)

    def forward(self, x):
        return F.relu(self.fc1(x))

class FullyConnected(nn.Module):
    def __init__(self, nodes_in, nodes_hidden, nodes_out, n_hidden):
        super(FullyConnected, self).__init__()
        self.net = nn.Sequential(nn.Linear(nodes_in, nodes_hidden),
                                 nn.ReLU(),
                                 *(basic_layer(nodes_hidden, nodes_hidden) for _ in  range(n_hidden-1)),
                                 nn.Linear(nodes_hidden, nodes_out))
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

class layer(nn.Module):
    def __init__(self, n_in, n_out, drop):
        super(layer, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return x


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
class Net2(nn.Module):
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
