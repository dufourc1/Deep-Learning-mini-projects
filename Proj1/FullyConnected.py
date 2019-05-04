import torch
from torch import nn
import torch.nn.functional as F

class layer(nn.Module):
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
                                 *(layer(nodes_hidden, nodes_hidden) for _ in  range(n_hidden-1)),
                                 nn.Linear(nodes_hidden, nodes_out))
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
