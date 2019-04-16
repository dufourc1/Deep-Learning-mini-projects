import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_in):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

class layer(nn.Module):
    def __init__(self, n_in, n_out):
        super(layer, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)

    def forward(self, x):
        return F.relu(self.fc1(x))

class Net2(nn.Module):
    def __init__(self, nodes_in, nodes_hidden, nodes_out, n_hidden):
        super(Net2, self).__init__()
        self.net = nn.Sequential(nn.Linear(nodes_in, nodes_hidden),
                                 nn.ReLU(),
                                 *(layer(nodes_hidden, nodes_hidden) for _ in  range(n_hidden-1)),
                                 nn.Linear(nodes_hidden, nodes_out))
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
