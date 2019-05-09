import torch
import torch.nn.functional as F
import torch.nn as nn

# input size [nb, 2, 14, 14]
class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)    # size [nb, 32, 10, 10]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)   # size [nb, 64, 4, 4]
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        
    def forward(self, x):        
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2)) # size [nb, 32, 5, 5]
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2)) # size [nb, 64, 2, 2] add stride=2 ?
        x = x.view(-1, 256) # size [nb, 256]
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)
        return x

