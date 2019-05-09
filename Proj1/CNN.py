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
        self.fc2 = nn.Linear(200, 2)
        
    def forward(self, x):        
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2)) # size [nb, 32, 5, 5]
        
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2)) # size [nb, 64, 2, 2] add stride=2 ?
        x = x.view(-1, 256) # size [nb, 256]
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)
        return x
    

    
class SimpleConv2(nn.Module):
    def __init__(self):
        super(SimpleConv2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)    # size [nb, 32, 10, 10]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)   # size [nb, 64, 4, 4]
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(20, 2)
        
    def forward(self, x):
        x1 = torch.reshape(x[:,0,:], (x.shape[0],1,14,14))
        x2 = torch.reshape(x[:,1,:], (x.shape[0],1,14,14))
        
        x1 = F.relu(F.max_pool2d(self.conv1(x1), kernel_size=2)) # size [nb, 32, 5, 5]
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2)) # size [nb, 64, 2, 2] add stride=2 ?
        x1 = x1.view(-1, 256) # size [nb, 256]
        x1 = F.relu(self.fc1(x1)) 
        x1 = self.fc2(x1)
        
        x2 = F.relu(F.max_pool2d(self.conv1(x2), kernel_size=2)) # size [nb, 32, 5, 5]
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2)) # size [nb, 64, 2, 2] add stride=2 ?
        x2 = x2.view(-1, 256) # size [nb, 256]
        x2 = F.relu(self.fc1(x2)) 
        x2 = self.fc2(x2)
        
        x = torch.cat((x1, x2), 1)
        x = self.fc3(x)
        return x


