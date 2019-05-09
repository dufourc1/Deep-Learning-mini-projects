import torch
import torchvision
from torch import nn

#do stuff 

from torch.autograd import Variable
from torch.nn import functional as F

import dlc_practical_prologue as prologue

######################################################################
# IMPORT DATA IN PAIR SETS
# in target we have 1 if first digit is lesser or equal to the second, and 0 otherwise

N = 1000
train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N)

train_input, train_target = Variable(train_input), Variable(train_target.float()) # use autograd
test_input, test_target = Variable(test_input), Variable(test_target.float())

######################################################################
# input size [1000, 2, 14, 14]
# Marche tres mal :

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size = 3)            # size after is [1000, 32, 12, 12]
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)           # size [1000, 64, 4, 4]
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 1)
        self.dropout = nn.Dropout() # default p = 0.5
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 2))  # size [1000, 32, 6, 6]
        x = self.dropout(x)
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 2))  # size [1000, 64, 2, 2]
        x = self.dropout(x)
        x = x.view(x.size(0), -1)                                 # size [1000, 256]
        x = F.relu(self.fc1(x))                                   # size [1000, 200]
        x = self.fc2(x)                                           # size [1000, 1]
        return x


######################################################################
# TRAINING FUNCTION
def train_model(model, train_input, train_target, mini_batch_size):
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    #criterion = nn.CrossEntropyLoss()                       # Cross-entropy loss
    criterion = nn.MSELoss()

    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)                           # Normalizing

    for e in range(nb_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), batch_size):
            output = model(train_input[b:b+batch_size])
            loss = criterion(output, train_target[b:b+batch_size])
            sum_loss = sum_loss + loss.item()

            #L2 PANALTIES
            #for p in model.parameters():
            #    loss += lambda1 * p.pow(2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(e, sum_loss)


    return


######################################################################
# TEST FUNCTION
# fonctionne pas encore correctement

def compute_nb_errors(model, test_input, test_target, mini_batch_size):
    nb_errors = 0

    for b in range(0, test_input.size(0), mini_batch_size):
        output = model(test_input[b:b+mini_batch_size]).view(-1)
        for k in range(mini_batch_size):
            print(output[k])
            print(test_target.data[b + k])
            #if output[k] <= 0:  # test_target.data[b + k] != output[k]:
            #    nb_errors += 1

    return nb_errors


######################################################################
lr = 1e-1
nb_epochs = 25
batch_size = 100
model = Net()
# lambda1 = 0.002 #for L2 penalties


train_model(model, train_input, train_target, batch_size)
nb_test_errors = compute_nb_errors(model, test_input, test_target, batch_size)
error = nb_test_errors / test_input.size(0) * 100
print( 'test error Net {:0.2f}% {:d}/{:d}'.format(error, nb_test_errors, test_input.size(0))  )
