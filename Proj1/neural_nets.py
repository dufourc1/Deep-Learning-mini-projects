'''
implementation of a neural net that is trained to recognize the number and then do the comparison
Use weight sharing , siamese type neural net
'''

################################################################################

import torch
import math

from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.nn import functional as F

import dlc_practical_prologue as prologue

################################################################################

#input size [1000,1,14,14]
#one image is then [1,14,14]

class Siamese_net(nn.Module):
    '''network that will be shared by the two "channels" of the bigger network '''
    def __init__(self):
        super(Siamese_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.fc1 = nn.Linear(2* 64, 100)
        self.fc2 = nn.Linear(100, 10)
        #to combine the ouput into a prediction at the output of the Siamese_net
        self.pooling = nn.Linear(20,2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, 2* 64)))
        x = self.fc2(x)
        return x

def split_channels(input, classes):
    '''Separate the two images and corresponding classes '''
    #separating the data so that we train on the two images separatly, and learn to classify them properly
    input1 = torch.reshape(input[:,0,:], (1000,1,14,14))
    classes1 = classes[:,0]

    input2 = torch.reshape(input[:,1,:], (1000,1,14,14))
    classes2 = classes[:,1]

    return input1,classes1, input2,classes2

def train(model,train_input,train_target, train_classes, nb_epochs = 25, verbose = True):

    #loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005)

    #separating the data so that we train on the two images separatly, and learn to classify them properly

    train_input1,train_classes1,train_input2,train_classes2 = split_channels(train_input, train_classes)

    for e in range(nb_epochs):
        out1 = model(train_input1)
        out2 = model(train_input2)

        #combine the two tensor on top o feach other
        out = torch.cat((out1, out2), 1)
        #perform actual prediction
        response = F.relu(model.pooling(out))

        #loss and optimization
        loss = criterion(response,train_target)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose:
            print("epoch {:3}, loss {:.5}".format(e,loss))

def accuracy(model,input,target,classes):

    input1,classes1, input2,classes2 = split_channels(input, classes)
    out1 = model(input1)
    out2 = model(input2)

    #combine the two tensor on top o feach other
    out = torch.cat((out1, out2), 1)
    #perform actual prediction
    response = F.relu(model.pooling(out))

    _, pred = torch.max(response,1)
    error = 0
    for tried,true in zip(pred,target):
        if tried != true: error+=1

    return error*100/pred.shape[0]


if __name__ == "__main__":

    #load the data
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)

    #definition of the model, loss and optimizer
    net = Siamese_net()
    train(net, train_input, train_target, train_classes, nb_epochs = 25, verbose = True)
    print(accuracy(net,test_input,test_target,test_classes))
