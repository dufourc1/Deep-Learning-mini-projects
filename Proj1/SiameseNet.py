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
from helpers import update_progress

################################################################################

#input size [1000,1,14,14]
#one image is then [1,14,14]


class SimpleNet(nn.Module):
    ''' basic branch of the siamese net'''

    def __init__(self):
        super(SimpleNet,self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(4*64 , 100)
        self.fc2 = nn.Linear(100, 10)
        #to combine the ouput into a prediction at the output of the SiameseNet

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, 4* 64)))
        x = self.fc2(x)
        return x

    def predict(self,x):
        _, predicted_class = torch.max(self.forward(x),1)
        return predicted_class

class SiameseNet(nn.Module):
    '''Siamese net module '''

    def __init__(self, branch = None):
        """ constructor for the siamese net, branch should be a module of input size [n,1,14,14] and ouput [n,10]

        Parameters
        ----------
        branch : nn.Module
            branch of the siamese net (net that is shared) (the default is None).
        >>>

        """
        super(SiameseNet, self).__init__()
        if branch == None:
            self.branch= SimpleNet()
        else:
            self.branch = branch
        self.pooling = nn.Linear(20,2)

    def forward(self,x):
        input1 = torch.reshape(x[:,0,:], (x.shape[0],1,14,14))
        input2 = torch.reshape(x[:,1,:], (x.shape[0],1,14,14))
        out1 = self.branch(input1)
        out2 = self.branch(input2)
        x = torch.cat((out2, out1), 1)
        x = self.pooling(x)
        return x

    def train(self,train_input, train_target, train_classes = None, auxiliary = False, verbose = True, nb_epochs = 50, batch_size=250, device='cpu', evolution = False, test_input = None, test_target = None):
        """ Training of the siamese module.
            if not auxiliary:
                usual training
            if auxiliary:
                first split the sample into the two images and use the branch to try to classify the numbers, compute the
                loss wrt to the classes of digits and update the parameters  of the branch.
                Then run the entire sample trough the sample and compute the loss against the train_target and update

        Parameters
        ----------
        train_input : [n,2,14,14]
            two images [14,14] representing two numbers
        train_target : [n,2]
            is the first digit bigger than the second
        train_classes : [n,2]
            values of the two digits
        auxiliary : bool
            use of the auxiliary loss (the default is False).
        verbose : bool
            if True print the training  `verbose` (the default is True).
        nb_epochs : int
            epochs to train (the default is 50).
        device : torch.device
            Torch device (the default is 'cpu').
        """

        if auxiliary:
            if verbose:
                print("training with auxiliary loss with {} epochs".format(nb_epochs))
            if train_classes is None: print("Error: if auxiliary loss, the model needs the classes of the training set")
        elif verbose:
            print("training with no auxiliary losses with {} epochs".format(nb_epochs))

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        optimizer = optim.Adam(self.parameters(),lr = 0.005)

        if evolution:
            acc_train = []
            acc_test = []

        for e in range(nb_epochs):
            for input, targets in zip(train_input.split(batch_size), train_target.split(batch_size)):

                if auxiliary:
                    #first pass
                    #separating the data so that we train on the two images separatly, and learn to classify them properly
                    train_input1,train_classes1,train_input2,train_classes2 = split_channels(train_input, train_classes)

                    #use the branch to perform handwritten digits classification
                    out1 = self.branch(train_input1)
                    out2 = self.branch(train_input2)

                    #auxiliary loss: learn to detect the handwritten digits directly
                    loss_aux = criterion(out1,train_classes1) + criterion(out2,train_classes2)

                    #optimize based on this
                    self.zero_grad()
                    loss_aux.backward(retain_graph=True)
                    optimizer.step()

                    #second pass
                    #loss and optimization of the whole model
                    response = self.forward(train_input)
                    loss = criterion(response,train_target)
                    self.zero_grad()
                    loss.backward()
                    optimizer.step()

                else:
                    response = self.forward(train_input)
                    loss = criterion(response,train_target)
                    self.zero_grad()
                    loss.backward()
                    optimizer.step()

            if verbose:
                acc = accuracy(self, train_input, train_target)
                print("epoch {:3}, loss {:7.4}, accuracy {:.2%}".format(e,loss,acc))
            else:
                update_progress((e+1)/nb_epochs, message="")

            if evolution:
                acc_train.append(accuracy(self, train_input, train_target))
                acc_test.append( accuracy(self, test_input, test_target))
        if evolution:
            return acc_train,acc_test



def split_channels(input, classes):
    '''Separate the two images and corresponding classes '''
    #separating the data so that we train on the two images separatly, and learn to classify them properly
    input1 = torch.reshape(input[:,0,:], (input.shape[0],1,14,14))
    classes1 = classes[:,0]

    input2 = torch.reshape(input[:,1,:], (input.shape[0],1,14,14))
    classes2 = classes[:,1]

    return input1,classes1, input2,classes2


def accuracy(model,input,target):

    #perform actual class prediction
    response = model(input).argmax(1)
    error = 0

    #compute the percentage of error
    for tried,true in zip(response,target):
        if tried != true: error+=1

    return 1-error/response.shape[0]
