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


class SimpleNet(nn.Module):

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

class SiameseNet_test(nn.Module):

    def __init__(self, branch = None):
        super(SiameseNet_test, self).__init__()
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

    def train(self,train_input, train_target, train_classes = None, auxilary = False, verbose = True, nb_epochs = 50):

        if auxilary:
            print("training with auxilary loss with {} epochs".format(nb_epochs))
            if train_classes is None: print("Error: if auxilary loss, the model needs the classes of the training set")
        else:
            print("training with no auxilary losses with {} epochs".format(nb_epochs))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(),lr = 0.005)

        for e in range(nb_epochs):

            if auxilary:

                #separating the data so that we train on the two images separatly, and learn to classify them properly
                train_input1,train_classes1,train_input2,train_classes2 = split_channels(train_input, train_classes)

                #use the branch to perform handwritten digits classification
                out1 = self.branch(train_input1)
                out2 = self.branch(train_input2)

                #auxilary loss: learn to detect the handwritten digits directly
                loss_aux = criterion(out1,train_classes1) + criterion(out2,train_classes2)

                #optimize based on this
                self.zero_grad()
                loss_aux.backward(retain_graph=True)
                optimizer.step()


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




class SiameseNet(nn.Module):

    def __init__(self):
        super(SiameseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(4*64 , 100)
        self.fc2 = nn.Linear(100, 10)
        #to combine the ouput into a prediction at the output of the SiameseNet
        self.pooling = nn.Linear(20,2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, 4* 64)))
        x = self.fc2(x)
        return x

    def predict(self,x):
        ''' method to predict from the network with original output if the first digit is bigger than the second '''
        input1 = torch.reshape(x[:,0,:], (x.shape[0],1,14,14))
        input2 = torch.reshape(x[:,1,:], (x.shape[0],1,14,14))
        out1 = self.forward(input1)
        out2 = self.forward(input2)

        #combine the two tensor on top o feach other
        out = torch.cat((out2, out1), 1)
        #perform actual prediction
        _, response = torch.max(self.pooling(out),1)

        return response

def split_channels(input, classes):
    '''Separate the two images and corresponding classes '''
    #separating the data so that we train on the two images separatly, and learn to classify them properly
    input1 = torch.reshape(input[:,0,:], (input.shape[0],1,14,14))
    classes1 = classes[:,0]

    input2 = torch.reshape(input[:,1,:], (input.shape[0],1,14,14))
    classes2 = classes[:,1]

    return input1,classes1, input2,classes2

def train_siamese(model,train_input,train_target, train_classes, nb_epochs = 25, verbose = True, aux = True):
    """    train the siamese network based on the input being two images of 14x14
    Parameters
    ----------
    model : SiameseNet
        model to train
    train_input : torch.tensor  [n,2,14,14]
        input of pairs of images shape
    train_target : torch.tensor [n]
        class to predict (1 if digit of first image is bigger than second one)
    train_classes : torch.tensor [n,2]
        classes of the actual digits on the two images
    nb_epochs : int
        (the default is 25).
    verbose : bool
        (the default is True).
    aux : bool
        If true train the model with the auxilary loss (the default is True).

    Examples
    -------
    >>> train_input, train_target, train_classes,\
        test_input, test_target, test_classes = prologue.generate_pair_sets(1000)
    >>> model = SiameseNet()
    >>> train_siamese(model,train_input,train_target, train_classes, nb_epochs = 75, verbose = False)

    """
    #loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = 0.005)

    #separating the data so that we train on the two images separatly, and learn to classify them properly
    train_input1,train_classes1,train_input2,train_classes2 = split_channels(train_input, train_classes)

    for e in range(nb_epochs):
        out1 = model(train_input1)
        out2 = model(train_input2)


        if aux:
            #auxilary loss: learn to detect the handwritten digits directly
            loss_aux = criterion(out1,train_classes1)+criterion(out2,train_classes2)
            model.zero_grad()
            loss_aux.backward(retain_graph=True)
            optimizer.step()


        #combine the two tensor on top o feach other
        out = torch.cat((out2, out1), 1)
        #perform actual prediction
        response = model.pooling(out)

        #loss and optimization
        loss = criterion(response,train_target)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose:
            acc = old_accuracy(model, train_input, train_target)
            print("epoch {:3}, loss {:7.4}, accuracy {:.2%}".format(e,loss,acc))

def train_model(model, train_input, train_target, mini_batch_size, verbose = False, lr = 0.005, nb_epochs = 25):
    ''' Simple on training for handwritten recognition'''

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = lr)

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            inter_target = train_target.narrow(0, b, mini_batch_size)
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size) )
            model.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose:
            print("Epoch {:3} loss {:7.4}".format(e, loss.data.numpy()))

def accuracy(model,input,target):

    #perform actual class prediction
    response = model(input).argmax(1)


    error = 0

    #compute the percentage of error
    for tried,true in zip(response,target):
        if tried != true: error+=1

    return 1-error/response.shape[0]

def old_accuracy(model,input,target):
    #perform actual class prediction
    response = model.predict(input)


    error = 0

    #compute the percentage of error
    for tried,true in zip(response,target):
        if tried != true: error+=1

    return 1-error/response.shape[0]
