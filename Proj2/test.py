import torch
import math
import sys

import matplotlib.pyplot as plt
import dlc_practical_prologue as helpers

from Sequential import Sequential
from Linear import Linear
from Functionnals import Relu,Tanh
import Optimizer
import Criterion

#setting the type of tensor
torch.set_default_dtype(torch.float32)


################################################################################
#Utility functions

def update_progress(progress,message=""):
    #function to see the training progression
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(20*progress))
    text = "\rLearning : [{0}] {1}% {2} {3}".format( "="*block + " "*(20-block), round(progress*100,2),
                                                    status,message)
    sys.stdout.write(text)
    sys.stdout.flush()

def train(model, criterion, optimizer, input, target, nb_epochs = 200):

    mini_batch_size = 100
    verbose = False

    loss_evolution = []
    for e in range(nb_epochs):
        loss_e = 0.
        for b in range(0, input.size(0), mini_batch_size):
            output = model.forward(input.narrow(0, b, mini_batch_size))
            loss = criterion.forward(output, target.narrow(0, b, mini_batch_size))
            loss_e += loss
            model.zero_grad()
            inter = criterion.backward()
            model.backward(inter)
            optimizer.step()
        if verbose:
            print("epoch {:3}, loss {:4}".format(e,loss_e))

        loss_evolution.append(loss_e)
        if not verbose:
            update_progress((e+1.)/nb_epochs)
    return loss_evolution

def compute_nb_errors(model, data_input, data_target):

    nb_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_errors += 1
    return nb_errors

################################################################################
# data functions

def generate_disc_data(n = 1000, one_hot_labels = False, long = True):
    input = torch.empty(n,2).uniform_(0,1)
    centered = input- torch.empty(n,2).fill_(0.5)
    if long:
        target = centered.pow(2).sum(1).sub_(1/(2*math.pi)).sign().add(1).div(2).long()
    else:
        target = centered.pow(2).sum(1).sub_(1/(2*math.pi)).sign().add(1).div(2)
    if one_hot_labels:
        target = helpers.convert_to_one_hot_labels(input,target)
    return input,target

if __name__ == "__main__":

###############################################################################
#                          test with crossentropy
###############################################################################

    train_input, train_target = generate_disc_data()
    test_input, test_target = generate_disc_data()
    mean,std = train_input.mean(), train_input.std()
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)


    model_relu = Sequential(Linear(2,25),Relu(),Linear(25,25),Linear(25,25),Relu(),Linear(25,2))
    model_tanh = Sequential(Linear(2,25),Tanh(),Linear(25,25),Linear(25,25),Tanh(),Linear(25,2))

    crossentropy = Criterion.CrossEntropy()

    lr = 1e-1

    optimizer_relu = Optimizer.SGD(model_relu.param(),lr = lr)
    optimizer_tanh = Optimizer.SGD(model_tanh.param(),lr = lr)

    loss_relu = train(model_relu, crossentropy, optimizer_relu, train_input, train_target, nb_epochs = 400)
    loss_tanh = train(model_tanh, crossentropy, optimizer_tanh, train_input, train_target, nb_epochs = 400)

    output = model_relu.forward(train_input)
    loss = crossentropy.forward(output, train_target)
    print("relu loss {}".format(loss))

    output = model_tanh.forward(train_input)
    loss = crossentropy.forward(output, train_target)
    print("tanh loss {}".format(loss))

    plt.plot(loss_relu, c = "red", label = "Relu", marker = ',')
    plt.plot(loss_tanh, c = "blue", label = "Tanh", marker = ',')
    plt.legend()
    plt.show()


################################################################################
#                         test with MSE
################################################################################

    train_input, train_target = generate_disc_data(one_hot_labels = True)
    test_input, test_target = generate_disc_data(one_hot_labels = True)
    mean,std = train_input.mean(), train_input.std()
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)


    model_relu = Sequential(Linear(2,25),Relu(),Linear(25,25),Relu(),Linear(25,25),Relu(),Linear(25,2))
    model_tanh = Sequential(Linear(2,25),Tanh(),Linear(25,25), Tanh(),Linear(25,25),Tanh(),Linear(25,2))

    mse = Criterion.MSE()

    lr = 1e-3

    optimizer_relu = Optimizer.SGD(model_relu.param(),lr = lr)
    optimizer_tanh = Optimizer.SGD(model_tanh.param(),lr = lr)

    loss_relu = train(model_relu, mse, optimizer_relu, train_input, train_target, nb_epochs = 600)
    loss_tanh = train(model_tanh, mse, optimizer_tanh, train_input, train_target, nb_epochs = 600)

    output = model_relu.forward(train_input)
    loss = mse.forward(output, train_target)
    print("relu loss {}".format(loss))

    output = model_tanh.forward(train_input)
    loss = mse.forward(output, train_target)
    print("tanh loss {}".format(loss))

    plt.plot(loss_relu[1:], c = "red", label = "Relu")
    plt.plot(loss_tanh[1:], c = "blue", label = "Tanh")
    plt.legend()
    plt.show()
