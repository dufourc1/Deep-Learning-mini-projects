#!/usr/bin/env python

######################################################################

import torch
torch.set_default_dtype(torch.float32)

import Criterion
from Sequential import Sequential
from Functionnals import Relu
from Linear import Linear
from Optimizer import SGD



import dlc_practical_prologue as prologue

train_input, train_target, test_input, test_target = \
    prologue.load_data(one_hot_labels = True, normalize = True, flatten = False)


train_input = train_input.view(1000,28*28)
test_input = test_input.view(1000,28*28)


def train_model(model, train_input, train_target, mini_batch_size):
    criterion = Criterion.MSE()
    optim = SGD(model.param(),lr = 1e-1)


    for e in range(200):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            inter = criterion.backward(output,train_target.narrow(0,b,mini_batch_size))
            model.backward(inter)
            optim.step()
            sum_loss = sum_loss + loss.item()

            #print(e, sum_loss)
    nb_train_errors = compute_nb_errors(model, train_input, train_target, 1000)
    print(model.forward(train_input))
    print('train error {:0.2f}% {:d}/{:d}'.format((100 * nb_train_errors) / train_input.size(0),
                                                      nb_train_errors, train_input.size(0)))

def compute_nb_errors(model, input, target, mini_batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model.forward(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.data.max(1)
        for k in range(mini_batch_size):
            if target.data[b + k, predicted_classes[k]] <= 0:
                nb_errors = nb_errors + 1

    return nb_errors

######################################################################

mini_batch_size = 100


for k in range(10):
    model = Sequential(Linear(28*28,200),Relu(),Linear(200,50),Relu(),Linear(50,10))
    print(model)
    print(model.forward(train_input))
    train_model(model, train_input, train_target, mini_batch_size)
    nb_train_errors = compute_nb_errors(model, train_input, train_target, 1000)
    print(model.forward(train_input))
    print('train error {:0.2f}% {:d}/{:d}'.format((100 * nb_train_errors) / train_input.size(0),
                                                      nb_train_errors, train_input.size(0)))
