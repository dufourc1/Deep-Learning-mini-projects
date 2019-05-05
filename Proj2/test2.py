#!/usr/bin/env python

######################################################################

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue

train_input, train_target, test_input, test_target = \
    prologue.load_data(one_hot_labels = True, normalize = True, flatten = False)



def train_model(model, train_input, train_target, mini_batch_size):
    criterion = nn.MSELoss()
    eta = 1e-1

    for e in range(25):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            sum_loss = sum_loss + loss.item()
            for p in model.parameters():
                p.data.sub_(eta * p.grad.data)
        print(e, sum_loss)

def compute_nb_errors(model, input, target, mini_batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.data.max(1)
        for k in range(mini_batch_size):
            if target.data[b + k, predicted_classes[k]] <= 0:
                nb_errors = nb_errors + 1

    return nb_errors

######################################################################

train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)


mini_batch_size = 100

######################################################################
# Question 2

for k in range(10):
    model = Net(200)
    train_model(model, train_input, train_target, mini_batch_size)
    nb_test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)
    print('test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))

######################################################################
# Question 3

for nh in [ 10, 50, 200, 500, 2500 ]:
    model = Net(nh)
    train_model(model, train_input, train_target, mini_batch_size)
    nb_test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)
    print('test error Net nh={:d} {:0.2f}%% {:d}/{:d}'.format(nh,
                                                              (100 * nb_test_errors) / test_input.size(0),
                                                              nb_test_errors, test_input.size(0)))

######################################################################
# Question 4

model = Net2()
train_model(model, train_input, train_target, mini_batch_size)
nb_test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)
print('test error Net2 {:0.2f}%% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                   nb_test_errors, test_input.size(0)))

print(model.parameters())
