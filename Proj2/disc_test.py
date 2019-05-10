#!/usr/bin/env python

######################################################################

import torch
import math
import time
import sys

# from torch import optim
# from torch import Tensor
# from torch.autograd import Variable
# from torch import nn


from Sequential import Sequential
from Linear import Linear
from Functionnals import Relu
import Optimizer
import Criterion



######################################################################

def generate_disc_set(nb):
    input = torch.Tensor(nb, 2).uniform_(-1, 1)
    target = input.pow(2).sum(1).sub(2 / math.pi).sign().add(1).div(2).long()
    return input, target

train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)

mean, std = train_input.mean(), train_input.std()

train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

# train_input, train_target = Variable(train_input), Variable(train_target)
# test_input, test_target = Variable(test_input), Variable(test_target)

mini_batch_size = 100

def update_progress(progress,message=""):
    # update_progress() : Displays or updates a console progress bar
    ## Accepts a float between 0 and 1. Any int will be converted to a float.
    ## A value under 0 represents a 'halt'.
    ## A value at 1 or bigger represents 100%
    barLength = 20 # Modify this to change the length of the progress bar
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
    block = int(round(barLength*progress))
    text = "\rLearning : [{0}] {1}% {2} {3}".format( "="*block + " "*(barLength-block), round(progress*100,2),
                                                    status,message)
    sys.stdout.write(text)
    sys.stdout.flush()
################################################################################

######################################################################


# def train_model(model, train_input, train_target):
#     start = time.time()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr = 1e-1)
#     nb_epochs = 100
#     time_opti = 0
#     for e in range(nb_epochs):
#         for b in range(0, train_input.size(0), mini_batch_size):
#             output = model(train_input.narrow(0, b, mini_batch_size))
#             loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
#             model.zero_grad()
#             loss.backward()
#             start_opti = time.time()
#             optimizer.step()
#             time_opti += time.time()-start_opti
#     end = time.time()
#     print("time: {:5.3}s".format(end-start))
#     print("mean_time_optimizer step {:5.3}s".format(time_opti/(e*e/b)))
#

def train_model_test(model,train_input,train_target, nb_epochs = 20):
    criterion = Criterion.CrossEntropy()
    optimizer = Optimizer.SGD(model.param(), lr = 1e-1)
    time_opti = 0
    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            inter = criterion.backward()
            model.backward(inter)
            optimizer.step()
        update_progress((e+1.)/nb_epochs)
######################################################################

def compute_nb_errors(model, data_input, data_target):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors
##########################################################################
# print("-------------------------------- Pytorch nn module --------------------")
# errors_train = []
# errors_test = []
# for k in range(10):
#     model = create_shallow_model()
#
#     std = 1e-0
#     for p in model.parameters(): p.data.normal_(0, std)
#
#     train_model(model, train_input, train_target)
#     errors_train.append(compute_nb_errors(model, train_input, train_target)/ test_input.size(0) * 100)
#     errors_test.append(compute_nb_errors(model, test_input, test_target) / test_input.size(0) * 100)
#
# print('train_error {:.02f}%  std {:.02f} \n test_error {:.02f}%  std {:.02f}'.format(
# torch.tensor(errors_train).mean(),torch.tensor(errors_train).std(),
# torch.tensor(errors_test).mean(),torch.tensor(errors_test).std(),
# )
# )

print("-------------------------------- my stupid  module --------------------")
errors_train = []
errors_test = []
for k in range(10):
    model = Sequential(Linear(2,128),Relu(),Linear(128,2))

    #std = 1e-0
    #for p in model.param(): p.value.normal_(0, std)

    train_model_test(model, train_input, train_target, nb_epochs = 400)
    errors_train.append(compute_nb_errors(model.forward, train_input, train_target)/ test_input.size(0) * 100)
    errors_test.append(compute_nb_errors(model.forward, test_input, test_target) / test_input.size(0) * 100)

print('train_error {:.02f}%  std {:.02f} \n test_error {:.02f}%  std {:.02f}'.format(
torch.tensor(errors_train).mean(),torch.tensor(errors_train).std(),
torch.tensor(errors_test).mean(),torch.tensor(errors_test).std(),
)
)
