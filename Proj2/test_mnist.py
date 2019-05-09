import torch
import dlc_practical_prologue as prologue

import Criterion
from Sequential import Sequential
from Linear import Linear
from Functionnals import Relu

train_input, train_target, test_input, test_target = \
    prologue.load_data(one_hot_labels = False, normalize = True, flatten = False)

def train_model(model, train_input, train_target, mini_batch_size):
    criterion = Criterion.CrossEntropy()
    eta = 1e-2

    for e in range(10000):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            inter = criterion.backward()
            model.backward(inter)
            sum_loss = sum_loss + loss.item()
            for p in model.param():
                p.value.sub_(eta * p.grad)
        if e%10 == 0:
            print(e, sum_loss)

def compute_nb_errors(model, input, target, mini_batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model.forward(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.data.max(1)
        for k in range(mini_batch_size):
            if target.data[b + k, predicted_classes[k]] <= 0:
                nb_errors = nb_errors + 1

    return nb_errors


mini_batch_size = 100

######################################################################
# Question 2
print(train_target.shape)

for k in range(1):
    print("-------------------------------- new run")
    model = Sequential(Linear(28*28,200), Relu(),Linear(200,10))
    train_input = train_input.view(1000,28*28)
    test_input = test_input.view(1000,28*28)
    train_model(model, train_input, train_target, mini_batch_size)
    #nb_test_errors = compute_nb_errors(model, train_input, train_target, mini_batch_size)
    #print('test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      #nb_test_errors, test_input.size(0)))
