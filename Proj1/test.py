
################################################################################

import torch
import math

from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.nn import functional as F

import dlc_practical_prologue as prologue

################################################################################

from neural_nets import *

from neural_nets import SiameseNet
from neural_nets import SimpleNet


#load the data
train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)

demo = 1

if demo == 2:

    net = SimpleNet()
    mini_batch_size = 1000

    neural_nets.train_model(net, train_input, train_target, mini_batch_size, verbose = True, lr = 0.05, nb_epochs = 50)
    acc_train = neural_nets.accuracy(net, train_input, train_target)
    acc_test  = neural_nets.accuracy(net, test_input, test_target)
    print("accuracy on train {:5.2%} and test set {:5.2%}".format(acc_train,acc_test))





if demo ==1:


    train_input1,train_classes1,train_input2,train_target2 =\
        split_channels(train_input, train_classes)
    test_input1,test_classes1,test_input2,test_target2 =\
        split_channels(test_input, test_classes)

    train_input, train_classes = Variable(train_input), Variable(train_classes)
    #print("shape of the classes {}".format(train_classes1.shape))

    mini_batch_size = 100
    model = SiameseNet()

    train_two_images(model,train_input,train_target, train_classes, nb_epochs = 75, verbose = True)


    size_visu = 36

    result = model.predict(train_input[0:size_visu,:])
    _, prediction = torch.max(result,1)
    print("predi {}".format(prediction.data.numpy()))
    print("truth {}".format(train_target[0:size_visu].data.numpy()))

    result_alt = model.predict(train_input)
    _, predicted_alt = torch.min(result_alt,1)
    error = 0

    print("flipp {}".format(predicted_alt[0:size_visu].data.numpy()))

    acc_test = accuracy(model,test_input,test_target)
    print("accuracy on test {}".format(acc_test))

    print("visualization of handwritten digits recognition on test")
    result = model(test_input1[0:size_visu])
    _, pred = result.data.max(1)
    print(pred.data.numpy())
    print(test_classes1[0:size_visu].data.numpy())
