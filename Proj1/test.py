
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




train_input1,train_classes1,train_input2,train_target2 =\
    split_channels(train_input, train_classes)
test_input1,test_classes1,test_input2,test_target2 =\
    split_channels(test_input, test_classes)
train_input, train_classes = Variable(train_input), Variable(train_classes)

mini_batch_size = 100
model = SiameseNet()

print("SiameseNet with auxilary loss function")
train_two_images(model,train_input,train_target, train_classes, nb_epochs = 75, verbose = False)
acc_train = accuracy(model,train_input,train_target)
acc_test = accuracy(model,test_input,test_target)
print("  accuracy on train {:4.2%} and on test {:4.2%}".format(acc_train,acc_test))
print("\n")
model = SiameseNet()
print("SiameseNet with no auxilary loss function")
train_two_images(model,train_input,train_target, train_classes, nb_epochs = 75, verbose = False, aux = False)
acc_train = accuracy(model,train_input,train_target)
acc_test = accuracy(model,test_input,test_target)
print("   accuracy on train {:4.2%} and on test {:4.2%}".format(acc_train,acc_test))
