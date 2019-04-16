
################################################################################

import torch
import math

from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.nn import functional as F

from dlc_practical_prologue import generate_pair_sets
from neural_nets import accuracy,SiameseNet, train_siamese

################################################################################


#load the data
train_input, train_target, train_classes,\
    test_input, test_target, test_classes = generate_pair_sets(1000)

train_input, train_classes = Variable(train_input), Variable(train_classes)

## testing with auxilary loss function (default setting)
model = SiameseNet()
print("SiameseNet with auxilary loss function")
train_siamese(model,train_input,train_target, train_classes, nb_epochs = 75, verbose = False)
acc_train = accuracy(model,train_input,train_target)
acc_test = accuracy(model,test_input,test_target)
print("  accuracy on train {:4.2%} and on test {:4.2%}".format(acc_train,acc_test))
print("\n")

## testing with
model = SiameseNet()
print("SiameseNet with no auxilary loss function")
train_siamese(model,train_input,train_target, train_classes, nb_epochs = 75, verbose = False, aux = False)
acc_train = accuracy(model,train_input,train_target)
acc_test = accuracy(model,test_input,test_target)
print("   accuracy on train {:4.2%} and on test {:4.2%}".format(acc_train,acc_test))
