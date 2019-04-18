
################################################################################

import torch
import math

from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.nn import functional as F

from dlc_practical_prologue import generate_pair_sets
from neural_nets import accuracy,SiameseNet, SimpleNet

################################################################################


#load the data
train_input, train_target, train_classes,\
    test_input, test_target, test_classes = generate_pair_sets(1000)

train_input, train_classes = Variable(train_input), Variable(train_classes)

epochs = 75

branch = SimpleNet()
model = SiameseNet(branch = branch)
model.train(train_input, train_target, train_classes = train_classes, auxilary = True, verbose = True, nb_epochs = epochs)
acc_train_with = accuracy(model,train_input,train_target)
acc_test_with = accuracy(model,test_input,test_target)


branch = SimpleNet()
model = SiameseNet(branch = branch)
model.train(train_input, train_target, train_classes = train_classes, auxilary = False, verbose = True, nb_epochs = epochs)
acc_train_withouth = accuracy(model,train_input,train_target)
acc_test_withouth = accuracy(model,test_input,test_target)
print("\n")


print("New module, auxilray:    accuracy on train {:4.2%} and on test {:4.2%}".format(acc_train_with,acc_test_with))
print("New module, no auxilary: accuracy on train {:4.2%} and on test {:4.2%}".format(acc_train_withouth,acc_test_withouth))
