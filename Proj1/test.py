
################################################################################

import torch
import math

from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.nn import functional as F

from dlc_practical_prologue import generate_pair_sets
from neural_nets import accuracy,SiameseNet, train_siamese, SiameseNet_test, train_model,SimpleNet, old_accuracy

################################################################################


#load the data
train_input, train_target, train_classes,\
    test_input, test_target, test_classes = generate_pair_sets(1000)

train_input, train_classes = Variable(train_input), Variable(train_classes)

epochs = 75

print("--------------- with new module ---------------------------------")
branch = SimpleNet()
model = SiameseNet_test(branch = branch)
model.train(train_input, train_target, train_classes = train_classes, auxilary = True, verbose = True, nb_epochs = epochs)
acc_train_new_with = accuracy(model,train_input,train_target)
acc_test_new_with = accuracy(model,test_input,test_target)


branch = SimpleNet()
model = SiameseNet_test(branch = branch)
model.train(train_input, train_target, train_classes = train_classes, auxilary = False, verbose = True, nb_epochs = epochs)
acc_train_new_withouth = accuracy(model,train_input,train_target)
acc_test_new_withouth = accuracy(model,test_input,test_target)
print("\n")
print("------------------------ with old version -----------------------------")

## testing with auxilary loss function (default setting)
model = SiameseNet()
print("training with auxilary loss function")
train_siamese(model,train_input,train_target, train_classes, nb_epochs = epochs, verbose = True)
acc_train_old_with = old_accuracy(model,train_input,train_target)
acc_test_old_with = old_accuracy(model,test_input,test_target)
## testing withouth auxilary loss
model = SiameseNet()
print("training with no auxilary loss function")
train_siamese(model,train_input,train_target, train_classes, nb_epochs = epochs, verbose = True, aux = False)
acc_train_old_withouth = old_accuracy(model,train_input,train_target)
acc_test_old_withouth = old_accuracy(model,test_input,test_target)

print("New module, avec:   accuracy on train {:4.2%} and on test {:4.2%}".format(acc_train_new_with,acc_test_new_with))
print("New module, sans:   accuracy on train {:4.2%} and on test {:4.2%}".format(acc_train_new_withouth,acc_test_new_withouth))
print("Old module, avec:   accuracy on train {:4.2%} and on test {:4.2%}".format(acc_train_old_with,acc_test_old_with))
print("Old module, sans:   accuracy on train {:4.2%} and on test {:4.2%}".format(acc_train_old_withouth,acc_test_old_withouth))
