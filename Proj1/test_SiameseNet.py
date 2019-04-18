
################################################################################

import torch
import math

from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.nn import functional as F

from dlc_practical_prologue import generate_pair_sets
from SiameseNet import accuracy,SiameseNet, SimpleNet
from ResNet import ResNet

################################################################################


#load the data
train_input, train_target, train_classes,\
    test_input, test_target, test_classes = generate_pair_sets(1000)

train_input, train_classes = Variable(train_input), Variable(train_classes)

def affiche_result(name, epochs, acc_train_with, acc_test_with, acc_train_withouth, acc_test_withouth):
    ''' helper to print the result '''
    print("------------------ Test results ------------------")
    print("Training on {} epochs: ".format(epochs))
    print(name +", auxilray:    accuracy on train {:4.2%} and on test {:4.2%}".format(acc_train_with,acc_test_with))
    print(name +", no auxiliary: accuracy on train {:4.2%} and on test {:4.2%}".format(acc_train_withouth,acc_test_withouth))
    print("--------------------------------------------------")
    print("\n")

def test(branch_init,name, epochs = 75, verbose = False,args = []):
    branch = branch_init(*args)
    model = SiameseNet(branch = branch)
    model.train(train_input, train_target, train_classes = train_classes, auxiliary = True, verbose = verbose, nb_epochs = epochs)
    acc_train_with = accuracy(model,train_input,train_target)
    acc_test_with = accuracy(model,test_input,test_target)


    branch = branch_init(*args)
    model = SiameseNet(branch = branch)
    model.train(train_input, train_target, train_classes = train_classes, auxiliary = False, verbose = verbose, nb_epochs = epochs)
    acc_train_withouth = accuracy(model,train_input,train_target)
    acc_test_withouth = accuracy(model,test_input,test_target)

    affiche_result(name, epochs, acc_train_with, acc_test_with, acc_train_withouth, acc_test_withouth)

if __name__ == '__main__':

    epochs = 50
    test(SimpleNet, "SimpleNet branch", epochs = epochs, verbose = False)
    test(ResNet, "ResNet branch", epochs = epochs, verbose = False,args= [12,5,3,1])
