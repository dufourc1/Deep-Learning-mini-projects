
################################################################################

import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.nn import functional as F

from dlc_practical_prologue import generate_pair_sets
from SiameseNet import accuracy,SiameseNet, SimpleNet
from ResNet import ResNet

################################################################################
def affiche_result(name, epochs, acc_train_with, acc_test_with, acc_train_withouth, acc_test_withouth):
    ''' helper to print the result '''
    print("------------------ Test results ------------------")
    print("Training on {} epochs: ".format(epochs))
    print(name +", auxiliray:    accuracy on train {:4.2%} and on test {:4.2%}".format(acc_train_with,acc_test_with))
    print(name +", no auxiliary: accuracy on train {:4.2%} and on test {:4.2%}".format(acc_train_withouth,acc_test_withouth))
    print("--------------------------------------------------")
    print("\n")

def test(input_train, target_train, classes_train, input_test, target_test, classes_test,\
 branch_init,name, epochs = 75, batch_size = 250, verbose = False,args = [], device = 'cpu'):
    branch = branch_init(*args)
    model = SiameseNet(branch = branch)

    model = model.to(device)

    input_train, target_train, classes_train = input_train.to(device), target_train.to(device), classes_train.to(device)
    input_test, target_test, classes_test = input_test.to(device), target_test.to(device), classes_test.to(device)

    acc_train, acc_test = model.train(input_train, target_train, train_classes = classes_train, auxiliary = True, verbose = verbose,\
                nb_epochs = epochs, batch_size= batch_size, device=device,evolution = True, test_input = input_test, test_target = target_test)


    plt.figure()
    plt.plot(acc_test, label = "validation accuracy")
    plt.plot(acc_train, label = "train accuracy")
    plt.legend()
    plt.title(name)
    plt.tight_layout()
    plt.savefig("data/"+name+"_accuracy.pdf")

    acc_train_with = accuracy(model,input_train,target_train)
    acc_test_with = accuracy(model,input_test,target_test)

    branch = branch_init(*args)
    model = SiameseNet(branch = branch)

    model = model.to(device)

    acc_train, acc_test = model.train(input_train, target_train, train_classes = classes_train, auxiliary = False, verbose = verbose,\
                nb_epochs = epochs, batch_size= batch_size, device=device,evolution = True, test_input = input_test, test_target = target_test)

    plt.figure()
    plt.plot(acc_test, label = "validation accuracy")
    plt.plot(acc_train, label = "train accuracy")
    plt.legend()
    plt.title(name)
    plt.tight_layout()
    plt.savefig("data/"+name+"_accuracy_no_auxiliary.pdf")

    acc_train_withouth = accuracy(model,input_train,target_train)
    acc_test_withouth = accuracy(model,input_test,target_test)

    affiche_result(name, epochs, acc_train_with, acc_test_with, acc_train_withouth, acc_test_withouth)

if __name__ == '__main__':

    #load the data
    input_train, target_train, classes_train,\
        input_test, target_test, classes_test = generate_pair_sets(1000)

    input_train, classes_train = Variable(input_train), Variable(classes_train)

    device = 'cpu'

    epochs = 75
    batch_size = 250

    test(input_train, target_train, classes_train,\
        input_test, target_test, classes_test, SimpleNet, "SimpleNet branch", epochs = epochs, verbose = False, device=device)
    test(input_train, target_train, classes_train,\
        input_test, target_test, classes_test, ResNet, "ResNet branch", epochs = epochs, verbose = False,args= [12,5,3,1], device=device)
