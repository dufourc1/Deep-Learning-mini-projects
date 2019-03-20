import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue

if __name__ == "__main__":

    #loading the data
    train_input, train_target, train_classes, train_input, train_target, train_classes= prologue.generate_pair_sets(1000)
