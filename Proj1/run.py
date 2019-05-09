from sys import exit
import argparse

import torch

from FullyConnected import FullyConnected
from BasicConvolutional import BasicConvolutional, BasicFullyConvolutional
from ResNet import ResNet

################################################################################
parser = argparse.ArgumentParser(description='Reproduction of our results for Project 1')

parser.add_argument('-m', '--model', type=str, default='NoModel',
                    help = 'The model to be tested.')

args = parser.parse_args()

################################################################################

def no_model():
    print("A model has to be chosen")
    exit(1)

def make_FullyConnected():
    return FullyConnected(nodes_in=2*14**2, nodes_hidden=1000, nodes_out=2, n_hidden=2)

def make_BasicConv():
    return BasicConvolutional(nb_channels_list= [2, 16, 16, 16, 32, 64],
        kernel_size_list= [3, 3, 3, 5, 4],
        activation_fc=torch.nn.functional.relu, linear_channels=64)

def make_BasicFullyConv():
    return BasicFullyConvolutional(nb_channels_list= [2, 16, 16, 16, 32, 64],
        kernel_size_list= [2, 2, 3, 3, 5, 4],
        activation_fc=torch.nn.functional.tanh)

def make_ResNet():
    return ResNet(nb_channels=27, kernel_size=5, nb_blocks=7)

model_makers = {'NoModel': no_model,
            'test': make_FullyConnected,
            'fcnn': make_FullyConnected,
            'basicconv': make_BasicConv,
            'basicfullyconv': make_BasicFullyConv,
            'resnet': make_ResNet}

if model_makers.get(args.model) is None:
    print("Invalid model")
    exit(1)

model_maker = model_makers.get(args.model)
model = model_maker()


from test import test

test(model, n_trials =10)
