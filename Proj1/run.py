from sys import exit
import argparse

import torch

from test import test

from FullyConnected import FullyConnected
from BasicConvolutional import BasicConvolutional, BasicFullyConvolutional
from ResNet import ResNet

################################################################################
parser = argparse.ArgumentParser(description='Reproduction of our results for Project 1')

parser.add_argument('-m', '--model', type=str, default='all',
                    help = 'The model to be tested.')

args = parser.parse_args()

################################################################################

def no_model():
    print("A model has to be chosen")
    exit(1)

def make_FullyConnected():
    return FullyConnected(nodes_in=2*14**2, nodes_hidden=1000, nodes_out=2, n_hidden=2)

def make_BasicConv():
    return BasicConvolutional(nb_channels_list= [2, 16, 16, 16],
        kernel_size_list= [5, 5, 6],
        activation_fc=torch.nn.functional.relu, linear_channels=16)

def make_BasicFullyConv():
    return BasicFullyConvolutional(nb_channels_list= [2, 8, 16, 32],
        kernel_size_list= [3, 5, 5, 4],
        activation_fc=torch.nn.functional.tanh)

def make_ResNet():
    return ResNet(nb_channels=27, kernel_size=5, nb_blocks=7)

model_makers = {'all': None,
            'NoModel': no_model,
            'fcnn': make_FullyConnected,
            'basicconv': make_BasicConv,
            'basicfullyconv': make_BasicFullyConv,
            'resnet': make_ResNet}

def run_all():
    n_trials = 10
    # output = 'results.csv'
    output = None

    if output is not None:
        with open(output, 'w') as f:
            f.write(';'.join(('ModelName', 'meanCELoss_tr', 'stdCELoss_tr',
             'meanCELoss_te', 'stdCELoss_te',
             'meanAccuracy_tr', 'stdAccuracy_tr',
             'meanAccuracy_te', 'stdAccuracy_tr')) + '\n')

    del model_makers['all']
    del model_makers['NoModel']

    for _, model in model_makers.items():
        test(model(), n_trials = n_trials, output_file= output)

    exit()

model_makers['all'] = run_all
################################################################################


if model_makers.get(args.model) is None:
    print("Invalid model")
    exit(1)

model_maker = model_makers.get(args.model)
model = model_maker()

test(model, n_trials =10)
