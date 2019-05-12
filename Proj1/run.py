from sys import exit
import argparse

import torch

from test import test

from FullyConnected import FullyConnected, DropoutFullyConnected
from BasicConvolutional import BasicConvolutional, BasicFullyConvolutional
from ResNet import ResNet
from SiameseNet import SiameseNet

################################################################################
parser = argparse.ArgumentParser(description='Reproduction of our results for Project 1')

parser.add_argument('-m', '--model', type=str, default='all',
                    help = 'The model to be tested.')

parser.add_argument('-o', '--output', type=str, default=None,
                    help = 'The output file where to save the results. If deafults print results on screen.')

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

def make_DropoutFullyConnected():
    dropout = 0.25
    return DropoutFullyConnected(nodes_in=2*14**2, nodes_hidden=1000, nodes_out=2, n_hidden=2, drop = dropout)

def make_DropoutFullyConnectedBatchNorm():
    dropout = 0.25
    return DropoutFullyConnected(nodes_in=2*14**2, nodes_hidden=1000, nodes_out=2, n_hidden=2, drop = dropout, with_batchnorm = True)

def make_ResNet():
    return ResNet(nb_channels=27, kernel_size=5, nb_blocks=7)

def make_SiameseResNet():
    return SiameseNet(branch = ResNet(nb_channels=12, kernel_size=5, nb_blocks=3, in_channels = 1, out_channels = 10))


model_makers = {'all': None,
            'NoModel': no_model,
            'fcnn': make_FullyConnected,
            'basicconv': make_BasicConv,
            'basicfullyconv': make_BasicFullyConv,
            'dropoutfc': make_DropoutFullyConnected,
            'dropoutfcbn': make_DropoutFullyConnectedBatchNorm,
            'resnet': make_ResNet,
            'siamese': SiameseNet,
            'siameseresnet': make_SiameseResNet}

model_lrs = {'all': None,
            'NoModel': no_model,
            'fcnn': 1e-3,
            'basicconv': 2e-4,
            'basicfullyconv': 7e-4,
            'dropoutfc': 2e-4,
            'dropoutfcbn': 2e-4,
            'resnet': 1e-3,
            'siamese': 5e-3,
            'siameseresnet': 5e-3}

model_infos = {'dropoutfcbn': 'BatchNorm',
                'siameseresnet': 'ResNet'}

def run_all(output = None):
    n_trials = 10
    # output = 'results.csv'
    # # output = None

    if output is not None:
        with open(output, 'w') as f:
            f.write(','.join(('ModelName', 'meanCELoss_tr', 'stdCELoss_tr',
             'meanCELoss_te', 'stdCELoss_te',
             'meanAccuracy_tr', 'stdAccuracy_tr',
             'meanAccuracy_te', 'stdAccuracy_tr')) + '\n')

    del model_makers['all']
    del model_makers['NoModel']

    for _, model_maker in model_makers.items():
        infos = model_infos.get(_, '')
        test(model_maker, n_trials = n_trials, output_file= output, lr = model_lrs[_], infos= infos)
        if _.find('siamese') >= 0:
            test(model_maker, n_trials = n_trials, output_file= output, lr = model_lrs[_], infos= infos, auxiliary= True)

    exit()

model_makers['all'] = lambda : run_all(output = args.output)
################################################################################


if model_makers.get(args.model) is None:
    print("Invalid model")
    exit(1)

n_trials = 10
model_maker = model_makers.get(args.model)
infos = model_infos.get(args.model, '')

test(model_maker, n_trials =n_trials, lr= model_lrs[args.model], infos= infos, output_file= args.output)
if args.model.find('siamese') >= 0:
    test(model_maker, n_trials = n_trials, lr = model_lrs[args.model],
            infos= infos, auxiliary= True, output_file= args.output)
