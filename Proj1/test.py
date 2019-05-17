from sys import exit
import argparse

import torch
from torch.nn.functional import relu, leaky_relu, tanh

from test_helpers import test

from FullyConnected import FullyConnected, DropoutFullyConnected
from BasicConvolutional import BasicConvolutional, BasicFullyConvolutional,\
                                BasicConvolutionalBN, BasicFullyConvolutionalBN
from ResNet import ResNet
from SiameseNet import SiameseNet

################################################################################
parser = argparse.ArgumentParser(description='Reproduction of our results for Project 1')

parser.add_argument('-m', '--model', type=str, default='all',
                    help = 'The model to be tested.')

parser.add_argument('-o', '--output', type=str, default=None,
                    help = 'The output file where to save the results. If deafults print results on screen.')

parser.add_argument('-a', '--activation_fc', type=str, default='relu',
                    help = 'The activation dunction to use in the net. (default: relu)')

args = parser.parse_args()

################################################################################

def make_FullyConnected(activation_fc= relu):
    """Wrapper of a FCNN constructor.

    Parameters
    ----------
    activation_fc : function
        The activation function to use in the model (the default is relu).

    Returns
    -------
    FullyConnected
        A fully connected neural network.

    """
    return FullyConnected(nodes_in=2*14**2, nodes_hidden=1000, nodes_out=2, n_hidden=2, activation_fc=activation_fc)

def make_BasicConv(activation_fc= relu):
    """Wrapper of a convolutional net constructor.

    Parameters
    ----------
    activation_fc : function
        The activation function to use in the model (the default is relu).

    Returns
    -------
    BasicConvolutional
        A three layer convolutional network with a final fully connected layer.

    """
    return BasicConvolutional(nb_channels_list= [2, 8, 16, 16],
        kernel_size_list= [3, 5, 5],
        activation_fc=activation_fc, linear_channels=4**2*16)

def make_BasicConvBN(activation_fc= relu):
    """Wrapper of a convolutional net constructor.

    Parameters
    ----------
    activation_fc : function
        The activation function to use in the model (the default is relu).

    Returns
    -------
    BasicConvolutionalBN
        A three layer convolutional network with batch normalization with a final fully connected layer.

    """
    return BasicConvolutionalBN(nb_channels_list= [2, 8, 16, 16],
        kernel_size_list= [3, 5, 5],
        activation_fc=activation_fc, linear_channels=4**2*16)

def make_BasicFullyConv(activation_fc= relu):
    """Wrapper of a fully convolutional net constructor.

    Parameters
    ----------
    activation_fc : function
        The activation function to use in the model (the default is relu).

    Returns
    -------
    BasicFullyConvolutional
        A three layer convolutional network with a final fully connected layer.

    """
    return BasicFullyConvolutional(nb_channels_list= [2, 8, 16, 16],
        kernel_size_list= [3, 5, 5, 4],
        activation_fc=activation_fc)

def make_BasicFullyConvBN(activation_fc= relu):
    """Wrapper of a fully convolutional net constructor.

    Parameters
    ----------
    activation_fc : function
        The activation function to use in the model (the default is relu).

    Returns
    -------
    BasicFullyConvolutional
        A four layer fully convolutional network with batch-normalization.

    """
    return BasicFullyConvolutionalBN(nb_channels_list= [2, 8, 16, 16],
                    kernel_size_list= [3, 5, 5, 4],
                    activation_fc= activation_fc)

def make_DropoutFullyConnected(*args,**kwargs):
    """Wrapper of a Fully Connected net with dropout constructor.

    Returns
    -------
    DropoutFullyConnected
        A fully connected nural network.

    """
    dropout = 0.25
    return DropoutFullyConnected(nodes_in=2*14**2, nodes_hidden=1000, nodes_out=2, n_hidden=2, drop = dropout)

def make_DropoutFullyConnectedBatchNorm(*args,**kwargs):
    """Wrapper of a Fully Connected net with dropout constructor and batch normalization\.

    Returns
    -------
    DropoutFullyConnected
        A fully connected nural network.

    """
    dropout = 0.25
    return DropoutFullyConnected(nodes_in=2*14**2, nodes_hidden=1000, nodes_out=2, n_hidden=2, drop = dropout, with_batchnorm = True)

def make_ResNet(*args,**kwargs):
    """Wrapper of a Residual Network constructor.

    Returns
    -------
    ResNet
        A residual notwork.

    """
    return ResNet(nb_channels=12, kernel_size=5, nb_blocks=6)

def make_SiameseNet(*args,**kwargs):
    """Wrapper of a Siames Network constructor. The two siamese network have\
two convolutional and two fully connected layers.

    Returns
    -------
    SiameseNet
        A siamese network.

    """
    return SiameseNet()


def make_SiameseResNet(*args,**kwargs):
    """Wrapper of a Siames Network constructor. The two siamese network are residual.

    Returns
    -------
    SiameseNet
        A siamese network.

    """
    return SiameseNet(branch = ResNet(nb_channels=12, kernel_size=5, nb_blocks=6, in_channels = 1, out_channels = 10))

################################################################################
# The following dictionaries are used to parse the arguments and define model-specific Parameters

model_makers = {'all': None,
            # 'NoModel': no_model,
            'fcnn': make_FullyConnected,
            'basicconv': make_BasicConv,
            'basicconvbn': make_BasicConvBN,
            'basicfullyconv': make_BasicFullyConv,
            'basicfullyconvbn': make_BasicFullyConvBN,
            'dropoutfc': make_DropoutFullyConnected,
            'dropoutfcbn': make_DropoutFullyConnectedBatchNorm,
            'resnet': make_ResNet,
            'siamese': make_SiameseNet,
            'siameseresnet': make_SiameseResNet}

model_lrs = {'all': None,
            # 'NoModel': no_model,
            'fcnn': 1e-3,
            'basicconv': 4e-4, #2e-4
            'basicconvbn': 4e-4,
            'basicfullyconv': 4e-4,
            'basicfullyconvbn': 4e-4,
            'dropoutfc': 2e-4,
            'dropoutfcbn': 2e-4,
            'resnet': 5e-3,
            'siamese': 5e-3,
            'siameseresnet': 5e-3}

activation_fcs = {'relu': relu,
            'leakyrelu': leaky_relu,
            'tanh': tanh}

model_infos = {'dropoutfcbn': 'BatchNorm',
                'siameseresnet': 'ResNet'}

def run_all(output = None):
    """Function that tests all models and outputs results on given support.

    Parameters
    ----------
    output : str
        The name of the file where to store the results. If default it prints onscreen (the default is None).
    """

    if output is not None:
        with open(output, 'w') as f:
            f.write(','.join(('ModelName', 'meanCELoss_tr', 'stdCELoss_tr',
             'meanCELoss_te', 'stdCELoss_te',
             'meanAccuracy_tr', 'stdAccuracy_tr',
             'meanAccuracy_te', 'stdAccuracy_te',
             'meanTime_tr', 'stdTime_tr')) + '\n')

    del model_makers['all'] #I delete from the dictionary the keywords that do not give "proper models"

    if not torch.cuda.is_available(): #the following models are very long to train so do not attempt to train them without gpu
        del model_makers['resnet']
        del model_makers['siamese']
        del model_makers['siameseresnet']

    activation_fc = activation_fcs['relu']

    for model, model_maker in model_makers.items():
        n_trials = 15 if (model.find('siamese') < 0 and model.find('resnet') < 0) else 10

        infos = model_infos.get(model, '')
        test(model_maker, activation_fc, n_trials = n_trials, output_file= output, lr = model_lrs[model], infos= infos)
        if model.find('siamese') >= 0:
            test(model_maker, activation_fc, n_trials = n_trials, output_file= output, lr = model_lrs[model], infos= infos, auxiliary= True)

    if torch.cuda.is_available():
        del model_makers['resnet']  #I do not test heavy models with activation fcts differetn from relu
        del model_makers['siamese']
        del model_makers['siameseresnet']

    activation_fc = activation_fcs['tanh']

    for model, model_maker in model_makers.items():
        n_trials = 15
        infos = model_infos.get(model, '')
        test(model_maker, activation_fc, n_trials = n_trials, output_file= output, lr = model_lrs[model], infos= infos + 'tanh')

    activation_fc = activation_fcs['leakyrelu']

    for model, model_maker in model_makers.items():
        n_trials = 15
        infos = model_infos.get(model, '')
        test(model_maker, activation_fc, n_trials = n_trials, output_file= output, lr = model_lrs[model], infos= infos + 'LeakyRelu')

    exit()

model_makers['all'] = lambda : run_all(output = args.output) #this add to the model parser the option to run everything
################################################################################
# The main body of the run is below

if model_makers.get(args.model) is None:
    print("Invalid model")
    exit(1)

#first is the case where we run all models
if args.model == 'all':
    model_makers[args.model]()

#here we test only one model
n_trials = 15 if (args.model.find('siamese') < 0 and args.model.find('resnet') < 0) else 10
model_maker = model_makers.get(args.model)
infos = model_infos.get(args.model, '')
activation_fc = activation_fcs.get(args.activation_fc)

test(model_maker, activation_fc= activation_fc, n_trials =n_trials, lr= model_lrs[args.model],
                infos= infos + args.activation_fc, output_file= args.output)
if args.model.find('siamese') >= 0:
    test(model_maker, n_trials = n_trials, lr = model_lrs[args.model],
            infos= infos, auxiliary= True, output_file= args.output)
