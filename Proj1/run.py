from sys import exit
import argparse

import torch

from FullyConnected import FullyConnected

################################################################################
parser = argparse.ArgumentParser(description='Reproduction of our results for Project 1')

parser.add_argument('model', type=str, default='NoModel',
                    help = 'The model to be tested.')

args = parser.parse_args()

################################################################################

def no_model():
    print("A model has to be chosen")
    exit(1)

def make_FullyConnected():
    return FullyConnected(nodes_in=2*14**2, nodes_hidden=1000, nodes_out=2, n_hidden=2)

model_makers = {'NoModel': no_model,
            'test': make_FullyConnected,
            'fcnn': make_FullyConnected}

if model_makers.get(args.model) is None:
    print("Invalid model")
    exit(1)

model_maker = model_makers.get(args.model)
model = model_maker()


from test import test

test(model)
