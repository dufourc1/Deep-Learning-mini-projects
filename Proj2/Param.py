import torch
from Module import Module

class Parameters(Module):

    def __init__(self,value):
        super(Parameters,self).__init__()
        self.value = value

        # initialization at 0 may raise issues later ?
        self.grad = 0

    def __str__(self):
        return "parameter of size {}".format(self.value.shape)