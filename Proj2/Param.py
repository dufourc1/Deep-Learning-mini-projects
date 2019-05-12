import torch
from Module import Module


class Parameters(Module):

    def __init__(self,value):
        super(Parameters,self).__init__()
        self.value = value
        self.grad = torch.zeros_like(self.value)

    def __str__(self):
        representation = "parameter of size  {}".format(self.value.shape)
        return representation
