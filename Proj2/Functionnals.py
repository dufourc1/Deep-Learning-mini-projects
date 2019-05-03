import torch

from Module import Module

class Relu(Module):

    def forward(self,x):
        #apply Relu
        return torch.max(x, torch.tensor([0.]))

    def backward(self,x):
        #apply derivative of relu
        inter = torch.max(x, torch.tensor([0.]))
        inter[inter <= 0.] = 0.
        inter[inter > 0.] = 1.
        return inter

class Tanh(Module):

    def forward(self,x):
        #apply tanh
        return x.tanh()

    def backward(self,x):
        #apply derivative of tanh
        return 4*(x.exp() + x.mul(-1).exp()).pow(-2)
