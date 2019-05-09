import torch

from Module import Module

# TODO: check if torch.relu can be used
# TODO: check derivative of relu

class Relu(Module):

    def __init__(self):
        super(Relu,self).__init__()
        self.input = torch.empty(1)


    def forward(self,x):
        #apply Relu
        #return torch.max(x, torch.zeros_like(x))
        self.input = x
        return torch.relu(x)

    def backward(self,dx):
        #apply derivative of relu in the backward propagation scheme
        # return the pointwise product of its derivative and the derivative of the next layer

        #compute the derivative of the relu
        inter = torch.max(self.input, torch.zeros_like(self.input))
        inter[inter <= 0.] = 0.
        inter[inter > 0.] = 1.

        #return hadamard product
        return inter*dx


    def __str__(self):
        return "Relu"

class Tanh(Module):

    def __init__(self):
        super(Tanh,self).__init__()
        self.input = torch.empty(1)

    def forward(self,x):
        self.input = x
        return x.tanh()

    def backward(self,dx):
        #apply derivative of tanh
        ds =  4*(self.input.exp() + self.input.mul(-1).exp()).pow(-2)
        return ds*dx

    def __str__(self):
        return "tanh"
