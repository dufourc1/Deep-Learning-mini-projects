import torch
from Module import Module
from Linear import Linear


class Sequential(Module):

    def __init__(self,*args):
        super(Sequential,self).__init__()
        self.layers = args

    def param(self):
        return self.layers

    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, next_derivative):
        #warning ! see scope of modifications and use copy if necessary
        inter = next_derivative
        for layer in self.layers[::-1]:
            print("inter in backward")
            print(inter)
            inter = layer.backward(inter)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def param(self):
        params = []
        for layer in self.layers:
            inter = layer.param()
            if len(inter)>0:
                for elt in inter:
                    params.append(elt)
        return params

    def __str__(self):
        result = "Sequential("
        for layer in self.layers:
            result += "\n "
            result += layer.__str__()
        result += "\n)"
        return result
