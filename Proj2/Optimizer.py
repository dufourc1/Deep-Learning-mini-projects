import torch


class Optimizer(object):

    def __init__(self, parameters ,lr):
        self.lr = lr
        self.parameters = parameters

    def step(self,*input):
        raise NotImplementedError

    def param(self):
        return self.lr,self.parameters


class SGD(Optimizer):

    def step(self):
        for param in self.parameters:
            param.value.sub_(self.lr*param.grad)
