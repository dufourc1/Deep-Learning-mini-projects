import torch


class Optimizer(object):

    def __init__(self, parameters ,lr):
        self.lr = lr
        self.parameters = parameters

    def step(self,*input):
        return NotImplementedError

    def param(self):
        return self.lr,self.parameters


class SGD(Optimizer):

    def step(self):
        return NotImplementedError
