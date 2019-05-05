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
        diff = 0
        for param in self.parameters:
            old = param.value
            param.value.sub_(self.lr*param.grad)
        #     diff += torch.max(old-param.value)
        # if diff < torch.tensor([1e-4]):
        #     print("not moving anymore")
