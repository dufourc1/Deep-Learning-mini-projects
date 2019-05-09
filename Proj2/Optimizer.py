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
            old = param.value.clone()
            param.value = param.value - self.lr*param.grad
            # if torch.max(old-param.value).item() <1e-14:
            #     print("ISSUE IN GRADIENT UPDATE: {} and norm of grad {}"
            #                 .format(torch.max(old-param.value),torch.torch.max(param.grad)))
            # else:
            #     print("update fine")
