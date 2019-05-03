import torch
from Module import Module

''''
Need for an abstract class criterion ? seems to be a double of Module then
'''

class MSE(Module):

    def forward(self,input, target):
        return torch.sum(torch.pow(input-target,2))/input.shape[0]

    def backward(self,input, target):
        '''
        return the derivative of the MSE loss with respect to the target
        '''
        return (target-input)*2/input.shape[0]

class CrossEntropy(Module):

    def forward(self, input, target):
        raise NotImplementedError

    def backward(self, input, target):
        raise NotImplementedError



#Not sure about this one, just in case of, should be moved
class Precision(Module):
    '''
    to compute precision in percentage: #right/#total * 100
    '''
    def forward(self, input, target):
        error = 0
        for tried,true in zip(input,target):
            if tried != true: error+=1

        return (1-error/input.shape[0])*100

    def backward(self, input, target):
        raise NotImplementedError
