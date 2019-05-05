import torch
from Module import Module


''''
Need for an abstract class criterion ? seems to be a double of Module then
'''

# TODO: check if all implementations are correct, in particular derivatives

class MSE(Module):

    def forward(self,input, target):
        input = input.view(target.size())
        return ((input-target)**2).mean()

    def backward(self,input, target):
        '''
        return the derivative of the MSE loss with respect to the target

        '''
        input = input.view(target.shape)
        derivative = (input-target)*2/input.shape[0]
        if len(derivative.shape) == 1:
            return derivative
        else:
            return derivative.t()

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
