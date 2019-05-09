import torch
from Module import Module


''''
Need for an abstract class criterion ? seems to be a double of Module then
'''

# TODO: check if all implementations are correct, in particular derivatives

class MSE(Module):

    def __init__(self):
        super(MSE,self).__init__()
        self.input = torch.empty(1)
        self.target = torch.empty(1)

    def forward(self,output, target):

        #resize to avoid automatic casting
        output = output.view(target.size())
        self.input = output
        self.target = target
        return ((output-target)**2).mean()

    def backward(self):
        '''
        return the derivative of the MSE loss with respect to the target
        '''
        derivative = 2*(self.input-self.target)/self.input.numel()
        return derivative

class CrossEntropy(Module):

    def __init__(self):
        super(CrossEntropy,self).__init__()
        self.input = torch.empty(1)
        self.target = torch.empty(1)

    def forward(self, output, target):

        #stabilization trick to avoid Nan:
        #     this is equivalent to multiply and divide by -max(output) in the ratio below
        self.input = output
        self.target = target

        #compute normalzed probabilities based on multi-class logistic classification
        proba = self.input.softmax(1)
        #proba = torch.exp(self.input[range(target.shape[0]),target])/torch.exp(self.input).sum(1)
        proba = proba[range(target.shape[0]),target]
        return -torch.log(proba).mean()

    def backward(self):

        inter = self.input.softmax(1)
        inter[range(self.target.shape[0]),self.target.long()] -= 1
        return inter/self.input.shape[0]


    def __repr__(self):
        print("Cross Entropy Loss")
