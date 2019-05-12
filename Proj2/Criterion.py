import torch
from Module import Module



class MSE(Module):
    """Implementation of the Mean Square Error loss

    Parameters
    ----------

    Examples
    -------
    >>> criterion = MSE()
    >>> criterion.forward(x,target) #compute the MSE between x and target
    # x and target should have same shape
    >>> criterion.backward() #compute the derivative of the loss wrt to the last input

    Attributes
    ----------
    input : tensor
        parameter reminding the last input in order to compute the gradient if needed
    target : tensor
        parameter reminding the last input in order to compute the gradient if needed
    """

    def __init__(self):
        super(MSE,self).__init__()
        self.input = torch.empty(1)
        self.target = torch.empty(1)

    def forward(self, output, target):
        #resize to avoid automatic casting
        output = output.view(target.size())
        self.input = output
        self.target = target
        return ((output-target)**2).mean()

    def backward(self):
        '''
        return the derivative of the MSE loss with respect to the target
        '''
        derivative = 2*(self.input - self.target)/self.input.numel()
        return derivative

    def __repr__(self):
        print("MSE loss")

class CrossEntropy(Module):
    """ Implementation of the Cross Entropy loss to compute loss for classification

    Parameters
    ----------

    Examples
    -------
    >>> criterion = CrossEntropy()
    >>> criterion.forward(x,target) #compute the Cross Entropy between x and target
    # x[0] should have length the number of possible value y can take
    >>> criterion.backward() #compute the derivative of the loss wrt to the last input

    Attributes
    ----------
    input : tensor
        parameter reminding the last input in order to compute the gradient if needed
    target : tensor
        parameter reminding the last input in order to compute the gradient if needed

    """

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
        proba = proba[range(target.shape[0]),target]
        return -torch.log(proba).mean()

    def backward(self):

        #compute derivative using chain rule
        inter = self.input.softmax(1)
        inter[range(self.target.shape[0]),self.target.long()] -= 1
        return inter/self.input.shape[0]


    def __repr__(self):
        print("Cross Entropy Loss")
