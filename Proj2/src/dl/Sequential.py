import torch
from Module import Module
from Linear import Linear


class Sequential(Module):
    """ Allows easy implementation of simple neural network

    Parameters
    ----------
    *args : list of Module
        list of Modules that will compose the sequential network, see examples

    Examples
    -------
    >>> model = Sequential(Linear(2,25),Relu(),Linear(25,25), Relu(),Linear(25,2))

    Attributes
    ----------
    layers : Module
        Etiher Linear layer of activation functions (as Functionnals)

    """

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
        """ backward operation for sequential model

        Parameters
        ----------
        next_derivative : tensor
            derivative of the loss with respect to the output of the sequential model

        """
        inter = next_derivative.clone()
        #print("BEGIN ")
        for layer in self.layers[::-1]:
            #print(inter)
            #print(layer)
            double_inter = layer.backward(inter)
            #print(double_inter)
            inter = double_inter
        #print("END")

    def zero_grad(self):
        '''
        Set all the gradients of the parameters to 0 in each layer of the sequential model
        '''
        for layer in self.layers:
            layer.zero_grad()

    def param(self):
        '''
        return a list of all the parameters in the sequential model beginning by the first one in forward order
        '''
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
            result += "\n    "
            result += layer.__str__()
        result += "\n)"
        return result
