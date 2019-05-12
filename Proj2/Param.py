import torch
from Module import Module


class Parameters(Module):
    """Short summary.

    Parameters
    ----------
    value : tensor
        value of the parameter

    Examples
    -------
    >>> param = Parameters(torch.tensor([1.,2.]))

    Attributes
    ----------
    grad : tensor
        attribute to keep track of the gradient of the parameter
    value: tensor
        value of the parameter

    """

    def __init__(self,value):
        super(Parameters,self).__init__()
        self.value = value
        self.grad = torch.zeros_like(self.value)

    def __str__(self):
        representation = "parameter of size  {}".format(self.value.shape)
        return representation
