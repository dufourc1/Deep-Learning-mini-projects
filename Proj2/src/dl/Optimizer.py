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
    """implementation of the stochastic gradient descent with the option to add momentum

    Parameters
    ----------
    parameters : list of Parameters
        the parameters of the model we want to update using this optimizer
    lr : float
        learning rate
    momentum : bool
        to add the momentum option (the default is False).
    mu : float
        momentum coefficient (the default is 0).

    Examples
    -------
    >>> model = Linear(100,20)
    >>> opti = SGD(model.param(), lr = 1e-1, momentum = True, mu = 0.2)
    >>> ... #do backward pass with model
    >>> opti.step()

    Attributes
    ----------
    states : list of tensor
        momentum to propagate from each update
    lr
    parameters
    momentum
    mu

    """


    def __init__(self, parameters ,lr, momentum = False, mu = 0):
        super(SGD,self).__init__(parameters,lr)
        self.lr = lr
        self.parameters = parameters
        self.momentum = momentum
        if self.momentum:
            self.mu = mu
            self.states = []
            for p in parameters:
                self.states.append(torch.zeros_like(p.value))


    def step(self):
        #perform the update step
        if self.momentum:
            # implement the formula p.value = p.value - lr*(mu*state + p.grad)
            # as it is in pytorch
            for param,state in zip(self.parameters,self.states):
                state.mul_(self.mu).add_(param.grad)
                param.value.sub_(self.lr,state)

        else:
            #perform the usual update p.value -= lr*p.grad
            for p in self.parameters:
                p.value.sub_(self.lr,p.grad)
