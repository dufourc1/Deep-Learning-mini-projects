import torch
from Module import Module
from Param import Parameters


class Linear(Module):


    def __init__(self, input_size, output_size):
        super(Linear,self).__init__()
        self.weights = Parameters(torch.empty(output_size,input_size,dtype=torch.double).normal_())
        self.bias = Parameters(torch.empty(output_size,dtype=torch.double).normal_())
        self.result = Parameters(torch.empty(output_size,dtype=torch.double))
        self.input = Parameters(torch.empty(input_size,dtype=torch.double))

    def forward(self,x):
        self.input.value = x
        self.result.value = self.weights.value@x + self.bias.value
        return self.result.value


    #if problems in backward it will probably comes from this
    '''
    backward propagation is wrong, do it again:
        the returned derivative is useless 
    '''
    def backward(self, next_derivative):
        self.weights.grad = torch.mm(next_derivative.view(next_derivative.shape[0],1),
                            self.input.value.view(1,self.input.value.shape[0]))

        self.bias.grad = next_derivative
        return self.bias.grad

    def zero_grad(self):
        self.weights.grad = torch.zeros(self.weights.value.shape)
        self.bias.grad = torch.zeros(self.bias.value.shape)

    #not sure about including the self.result at the end
    def param(self):
        return self.weights, self.bias, self.result

    def __str__(self):
        return "Linear(in_features= {}, out_features= {})".format(self.input.value.shape[0],
                                                                self.result.value.shape[0])
