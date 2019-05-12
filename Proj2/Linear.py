import torch
from Module import Module
from Param import Parameters


import sys

class Linear(Module):


    def __init__(self, input_size, output_size):
        super(Linear,self).__init__()
        type = torch.float32
        self.weights = Parameters(torch.empty(output_size,input_size,dtype= type).normal_(0,1e-0))
        self.bias = Parameters(torch.zeros(output_size,dtype=type))
        self.result = Parameters(torch.empty(output_size,dtype=type))
        self.input = Parameters(torch.empty(input_size,dtype=type))

    def forward(self,x):
        """perform the forward pass for a fully connected linear layer of sizes input_size -> output_size

        Parameters
        ----------
        x : input tensor of size [number_of_training_points,input_size]
            training data points in minibatch

        Returns
        -------
        torch float64 tensor
            result of the forward pass of dimension [number_of_training_points,output_size]
        """


        #we check if we have a single datapoint entry, in which case we do not have to worry about the dimensions
        if len(x.shape)>1:
            self.input.value = x
            #expand the bias vector so it matches the size of the mini batches
            B = self.bias.value.repeat(1,self.input.value.t().shape[1]).view(-1,self.bias.value.shape[0]).t()
            #do the actual forward pass
            self.result.value = x.matmul(self.weights.value.t()) + self.bias.value
            #self.result.value = self.result.value.t()


        else:
            self.input.value = x
            self.result.value = torch.mv(self.weights.value,self.input.value) + self.bias.value

        return self.result.value



    def backward(self, derivative):
        """ backward operation for fully connected layer

        Parameters
        ----------
        derivative : tensor
            supposing the layer returns x_(i+1) and takes as input x_i, derivative
            is dloss/dx_(i+1)

        Returns
        -------
        tensor
            return dloss/dx_i so that the next layer can perform backward propagation

        """

        #add a new dimension if necessary so that the derivative is a matrix not a vector
        if len(derivative.shape) == 1:
            if self.weights.value.shape[0] == 1:
                derivative = derivative.view(derivative.shape[0],1)
            else:
                derivative = derivative.view(1,derivative.shape[0])
        if len(self.input.value.shape) == 1:
            x_i = self.input.value.view(1,self.input.value.shape[0])
        else:
            x_i = self.input.value

        #debugging
        # print("backward for {}".format(self))
        # print("derivative received is {}".format(next_derivative.shape))
        # print("input was of size {}".format(self.input.value.shape))

        # derivative with respect to the weights: dloss/dw
        self.weights.grad += torch.mm(derivative.t(),x_i)

        #derivative with respect to the bias: dloss/dbias
        self.bias.grad += derivative.sum(0)


        #derivative if the loss with respect to the input of the layer:  dloss/dx_i pass it to the next layer
        next_derivative = torch.mm(derivative,self.weights.value)

        # if (derivative != derivative).sum().item() > 0:
        #     print("derivative passed to {}, issue nan detected".format(self))
        #     print(self.input.value)
        #     print(self.weights.value)
        #     print(self.bias.value)
        #     print(derivative)
        #     sys.exit()
        #
        # if (next_derivative != next_derivative).sum().item() > 0:
        #     print("derivative output of {}".format(self))
        #     print(self.input.value)
        #     print(self.weights.value)
        #     print(self.bias.value)
        #     print(next_derivative)
        #     sys.exit()
        # return dloss/dx_i so that it can be passed to the next layer
        return next_derivative


    def zero_grad(self):
        '''
        Set all the parameters' gradient to 0
        '''
        self.weights.grad.zero_()
        self.bias.grad.zero_()


    def param(self):
        return self.weights, self.bias

    def __str__(self):
        return "Linear(in_features= {}, out_features= {})".format(self.weights.value.shape[1],
                                                                self.weights.value.shape[0])
