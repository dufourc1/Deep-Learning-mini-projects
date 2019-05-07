import torch
from Module import Module
from Param import Parameters


# TODO: self.result is not usefull and should be discarded


class Linear(Module):


    def __init__(self, input_size, output_size):
        super(Linear,self).__init__()
        self.weights = Parameters(torch.empty(output_size,input_size,dtype=torch.double).normal_())
        self.bias = Parameters(torch.empty(output_size,dtype=torch.double).normal_())
        self.result = Parameters(torch.empty(output_size,dtype=torch.double))
        self.input = Parameters(torch.empty(input_size,dtype=torch.double))

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
            self.input.value = x.t()

            #expand the bias vector so it matches the size of the mini batches
            B = self.bias.value.repeat(1,self.input.value.shape[1]).view(-1,self.bias.value.shape[0]).t()
            #do the actual forward pass
            self.result.value = torch.mm(self.weights.value,self.input.value) + B
            self.result.value = self.result.value.t()


        else:
            self.input.value = x
            self.result.value = torch.mv(self.weights.value,self.input.value) + self.bias.value

        return self.result.value



    def backward(self, next_derivative):
        """ backward operation for fully connected layer

        Parameters
        ----------
        next_derivative : tensor
            supposing the layer returns x_(i+1) and takes as input x_i, next_derivative
            is dloss/dx_(i+1)

        Returns
        -------
        tensor
            return dloss/dx_i so that the next layer can perform backward propagation

        """
        print(next_derivative.shape)

        if len(next_derivative.shape) == 1:
            #add a new dimension so that the derivative is a matrix not a vector
            next_derivative = next_derivative.view(next_derivative.shape[0],1)
        if len(self.input.value.shape) == 1:
            #add a new dimension so that the derivative is a matrix not a vector
            x_i = self.input.value.view(self.input.value.shape[0],1)
        else:
            x_i = self.input.value

        print(x_i.shape)
        print(next_derivative.shape)
        # derivative with respect to the weights: dloss/dw
        self.weights.grad += torch.mm(next_derivative,x_i.t())
        #derivative with respect to the bias: dloss/dbias
        self.bias.grad += torch.sum(next_derivative, dim=1)
        #derivative if the loss with respect to the input of the layer:  dloss/dx_i pass it to the next layer
        print(self)
        print(self.weights)
        print(next_derivative.shape)
        new_next_derivative = torch.mm(self.weights.value.t(),next_derivative)

        # return dloss/dx_i so that it can be passed to the next layer
        return new_next_derivative




        # for i in range(self.input.value.shape[1]):
        #
        #     #pick input x_i and dloss/dx_i
        #     inter_input = self.input.value[:,i]
        #     inter_derivative = next_derivative[:,i]
        #
        #     #debugging print
        #     # print("debuging backward linear")
        #     # print("derivative recieved as input shape {}".format(next_derivative.shape))
        #     # print("inter input shape: {}".format(inter_input.shape))
        #     # print("inter_derivative shape: {}".format(inter_derivative.shape))
        #     # print("{} weights grad shape: {}".format(self.__str__(),self.weights.grad.shape))
        #     # print("end debug backward linear")
        #
        #     #apply course formula to get the dloss/dw and dloss/dbias
        #     self.weights.grad += torch.mm(inter_derivative.view(inter_derivative.shape[0],1),
        #                                 inter_input.view(1,inter_input.shape[0]))
        #     self.bias.grad += inter_derivative.view(-1)
        #
        #     #create dloss/dx_i in order to pass it to the next layer
        #     #resize to avoid later issues of compatibility
        #     if len(inter_derivative.shape) == 1:
        #         new_next_derivative[:,i] = torch.mv(self.weights.value.t(), inter_derivative)
        #     else :
        #         new_next_derivative[:,i] = torch.mm(self.weights.value.t(), inter_derivative)
        #
        # return new_next_derivative

    def zero_grad(self):
        '''
        Set all the parameters' gradient to 0
        '''
        self.weights.grad = torch.zeros(self.weights.value.shape)
        self.bias.grad = torch.zeros(self.bias.value.shape)


    def param(self):
        return self.weights, self.bias

    def __str__(self):
        return "Linear(in_features= {}, out_features= {})".format(self.weights.value.shape[1],
                                                                self.weights.value.shape[0])
