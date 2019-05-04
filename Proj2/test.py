'''
Test the function during the process of writing the module
'''
import torch
import torch.nn as nn

import Functionnals
import Param
import Criterion
from optim import SGD
from Linear import Linear
from Sequential import Sequential

torch.set_default_dtype(torch.float64)

which_test = "sequential"




if which_test == "sequential":

    fc1 = Linear(256,124)
    fc2 = Linear(124,45)
    fc3 = Linear(45,2)
    test = Sequential(fc1,fc2,fc3)

    class test_sequential(nn.Module):
        def __init__(self):
            super(test_sequential, self).__init__()
            self.fc1 = nn.Linear(256, 124)
            self.fc2 = nn.Linear(124,45)
            self.fc3 = nn.Linear(45,2)
            self.fc1.weight = torch.nn.parameter.Parameter(fc1.weights.value)
            self.fc1.bias = torch.nn.parameter.Parameter(fc1.bias.value)
            self.fc2.weight = torch.nn.parameter.Parameter(fc2.weights.value)
            self.fc2.bias = torch.nn.parameter.Parameter(fc2.bias.value)
            self.fc3.weight = torch.nn.parameter.Parameter(fc3.weights.value)
            self.fc3.bias = torch.nn.parameter.Parameter(fc3.bias.value)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x


    model = test_sequential()

    x = torch.empty(256).fill_(12.)

    criterion_test = Criterion.MSE()
    criterion_ref = nn.MSELoss()

    target = torch.tensor([1.,2.])
    y_test = test.forward(x)
    y_ref = model(x)

    print("test for forward")
    print(y_ref)
    print(y_test)
    print(torch.max(y_ref-y_test))

    loss_ref = criterion_ref(y_ref, target)
    model.zero_grad()
    loss_ref.backward()

    loss_test = criterion_test.forward(y_test,target)
    test.zero_grad()
    test.backward(criterion_test.backward(y_test,target))

    print("differences between the loss")
    print(torch.max(loss_test-loss_ref))
    print("\n")

    print("difference between the gradient")
    print(model.fc1.weight.grad.shape)
    print(fc1.weights.grad.shape)
    print(torch.max(model.fc1.weight.grad- fc1.weights.grad))

if which_test == "gradient linear":
    '''
    diff of 1e-14 --> don't know where it is from
    '''
    criterion_test = Criterion.MSE()
    criterion_ref = nn.MSELoss()

    layer_test = Linear(10,10)
    layer_test.weights.value = torch.empty(10,10).normal_(10)
    layer_test.bias.value = torch.empty(10).fill_(12.)

    class test_linear_layer(nn.Module):
        def __init__(self):
            super(test_linear_layer, self).__init__()
            self.fc1 = nn.Linear(10, 10)
            self.fc1.weight = torch.nn.parameter.Parameter(layer_test.weights.value)
            self.fc1.bias = torch.nn.parameter.Parameter(layer_test.bias.value)

        def forward(self, x):
            x = self.fc1(x)
            return x

    layer_ref = test_linear_layer()


    print("test for identical parameters")
    print(torch.max(layer_ref.fc1.weight-layer_test.weights.value))
    print(torch.max(layer_ref.fc1.bias-layer_test.bias.value))
    print("\n")

    x_test = torch.empty(10).fill_(10)
    target = torch.ones(10)
    y_ref = layer_ref(x_test)
    y_test = layer_test.forward(x_test)
    print(y_ref)
    print(y_test)

    print("test for forward pass")
    print(torch.max(y_ref-y_test))
    print("\n")

    loss_ref = criterion_ref(y_ref, target)
    layer_ref.zero_grad()
    loss_ref.backward()

    loss_test = criterion_test.forward(y_test,target)
    layer_test.zero_grad()
    layer_test.backward(criterion_test.backward(y_test,target))

    print("differences between the loss")
    print(torch.max(loss_test-loss_ref))
    print("\n")


    print("max difference between the two gradient")
    diff = torch.max(layer_ref.fc1.weight.grad-layer_test.weights.grad)
    if diff >0:
        print("difference of {}".format(diff))
        index = torch.argmax(layer_ref.fc1.weight.grad-layer_test.weights.grad)
        print(index)
        print(layer_ref.fc1.weight.grad[int(index/10),index-int(index/10)*10].item())
        print(layer_test.weights.grad[int(index/10),index-int(index/10)*10].item())
        print(layer_test.weights.grad)
        print(layer_ref.fc1.weight.grad)
        print(layer_test.weights.grad.type())
        print(layer_ref.fc1.weight.grad.type())
    else:
        print("identical")
    print("\n")

    print("max difference between the two biases gradient")
    print(torch.max(layer_ref.fc1.bias.grad-layer_test.bias.grad))
