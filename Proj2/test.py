'''
Test the function during the process of writing the module
'''


##########################################################################################3
import torch
from torch import nn
from torch.nn import functional as F
import math


import Functionnals
import Param
import Criterion
from Linear import Linear
from Sequential import Sequential
from Optimizer import SGD
from Functionnals import Relu

###########################################################################################

torch.set_default_dtype(torch.float64)

def train_model(model, train_input, train_target):
    criterion = Criterion.MSE()
    optimizer = SGD(model.param(), lr = 0.2)
    nb_epochs = 12
    mini_batch_size = 100
    old_loss = 0
    loss = 0
    for e in range(nb_epochs):
        old_loss = loss
        loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            loss += criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            inter = criterion.backward(output,train_target.narrow(0, b, mini_batch_size))
            model.backward(inter)
            optimizer.step()
        print("loss at epoch {:4}: {:10.6}, difference with previous {:6.4}".format(e,loss, abs(loss-old_loss)))
def train_model_per_data_point(model, train_input, train_target):
    criterion = Criterion.MSE()
    optimizer = SGD(model.param(), lr = 1e-1)
    nb_epochs = 250
    mini_batch_size = 100

    for e in range(nb_epochs):
        for b in range(train_input.size(0)):
            print(b)
            print("input shape: {}".format(train_input[b].shape))
            output = model.forward(train_input[b])
            loss = criterion.forward(output, train_target[b])
            model.zero_grad()
            inter = criterion.backward(output,train_target[b])
            model.backward(inter)
            optimizer.step()
        print("loss at epoch {}: {:4}".format(e,loss))
def generate_disc_set(nb):
    input = torch.Tensor(nb, 2).uniform_(-1, 1)
    target = input.pow(2).sum(1).sub(2 / math.pi).sign().add(1).div(2)
    return input, target

############################################################################################

# TODO: check why there are differences when evaluating the two models in subsamples of the data
#        but not in the all data set

which_test = "gradient linear"



#############################################################################################
if which_test == "training":


    train_input, train_target = generate_disc_set(1000)
    test_input, test_target = generate_disc_set(1000)

    mean, std = train_input.mean(), train_input.std()

    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)


    model = Sequential(Linear(2,200),
                      Relu(),
                      Linear(200,50),
                      Relu(),
                      Linear(50,1))



    print("\n")
    print("--------------------------------------------------------training on minibatch")
    print(model)
    print(train_input.shape)
    train_model(model,train_input,train_target)

##############################################################################################
if which_test == "sequential":


    #creation of model from our module to test
    fc1 = Linear(256,124)
    fc2 = Linear(124,45)
    fc3 = Linear(45,2)
    sigma1 = Functionnals.Relu()
    sigma2 = Functionnals.Relu()
    test = Sequential(fc1,sigma1,fc2,sigma2,fc3)

    #creation of model from pytorch module, with same initialization
    class test_sequential(nn.Module):
        def __init__(self):
            super(test_sequential, self).__init__()
            self.Fc1 = nn.Linear(256, 124)
            self.Fc2 = nn.Linear(124,45)
            self.Fc3 = nn.Linear(45,2)
            self.Fc1.weight = torch.nn.parameter.Parameter(fc1.weights.value)
            self.Fc1.bias = torch.nn.parameter.Parameter(fc1.bias.value)
            self.Fc2.weight = torch.nn.parameter.Parameter(fc2.weights.value)
            self.Fc2.bias = torch.nn.parameter.Parameter(fc2.bias.value)
            self.Fc3.weight = torch.nn.parameter.Parameter(fc3.weights.value)
            self.Fc3.bias = torch.nn.parameter.Parameter(fc3.bias.value)

        def forward(self, x):
            x = F.relu(self.Fc1(x))
            x = F.relu(self.Fc2(x))
            x = self.Fc3(x)
            return x
    model = test_sequential()

    #creation of dummy data_set and target
    X = torch.empty(1000,256).normal_(12.)
    target = torch.empty(1000,2).normal_(3)

    #creation of criterion for the two models
    criterion_test = Criterion.MSE()
    criterion_ref = nn.MSELoss()

    #creation of the optimizers for the two models
    lr = 0.1
    optim_test = SGD(test.param(), lr = lr)
    optim_ref = torch.optim.SGD(model.parameters(), lr = lr)


    #actual training in parallel with comparison of data along the way to monitor the differences
    nb_epochs = 100
    mini_batch_size = 200


    #check if initialization is the same for both models
    print("-----------------------------------difference between the initial values")
    print(torch.max(model.Fc1.weight- fc1.weights.value))
    print(torch.max(model.Fc2.weight- fc2.weights.value))
    print(torch.max(model.Fc3.weight- fc3.weights.value))
    print(torch.max(model.Fc1.bias- fc1.bias.value))
    print(torch.max(model.Fc2.bias- fc2.bias.value))
    print(torch.max(model.Fc3.bias- fc3.bias.value))

    for e in range(nb_epochs):
        loss_epoch_test = 0
        loss_epoch_ref = 0
        for b in range(0, X.size(0), mini_batch_size):

            #my framework
            input_narrow = X.narrow(0, b, mini_batch_size)
            output_test = test.forward(input_narrow)
            loss_test = criterion_test.forward(output_test, target.narrow(0, b, mini_batch_size))
            loss_epoch_test += loss_test
            test.zero_grad()
            inter_test = criterion_test.backward(output_test,target.narrow(0, b, mini_batch_size))
            test.backward(inter_test)
            optim_test.step()

            #the reference framework

            output_ref = model(input_narrow)
            if torch.max(output_ref-output_test)>torch.tensor([0.]):
                print("Warning silent failure may have occured {}".format(torch.max(output_ref-output_test)))
                #print(torch.cat([output_ref,output_test]))
                #break
            loss_ref = criterion_ref(output_ref, target.narrow(0, b, mini_batch_size))
            loss_epoch_ref += loss_ref
            model.zero_grad()
            loss_ref.backward()
            optim_ref.step()


        print("")
        print("----------------------------------- global evolution")
        print("My framework loss at epoch {:4}: {:10.6}".format(e,loss_epoch_test))
        print("The ref      loss at epoch {:4}: {:10.6}".format(e,loss_epoch_ref))
        print("----------------------------------- comparison of output")
        print(torch.max(test.forward(X)-model(X)))

        print("-----------------------------------differences between the loss")
        print(torch.max(loss_epoch_test-loss_epoch_ref))
        print("-----------------------------------difference between the gradients")
        print(torch.max(model.Fc1.weight.grad- fc1.weights.grad))
        print(torch.max(model.Fc2.weight.grad- fc2.weights.grad))
        print(torch.max(model.Fc3.weight.grad- fc3.weights.grad))
        print(torch.max(model.Fc1.bias.grad- fc1.bias.grad))
        print(torch.max(model.Fc2.bias.grad- fc2.bias.grad))
        print(torch.max(model.Fc3.bias.grad- fc3.bias.grad))

###############################################################################################
if which_test == "gradient linear":

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
