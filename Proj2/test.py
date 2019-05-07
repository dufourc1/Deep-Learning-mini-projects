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

torch.set_default_dtype(torch.double)

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

which_test = "sequential"


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




    #creation of model from pytorch module, with same initialization
    class test_sequential(nn.Module):
        def __init__(self):
            super(test_sequential, self).__init__()
            self.Fc1 = nn.Linear(2, 124)
            self.Fc2 = nn.Linear(124,45)
            self.Fc3 = nn.Linear(45,1)

        def forward(self, x):
            x = F.relu(self.Fc1(x))
            x = F.relu(self.Fc2(x))
            x = self.Fc3(x)
            return x


    model = test_sequential()


    #creation of model from our module to test with the same initialization as the reference model
    fc1 = Linear(2,124)
    fc2 = Linear(124,45)
    fc3 = Linear(45,1)
    fc1.weights.value = model.Fc1.weight.data.clone()
    fc2.weights.value = model.Fc2.weight.data.clone()
    fc3.weights.value = model.Fc3.weight.data.clone()
    fc1.bias.value = model.Fc1.bias.data.clone()
    fc2.bias.value = model.Fc2.bias.data.clone()
    fc3.bias.value = model.Fc3.bias.data.clone()
    sigma1 = Functionnals.Relu()
    sigma2 = Functionnals.Relu()
    test = Sequential(fc1,sigma1,fc2,sigma2,fc3)

    print(model)
    print(test)



    #creation of dummy data_set and target
    X,target = generate_disc_set(1000)

    #normalization
    mean, std = X.mean(), X.std()
    X.sub_(mean).div_(std)



    #creation of criterion for the two models
    criterion_test = Criterion.MSE()
    criterion_ref = nn.MSELoss()

    #creation of the optimizers for the two models
    lr = 0.1
    optim_test = SGD(test.param(), lr = lr)
    optim_ref = torch.optim.SGD(model.parameters(), lr = lr)


    #actual training in parallel with comparison of data along the way to monitor the differences
    nb_epochs = 100
    mini_batch_size = 100


    #plots_evolution
    plot_ref = []
    plot_test = []


    #check if initialization is the same for both models
    print("-----------------------------------difference between the initial values")
    print("weights")
    print(torch.max(model.Fc1.weight- fc1.weights.value))
    print(torch.max(model.Fc2.weight- fc2.weights.value))
    print(torch.max(model.Fc3.weight- fc3.weights.value))
    print("biases")
    print(torch.max(model.Fc1.bias- fc1.bias.value))
    print(torch.max(model.Fc2.bias- fc2.bias.value))
    print(torch.max(model.Fc3.bias- fc3.bias.value))

    for e in range(nb_epochs):
        loss_epoch_test = 0
        loss_epoch_ref = 0
        print("-----------------------------------------------------------------  epoch {}".format(e))
        for b in range(0, X.size(0), mini_batch_size):

            input_narrow = X.narrow(0, b, mini_batch_size)

            #my framework
            output_test = test.forward(input_narrow)
            loss_test = criterion_test.forward(output_test, target.narrow(0, b, mini_batch_size))
            plot_test.append(loss_test)
            loss_epoch_test += loss_test
            test.zero_grad()
            inter_test = criterion_test.backward(output_test,target.narrow(0, b, mini_batch_size))
            test.backward(inter_test)
            optim_test.step()


            #the reference framework
            output_ref = model(input_narrow)
            loss_ref = criterion_ref(output_ref, target.narrow(0, b, mini_batch_size))
            plot_ref.append(loss_ref)
            loss_epoch_ref += loss_ref
            model.zero_grad()
            loss_ref.backward()
            optim_ref.step()






        print("")
        print("----------------------------------- global evolution")
        print("My framework loss at epoch {:4}: {:10.6}".format(e,loss_epoch_test))
        print("The ref      loss at epoch {:4}: {:10.6}".format(e,loss_epoch_ref))
        print("----------------------------------- comparison of output")
        print(torch.max(test.forward(X[0:200,:])-model(X[0:200,:])))

        print("-----------------------------------differences between the loss")
        print(torch.max(loss_epoch_test-loss_epoch_ref))
        print("-----------------------------------difference between the gradients")
        print("weights")
        print(torch.max(model.Fc1.weight.grad- fc1.weights.grad))
        print(torch.max(model.Fc2.weight.grad- fc2.weights.grad))
        print(torch.max(model.Fc3.weight.grad- fc3.weights.grad))
        print("biases")
        print(torch.max(model.Fc1.bias.grad- fc1.bias.grad))
        print(torch.max(model.Fc2.bias.grad- fc2.bias.grad))
        print(torch.max(model.Fc3.bias.grad- fc3.bias.grad))


    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(plot_test, label = "test", marker = ".")
    plt.plot(plot_ref, label = "ref", marker = ".")
    plt.legend()
    plt.show()
    plt.plot(np.array(plot_test)-np.array(plot_ref), label = "diff", marker = ".")
    plt.plot(np.array(plot_test)/np.array(plot_ref), label = "ratio", marker = ".")
    plt.legend()
    plt.show()

###############################################################################################
if which_test == "gradient linear":

    in_ = 6
    out = 9

    criterion_test = Criterion.MSE()
    criterion_ref = nn.MSELoss()



    class test_linear_layer(nn.Module):
        def __init__(self):
            super(test_linear_layer, self).__init__()
            self.fc1 = nn.Linear(in_, out)

        def forward(self, x):
            x = self.fc1(x)
            return x

        def affiche_gradient(self):
            for p in self.parameters():
                print(p.grad)

    layer_ref = test_linear_layer()

    layer_test = Linear(in_,out)
    layer_test.weights.value = layer_ref.fc1.weight.data.clone()
    layer_test.bias.value = layer_ref.fc1.bias.data.clone()

    x_test = torch.empty(1000,in_).normal_(10)
    target = torch.ones(1000,out)
    y_ref = layer_ref(x_test)
    y_test = layer_test.forward(x_test)


    loss_ref = criterion_ref(y_ref, target)
    layer_ref.zero_grad()
    loss_ref.backward()




    loss_test = criterion_test.forward(y_test,target)
    layer_test.zero_grad()
    loss_test_grad = criterion_test.backward(y_test,target)
    layer_test.backward(loss_test_grad)


    print("differences between the loss")
    print(torch.max(loss_test-loss_ref))
    print("")


    print("max difference between the two gradient")
    diff = torch.max(layer_ref.fc1.weight.grad-layer_test.weights.grad)
    if diff >0:
        print("max difference of {}".format(diff))
        index = torch.argmax(layer_ref.fc1.weight.grad-layer_test.weights.grad)
        print("located at {}".format(index))
        print("actual gradient for the weights")
        print(layer_ref.fc1.weight.grad)
        print(layer_test.weights.grad)
        print("actual gradient for the biases")
        print(layer_ref.fc1.bias.grad)
        print(layer_test.bias.grad)

    else:
        print("identical")
    print("")

    print("max difference between the two biases gradient")
    print(torch.max(layer_ref.fc1.bias.grad-layer_test.bias.grad))

    print(in_,out,target.shape)
