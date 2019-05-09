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
        print("loss at epoch {:4}: {:10.6}, difference with previous {:6.4}".
                format(e,loss, abs(loss-old_loss)))
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
def generate_disc_set(nb, loss = "cross"):
    input = torch.Tensor(nb, 2).uniform_(-1, 1)
    if loss == "mse":
        target = input.pow(2).sum(1).sub(2 / math.pi).sign().add(1).div(2)
    else:
        target = input.pow(2).sum(1).sub(2 / math.pi).sign().add(1).div(2).long()
    return input, target
def accuracy(model,input,target):

    #perform actual class prediction
    response = model.forward(input).argmax(1)
    error = 0

    #compute the percentage of error
    for tried,true in zip(response,target):
        if tried.item() != true.item(): error+=1

    return 1-error/response.shape[0]
def accuracy_ref(model,input,target):

    #perform actual class prediction
    response = model.forward(input).argmax(1)
    error = 0

    #compute the percentage of error
    for tried,true in zip(response,target):
        if tried.item() != true.item(): error+=1

    return 1-error/response.shape[0]

############################################################################################

# TODO: check why there are differences when evaluating the two models in subsamples of the data
#        but not in the all data set

which_test = "sequential"

x = torch.tensor([[-1.,-3.,4.],[-3.,3.,-1.]])
y = torch.tensor([0,1])

print(x.softmax(1))
print((x-torch.max(x)).softmax(1))

print("mhh")
print(x[range(y.shape[0]),y])

test = Criterion.CrossEntropy()
print(test.forward(x,y))




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

    out = 1


    #creation of model from pytorch module, with same initialization
    class test_sequential(nn.Module):
        def __init__(self):
            super(test_sequential, self).__init__()
            self.Fc1 = nn.Linear(2, 2)
            #self.Fc2 = nn.Linear(300,100)
            #self.Fc3 = nn.Linear(100,45)
            self.Fc4 = nn.Linear(2,out)

        def forward(self, x):
            x = F.relu(self.Fc1(x))
            #x = F.relu(self.Fc2(x))
            #x = F.relu(self.Fc3(x))
            return self.Fc4(x)


    model = test_sequential()


    #creation of model from our module to test with the same initialization as the reference model
    fc1 = Linear(2,2)
    #fc2 = Linear(300,100)
    #fc3 = Linear(100,45)
    fc4 = Linear(2,out)
    fc1.weights.value = model.Fc1.weight.data.clone()
    # fc2.weights.value = model.Fc2.weight.data.clone()
    # fc3.weights.value = model.Fc3.weight.data.clone()
    fc4.weights.value = model.Fc4.weight.data.clone()
    fc1.bias.value = model.Fc1.bias.data.clone()
    # fc2.bias.value = model.Fc2.bias.data.clone()
    # fc3.bias.value = model.Fc3.bias.data.clone()
    fc4.bias.value = model.Fc4.bias.data.clone()
    sigma1 = Functionnals.Relu()
    sigma2 = Functionnals.Relu()
    sigma3 = Functionnals.Relu()
    #test = Sequential(fc1,sigma1,fc2,sigma2,fc3,sigma3,fc4)
    test = Sequential(fc1,sigma1,fc4)

    print(model)
    print(test)



    #creation of dummy data_set and target
    X,target = generate_disc_set(1000,"mse")

    #normalization
    #mean, std = X.mean(), X.std()
    #X.sub_(mean).div_(std)




    #creation of criterion for the two models
    criterion_test = Criterion.MSE()
    criterion_ref = nn.MSELoss()

    #creation of the optimizers for the two models
    lr = 0.9
    optim_test = SGD(test.param(), lr = lr)
    optim_ref = torch.optim.SGD(model.parameters(), lr = lr)


    #actual training in parallel with comparison of data along the way to monitor the differences
    nb_epochs = 200
    mini_batch_size = 100



    #check if initialization is the same for both models
    # print("-----------------------------------difference between the initial values")
    # print("weights")
    # print(torch.max(model.Fc1.weight- fc1.weights.value))
    # print(torch.max(model.Fc2.weight- fc2.weights.value))
    # print(torch.max(model.Fc3.weight- fc3.weights.value))
    # print("biases")
    # print(torch.max(model.Fc1.bias- fc1.bias.value))
    # print(torch.max(model.Fc2.bias- fc2.bias.value))
    # print(torch.max(model.Fc3.bias- fc3.bias.value))

    for e in range(nb_epochs):
        loss_epoch_test = 0
        loss_epoch_ref = 0
        #print("-----------------------------------------------------------------  epoch {}".format(e))
        for b in range(0, X.size(0), mini_batch_size):

            input_narrow_test = X.narrow(0, b, mini_batch_size)
            input_narrow_ref = input_narrow_test.clone()

            #the reference framework
            output_ref = model(input_narrow_ref)
            # if e>1:
            #     print(torch.max(old_ref-output_ref).item())
            old_ref = output_ref
            loss_ref = criterion_ref(output_ref, target.narrow(0, b, mini_batch_size))
            loss_epoch_ref += loss_ref
            model.zero_grad()
            loss_ref.backward()
            optim_ref.step()


            #my framework
            output_test = test.forward(input_narrow_test)
            # if e>1:
            #     print(torch.max(output_test-old).item())
            old = output_test
            #print(torch.cat([output_ref.t(),output_test.t()]))
            loss_test = criterion_test.forward(output_test, target.narrow(0, b, mini_batch_size))
            #print(loss_test)
            loss_epoch_test += loss_test
            test.zero_grad()
            inter_test = criterion_test.backward(output_test,target.narrow(0, b, mini_batch_size))
            test.backward(inter_test)

            # for param in test.param():
            #     print(torch.max(param.grad))
            #     param.value -= lr*param.grad
            optim_test.step()

        print("epoch {:4}, loss_ref {:9.9}, loss_test {:9.9}".
                format(e,loss_epoch_ref,loss_epoch_test))

        predicted_test = accuracy(test,X,target)
        predicted_ref = accuracy_ref(model,X,target)
        print("precision ref {:.4%}, precision test {:.4%}".format(predicted_ref,predicted_test))






        # print("")
        # print("----------------------------------- global evolution")
        # print("My framework loss at epoch {:4}: {:10.6}".format(e,loss_epoch_test))
        # print("The ref      loss at epoch {:4}: {:10.6}".format(e,loss_epoch_ref))
        # print("----------------------------------- comparison of output")
        # print(torch.max(test.forward(X[0:200,:])-model(X[0:200,:])))
        #
        # print("-----------------------------------differences between the loss")
        # print(torch.max(loss_epoch_test-loss_epoch_ref))
        # print("-----------------------------------difference between the gradients")
        # print("weights")
        # print(torch.max(model.Fc1.weight.grad- fc1.weights.grad))
        # print(torch.max(model.Fc2.weight.grad- fc2.weights.grad))
        # print(torch.max(model.Fc3.weight.grad- fc3.weights.grad))
        # print("biases")
        # print(torch.max(model.Fc1.bias.grad- fc1.bias.grad))
        # print(torch.max(model.Fc2.bias.grad- fc2.bias.grad))
        # print(torch.max(model.Fc3.bias.grad- fc3.bias.grad))




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
