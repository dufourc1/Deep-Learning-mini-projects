import torch
import math
import sys

#importing our framework
sys.path.append('dl/')
from Sequential import Sequential
from Linear import Linear
from Functionnals import Relu,Tanh
import Optimizer
import Criterion
import helpers

#import the nn module
from torch import nn
import torch.functional as F
from torch import optim


#setting the type of tensor
torch.set_default_dtype(torch.float32)

#def a fnunction to train pytorch models
def train_ref(model, criterion, optimizer, input, target, nb_epochs = 200, verbose = False):


    mini_batch_size = 100

    #empty recipient
    loss_evolution = []
    precision_evolution = []

    #actual training
    for e in range(nb_epochs):
        loss_e = 0.
        for b in range(0, input.size(0), mini_batch_size):
            output = model(input.narrow(0, b, mini_batch_size))
            loss = criterion(output, target.narrow(0, b, mini_batch_size))
            loss_e += loss
            model.zero_grad()
            loss.backward()
            optimizer.step()

        #record the data
        precision_evolution.append(helpers.compute_accuracy(model,
                                                input,target)/target.shape[0] * 100)
        loss_evolution.append(loss_e)

        if verbose:
            message = "epoch {:3}, loss {:10.4}".format(e,loss_e)
            helpers.update_progress((e+1.)/nb_epochs, message= message)

    return loss_evolution, precision_evolution


#parameters for the test
rep = 20
epochs = 200
lr = 1e-1
mu = 0.2


#saving and plotting bool
plots = False
show_plots = False
save_csv = False


#create models
def model_relu():
    return  Sequential(Linear(2,25),Relu(),Linear(25,25),
                        Relu(),Linear(25,25), Relu(),Linear(25,2))
def model_tanh():
    return Sequential(Linear(2,25),Tanh(),Linear(25,25),
                        Tanh(), Linear(25,25), Tanh(),Linear(25,2))
def model_relu_nn():
    return  nn.Sequential(nn.Linear(2,25), nn.ReLU(), nn.Linear(25,25),
                        nn.ReLU(), nn.Linear(25,25), nn.ReLU(), nn.Linear(25,2))
def model_tanh_nn():
    return nn.Sequential(nn.Linear(2,25),nn.Tanh(),nn.Linear(25,25),
                        nn.Tanh(), nn.Linear(25,25), nn.Tanh(), nn.Linear(25,2))

#create optimizers
def opti(model):
    return Optimizer.SGD(model.param(),lr = lr)
def opti_mom(model):
    return Optimizer.SGD(model.param(), lr = lr, momentum = True, mu = mu)
def opti_nn(model):
    return optim.SGD(model.parameters(),lr = lr)
def opti_mom_nn(model):
    return optim.SGD(model.parameters(), lr = lr, momentum = mu)

#create criterions
CE = Criterion.CrossEntropy()
MSE = Criterion.MSE()
CE_nn = nn.CrossEntropyLoss()
MSE_nn = nn.MSELoss()



#actual comparison
helpers.test(model_relu, model_relu_nn, opti, opti_nn, CE, CE_nn, "relu CE", "relu nn CE",
            repetitions = rep, message = "Relu vs Relu nn with crossentropy",
            plots = plots, show_plots = show_plots, chrono = True,
            training2 = train_ref, title_plots = "comparison_CE_",
            save_result_csv = save_csv, filename = "../results/csv/CE_comparison.csv")

helpers.test(model_relu, model_relu_nn, opti_mom, opti_mom_nn, CE, CE_nn, "relu CE with momentum", "relu nn CE with momentum",
            repetitions = rep, message = "Relu vs Relu ref with crossentropy and momentum",
            plots = plots, show_plots = show_plots, chrono = True,
            training2 = train_ref, title_plots = "comparison_CE_MOM_",
            save_result_csv = save_csv, filename = "../results/csv/CE_comparison.csv")

helpers.test(model_relu, model_relu_nn, opti, opti_nn, MSE, MSE_nn, "relu MSE", "relu nn MSE",
            repetitions = rep, message = "Relu vs Relu nn with MSE",
            plots = plots, show_plots = show_plots, one_hot =True, chrono = True,
            training2 = train_ref, title_plots = "comparison_MSE_",
            save_result_csv = save_csv, filename = "../results/csv/MSE_comparison.csv")

helpers.test(model_relu, model_relu_nn, opti_mom, opti_mom_nn, MSE, MSE_nn, "relu MSE with momentum", "relu nn MSE with momentum",
            repetitions = rep, message = "Relu vs Relu nn with MSE with momentum",
            plots = plots, show_plots = show_plots, one_hot =True, chrono = True,
            training2 = train_ref, title_plots = "comparison_MSE_MOM_",
            save_result_csv = save_csv, filename = "../results/csv/MSE_comparison.csv")
