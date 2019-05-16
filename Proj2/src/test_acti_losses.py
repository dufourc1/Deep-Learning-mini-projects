import torch
import math
import sys

#importing our module
sys.path.append('dl/')
from Sequential import Sequential
from Linear import Linear
from Functionnals import Relu,Tanh
import Optimizer
import Criterion
from helpers import test

#setting the type of tensor
torch.set_default_dtype(torch.float32)

#disable autograd
torch.set_grad_enabled(False)



rep = 20
epochs = 200

lr = 1e-1
mu = 0.2

plots = False
show_plots = False
save_csv = False

#create models and optimizers
def model_relu():
    return  Sequential(Linear(2,25),Relu(),Linear(25,25),
                        Relu(),Linear(25,25), Relu(),Linear(25,2))
def model_tanh():
    return Sequential(Linear(2,25),Tanh(),Linear(25,25),
                        Tanh(), Linear(25,25), Tanh(),Linear(25,2))

def opti(model):
    return Optimizer.SGD(model.param(),lr = lr)
def opti_mom(model):
    return Optimizer.SGD(model.param(), lr = lr, momentum = True, mu = mu)

CE = Criterion.CrossEntropy()
MSE = Criterion.MSE()


##############################################################################
#                    test Relu vs Tanh with crossentropy
##############################################################################

test(model_relu, model_tanh, opti, opti, CE, CE, "relu", "tanh",
    repetitions = rep, message = "Relu vs Tanh with crossentropy",
    plots = plots, show_plots = show_plots, title_plots = "CE",
    save_result_csv = save_csv, filename = "../results/csv/CE.csv")

###############################################################################
#               test Relu vs Tanh with crossentropy and momentum
###############################################################################

test(model_relu, model_tanh, opti_mom, opti_mom, CE, CE, "relu with momentum", "tanh with momentum",
    repetitions = rep, message = "Relu vs Tanh with crossentropy and momentum",
    plots = plots, show_plots = show_plots, title_plots = "CE_MOM",
    save_result_csv = save_csv, filename = "../results/csv/CE.csv")

##############################################################################
#                    test Relu vs Tanh with MSE
##############################################################################

test(model_relu, model_tanh, opti, opti, MSE, MSE, "relu", "tanh", one_hot = True,
    repetitions = rep, message = "Relu vs Tanh with Mean Square error",
    plots = plots, show_plots = show_plots, title_plots = "MSE",
    save_result_csv = save_csv, filename = "../results/csv/MSE.csv")

##############################################################################
#                    test Relu vs Tanh with MSE with mom
##############################################################################

test(model_relu, model_tanh, opti_mom, opti_mom, MSE, MSE, "relu with momentum", "tanh with momentum",
    one_hot =True, repetitions = rep, message = "Relu vs Tanh with Mean Square error and momentum",
    plots = plots, show_plots = show_plots, title_plots = "MSE_MOM",
    save_result_csv = save_csv, filename = "../results/csv/MSE.csv")
