import torch
import math
import sys

import matplotlib.pyplot as plt

from Sequential import Sequential
from Linear import Linear
from Functionnals import Relu,Tanh
import Optimizer
import Criterion
import helpers

#setting the type of tensor
torch.set_default_dtype(torch.float32)


################################################################################

def train(model, criterion, optimizer, input, target, nb_epochs = 200, verbose = False):

    mini_batch_size = 100

    #empty recipient
    loss_evolution = []
    precision_evolution = []

    #actual training
    for e in range(nb_epochs):
        loss_e = 0.
        for b in range(0, input.size(0), mini_batch_size):
            output = model.forward(input.narrow(0, b, mini_batch_size))
            loss = criterion.forward(output, target.narrow(0, b, mini_batch_size))
            loss_e += loss
            model.zero_grad()
            inter = criterion.backward()
            model.backward(inter)
            optimizer.step()

        #record the data
        precision_evolution.append(helpers.compute_nb_errors(model.forward,
                                                input,target)/target.shape[0] * 100)
        loss_evolution.append(loss_e)

        if verbose:
            message = "epoch {:3}, loss {:10.4}".format(e,loss_e)
            helpers.update_progress((e+1.)/nb_epochs, message= message)

    return loss_evolution, precision_evolution



def test(Model1,Model2, Optimizer1, Optimizer2, loss1, loss2, repetitions = 10, epochs = 200, message = "",one_hot = False, long = True):
    '''
    run 20 times the training of each model for 200 epochs and return the loss and the error rate on the training set
    and also estimate for the test loss and test error rate
    '''

    losses_model1 = []
    losses_model2 = []
    precision1_all = []
    precision2_all = []
    loss_test_model1 = []
    loss_test_model2 = []
    error_test_model1 = []
    error_test_model2 = []

    print("begining comparison for 20 runs \n    " + message)
    helpers.update_progress((0.)/repetitions)

    for k in range(repetitions):

        #create a dataset
        train_input, train_target = helpers.generate_disc_data(one_hot_labels = one_hot, long = long)
        test_input, test_target = helpers.generate_disc_data(one_hot_labels = one_hot, long = long)
        #normalize the data
        mean,std = train_input.mean(), train_input.std()
        train_input.sub_(mean).div_(std)
        test_input.sub_(mean).div_(std)

        #create new models
        model1 = Model1()
        model2 = Model2()
        #create new optimizers
        optimizer1 = Optimizer1(model1)
        optimizer2 = Optimizer1(model2)


        #training and recording of data
        loss_model_1, precision1 = train(model1, loss1, optimizer1,
                            train_input, train_target, nb_epochs = epochs)

        loss_model_2, precision2 = train(model2, loss2, optimizer2,
                            train_input, train_target, nb_epochs = epochs)
        #to avoid nans:
        if (torch.tensor(loss_model_1) != torch.tensor(loss_model_1)).sum() == 0:
            losses_model1.append(loss_model_1)
        if (torch.tensor(loss_model_2) != torch.tensor(loss_model_2)).sum() == 0:
            losses_model2.append(loss_model_2)
        precision1_all.append(precision1)
        precision2_all.append(precision2)


        loss_test_model1.append(loss1.forward(model1.forward(test_input),test_target))
        error_test_model1.append(helpers.compute_nb_errors(model1.forward,
                                                test_input,test_target)/test_target.shape[0] * 100)
        loss_test_model2.append(loss1.forward(model2.forward(test_input),test_target))
        error_test_model2.append(helpers.compute_nb_errors(model2.forward,
                                                test_input,test_target)/test_target.shape[0] * 100)


        helpers.update_progress((k+1.)/repetitions)

    return losses_model1, losses_model2, precision1_all, precision2_all,\
                loss_test_model1, loss_test_model2, error_test_model1, error_test_model2



################################################################################

if __name__ == "__main__":


    rep = 20

    def model_relu():
        return  Sequential(Linear(2,25),Relu(),Linear(25,25),Linear(25,25),
                            Relu(),Linear(25,2))
    def model_tanh():
        return Sequential(Linear(2,25),Tanh(),Linear(25,25),Linear(25,25),
                            Tanh(),Linear(25,2))

    lr = 1e-1

    def opti(model):
        return Optimizer.SGD(model.param(),lr = lr)

    def opti_mom(model):
        return Optimizer.SGD(model.param(), lr = lr, momentum = True, mu = 0.2)

    crossentropy = Criterion.CrossEntropy()
    mse = Criterion.MSE()


    save = False


##############################################################################
#                    test Relu vs Tanh with crossentropy
##############################################################################

    loss_relu, loss_tanh, precision_relu, precision_tanh,\
    loss_test_model1, loss_test_model2, error_test_model1, error_test_model2 = \
    test(model_relu, model_tanh, opti,opti, crossentropy, crossentropy,
        repetitions = rep, message = "test Relu vs Tanh with crossentropy")


    helpers.print_(loss_test_model1, error_test_model1, " relu with Crossentropy")
    helpers.print_(loss_test_model2, error_test_model2, " tanh with CrossEntropy")


    helpers.plot_loss([loss_relu, loss_tanh],
                        ["relu with CrossEntropy, 200 epochs", "tanh with CrossEntropy, 200 epochs"],
                        ["red", "blue"],
                        title = "Loss CE",
                        show = False, save = save)

    helpers.plot_loss([precision_relu, precision_tanh],
                        ["relu with CrossEntropy, 200 epochs", "tanh with CrossEntropy, 200 epochs"],
                        ["red", "blue"],
                        title = "Error rate CE",
                        show = False, save = save)

###############################################################################
#               test Relu vs Tanh with crossentropy and momentum
###############################################################################
    loss_relu_mom, loss_tanh_mom, precision_relu_mom, precision_tanh_mom, \
    loss_test_model1, loss_test_model2, error_test_model1, error_test_model2 = \
    test(model_relu, model_tanh, opti_mom,opti_mom, crossentropy, crossentropy,
        repetitions = rep,  message = "test Relu vs Tanh with crossentropy and momentum")

    helpers.print_(loss_test_model1, error_test_model1, " relu with Crossentropy and momentum")
    helpers.print_(loss_test_model2, error_test_model2, " tanh with CrossEntropy and momentum")

    helpers.plot_loss([loss_relu_mom, loss_tanh_mom],
                        ["relu with CrossEntropy and momentum, 200 epochs",
                           "tanh with CrossEntropy and momentum, 200 epochs"],
                        ["red", "blue"],
                        title = "Loss CE and MOM",
                        show = False, save = save)

    helpers.plot_loss([precision_relu_mom, precision_tanh_mom],
                        ["relu with CrossEntropy and momentum, 200 epochs",
                           "tanh with CrossEntropy and momentum, 200 epochs"],
                        ["red", "blue"],
                        title = "Error rate Ce and MOM",
                        show = False, save = save)

##############################################################################
#                    test Relu vs Tanh with MSE
##############################################################################

    loss_relu, loss_tanh, precision_relu, precision_tanh,\
    loss_test_model1, loss_test_model2, error_test_model1, error_test_model2 = \
    test(model_relu, model_tanh, opti,opti, mse, mse,
        repetitions = rep, message = "test Relu vs Tanh with mse",one_hot = True, long = True)

    helpers.print_(loss_test_model1, error_test_model1, " relu with MSE")
    helpers.print_(loss_test_model2, error_test_model2, " tanh with MSE")



    helpers.plot_loss([loss_relu, loss_tanh],
                        ["relu with mse, 200 epochs", "tanh with mse, 200 epochs"],
                        ["red", "blue"],
                        title = "Loss MSE",
                        show = False, save = save, smaller = True)

    helpers.plot_loss([precision_relu, precision_tanh],
                        ["relu with mse, 200 epochs", "tanh with mse, 200 epochs"],
                        ["red", "blue"],
                        title = "Error rate MSE",
                        show = False, save = save)

##############################################################################
#                    test Relu vs Tanh with MSE with mom
##############################################################################


    loss_relu, loss_tanh, precision_relu, precision_tanh,\
    loss_test_model1, loss_test_model2, error_test_model1, error_test_model2 = \
    test(model_relu, model_tanh, opti_mom,opti_mom, mse, mse,
        repetitions = rep, message = "test Relu vs Tanh with mse and mom",one_hot = True, long = True)

    helpers.print_(loss_test_model1, error_test_model1, " relu with MSE and momentum")
    helpers.print_(loss_test_model2, error_test_model2, " tanh with MSE and momentum")




    helpers.plot_loss([loss_relu, loss_tanh],
                            ["relu with mse, 200 epochs", "tanh with mse, 200 epochs"],
                            ["red", "blue"],
                            title = "Loss MSE and Mom",
                            show = False, save = save, smaller = True)

    helpers.plot_loss([precision_relu, precision_tanh],
                            ["relu with mse, 200 epochs", "tanh with mse, 200 epochs"],
                            ["red", "blue"],
                            title = "Error rate MSE and Mom",
                            show = True, save = save)
