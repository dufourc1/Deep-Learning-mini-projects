'''
Various small helpers function for printing, plotting,...
'''

import torch
import math
import sys
import time

import matplotlib.pyplot as plt


################################################################################
#data helpers
################################################################################

def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0),target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0).type(torch.float32)
    return tmp

def generate_disc_data(n = 1000, one_hot_labels = False, long = True):
    #generate points uniformly with label 0 if it falls outside the disk of radius 1/ 2π and 1 inside
    input = torch.empty(n,2).uniform_(0,1)
    centered = input- torch.empty(n,2).fill_(0.5)
    target = centered.pow(2).sum(1).sub_(1/(2*math.pi)).sign().add(1).div(2).long()
    target = (-1.*(target-1.)).long()
    if one_hot_labels:
        target = convert_to_one_hot_labels(input,target)
    return input,target

def create_results_csv():
    with open("../results/csv/CE.csv", 'w') as f:
        f.write(','.join(('ModelName', 'meanCELoss_tr', 'stdCELoss_tr',
                'meanCELoss_te', 'stdCELoss_te',
                'meanAccuracy_tr', 'stdAccuracy_tr',
                'meanAccuracy_te', 'stdAccuracy_tr')) + '\n')

    with open("../results/csv/MSE.csv", 'w') as f:
        f.write(','.join(('ModelName', 'meanMSELoss_tr', 'stdMSELoss_tr',
                'meanMSELoss_te', 'stdMSELoss_te',
                'meanAccuracy_tr', 'stdAccuracy_tr',
                'meanAccuracy_te', 'stdAccuracy_tr')) + '\n')

################################################################################
# model relative functions
################################################################################

def train(model, criterion, optimizer, input, target, nb_epochs = 200, verbose = False):
    #train a model form our framework
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
            loss_e += loss.item()
            model.zero_grad()
            inter = criterion.backward()
            model.backward(inter)
            optimizer.step()

        #record the data
        precision_evolution.append(compute_accuracy(model.forward,
                                                input,target)/target.shape[0] * 100)
        loss_evolution.append(loss_e)

        if verbose:
            message = "epoch {:3}, loss {:10.4}".format(e,loss_e)
            update_progress(e/nb_epochs, message= message)

    return loss_evolution, precision_evolution

def compute_accuracy(model, data_input, data_target):

    #compute accuracy for classification task

    mini_batch_size = 100
    nb_errors = 0

    #transformation needed if the encoding is one hot
    if len(data_target.shape) > 1:
        _,target = torch.max(data_target.data, 1)
    else: target =data_target.data

    #compute the number of errors
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if target[b + k] != predicted_classes[k]:
                nb_errors += 1
    return target.shape[0]-nb_errors

################################################################################
# actual test function
################################################################################

def test(Model1, Model2, Optimizer1, Optimizer2, loss1, loss2, name1, name2, repetitions = 10,
        epochs = 200, message = "", plots = False, save_result_csv = False, one_hot = False,chrono = False,
        filename = None, show_plots = False, training1 = train, training2 = train, title_plots = ""):

    """
        run repetitions times the training of tow models for 'epochs' epochs
        and return the loss and the error rate on the training set
        and also estimate for the test loss and test error rate.

    Parameters
    ----------
    Model1 : function
        when call return a Sequential model

    Model2 : function
        when call return a Sequential model
    Optimizer1 : function
        when call with opti = Optimizer1(model.param()), return an optimizer on
        this model
    Optimizer2 : function
        same as Optimizer1
    loss1 : Criterion
        Criterion (loss) for model1
    loss2 : Criterion
        criterion (loss) for model2
    name1 : string
        name of the first model
    name2 : string
        name of the second model
    repetitions : int
        number of repetitions to estimate mean and std (the default is 10).
    epochs : int
        number of epochs to train the models (the default is 200).
    message : string
        to print while training (the default is "").
    plots : bool
        if True, show plots of training and save them (the default is False).
    save_result_csv : bool
        if True save statistics of results to a csv file (the default is False).
    one_hot : bool
        encoding of the data generated (the default is False).
    chrono: bool
        print the mean time to train (the default is False).
    filename : string
        name of the csv file to save the results (the default is None).
    show_plots : bool
        trick to allow multiple test withouth interuption by matplotlib (the default is False).
    training: function
        function to train the models, if different ones are needed (for ex when comparing nn
        and this framework)
    title_plots: string
        to save and retrieve the plots


    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>> see run.py

    """


    if filename is None and save_result_csv:
        print("missing filename to save data, abort")
        sys.exit(1)

    losses_model1 = []
    losses_model2 = []

    accuracy_tr1 = []
    accuracy_tr2 = []

    loss_test_model1 = []
    loss_test_model2 = []

    accuracy_te1 = []
    accuracy_te2 = []


    print("--------------------------------Comparison for {} runs of ".format(repetitions) + message)
    update_progress((0.)/repetitions)

    if chrono:
        times1 = []
        times2 = []
    for k in range(repetitions):

        #create a dataset
        train_input, train_target = generate_disc_data(one_hot_labels = one_hot)
        test_input, test_target = generate_disc_data(one_hot_labels = one_hot)

        #normalize the data
        mean,std = train_input.mean(), train_input.std()
        train_input.sub_(mean).div_(std)
        test_input.sub_(mean).div_(std)

        #create new models
        model1 = Model1()
        model2 = Model2()

        #create new optimizers
        optimizer1 = Optimizer1(model1)
        optimizer2 = Optimizer2(model2)


        #training and recording of data
        if chrono:  start = time.time()
        loss_model_1, accuracy_model1 = training1(model1, loss1, optimizer1,
                            train_input, train_target, nb_epochs = epochs)
        if chrono: times1.append(time.time()-start)

        if chrono: start = time.time()
        loss_model_2, accuracy_model2 = training2(model2, loss2, optimizer2,
                            train_input, train_target, nb_epochs = epochs)
        if chrono: times2.append(time.time()-start)

        #save the data at each epochs
        losses_model1.append(loss_model_1)
        losses_model2.append(loss_model_2)
        accuracy_tr1.append(accuracy_model1)
        accuracy_tr2.append(accuracy_model2)

        #model1 performance on test
        loss_test_model1.append(loss1.forward(model1.forward(test_input),test_target))
        accuracy_te1.append(
            compute_accuracy(model1.forward, test_input, test_target)/test_target.shape[0] * 100)

        #model2 performance on test
        loss_test_model2.append(loss1.forward(model2.forward(test_input),test_target))
        accuracy_te2.append(
            compute_accuracy(model2.forward, test_input, test_target)/test_target.shape[0] * 100)


        update_progress((k+1.)/repetitions)

    #terminal printing of results
    print_(loss_test_model1, accuracy_tr1, name1)
    print_(loss_test_model2, accuracy_tr2, name2)

    if chrono:
        print(name1," mean time to train on {} epochs: {:4.4}s".format(epochs, torch.tensor(times1).mean()))
        print(name2," mean time to train on {} epochs: {:4.4}s".format(epochs, torch.tensor(times2).mean()))

    losses_model1_t = torch.tensor(losses_model1)
    losses_model2_t = torch.tensor(losses_model2)
    accuracy_tr1_t = torch.tensor(accuracy_tr1)
    accuracy_tr2_t = torch.tensor(accuracy_tr2)

    #save the results in csv
    if save_result_csv:
        save_results(filename, name1, losses_model1_t[:,-1], loss_test_model1,
                    accuracy_tr1_t[:,-1], accuracy_te1)
        save_results(filename, name2, losses_model2_t[:,-1], loss_test_model2,
                    accuracy_tr2_t[:,-1], accuracy_te2)

    #plot training and save the plots
    if plots:
        plot_loss([losses_model1, losses_model2],
                    [name1 +", {} epochs".format(epochs), name2 + ", {} epochs".format(epochs)],
                    ["red", "blue"],
                    title = title_plots+ "loss",
                    show = show_plots, save = True)

        plot_loss([accuracy_tr1, accuracy_tr2],
                    [name1 +", {} epochs".format(epochs), name2 + ", {} epochs".format(epochs)],
                    ["red", "blue"],
                    title = title_plots +"accuracy",
                    show = show_plots, save = True)

################################################################################
# visualization helpers
################################################################################

def update_progress(progress,message=""):
    #function to see the training progression
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(20*progress))
    text = "\rLearning : [{0}] {1}% {2} {3}".format( "="*block + " "*(20-block), round(progress*100,2),
                                                    status,message)
    sys.stdout.write(text)
    sys.stdout.flush()

def plot_loss(losses,names, colors, title = "", show = True,save = False, smaller = False):
    #plot for multiple training curves for multiples different models
    plt.figure()
    for loss,name, color in zip(losses,names, colors):
        mean = torch.tensor(loss).mean(0)

        plt.plot(mean.numpy(), c = color, label = name)

        for i in range(len(loss)):
            plt.plot(loss[i],  alpha = 0.1 ,c = color)

    plt.xlabel("epoch")
    plt.legend()
    plt.title(title)
    if save:
        plt.savefig("../results/plots/"+title, bbox_inches = "tight")
    if show:
        plt.show()

def print_(loss,error, title = ""):
    loss = torch.tensor(loss)
    error = torch.tensor(error)
    print(title + " loss test {:4.4}, std {:4.4}".format(loss.mean(),loss.std()))
    print(title + " accuracy test {:4.4}, std {:4.4}".format(error.mean(),error.std()))

def save_results(filename, model_name, Loss_tr, Loss_te, Error_rate_tr, Error_rate_te):
    # warning training results are supposed to be tensor and test results list
    #(due to the fact we don't use numpy)
    with open(filename, 'a') as f:
        f.write('{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n'.format(
         model_name,
         Loss_tr.mean().item(), Loss_tr.std().item(),
         torch.tensor(Loss_te).mean().item(), torch.tensor(Loss_te).std().item(),
         Error_rate_tr.mean().item(), Error_rate_tr.std().item(),
         torch.tensor(Error_rate_te).mean().item(), torch.tensor(Error_rate_te).std().item()))
