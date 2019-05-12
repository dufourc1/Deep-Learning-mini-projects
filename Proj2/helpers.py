'''
Various small helpers function for printing, plotting,...
'''

import torch
import math
import sys

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
    if long:
        target = centered.pow(2).sum(1).sub_(1/(2*math.pi)).sign().add(1).div(2).long()
    else:
        target = centered.pow(2).sum(1).sub_(1/(2*math.pi)).sign().add(1).div(2)
    if one_hot_labels:
        target = convert_to_one_hot_labels(input,target)
    return input,target

################################################################################

def compute_nb_errors(model, data_input, data_target):

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
    return nb_errors

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
        loss_torch = torch.tensor(loss)
        mean = loss_torch.mean(0)

        plt.plot(mean.numpy(), c = color, label = name)

        for i in range(len(loss)):
            plt.plot(loss[i],  alpha = 0.1 ,c = color)

    plt.xlabel("epoch")
    plt.legend()
    plt.title(title)
    if save:
        plt.savefig("results/"+title, bbox_inches = "tight")
    if show:
        plt.show()


def print_(loss,error, title = ""):
    print("")
    loss = torch.tensor(loss)
    error = torch.tensor(error)

    print(title + " loss test {:4.4}, std {:4.4}".format(loss.mean(),loss.std()))
    print(title + " error_rate test {:4.4}, std {:4.4}".format(error.mean(),error.std()))
    print("")
