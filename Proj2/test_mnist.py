import torch
import dlc_practical_prologue as prologue
import math
import sys


from torch import optim
from torch import Tensor
from torch.autograd import Variable
from torch import nn



from Sequential import Sequential
from Linear import Linear
from Functionnals import Relu
import Optimizer
import Criterion

torch.set_default_dtype(torch.float32)


################################################################################
#load the data
train_input, train_target, test_input, test_target = \
    prologue.load_data(one_hot_labels = False, normalize = True, flatten = False)

#reshape them to pass them to a linear neural network
train_input = train_input.view(train_input.shape[0],28*28).type(torch.float32)
test_input = train_input.view(test_input.shape[0],28*28).type(torch.float32)

#normalize the input
mean, std = train_input.mean(), train_input.std()

train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)


################################################################################
mini_batch_size = 100

def update_progress(progress,message=""):
    # update_progress() : Displays or updates a console progress bar
    ## Accepts a float between 0 and 1. Any int will be converted to a float.
    ## A value under 0 represents a 'halt'.
    ## A value at 1 or bigger represents 100%
    barLength = 20 # Modify this to change the length of the progress bar
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
    block = int(round(barLength*progress))
    text = "\rLearning : [{0}] {1}% {2} {3}".format( "="*block + " "*(barLength-block), round(progress*100,2),
                                                    status,message)
    sys.stdout.write(text)
    sys.stdout.flush()
################################################################################

def train_model_test(model,train_input,train_target, nb_epochs = 20):
    criterion = Criterion.CrossEntropy()
    optimizer = Optimizer.SGD(model.param(), lr = 1e-1)
    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            inter = criterion.backward()
            model.backward(inter)
            optimizer.step()
        update_progress((e+1.)/nb_epochs)

def train_model(model, train_input, train_target,nb_epochs = 20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 1e-1)
    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()
        update_progress((e+1.)/nb_epochs)




def compute_nb_errors(model, data_input, data_target):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

################################################################################


################################################################################
print("-------------------------------- performance evaluation --------------------")
errors_train = []
errors_test = []
errors_test_ref = []
for nb_epochs in  [200]:# [20,40,80,120,150,200]:
    print("")
    print("                                   with {} epochs \n".format(nb_epochs))
    for k in range(10):
        model = Sequential(Linear(28*28,20),Relu(),Linear(20,10))
        model_2 = Sequential(Linear(28*28,20),Relu(),Linear(20,10))
        #std = 1e-0
        #for p in model.param(): p.value.normal_(0, std)

        train_model_test(model, train_input, train_target,nb_epochs = nb_epochs)
        errors_train.append(compute_nb_errors(model.forward, train_input, train_target)/ test_input.size(0) * 100)
        errors_test.append(compute_nb_errors(model.forward, test_input, test_target) / test_input.size(0) * 100)
        errors_test_ref.append(compute_nb_errors(model_2.forward, test_input, test_target) / test_input.size(0) * 100)

    print("")
    print('train_error {:.02f}%  std {:.02f} \ntest_error {:.02f}%  std {:.02f}'.format(
    torch.tensor(errors_train).mean(),torch.tensor(errors_train).std(),
    torch.tensor(errors_test).mean(),torch.tensor(errors_test).std(),
    )
    )
    print("error ref {:.02f}% std {:.02f}".format(
            torch.tensor(errors_test_ref).mean(),torch.tensor(errors_test_ref).std()))

print("-------------------------------- performance reference --------------------")
errors_train = []
errors_test = []
errors_test_ref = []
for nb_epochs in  [200]:#[20,40,80,120,150,200]:
    print("")
    print("                                   with {} epochs \n".format(nb_epochs))
    for k in range(10):
        model = nn.Sequential(nn.Linear(28*28,20),nn.ReLU(),nn.Linear(20,10))
        model_2 = nn.Sequential(nn.Linear(28*28,20),nn.ReLU(),nn.Linear(20,10))
        std = 1e-0
        for p in model.parameters(): p.data.normal_(0, std)
        for p in model_2.parameters(): p.data.normal_(0, std)

        train_model(model, train_input, train_target,nb_epochs = nb_epochs)
        errors_train.append(compute_nb_errors(model, train_input, train_target)/ test_input.size(0) * 100)
        errors_test.append(compute_nb_errors(model, test_input, test_target) / test_input.size(0) * 100)
        errors_test_ref.append(compute_nb_errors(model_2, test_input, test_target) / test_input.size(0) * 100)

    print("")
    print('train_error {:.02f}%  std {:.02f} \ntest_error {:.02f}%  std {:.02f}'.format(
    torch.tensor(errors_train).mean(),torch.tensor(errors_train).std(),
    torch.tensor(errors_test).mean(),torch.tensor(errors_test).std(),
    )
    )
    print("error ref {:.02f}% std {:.02f}".format(
            torch.tensor(errors_test_ref).mean(),torch.tensor(errors_test_ref).std()))
