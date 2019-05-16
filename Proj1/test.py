import time
import torch
from torch.nn.functional import relu

from dlc_practical_prologue import generate_pair_sets
from SiameseNet import split_channels

def nn_accuracy_score(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).sum().item() / len(y)

def score_printing(CELoss_tr, CELoss_te, Accuracy_tr, Accuracy_te, time_tr, model_name='Network', output= None):
    """Printing funtion for the test results.

    Parameters
    ----------
    CELoss_tr : iterable
        List of cross entropy scores on train set.
    CELoss_te : iterable
        List of cross entropy scores on test set.
    Accuracy_tr : iterable
        List of accuracy scores on train set.
    Accuracy_te : iterable
        List of accuracy scores on test set.
    time_tr : iterable
        List of training times in seconds.
    model_name : str
        Name of the network that gave the results above (the default is 'Network').
    output : str
        Name of the file where to save the results. If empty it prints on screen (the default is None).
    """
    if output is None: #print onscreen
        print('\n\
    Cross Entropy Loss on TRAIN :\t{:.4}'.format(torch.tensor(CELoss_tr).mean().item()), u"\u00B1", '{:.4}'.format(torch.tensor(CELoss_tr).std().item()), '\n\
    Cross Entropy Loss on TEST :\t{:.4}'.format(torch.tensor(CELoss_te).mean().item()), u"\u00B1", '{:.4}'.format(torch.tensor(CELoss_te).std().item()), '\n\
    Accuracy score on TRAIN :\t\t{:.4}'.format(torch.tensor(Accuracy_tr).mean().item()), u"\u00B1", '{:.4}'.format(torch.tensor(Accuracy_tr).std().item()), '\n\
    Accuracy score on TEST :\t\t{:.4}'.format(torch.tensor(Accuracy_te).mean().item()), u"\u00B1", '{:.4}'.format(torch.tensor(Accuracy_te).std().item()), '\n\
    Training time (s):\t\t\t{:.4}'.format(torch.tensor(time_tr).mean().item()), u"\u00B1", '{:.4}'.format(torch.tensor(time_tr).std().item()))
        return

    with open(output, 'a') as f: #print on file
        f.write('{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n'.format(model_name, torch.tensor(CELoss_tr).mean().item(), torch.tensor(CELoss_tr).std().item(),
         torch.tensor(CELoss_te).mean().item(), torch.tensor(CELoss_te).std().item(),
         torch.tensor(Accuracy_tr).mean().item(), torch.tensor(Accuracy_tr).std().item(),
         torch.tensor(Accuracy_te).mean().item(), torch.tensor(Accuracy_te).std().item(),
         torch.tensor(time_tr).mean().item(), torch.tensor(time_tr).std().item()))
    print('\n')

def test(model_maker, activation_fc= relu, mean=True, n_trials = 5, device=None, output_file= None, lr =10e-3, nb_epochs=50, batch_size =250, infos='', auxiliary= False):
    """Function that test multiple times a given network on MNIST and save the results.

    Parameters
    ----------
    model_maker : function
        A function that returns a torch.nn.Module. This function should be able to accept arguments, even if not used.
    activation_fc : torch.nn.functional
        Activation function to be passed in model_maker (the default is relu).
    mean : bool
        If to compute the mean over multiple trials (the default is True).
    n_trials : int
        Number of times the model is to be tested, ignored if mean is False (the default is 5).
    device : torch.device
        The device to be used, by default is chosen according to computer resources (the default is None).
    output_file : str
        Name of the file where to save the results. If empty it prints on screen (the default is None).
    lr : double
        Learning rate (the default is 10e-3).
    nb_epochs : int
        Number of epochs (the default is 50).
    batch_size : int
        Batch size (the default is 100).
    infos : str
        Additional model infromations to be printed (the default is '').
    auxiliary : bool
        Wether to use auxiliary loss, it works only for siames net (the default is False).
    """

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not mean:
        n_trials = 1

    CELoss_tr = []
    CELoss_te = []
    Accuracy_tr = []
    Accuracy_te = []
    time_tr = []

    model = model_maker(activation_fc)
    model_name = type(model).__name__ + infos

    if type(model).__name__ == 'SiameseNet' and auxiliary:
        model_name += 'Auxiliary'

    print('Training {}:'.format(model_name))

################################################################################
# The model testing follows

    for trial in range(n_trials):
        input_train, target_train, classes_train, input_test, target_test, classes_test = generate_pair_sets(1000)

        criterion = torch.nn.CrossEntropyLoss()

        model = model_maker().to(device)

        criterion = criterion.to(device)

        optimizer = torch.optim.Adam(model.parameters(),lr)

        input_train, target_train = input_train.to(device), target_train.to(device)
        input_test, target_test = input_test.to(device), target_test.to(device)
        if auxiliary:
            classes_train, classes_test = classes_train.to(device), classes_test.to(device)

        ########################################################################
        #A model is trained for each trial

        start_time = time.time()

        for e in range(nb_epochs):
            for input, targets in zip(input_train.split(batch_size), target_train.split(batch_size)):

                if type(model).__name__ != 'SiameseNet' or not auxiliary:
                    #the standart training
                    optimizer.zero_grad()
                    output = model(input)
                    loss = criterion(output, targets)
                    loss.backward()
                    optimizer.step()
                else:
                    #first pass
                    #separating the data so that we train on the two images separatly, and learn to classify them properly
                    input_train1,classes_train1,input_train2,classes_train2 = split_channels(input_train, classes_train)

                    #use the branch to perform handwritten digits classification
                    out1 = model.branch(input_train1)
                    out2 = model.branch(input_train2)

                    #auxiliary loss: learn to detect the handwritten digits directly
                    loss_aux = criterion(out1,classes_train1) + criterion(out2,classes_train2)

                    #optimize based on this
                    model.zero_grad()
                    loss_aux.backward(retain_graph=True)
                    optimizer.step()

                    #second pass
                    #loss and optimization of the whole model
                    response = model.forward(input_train)
                    loss = criterion(response,target_train)
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                print('{:3}%| Trial {:>2}/{} - Iteration #{:>3}/{}:\t'.format(int((trial*nb_epochs + e+1)/n_trials/nb_epochs*100), trial+1, n_trials, e+1, nb_epochs),
                'Cross Entropy Loss on TRAIN :\t{:11.5}'.format(criterion(model(input_train),target_train).item()), end='\r')

        ########################################################################
        # model evaluation

        with torch.no_grad():
            elapsed_time = time.time() - start_time

            model.train(False)
            
            this_CELoss_tr = criterion(model(input_train),target_train).item()
            this_CELoss_te = criterion(model(input_test),target_test).item()
            this_Accuracy_tr = nn_accuracy_score(model, input_train, target_train)
            this_Accuracy_te =  nn_accuracy_score(model, input_test, target_test)
            if not mean:
                print('Cross Entropy Loss on TRAIN :\t', this_CELoss_tr,'\n\
                Cross Entropy Loss on TEST :\t', this_CELoss_te,'\n\
                Accuracy score on TRAIN :\t', this_Accuracy_tr,'\n\
                Accuracy score on TEST :\t', this_Accuracy_te)

            CELoss_tr.append(this_CELoss_tr)
            CELoss_te.append(this_CELoss_te)
            Accuracy_tr.append(this_Accuracy_tr)
            Accuracy_te.append(this_Accuracy_te)
            time_tr.append(elapsed_time)

    with torch.no_grad():
        score_printing(CELoss_tr, CELoss_te, Accuracy_tr, Accuracy_te, time_tr, model_name = model_name, output = output_file)
    return
