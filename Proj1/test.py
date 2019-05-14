import time
import torch
from torch.nn.functional import relu

from dlc_practical_prologue import generate_pair_sets
from SiameseNet import split_channels

def nn_accuracy_score(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).sum().item() / len(y)

def score_printing(CELoss_tr, CELoss_te, Accuracy_tr, Accuracy_te, time_tr, model_name='Network', output= None):
    if output is None:
        print('\n\
    Cross Entropy Loss on TRAIN :\t{:.4}'.format(torch.tensor(CELoss_tr).mean().item()), u"\u00B1", '{:.4}'.format(torch.tensor(CELoss_tr).std().item()), '\n\
    Cross Entropy Loss on TEST :\t{:.4}'.format(torch.tensor(CELoss_te).mean().item()), u"\u00B1", '{:.4}'.format(torch.tensor(CELoss_te).std().item()), '\n\
    Accuracy score on TRAIN :\t\t{:.4}'.format(torch.tensor(Accuracy_tr).mean().item()), u"\u00B1", '{:.4}'.format(torch.tensor(Accuracy_tr).std().item()), '\n\
    Accuracy score on TEST :\t\t{:.4}'.format(torch.tensor(Accuracy_te).mean().item()), u"\u00B1", '{:.4}'.format(torch.tensor(Accuracy_te).std().item()), '\n\
    Training time (s):\t\t\t{:.4}'.format(torch.tensor(time_tr).mean().item()), u"\u00B1", '{:.4}'.format(torch.tensor(time_tr).std().item()))
        return

    with open(output, 'a') as f:
        f.write('{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n'.format(model_name, torch.tensor(CELoss_tr).mean().item(), torch.tensor(CELoss_tr).std().item(),
         torch.tensor(CELoss_te).mean().item(), torch.tensor(CELoss_te).std().item(),
         torch.tensor(Accuracy_tr).mean().item(), torch.tensor(Accuracy_tr).std().item(),
         torch.tensor(Accuracy_te).mean().item(), torch.tensor(Accuracy_te).std().item(),
         torch.tensor(time_tr).mean().item(), torch.tensor(time_tr).std().item()))
    print('\n')

def test(model_maker, activation_fc= relu, mean=True, n_trials = 5, device=None, output_file= None, lr =10e-3, nb_epochs=75, batch_size =250, infos='', auxiliary= False):

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

    for trial in range(n_trials):
        input_train, target_train, classes_train, input_test, target_test, classes_test = generate_pair_sets(1000)

        criterion = torch.nn.CrossEntropyLoss()

        model = model_maker().to(device)

        criterion = criterion.to(device)

        optimizer = torch.optim.Adam(model.parameters(),lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr= 10e-2)


        # optimizer.to(device)
        input_train, target_train = input_train.to(device), target_train.to(device)
        input_test, target_test = input_test.to(device), target_test.to(device)
        if auxiliary:
            classes_train, classes_test = classes_train.to(device), classes_test.to(device)


        # input_train, target_train = input_train[:,0,:,:].to(device), classes_train[:,0].to(device)
        # input_test, target_test = input_test[:,0,:,:].to(device), classes_test[:,0].to(device)
        start_time = time.time()

        for e in range(nb_epochs):
            for input, targets in zip(input_train.split(batch_size), target_train.split(batch_size)):
                if type(model).__name__ != 'SiameseNet' or not auxiliary:
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

        with torch.no_grad():
            elapsed_time = time.time() - start_time
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
