import torch

from dlc_practical_prologue import generate_pair_sets

def nn_accuracy_score(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).sum().item() / len(y)

def test(model, mean=True, n_trials = 5, device=None):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not mean:
        n_trials = 1

    CELoss_tr = []
    CELoss_te = []
    Accuracy_tr = []
    Accuracy_te = []

    for trial in range(n_trials):
        input_train, target_train, classes_train, input_test, target_test, classes_test = generate_pair_sets(1000)

        criterion = torch.nn.CrossEntropyLoss()

        lr, nb_epochs, batch_size = 10e-3, 50, 250

        model = model.to(device)
        criterion = criterion.to(device)

        optimizer = torch.optim.Adam(model.parameters(),lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr= 10e-2)

        # optimizer.to(device)
        input_train, target_train = input_train.to(device), target_train.to(device)
        input_test, target_test = input_test.to(device), target_test.to(device)

        # input_train, target_train = input_train[:,0,:,:].to(device), classes_train[:,0].to(device)
        # input_test, target_test = input_test[:,0,:,:].to(device), classes_test[:,0].to(device)

        for e in range(nb_epochs):
            for input, targets in zip(input_train.split(batch_size), target_train.split(batch_size)):
                optimizer.zero_grad()
                output = model(input)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                print('Trial {:>2}/{} - Iteration #{:>3}/{}:\t'.format(trial+1, n_trials, e+1, nb_epochs),
                'Cross Entropy Loss on TRAIN :\t{:11.5}'.format(criterion(model(input_train),target_train).item()), end='\r')

        with torch.no_grad():
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

    with torch.no_grad():
        print('\n\
    Cross Entropy Loss on TRAIN :\t', torch.tensor(CELoss_tr).mean().item(), u"\u00B1", torch.tensor(CELoss_tr).std().item(), ' \n\
    Cross Entropy Loss on TEST :\t', torch.tensor(CELoss_te).mean().item(), u"\u00B1", torch.tensor(CELoss_te).std().item(),'\n\
    Accuracy score on TRAIN :\t', torch.tensor(Accuracy_tr).mean().item(), u"\u00B1", torch.tensor(Accuracy_tr).std().item(),'\n\
    Accuracy score on TEST :\t', torch.tensor(Accuracy_te).mean().item(), u"\u00B1", torch.tensor(Accuracy_te).std().item())
    return model
