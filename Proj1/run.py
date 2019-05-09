import torch
from dlc_practical_prologue import generate_pair_sets
from ResNet import ResNet
from CNN import SimpleConv, SimpleConv2
from FullyConnected import Net2


def test(input_train, target_train, classes_train, input_test, target_test, classes_test,\
 model, epochs = 150, batch_size = 250, device = 'cpu'):
    lr = 10e-3
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr)

    model = model.to(device)
    criterion = criterion.to(device)
    input_train, target_train = input_train.to(device), target_train.to(device)
    input_test, target_test = input_test.to(device), target_test.to(device)

    for e in range(nb_epochs):
        for input, targets in zip(input_train.split(batch_size), target_train.split(batch_size)):
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            the_loss = criterion(model(input_train),target_train).item()
            print('  Iteration {:>3}:\t'.format(e+1), 'Loss on TRAIN: {:0.7f}\t'.format(the_loss), end='\r')

    with torch.no_grad():
        train_loss = criterion(model(input_train),target_train).item()
        test_loss  = criterion(model(input_test),target_test).item()
        train_accuracy = (model(input_train).argmax(dim=1) == target_train).sum().item() / len(target_train)
        test_accuracy  = (model(input_test).argmax(dim=1) == target_test).sum().item() / len(target_test)
        print('\n  Loss on TRAIN :\t', train_loss,
              '\n  Loss on TEST : \t', test_loss,
              '\n  Accuracy on TRAIN :\t', train_accuracy,
              '\n  Accuracy on TEST :\t', test_accuracy)




##### RUN #####
input_train, target_train, classes_train, input_test, target_test, classes_test = generate_pair_sets(1000)
nb_epochs, batch_size = 25, 250
device = 'cpu'

print('Model 1 : Convolution and max-pool layer')
model = SimpleConv()
test(input_train, target_train, classes_train,\
        input_test, target_test, classes_test, model, epochs = nb_epochs, batch_size = batch_size, device = device)


print('Model 2 : Convolution and max-pool layer (other formulation)')
model = SimpleConv2()
test(input_train, target_train, classes_train,\
        input_test, target_test, classes_test, model, epochs = nb_epochs, batch_size = batch_size, device = device)



nb_epochs = 50
print('Model 3 : Fully Connected layers (no dropout)')
model = Net2(nodes_in=2*14**2, nodes_hidden=1000, nodes_out=2, n_hidden=2)
test(input_train, target_train, classes_train,\
        input_test, target_test, classes_test, model, epochs = nb_epochs, batch_size = batch_size, device = device)

dropout = 0.25
print('Model 4 : Fully Connected layers with dropout ({:0.2f})'.format(dropout))
model = Net2(nodes_in=2*14**2, nodes_hidden=1000, nodes_out=2, n_hidden=2, drop = dropout)
test(input_train, target_train, classes_train,\
        input_test, target_test, classes_test, model, epochs = nb_epochs, batch_size = batch_size, device = device)


nb_epochs = 20
# bon résultat avec 20 itérations (augmenter le nombre d'iération n'améliore pas sensiblement "Accuracy on TEST")
print('Model 5 : Fully Connected layers with dropout ({:0.2f}) and batch normalization'.format(dropout))
model = Net2(nodes_in=2*14**2, nodes_hidden=1000, nodes_out=2, n_hidden=2, drop = dropout, with_batchnorm = True)
test(input_train, target_train, classes_train,\
        input_test, target_test, classes_test, model, epochs = nb_epochs, batch_size = batch_size, device = device)



