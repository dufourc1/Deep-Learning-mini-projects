import torch
# import numpy as np
# import pandas as pd

from dlc_practical_prologue import generate_pair_sets
from ResNet import ResNet
from FullyConnected import Net2

input_train, target_train, classes_train, input_test, target_test, classes_test = generate_pair_sets(1000)


device = 'cuda'

criterion = torch.nn.CrossEntropyLoss()
model = ResNet(nb_channels=50, kernel_size=5, nb_blocks=7)
# model = Net2(nodes_in=2*14**2, nodes_hidden=1000, nodes_out=2, n_hidden=2)

lr, nb_epochs, batch_size = 10e-3, 150, 250

model = model.to(device)
criterion = criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr)

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
        print('Iteration #{:>3}:\t'.format(e),
'Cross Entropy Loss on TRAIN :\t', criterion(model(input_train),target_train).item(), end='\r')

def nn_accuracy_score(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).sum().item() / len(y)

with torch.no_grad():
    print('\n Cross Entropy Loss on TRAIN :\t', criterion(model(input_train),target_train).item(),'\n\
Cross Entropy Loss on TEST :\t', criterion(model(input_test),target_test).item(),'\n\
Accuracy score on TRAIN :\t', nn_accuracy_score(model, input_train, target_train),'\n\
Accuracy score on TEST :\t', nn_accuracy_score(model, input_test, target_test))
