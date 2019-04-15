import torch
# import numpy as np
# import pandas as pd

from dlc_practical_prologue import generate_pair_sets
from ResNet import ResNet

input_train, target_train, classes_train, input_test, target_test, classes_test = generate_pair_sets(1000)

device = 'cuda'

criterion = torch.nn.CrossEntropyLoss()
model = ResNet(nb_channels=10, kernel_size=1, nb_blocks=20)

lr, nb_epochs, batch_size = 1, 200, 50

optimizer = torch.optim.Adam(model.parameters())

model.to(device)
criterion.to(device)
# optimizer.to(device)
input_train, target_train = input_train.to(device), target_train.to(device)
input_test, target_test = input_test.to(device), target_test.to(device)

for e in range(nb_epochs):
    for input, targets in zip(input_train.split(batch_size), target_train.split(batch_size)):
        output = model(input)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        print('Iteration #{:>3}:\t'.format(e),
'Cross Entropy Loss on TRAIN :\t', criterion(model(input_train),target_train).item(), end='\r')

with torch.no_grad():
    print('\n Cross Entropy Loss on TRAIN :\t', criterion(model(input_train),target_train).item(),'\n\
Cross Entropy Loss on TEST :\t', criterion(model(input_test),target_test).item())
