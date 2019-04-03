import torch
from torch.autograd import Variable
from torch import nn
from torch import optim

from torch.nn import functional as F

import dlc_practical_prologue as prologue


class model_1(nn.Module):
    def __init__(self):
        super(model_1, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64, 40)
        self.fc2 = nn.Linear(40, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 64)))
        x = self.fc2(x)
        return x

    def predict(self,x):
        return torch.argmax(self(x),dim = 1)


def comppute_nb_errors(predicted,true):
    nb_errors = 0
    for true,tried in zip(train_target,predicted):
        if true.item() != tried.item():
            nb_errors += 1

    return nb_errors

def train_model(model, train_input, train_target,test_input,test_target):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 1e-1)
    nb_epochs = 20
    mini_batch_size = int(train_input.size(0)/200)

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch {}, precision on test set {}".format(e, comppute_nb_errors(model.predict(test_input),test_target)/test_target.size(0)))

if __name__ == "__main__":

    #loading the data
    train_input, train_target, train_classes, test_input, test_target, test_classes= prologue.generate_pair_sets(1000)
    net = model_1()
    out = net(train_input)
    predicted = torch.argmax(out, dim = 1)

    train_model(net,train_input,train_target,test_input,test_target)
