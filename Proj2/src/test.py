import torch
import math
import sys
import matplotlib.pyplot as plt

#importing our module
sys.path.append('dl/')
from Sequential import Sequential
from Linear import Linear
from Functionnals import Relu
import Optimizer
import Criterion
from helpers import train, generate_disc_data, compute_accuracy

#setting the type of tensor
torch.set_default_dtype(torch.float32)

#disable autograd
torch.set_grad_enabled(False)

#create model
model = Sequential(Linear(2,25),Relu(),Linear(25,25),Relu(),Linear(25,25), Relu(),Linear(25,2))

#create data_sets with one hot encoding for MSE
train_input, train_target = generate_disc_data(one_hot_labels = True)
test_input, test_target = generate_disc_data(one_hot_labels = True)

#normalize the data
mean,std = train_input.mean(), train_input.std()
train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

#define loss
criterion = Criterion.MSE()

#define optimizer
optim = Optimizer.SGD(parameters = model.param(), lr = 1e-1)

#train the model
loss, accuracy = train(model,criterion,optim,train_input,train_target, nb_epochs = 200, verbose = True)

#compute statistics on test
output = model.forward(test_input)
loss_test = criterion.forward(output,test_target)
accuracy_test = compute_accuracy(model.forward,test_input,test_target)

print("")
print("TRAIN:  accuracy {:.4}%, loss {:.4}".format(accuracy[-1],loss[-1]))
print("TEST :  accuracy {:.4}%, loss {:.4}".format( accuracy_test/test_target.shape[0]*100, loss_test))

#vizualisation
plt.subplot(121)
plt.plot(loss, label = "loss", c = "orange")
plt.legend()
plt.subplot(122)
plt.plot(accuracy, label = "accuracy", c = "blue")
plt.legend()
plt.savefig("../results/plots/test.png", bbox_inches = "tight")
plt.show()
