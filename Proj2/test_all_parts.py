################################################################################
import torch
from torch import nn
from torch.nn import functional as F
import math


import Functionnals
import Param
import Criterion
from Linear import Linear
from Sequential import Sequential
from Optimizer import SGD
from Functionnals import Relu
from Functionnals import Tanh

################################################################################
number_tests = 1000
# have to precise the type of tensor otherwise gonna be issues
torch.set_default_dtype(torch.double)
################################################################################

'test for MSE function'
if False:
    print('test for MSE function')
    MSE_test = Criterion.MSE()
    MSE_ref = nn.MSELoss()

    n1 = 10
    n2 = 5

    mean_error = 0
    mean_grad_diff = 0

    for i in range(number_tests):
        x_test = torch.empty(n1,n2).normal_(10)
        y_test = torch.empty(n1,n2).normal_(3)

        x_ref = x_test.clone()
        y_ref = y_test.clone()
        x_ref.requires_grad = True

        loss_ref = MSE_ref(x_ref,y_ref)
        loss_test = MSE_test.forward(x_test,y_test)

        diff_loss = torch.max(loss_ref-loss_test)

        if diff_loss > torch.tensor([0.]):
            mean_error += diff_loss

        derivative_test = MSE_test.backward(x_test,y_test)
        x_ref.grad = None
        loss_ref.backward()
        derivative_ref = x_ref.grad

        diff_grad = torch.max(derivative_test-derivative_ref)

        if diff_grad > torch.tensor([0.]):
            mean_grad_diff += diff_grad
    print("mean error for losses {:.15}".format(mean_error/number_tests))
    print("mean error for gradient {:.15}".format(mean_grad_diff/number_tests))


'evidence of computation issue below 1e-13'

################################################################################

'test for Cross entropy loss'
if False:
    print('test for Cross entropy loss')
    MSE_test = Criterion.CrossEntropy()
    MSE_ref = nn.CrossEntropyLoss()

    n1 = 10
    n2 = 5

    mean_error = 0
    mean_grad_diff = 0

    for i in range(number_tests):
        x_test = torch.empty(n1,n2).normal_(10)
        y_test = torch.LongTensor(10).random_(0, 5)

        x_ref = x_test.clone()
        y_ref = y_test.clone()
        x_ref.requires_grad = True

        loss_ref = MSE_ref(x_ref,y_ref)
        loss_test = MSE_test.forward(x_test,y_test)

        diff_loss = torch.max(loss_ref-loss_test)

        if diff_loss > torch.tensor([0.]):
            mean_error += diff_loss

        derivative_test = MSE_test.backward(x_test,y_test)
        x_ref.grad = None
        loss_ref.backward()
        derivative_ref = x_ref.grad

        diff_grad = torch.max(derivative_test-derivative_ref)

        if diff_grad > torch.tensor([0.]):
            mean_grad_diff += diff_grad
    print("mean error for losses   {:.15}".format(mean_error/number_tests))
    print("mean error for gradient {:.15}".format(mean_grad_diff/number_tests))


'evidence of difference of the order of 1e-16 and 1e-17'
################################################################################
'test for Linear'
if False:
    print('test for Linear')

    n1 = 10
    n2 = 5

    mean_error = 0
    mean_error_loss = 0
    mean_grad_w_diff = 0
    mean_grad_b_diff = 0


    criterion_test = Criterion.MSE()
    criterion_ref = nn.MSELoss()

    for i in range(number_tests):


        test = Linear(n1,n2)
        ref = nn.Linear(n1,n2)

        test.weights.value = ref.weight.data.clone()
        test.bias.value = ref.bias.data.clone()

        #building the summy datasets
        x_test = torch.empty(n1).normal_(10)
        y_test = torch.empty(n2).random_(0, 5)
        x_ref = x_test.clone()
        y_ref = y_test.clone()
        x_ref.requires_grad = True

        #comparing output
        output_ref = ref(x_ref)
        output_test = test.forward(x_test)
        mean_error += torch.max(output_test-output_ref)


        #comparing the losses again (in case of)
        loss_test = criterion_test.forward(output_test,y_test)
        loss_ref = criterion_ref(output_ref,y_ref)
        mean_error_loss += torch.max(loss_test-loss_ref)

        #comparing the gradient
        test.zero_grad()
        ref.zero_grad()
        loss_ref.backward()
        inter = criterion_test.backward(output_test,y_test)
        test.backward(inter)

        grad_w_ref = ref.weight.grad
        grad_b_ref = ref.bias.grad
        grad_w_test = test.weights.grad
        grad_b_test = test.bias.grad

        mean_grad_w_diff += torch.max(grad_w_test-grad_w_ref)
        mean_grad_b_diff += torch.max(grad_b_test - grad_b_ref)

    print("mean error for output   {}".format(mean_error/number_tests))
    print("mean error for loss     {}".format(mean_error_loss/number_tests))
    print("mean error for grad w   {}".format(mean_grad_w_diff/number_tests))
    print("mean error for grad b   {}".format(mean_grad_b_diff/number_tests))

################################################################################
'test for sequential'
if True:
    print('test for Sequential')

    n1 = 10
    n2 = 5

    mean_error = 0
    mean_error_loss = 0
    mean_grad_w_diff1 = 0
    mean_grad_b_diff1 = 0
    mean_grad_w_diff2 = 0
    mean_grad_b_diff2 = 0


    criterion_test = Criterion.MSE()
    criterion_ref = nn.MSELoss()


    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.Fc1 = nn.Linear(n1, 100)
            self.Fc2 = nn.Linear(100, n2)

        def forward(self, x):
            x = torch.tanh(self.Fc1(x))
            x = self.Fc2(x)
            return x

    for i in range(number_tests):


        test = Sequential(Linear(n1,100),Tanh(),Linear(100,n2))

        ref = Model()

        layer1_test = test.layers[0]
        layer2_test = test.layers[-1]
        layer1_test.weights.value = ref.Fc1.weight.data.clone()
        layer1_test.bias.value = ref.Fc1.bias.data.clone()
        layer2_test.weights.value = ref.Fc2.weight.data.clone()
        layer2_test.bias.value = ref.Fc2.bias.data.clone()

        #building the summy datasets
        x_test = torch.empty(n1).normal_(10)
        y_test = torch.empty(n2).random_(0, 5)
        x_ref = x_test.clone()
        y_ref = y_test.clone()
        x_ref.requires_grad = True

        #comparing output
        output_ref = ref(x_ref)
        output_test = test.forward(x_test)
        mean_error += torch.max(output_test-output_ref)


        #comparing the losses again (in case of)
        loss_test = criterion_test.forward(output_test,y_test)
        loss_ref = criterion_ref(output_ref,y_ref)
        mean_error_loss += torch.max(loss_test-loss_ref)

        #comparing the gradient
        test.zero_grad()
        ref.zero_grad()
        loss_ref.backward()
        inter = criterion_test.backward()
        test.backward(inter)

        grad_w_ref_layer1 = ref.Fc1.weight.grad
        grad_b_ref_layer1 = ref.Fc1.bias.grad
        grad_w_test_layer1 = layer1_test.weights.grad
        grad_b_test_layer1 = layer1_test.bias.grad

        grad_w_ref_layer2 = ref.Fc2.weight.grad
        grad_b_ref_layer2 = ref.Fc2.bias.grad
        grad_w_test_layer2 = layer2_test.weights.grad
        grad_b_test_layer2 = layer2_test.bias.grad

        mean_grad_w_diff1 += torch.max(grad_w_test_layer1-grad_w_ref_layer1)
        mean_grad_b_diff1 += torch.max(grad_b_test_layer1 - grad_b_ref_layer1)
        mean_grad_w_diff2 += torch.max(grad_w_test_layer1-grad_w_ref_layer1)
        mean_grad_b_diff2 += torch.max(grad_b_test_layer1 - grad_b_ref_layer1)


    print("mean error for output   {}".format(mean_error/number_tests))
    print("mean error for loss     {}".format(mean_error_loss/number_tests))
    print("mean error for grad w 1 {}".format(mean_grad_w_diff1/number_tests))
    print("mean error for grad b 1 {}".format(mean_grad_b_diff1/number_tests))
    print("mean error for grad w 2 {}".format(mean_grad_w_diff2/number_tests))
    print("mean error for grad b 2 {}".format(mean_grad_b_diff2/number_tests))
