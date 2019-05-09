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

################################################################################
number_tests = 1000
# have to precise the type of tensor otherwise gonna be issues
torch.set_default_dtype(torch.double)
################################################################################


'test for MSE function'

if True:
    MSE_test = Criterion.MSE()
    MSE_ref = nn.MSELoss()

    n1 = 10
    n2 = 5

    mean_error = 0

    for i in range(number_tests):
        x_test = torch.empty(n1,n2).normal_(10)
        y_test = torch.empty(n1,n2).normal_(3)

        x_ref = x_test.clone()
        y_ref = y_test.clone()
        x_ref.requires_grad = True
        y_ref.requires_grad = True

        loss_ref = MSE_test.forward(x_ref,y_ref)
        loss_test = MSE_ref(x_test,y_test)

        diff_loss = torch.max(loss_ref-loss_test)

        if diff_loss > torch.tensor([0.]):
            mean_error += diff_loss

        derivative_test = MSE_test.backward(x_ref,y_ref)
        x_ref.grad = None
        loss_ref.backward()
        derivative_ref = x_ref.grad

        diff_grad = torch.max(derivative_test-derivative_ref)

        if diff_grad > torch.tensor([0.]):
            print("diff in gradient")
            print(diff_grad)
    print("mean error for losses {:.15%}".format(mean_error/number_tests))

'evidence of computation issue below 1e-13'
################################################################################
