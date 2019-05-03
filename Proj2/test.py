'''
Test the function during the process of writing the module
'''
import torch
import torch.nn as nn

import Functionnals
import Param
import Criterion
from optim import SGD

x = Param.Parameters(torch.tensor([12.,4.]))
y = Param.Parameters(torch.tensor([-3.,4.]))

test = SGD(2,3)
print(test.parameters,test.lr)
print(test.param())
