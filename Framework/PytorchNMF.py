from __future__ import print_function
import torch
from sympy.tensor.tensor import Tensor
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
learningRate = .0001
numUsers = 100
numItems = 100
latentFactorDimension = 60

dense = torch.randn(100,100)
dense[[0,0,1], [1,2,0]] = 0 # make sparse
indices = torch.nonzero(dense).t()
print(indices)
values = dense[indices[0], indices[1]] # modify this based on dimensionality
torch.sparse.FloatTensor(indices, values, dense.size())

U = Variable(torch.rand(numUsers,latentFactorDimension),requires_grad=True)
V = Variable(torch.rand(latentFactorDimension,numItems),requires_grad=True)
parameters = [U,V]
loss = 0
baseline = Variable(values)
entries = [list(x) for x in zip(indices[0,:],indices[1,:])]
print(indices)
for x in range(100):
    print(baseline.size())
    for entry in entries:
        x,y = entry
        xhat = U[x,:]
        yhat = V[:,y]
        pred = torch.dot(U,V)
        data = [dense[entry[0],entry[1]] for entry in entries]
        print(data)
        loss = loss+ torch.pow(data-pred[entries],2)
    loss = loss/(numUsers*numItems)
    print("Loss", loss)
    loss.backward(retain_graph=True)
    learning_rate = 1
    print("GRAD",U.grad)
    for param in parameters:

        param.data.sub_(learning_rate*param.grad.data)
        param.grad.data.zero_()
    loss = 0
    # U =
# print(U)
# print(torch.mm(U,V))

