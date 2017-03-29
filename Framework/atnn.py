#Main method Rfor running models
import numpy as np
from scipy.sparse import *
from sklearn.decomposition import NMF
import os
from DataLoader import *
from NMF_ATNN import *
from autograd.optimizers import sgd
import utils
from utils import *
import NMF_ATNN
full_data  = DataLoader().LoadData(file_path="../Data/ml-100k/u.data",data_type=DataLoader.MOVIELENS)

#Reduce the matrix to toy size
train, test = full_data[:200,:100], full_data[:400,:100]
row_size, col_size = train.shape

#Model Parameters

#Training Parameters
num_epochs = 60
num_iters = 5
step_size = 0.01

#model = NMF_ATNN(n_components=80,data=train)
parameters = build_params(train.shape)

grads = lossGrad(train)
for iter in range(1):
    parameters = adam(grads,parameters, step_size=step_size,
                            num_iters=num_epochs, callback=dataCallback(train))
    print "Test performance:"
    print_perf(parameters,data=test)

#Print performance on each model
invtrans = getInferredMatrix(parameters,train)
print "\n".join([str(x) for x in ["Train", print_perf(parameters,data=train), train, np.round(invtrans)]])

invtrans = getInferredMatrix(parameters,test)
print "\n".join([str(x) for x in ["Test", print_perf(parameters,data=test), test, np.round(invtrans)]])

