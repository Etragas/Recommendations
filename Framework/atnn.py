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
import train
from train import *
import NMF_ATNN
from sklearn.decomposition import NMF
full_data  = DataLoader().LoadData(file_path="../Data/ml-100k/u.data",data_type=DataLoader.MOVIELENS)
full_data = full_data[:1000,:1700]
#Reduce the matrix to toy size
pre_train, pre_test = full_data[:200,:100], full_data[:400,:100]
print pre_train

#[Model Parameters
train_user_size = 200
train_movie_size = 100

#Training Parameters
#num_epochs = 20
step_size = 0.005
num_iters = 20

parameters = build_params([train_user_size + num_user_latents, train_movie_size + num_movie_latents])

parameters = train(full_data, [train_user_size, train_movie_size], parameters, p1=True, p2=True, trainArgs = [step_size, num_iters])

#Print performance on each model - MOVED INTO TRAIN.PY
'''
invtrans = getInferredMatrix(parameters,train)
print "\n".join([str(x) for x in ["Train", print_perf(parameters,data=train), train, np.round(invtrans)]])

invtrans = getInferredMatrix(parameters,test)
print "\n".join([str(x) for x in ["Test", print_perf(parameters,data=test), test, np.round(invtrans)]])
'''
