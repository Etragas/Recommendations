#Main method Rfor running models
import numpy as np
from scipy.sparse import *
from sklearn.decomposition import NMF
import os
from DataLoader import *
from NMF_ATNN import *
from autograd.optimizers import sgd
import utils
import utils
import NMF_ATNN
from sklearn.decomposition import NMF
import multiprocessing as mp
import os

print os.getcwd()
full_data  = DataLoader().LoadData(file_path="Data/ml-100k/u.data",data_type=DataLoader.MOVIELENS)
print full_data.shape
full_data = full_data[:100,:100]
#Reduce the matrix to toy size
full_shape = full_data.shape
utils.num_user_latents = int(.1*full_shape[0])
utils.num_movie_latents = int(.1*full_shape[1])

#[Model Parameters
splitData(full_data)
train_idx, test_idx = splitData(full_data)

print train_idx
can_usr_idx, can_mov_idx = get_canonical_indices(full_data, [utils.num_user_latents, utils.num_movie_latents])
train = fill_in_gaps([can_usr_idx, can_mov_idx], train_idx, full_data)
test = fill_in_gaps([can_usr_idx, can_mov_idx], test_idx, full_data)
print train.shape

#Training Parameters
num_epochs = 20
num_iters = 5
step_size = 0.005

parameters = build_params(train.shape)
#Pretrain rating net and latents
grads = lossGrad(train)
num_proc =  2#mp.cpu_count()
NMF_ATNN.wipe_caches()
#parameters = black_adam(grad_funs,parameters,step_size=step_size,
#                        num_iters=num_epochs, callback=dataCallback(train),num_proc=num_proc)
parameters = adam(grads,parameters, step_size=step_size,
                         num_iters=num_epochs, callback=dataCallback(train))
#Print performance on each model
invtrans = getInferredMatrix(parameters,train)
print "\n".join([str(x) for x in ["Train", print_perf(parameters,data=train), train, np.round(invtrans)]])

invtrans = getInferredMatrix(parameters,test)
print "\n".join([str(x) for x in ["Test", print_perf(parameters,data=test), test, np.round(invtrans)]])
