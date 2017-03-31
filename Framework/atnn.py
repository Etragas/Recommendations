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
from sklearn.decomposition import NMF
full_data  = DataLoader().LoadData(file_path="../Data/ml-100k/u.data",data_type=DataLoader.MOVIELENS)

#Reduce the matrix to toy size
pre_train, pre_test = full_data[:200,:100], full_data[:400,:100]
print pre_train
#[Model Parameters
train_indx = [(np.array(range(200))),(np.array(range(100)))]
test_idndx = [np.array(range(1000,1040)),np.array(range(1000,1040))]

user_idx, movie_idx = get_canonical_indices(full_data,[utils.num_user_latents,utils.num_movie_latents])

train = fill_in_gaps([user_idx, movie_idx], train_indx, full_data)
test = fill_in_gaps([user_idx, movie_idx], test_idndx, full_data)
print "user idx ", user_idx
print "movie idx", movie_idx
np.savetxt('train_special.out', train.astype(int), fmt = '%i',delimiter=',')
#Training Parameters
num_epochs = 300
num_iters = 5
step_size = 0.01
print train[50:,:50]

#model = NMF_ATNN(n_components=80,data=train)
parameters = build_params(train.shape)
#X = train[:utils.num_user_latents,:utils.num_movie_latents]
#model = NMF(n_components=40,init='random',random_state = 0)
#parameters[keys_row_latents]= model.fit_transform(X)
#parameters[keys_col_latents] = model.components_
#Reduce the matrix to toy size
np.savetxt('train_normal', train.astype(int),fmt = '%i', delimiter=',')
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

