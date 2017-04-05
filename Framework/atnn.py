#Main method Rfor running models
import numpy as np
from scipy.sparse import *
from sklearn.decomposition import NMF
import os
from DataLoader import *
import NMF_ATNN
from NMF_ATNN import *
import utils
from utils import *
import train
from train import *

#Load the data using DataLoader
full_data = DataLoader().LoadData(file_path="../Data/ml-100k/u.data",data_type=DataLoader.MOVIELENS)
#Our dataset only has 1000 users and 1700 movies
full_data = full_data[:1000,:1700]

#Reduce the matrix to toy size
pre_train, pre_test = full_data[:200,:100], full_data[:400,:100]
#print pre_train

#Model Parameters
train_user_size = 200
train_movie_size = 100

#Training Parameters
step_size = 0.005
num_iters = 20
hyper = [step_size, num_iters]

#Build the dictionary of parameters for the nets, etc.
parameters = build_params([train_user_size + num_user_latents, train_movie_size + num_movie_latents])

#Train the parameters.  Pretraining the nets and canon latents are optional.
parameters = train(full_data, [train_user_size, train_movie_size], parameters, 
                                  p1=True, p1Args = hyper, p2=True, p2Args = hyper, trainArgs = hyper)
