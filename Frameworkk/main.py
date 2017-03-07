#Main method Rfor running models

#Start by loading the data
import numpy as np
from scipy.sparse import *
from sklearn.decomposition import NMF
import os
from DataLoader import *
from NMF import *


#Load all data
X = DataLoader().LoadData(file_path="../Data/ml-100k/u.data",data_type=DataLoader.MOVIELENS)

#Reduce the matrix to toy size
X = X[:100,:100]
print(X)
#Construct Model
model = NMF(80,data=X)

#Random batch size
batch_size = X.shape[1]

#Train the model, using batch learning

for counter in range(0,1000,1):
    #Generate indices for users
    data_indices = range(X.shape[1])
    #Shuffle the indices so we can have random batches
    np.random.shuffle(data_indices)
    #Chunks represent units of users we'll test against
    chunks = [data_indices[x:x+batch_size] for x in range(0,len(data_indices),batch_size)]

    for user_indices in chunks:
        #Create data-matrix for chosen users
        batch, user_indices, movie_indices = DataLoader().getBatch(X, user_indices, 'u')
        model.train(latent_indices=[user_indices,movie_indices],data=batch)
        #Print loss
        if (counter%100 == 0):
            print("final loss is ",model.loss(model.parameters,X))

#Do Inference
invtrans = model.inference(model.parameters)
print "new model"

print model.parameters[param_colLatents].shape
print model.parameters[param_rowLatents].shape

print(X)
print(np.round(invtrans))
