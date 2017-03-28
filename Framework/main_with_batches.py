#Main method Rfor running models

#Start by loading the data
import numpy as np
from scipy.sparse import *
from sklearn.decomposition import NMF
import os
from DataLoader import *
from NMF import *
from NMF_NN import *


#Load all data
full_data  = DataLoader().LoadData(file_path="../Data/ml-100k/u.data",data_type=DataLoader.MOVIELENS)

#Reduce the matrix to toy size
X = full_data[:100,:100]
print(X)
#Construct Model
latent_size = 80
model = NMF_NN(n_components=latent_size,data=X,layer_sizes=[latent_size,40,latent_size])
model = NMF(n_components=latent_size,data=X)
model.inference = model.rowlessInference
#Random batch size
batch_size = X.shape[1]

#Train the model, using batch learning


for counter in range(0,1000):
    model.train()
    #Generate indices for users
    #data_indices = range(X.shape[1])
    # model.inference = model.rowlessInference
    # model.train()
    # #Shuffle the indices so we can have random batches
    # np.random.shuffle(data_indices)
    #
    # #Chunks represent units of users we'll test against
    # chunks = [data_indices[x:x+batch_size] for x in range(0,len(data_indices),batch_size)]
    #
    # for user_indices in chunks:
    #     #Create data-matrix for chosen users
    #     batch, user_indices, movie_indices = DataLoader().getBatch(X, user_indices, 'u')
    #
    #     model.inference = model.neural_net_inference
    #     model.train_neural_net(latent_indices=[user_indices,movie_indices],data=batch)
    #     #Print loss
    if (counter%100 == 0):
        #Mean square error is
        invtrans = model.inference(model.parameters)
        print("MSE is ",(abs(X-invtrans).sum())/(X.shape[0]*X.shape[1]))
        print(model.loss(model.parameters,model.data))

#Do Inference
invtrans = model.inference(model.parameters)
print "new model"

# print model.parameters[param_colLatents].shape
# print model.parameters[param_rowLatents].shape

print(X)
print(np.round(invtrans))

X = full_data[100:200,:100]
model.data = X
print model.loss(model.parameters,X)/sum(X.shape)
print(X)
print(np.round(invtrans))
#Time for testing
