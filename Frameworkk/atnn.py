#Main method Rfor running models

#Start by loading the data
import numpy as np
from scipy.sparse import *
from sklearn.decomposition import NMF
import os
from DataLoader import *
from NMF_ATNN import *


#Load all data
full_data  = DataLoader().LoadData(file_path="../Data/ml-100k/u.data",data_type=DataLoader.MOVIELENS)

#Reduce the matrix to toy size
X = full_data[:100,:100]
print(X)
#Construct Model
latent_size = 40

model = NMF_ATNN(n_components=latent_size,data=X,layer_sizes=[latent_size+1, 80, 1])
model.inference = model.neural_net_inference
model.loss = model.nnLoss
for counter in range(0,200):
    model.train()

    print(counter)
    if (counter%1 == 0):
        #Mean square error is
        predicted_data = model.inference(model.parameters)
        print("MSE is ",(abs(X-predicted_data).sum())/(X.shape[0]*X.shape[1]))
        print(model.loss(model.parameters,model.data))

#Do Inference
invtrans = model.inference(model.parameters)
print "new model"

print(X)
print(np.round(invtrans))
print(X*(X>0))
print(np.round(invtrans) * (X > 0))

#Test rowless on novel users.
X = full_data[100:200,:100]
model.data = X
print model.loss(model.parameters,X)/sum(X.shape)
print(X)
print(np.round(invtrans))
print(X*(X>0))
print(np.round(invtrans) * (invtrans > 0))
#Time for testing
