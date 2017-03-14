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
train,test = full_data[:400,:100], full_data[100:200,:100]
print(train)
#Construct Model
latent_size = 80
#model = NMF_NN(n_components=latent_size,data=train,layer_sizes=[latent_size,200,latent_size])
model = NMF(n_components=latent_size,data=train)
model.inference = model.rowlessInference


for counter in range(0,1000):
    model.train()

    if (counter%100 == 0):
        #Mean square error is
        predicted_data = model.inference(model.parameters)
        print("MSE Train is ",(abs(train-predicted_data).sum())/(train.shape[0]*train.shape[1]))
        print(model.loss(model.parameters,model.data))

        predicted_data = model.inference(model.parameters,data=test)
        print("MSE Test is ",(abs(test-predicted_data).sum())/(test.shape[0]*test.shape[1]))
        print(model.loss(model.parameters,test))

#Do Inference
invtrans = model.inference(model.parameters)
print "new model"

print(train)
print(np.round(invtrans))
#Test rowless on novel users.
X = full_data[100:200,:100]
model.data = X
print model.loss(model.parameters,X)/sum(X.shape)
print(X)
print(np.round(model.inference(model.parameters,data=test)))
#Time for testing
