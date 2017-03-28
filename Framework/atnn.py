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
train, test = full_data[:400,:100], full_data[400:500,:100]
X = train
print(X)
#Construct Model
latent_size = 40

model = NMF_ATNN(n_components=latent_size,data=X,layer_sizes=[latent_size+1, 200,80, 1])
model.inference = model.neural_net_inference
model.loss = model.nnLoss
for counter in range(0,100):
    model.train()

    print(counter)
    if (counter%1 == 0):
        #Mean square error is
        predicted_data = model.inference(model.parameters,data=train)
        print("MSE is ",(abs(X-(X>0)*predicted_data).sum())/((X>0).sum()))
        print(model.loss(model.parameters,model.data))

        predicted_data = model.inference(model.parameters,data=test)
        print("MSE Test is ",(abs(test-(test>0)*predicted_data).sum())/((test>0).sum()))
        print(model.loss(model.parameters,test))


#Do Inference
invtrans = model.inference(model.parameters, data = train)
print "new model"

print(X)
print(np.round(invtrans))
print "with keeps"
print(X*(X>0))
print "inferred"
print(np.round(invtrans) * (X > 0))

#Test rowless on novel users.
X = test
print model.loss(model.parameters,X)/sum(X.shape)
print(X)
invtrans = model.inference(model.parameters,data=test)
print(np.round(invtrans))
print "with keeps"
print(X*(X>0))
print "inferred"
print(np.round(invtrans) * (X > 0))
#Time for testing
