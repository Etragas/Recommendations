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
test_size = (100,100)
train, test = full_data[:100,:100], full_data[400:500,:100]
step_size = 0.01
latent_size = 80

model = NMF_ATNN(n_components=latent_size,data=train,layer_sizes=[latent_size+1,120,40])
model.inference = model.neural_net_inference
model.loss = model.nnLoss
for iter in range(5):
    model.parameters = adam(grad(model.loss,0),model.parameters, step_size=step_size,
                            num_iters=100, callback=model.print_perf)
    print "Test performance:"
    model.print_perf(model.parameters,data=test)



#Do Inference
invtrans = model.getInferredMatrix(model.parameters,model.data)
print "new model"


print "Train"
print (model.print_perf(model.parameters,data=train))
print(train)
print(np.round(invtrans))

#Test rowless on novel users.
print "Test"
print (model.print_perf(model.parameters,data=test))
print(test)
invtrans = model.getInferredMatrix(model.parameters,test)
print(np.round(invtrans))

