#Main method Rfor running models
import numpy as np
from scipy.sparse import *
from sklearn.decomposition import NMF
import os
from DataLoader import *
from NMF_ATNN import *
full_data  = DataLoader().LoadData(file_path="../Data/ml-100k/u.data",data_type=DataLoader.MOVIELENS)

#Reduce the matrix to toy size
train, test = full_data[:100,:100], full_data[400:500,:100]
#Model Parameters
latent_size = 80
layer_size = [latent_size+1, 120, 40]

#Training Parameters
num_epochs = 100
num_iters = 5
step_size = 0.01

model = NMF_ATNN(n_components=latent_size,data=train,layer_sizes=layer_size)

for iter in range(1):
    model.parameters = adam(grad(model.loss,0),model.parameters, step_size=step_size,
                            num_iters=num_epochs, callback=model.print_perf)
    print "Test performance:"
    model.print_perf(model.parameters,data=test)

#Print performance on each model
invtrans = model.getInferredMatrix(model.parameters,model.data)
print "\n".join([str(x) for x in ["Train", model.print_perf(model.parameters,data=train), train, np.round(invtrans)]])

invtrans = model.getInferredMatrix(model.parameters,test)
print "\n".join([str(x) for x in ["Test", model.print_perf(model.parameters,data=test), test, np.round(invtrans)]])