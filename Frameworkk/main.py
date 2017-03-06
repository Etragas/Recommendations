#Main method for running models

#Start by loading the data
import numpy as np
from scipy.sparse import *
from sklearn.decomposition import NMF
import os
from DataLoader import *
from NMF import *


#Load all data
X = DataLoader().LoadData(file_path="../Data/ml-100k/u.data",data_type=DataLoader.MOVIELENS)

denseX = X[:100,:100]
print X.shape
X = dok_matrix(denseX)

#Construct Model
model = NMF(80,100,100,data=X)

#Train Model
model.train()

#Do Inference
invtrans = model.do_inference()
print "new model"

print model.parameters.get("colLatents").shape
print model.parameters.get("rowLatents").shape
# Report Error
sse = 0
for x in range(100):
    for y in range(100):
        sse += (denseX[x,y]-invtrans[x,y])**2


print sse
#Then construct the model
#Run training on the model
#Run inference on the model
#Do some visualizations