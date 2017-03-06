import numpy as np
from scipy.sparse import *
from sklearn.decomposition import NMF
import os



#Load all data
f = open("../Data/ml-100k/u.data",'r')
X = np.zeros((100001,100001))

for elem in f.readlines():
    user, item, rating, _ = [int(x) for x in elem.split()]
    X[user,item] = rating

#Keep only first 100 vectors
denseX = X[:100,:100]
print X.shape
X = dok_matrix(denseX)

#Generate latents using matrix factorization
model = NMF(n_components=80, init='random', random_state=0, max_iter = 1000, alpha=.001, l1_ratio=.1)
print denseX
rowLatents = model.fit_transform(X)
colLatents = model.components_
invtrans = np.dot(rowLatents,colLatents)
print "new model"
print model.reconstruction_err_
print model.n_iter_
print colLatents.shape
print rowLatents.shape

sse = 0
for x in range(100):
    for y in range(100):
        sse += (denseX[x,y]-invtrans[x,y])**2


print sse