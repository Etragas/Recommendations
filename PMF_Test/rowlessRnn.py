import numpy as np
from scipy.sparse import *
from sklearn.decomposition import NMF
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper, MultiRNNCell

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
n_components = 50
model = NMF(n_components=n_components, init='random', random_state=0, max_iter = 1000, alpha=.001, l1_ratio=.1)
print denseX
rowLatents = model.fit_transform(X)
colLatents = model.components_

#Recompute predictions
invtrans = np.dot(rowLatents,colLatents)
print "new model"
print model.reconstruction_err_
print colLatents.shape
print rowLatents.shape

#Just print the sum of squared errors as a sanity check.
sse = 0
for x in range(100):
    for y in range(100):
        sse += (denseX[x,y]-invtrans[x,y])**2

print sse

#Start an RNN

sess = tf.Session()


num_neurons = 200
num_layers = 1
#dropout = tf.placeholder(tf.float64)

cell = LSTMCell(num_neurons)  # Or LSTMCell(num_neurons)
cell = DropoutWrapper(cell)#, output_keep_prob=dropout)
cell = MultiRNNCell([cell] * num_layers)

max_length = 100
# Batch size x time steps x features.
#Dimensions of data are #num examples, max length, input size
data = tf.placeholder(tf.float64, [None, max_length, n_components])
X = np.random.randn(3,2,3)
X_lengths = [2,2,2]
output, last_state = tf.nn.dynamic_rnn(cell, sequence_length=X_lengths, inputs = X, dtype=tf.float64)

print output.shape

output = tf.transpose(output,[1,0,2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)
print last.shape

target = tf.constant([[1,0,1,2],[1,0,1,2],[1,0,1,2]],dtype=tf.float64)

print target.get_shape()
out_size = int(target.get_shape()[1])
weight = tf.Variable(tf.truncated_normal([num_neurons, out_size], stddev=0.1, dtype=tf.float64))
bias = tf.Variable(tf.constant(0.1, shape=[out_size],dtype=tf.float64))
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
c = tf.Print(prediction,[prediction])
cross_entropy = -tf.reduce_sum(target * tf.log(prediction))
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
optimizer = tf.train.GradientDescentOptimizer(.1)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
sess.run(init)
for x in [1,2,3]:
    sess.run(train_op)
    print(sess.run(loss))
