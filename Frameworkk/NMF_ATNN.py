import autograd.numpy as np
from autograd import grad
from NMF import NMF
from autograd.util import flatten
import time
from autograd.core import primitive
from autograd.scipy.misc import logsumexp
from autograd.optimizers import adam



rs=np.random.RandomState(0)
swap = 0
class NMF_ATNN(NMF):

    def __init__(self,n_components=0, data = None, scale = .1, layer_sizes = []):
        NMF.__init__(self,n_components,data)
        self.row_size, self.col_size = data.shape
        self.parameters = self.init_random_params(scale,layer_sizes)#parameters is the [[w, b],...] so far
        print(len(self.parameters))
        self.NET_DEPTH = len(self.parameters)
        print(self.parameters)
        #Append column latents to end.
        self.parameters.append(scale * np.random.rand(n_components,self.col_size))#parameters is the [[w, b],...]
        self.parameters.append(scale *  np.ones(layer_sizes[-1]))
        self.parameters = list(self.parameters)
        self.train = self.train_neural_net
        self.loss = self.nnLoss
        self.inference = self.neural_net_inference

    def nnLoss(self,parameters,iter=0,data=None):
        """
        Compute simplified version of squared loss with penalty on vector norms
        :param parameters: Same as class parameter, here for autograd
        :param data:
        :return: A scalar denoting the loss
        """
        #Frobenius Norm squared error term
        data = self.data if (data is None or data is 0) else data
        loss = 0
        regTerms = .001*np.square(flatten(self.parameters)[0]).sum()
        inferred = self.inference(parameters,data=data)
        for i in range(data.shape[0]):
            temp_data = data[i,:]
            keep = temp_data > 0
            loss = loss + np.square(temp_data[keep]- inferred[i]).sum()
        return loss + regTerms

    def getInferredMatrix(self,parameters,data):
        inferred = self.inference(parameters,data=data)
        newarray = np.zeros((data.shape))
        for i in range(data.shape[0]):
            ratings_high = data[i,:]>0
            newarray[i,ratings_high] = inferred[i]
        return newarray

#Credit to David Duvenaud for sleek init code
    def init_random_params(self, scale, layer_sizes, rs=np.random.RandomState(0)):
        """Build a list of (weights, biases) tuples,
           one for each layer in the net."""
        return [(scale * rs.randn(m, n),   # weight matrix
                  scale *  rs.randn(n))      # bias vector
                for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]


    def print_perf(self,params, iter=0, gradient=0, data = None):
        data = self.data if (data is None or data is 0) else data
        predicted_data = self.getInferredMatrix(params,data)
        print("iter is ", iter)
        print("MSE is ",(abs(data-predicted_data).sum())/((data>0).sum()))
        print(self.loss(parameters=params,data=data))


    def train_neural_net(self,alpha = .00003, max_iter = 1,latent_indices = None,data = None, counter = 0):
        global swap
        train_data = self.data if data is None else data


    def neural_net_inference(self,parameters,iter = 0, data = None):
            global swap
            data = self.data if (data is None or data is 0) else data

            #users, movies = data.shape
            predictions = [0]*data.shape[0]
            colLatents = parameters[self.NET_DEPTH]
            mult = parameters[self.NET_DEPTH+1]
            for i in range(data.shape[0]):
                cur_data = data[i,:]
                net_parameters = parameters[:self.NET_DEPTH]
                #Data preproc
                ratings_high = flatten(cur_data > 0)[0] #Data preproc
                if ratings_high.sum() == 0:
                    predictions[i] = np.array(0)
                    continue
                reduced_latents = colLatents[:,ratings_high] #Data prepoc
                reduced_ratings = cur_data[ratings_high].reshape((1,ratings_high.sum())) #Data prepoc
                temp_reduced_colLatents = np.concatenate((reduced_latents,reduced_ratings),axis = 0 ) #Data prepoc
                prediction = self.neural_net_predict(net_parameters,np.transpose(temp_reduced_colLatents))
                dense_attention = np.dot(prediction,mult)
                predictions[i] = np.dot(np.transpose(np.dot(reduced_latents, dense_attention)),reduced_latents)

            return predictions #Actual inference

    def neural_net_predict(self,params, inputs):
        """Implements a deep neural network for classification.
           params is a list of (weights, bias) tuples.
           inputs is an (N x D) matrix.
           returns normalized class log-probabilities."""
        for W, b in params:
            outputs = np.dot(inputs, W) + b
            inputs = relu(outputs)
        return outputs #- logsumexp(outputs, keepdims=True)

def softmax(x):
    #Compute softmax values for every element in input_data
    if x.sum() == 0:
        return np.array(0)
    return x / logsumexp(x)

def logsumexp(x):
    """Numerically stable log(sum(exp(x)))"""
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

def relu(data):
    return data * (data > 0)
