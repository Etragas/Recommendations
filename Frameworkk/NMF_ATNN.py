import autograd.numpy as np
from autograd import grad
from NMF import NMF
import time
from autograd.core import primitive
from autograd.scipy.misc import logsumexp



class NMF_ATNN(NMF):

    def __init__(self,n_components=0, data = None, scale = .1, layer_sizes = []):
        NMF.__init__(self,n_components,data)
        self.row_size, self.col_size = data.shape
        self.parameters = self.init_random_params(scale,layer_sizes)#parameters is the [[w, b],...] so far
        self.parameters+=(self.init_random_params(scale,[n_components,80,n_components]))#parameters is the [[w, b],...] so far
        print(len(self.parameters))
        self.NET_DEPTH = len(self.parameters)
        print(self.parameters)
        #Append column latents to end.
        self.parameters.append(scale * np.random.rand(n_components,self.col_size))#parameters is the [[w, b],...]
        self.parameters.append(data)
        self.parameters = list(self.parameters)
        self.train = self.train_neural_net
        self.loss = self.nnLoss
        self.inference = self.neural_net_inference

    def nnLoss(self,parameters,data):
        """
        Compute simplified version of squared loss with penalty on vector norms
        :param parameters: Same as class parameter, here for autograd
        :param data:
        :return: A scalar denoting the loss
        """
        #Frobenius Norm squared error term
        regTerms = 0
        for i in range (self.NET_DEPTH):
            regTerms = regTerms + np.square(self.parameters[i][0]).sum() + np.square(self.parameters[i][1]).sum()
        regTerms = regTerms + np.square(self.parameters[self.NET_DEPTH]).sum()
        keep = data > 0
        loss = np.square(data*keep-self.inference(parameters)).sum() + .001*regTerms
        return loss

#Credit to David Duvenaud for sleek init code
    def init_random_params(self, scale, layer_sizes, rs=np.random.RandomState(0)):
        """Build a list of (weights, biases) tuples,
           one for each layer in the net."""
        return [[ scale + scale * rs.randn(m, n),   # weight matrix
                 scale + scale *  rs.randn(n)]     # bias vector
                for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

    def train_neural_net(self,alpha = .000003, max_iter = 1,latent_indices = None,data = None):
        train_data = self.data if data is None else data
        colLatents = self.parameters[self.NET_DEPTH]
        grads = None

        print "before"
        for iter in range(0,max_iter):
            #Do for every user
            loss = 0
            start = time.time()
            op_total = 0
            grad_total = 0
            lgrad = grad(self.loss,0)

            grad_start = time.time()
            grads = lgrad(self.parameters, train_data)
            print loss
            print "end", time.time() - start
            print "op total", op_total
            print "grad total", grad_total
            #Get gradients
            #Update parameters
            for i in range (self.NET_DEPTH):
                #Updating net weights
                self.parameters[i] = [self.parameters[i][0] -alpha*grads[i][0], self.parameters[i][1] -alpha*grads[i][1]]
            #Updating col_latents
            self.parameters[self.NET_DEPTH] += -alpha*grads[self.NET_DEPTH]

    def neural_net_inference(self,parameters, data = None):
        colLatents = self.parameters[self.NET_DEPTH]
        op_total = 0
        for i in range(self.row_size):
            op_start = time.time()
            ratings = self.data[i,:].reshape([1,100])
            ratings_high = ratings > 0
            reduced_latents = colLatents[:,np.ndarray.flatten(ratings_high)] #Data prepoc
            reduced_ratings = ratings[:,np.ndarray.flatten(ratings_high)] #Data prepoc
            op_total += time.time() - op_start
            temp_reduced_colLatents = np.concatenate((reduced_latents,reduced_ratings), axis=0) #Data prepoc

            net_parameters = parameters[:2]
            colLatents = parameters[self.NET_DEPTH]
            reducedLatents = colLatents[:,ratings_high[0,:]]
            num_dense = ratings_high.sum()
            dense_attention = softmax(self.neural_net_predict(net_parameters,np.transpose(temp_reduced_colLatents))).reshape([num_dense]) #Inference
            #sparse_attention = np.array((self.listify(ratings_high[0,:],np.transpose(dense_attention)))) #Inference
            attention = ((np.array(dense_attention)))
            return np.dot(np.transpose(np.dot(reducedLatents, attention)),colLatents) #Actual inference

# Final ideal
#             net_parameters = parameters[:2]
#             colLatents = parameters[self.NET_DEPTH]
#             dense_attention = softmax(self.neural_net_predict(net_parameters,np.transpose(temp_reduced_colLatents)))[:,0] #Inference
#             sparse_attention = np.array((self.listify(ratings_high[0,:],np.transpose(dense_attention)))) #Inference
#             attention = np.transpose((np.array(sparse_attention)))
#             return np.dot(np.transpose(np.dot(colLatents, attention)),colLatents) #Actual inference


    def listify(self,indicator, data):
        data_ind = 0
        final_list = []
        for bool in indicator:
            if bool:
                final_list.append(data[data_ind])
                data_ind+=1
            else:
                final_list.append(0)
        return final_list

    def neural_net_predict(self,net_parameters,input_data):
        #Assume 3 layer net, so net_parameters
        W1 , b1 = net_parameters[0]
        W2, b2 = net_parameters[1]
        layer2 = relu(np.dot(input_data,W1) + b1)
        layer3 = relu(np.dot(layer2,W2) + b2)
        return layer3

def softmax(x):
    #Compute softmax values for every element in input_data
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def nested_sum(l1,l2):
    #Do rec over one, adding from other
    for idx in range(len(l1)):
        if type(l1) is np.numpy_extra.ArrayNode or type(l1) is np.ndarray:
            l1[idx] = l1[idx] + l2[idx]
        else:
            l1[idx] = nested_sum(l1[idx],l2[idx])
    return l1


def relu(data):
    return data * (data > 0)
