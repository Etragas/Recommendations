import autograd.numpy as np
from autograd.util import flatten
from autograd import grad
from NMF import NMF
import time
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
            regTerms += np.square(self.parameters[i][0]).sum() + np.square(self.parameters[i][1]).sum()
        regTerms += np.square(self.parameters[self.NET_DEPTH]).sum()
        keep = data > 0
        loss = np.square(data- keep*self.inference(parameters,data=data)).sum() + regTerms
        return loss

#Credit to David Duvenaud for sleek init code
    def init_random_params(self, scale, layer_sizes, rs=np.random.RandomState(0)):
        """Build a list of (weights, biases) tuples,
           one for each layer in the net."""
        return [( scale + scale * rs.randn(m, n),   # weight matrix
                 scale + scale *  rs.randn(n))      # bias vector
                for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

    def train_neural_net(self,alpha = .00001, max_iter = 1,latent_indices = None,data = None):
        train_data = self.data if data is None else data
        prev_grads = []
        cur_grad = []
        prev_grad_roll = None
        alt = 0
        for iter in range(0,10):
            predicted_data = self.inference(self.parameters,train_data)
            print("MSE is ",(abs(train_data-(train_data>0)*predicted_data).sum())/((train_data>0).sum()))
            print(self.loss(self.parameters,self.data))

            start = time.time()
            print "before", start
            grads = grad(self.loss,0)(self.parameters, train_data)
            print "after", time.time() - start
            #Get gradients
            #Update parameters
            if not prev_grads:
                prev_grads, grad_roll = flatten(grads)
                prev_grads = grad_roll(prev_grads)
            else:
                a = flatten(grads)[0]
                prev_grads = grad_roll(flatten(prev_grads)[0] * .95 + a)
            cur_grad = prev_grads

            if alt == 0:
                for i in range (self.NET_DEPTH):
                    #Updating net weights
                    self.parameters[i] = [self.parameters[i][l] -alpha*cur_grad[i][l] for l in range(len(self.parameters[i]))]
                print "net_pass"
                alt = 1
            else:
            #Updating col_latents
                print "col_pass"
                self.parameters[self.NET_DEPTH] += -alpha*cur_grad[self.NET_DEPTH]
                alt = 0


    def neural_net_inference(self,parameters, data = None):
            users, movies = data.shape
            net_parameters = parameters[:3]
            colLatents = parameters[self.NET_DEPTH]
            temp_attention = []
            colLatents = parameters[self.NET_DEPTH]
            for i in range(users):
              ratings = self.data[i,:].reshape([1,self.col_size]) #Data preproc
              ratings_high = ratings > 0 #Data preproc
              reduced_latents = colLatents[:,np.ndarray.flatten(ratings_high)] #Data prepoc
              reduced_ratings = ratings[:,np.ndarray.flatten(ratings_high)] #Data prepoc
              temp_colLatents = np.concatenate((colLatents,ratings), axis=0) * (ratings > 0) #Data prepoc, this has zeros
              temp_reduced_colLatents = np.concatenate((reduced_latents,reduced_ratings), axis=0) #Data prepoc

              #All above, outside grad.
              #
              #This is ugly
              dense_attention = softmax(self.neural_net_predict(net_parameters,np.transpose(temp_reduced_colLatents)))[:,0] #Inference
              sparse_attention = np.array((self.listify(ratings_high[0,:],np.transpose(dense_attention)))) #Inference
              #sparse_latents = np.array((self.listify(ratings_high[0,:],np.transpose(reduced_latents)))) #Inference
              temp_attention.append(sparse_attention) #Inference

            attention = np.transpose((np.array(temp_attention)))
            return np.dot(np.transpose(np.dot(colLatents, attention)),colLatents) #Actual inference

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
        W3, b3 = net_parameters[2]
        layer2 = relu(np.dot(input_data,W1) + b1)
        layer3 = relu(np.dot(layer2,W2) + b2)
        return relu(np.dot(layer3,W3) + b3)


def softmax(x):
    #Compute softmax values for every element in input_data
    return np.exp(x) / np.sum(np.exp(x), axis=0)
def relu(data):
    return np.maximum(data,0)

