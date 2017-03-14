import autograd.numpy as np
from autograd import grad
from Frameworkk.NMF import NMF

class NMF_NN(NMF):

    def __init__(self,n_components=0, data = None, scale = .1, layer_sizes = []):
        NMF.__init__(self,n_components,data)
        self.NET_DEPTH = len(layer_sizes)-1
        row_size,col_size = data.shape
        self.parameters = self.init_random_params(scale,layer_sizes)
        #Append column latents to end.
        self.parameters.append(scale*np.random.rand(n_components,col_size))
        self.parameters.append(self.data)
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
        keeper = data > 0
        for i in range (self.NET_DEPTH):
            regTerms = np.square(self.parameters[i][0]).sum() + np.square(self.parameters[i][1]).sum()
        regTerms += np.square(self.parameters[self.NET_DEPTH]).sum()
        loss = np.square(data-keeper*self.inference(parameters)).sum() + .1*regTerms
        return loss

#Credit to David Duvenaud for sleek init code
    def init_random_params(self, scale, layer_sizes, rs=np.random.RandomState(0)):
        """Build a list of (weights, biases) tuples,
           one for each layer in the net."""
        return [(scale * rs.randn(m, n),   # weight matrix
                 scale * rs.randn(n))      # bias vector
                for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

    def train_neural_net(self,alpha = .0001, max_iter = 20,latent_indices = None,data = None):
        train_data = self.data if data is None else data
        for iter in range(0,max_iter):
            #Get gradients
            grads = grad(self.loss,0)(self.parameters ,train_data)

            #Update parameters
            for i in range (self.NET_DEPTH):
                #Updating net weights
                self.parameters[i] = [self.parameters[i][0] -alpha*grads[i][0], self.parameters[i][1] -alpha*grads[i][1]]
            #Updating col_latents
            self.parameters[self.NET_DEPTH] += -alpha*grads[self.NET_DEPTH]

    def neural_net_inference(self,parameters):
            net_parameters = parameters[:self.NET_DEPTH]
            colLatents = parameters[self.NET_DEPTH]

            #This is broken and stupid
            unweighted_user_latents = self.neural_net_predict(net_parameters,np.transpose(colLatents))
            return np.dot(unweighted_user_latents,colLatents)

    def neural_net_predict(self,net_parameters,input_data):
        #Assume 3 layer net, so net_parameters
        W1 , b1 = net_parameters[0]
        W2, b2 = net_parameters[1]
        layer2 = relu(np.dot(input_data,W1) + b1)
        layer3 = relu(np.dot(layer2,W2) + b2)
        return layer3

def relu(data):
    return np.maximum(data,0)

