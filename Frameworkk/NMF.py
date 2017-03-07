import Model
from sklearn.decomposition import NMF as NMFSCIKIT
import autograd.numpy as np
from autograd import grad
import inspect
param_rowLatents = 0
param_colLatents = 1

class NMF():

    def __init__(self,n_components=0, data = None):
        row_size,col_size = data.shape
        self.components = n_components
        self.data = data
        self.parameters = [np.random.rand(row_size,n_components), np.random.rand(n_components,col_size)]
        self.loss = self.defaultLoss
        self.inference = self.defaultInference

    def defaultLoss(self,parameters,data):
        """
        Compute simplified version of squared loss with penalty on vector norms
        :param parameters: Same as class parameter, here for autograd
        :param data:
        :return: A scalar denoting the loss
        """
        #Frobenius Norm squared error term
        loss = np.square(data-self.inference(parameters)).sum()
        + .1*(np.square(parameters[0].sum()) + np.square(parameters[1].sum()))
        return loss

    def defaultInference(self, parameters):
        """
        Default inference method. In this case, just inner product.
        :param parameters: Same as class parameter, here for autograd
        :return: Scalar denoting rating
        """
        rowLatents = parameters[0]
        colLatents = parameters[1]
        #If indices aren't zero, we take the appropriate subsets
        pred = np.dot(rowLatents,colLatents)
        return pred

    def train(self,alpha = .0005, max_iter = 20,latent_indices = None,data = None):
        """
        This method just runs training with some special functions to support batch learning.
        It uses the latent_indices to only use the parameters necessary for the batch.

        :param alpha: Gradient penalty
        :param latent_indices: In the case of batch learning, these are the user,movie latents needed
        """
        #Old way of training, useful for debugging
        #self.parameters[param_rowLatents] = self.model.fit_transform(self.data)
        #self.parameters[param_colLatents] = self.model.components_

        train_data = self.data if data is None else data
        parameters = [[],[]]

        if latent_indices is None:
            parameters = self.parameters
        else:
            parameters[0] = self.parameters[0][latent_indices[0],:]
            parameters[1] = self.parameters[1][:,latent_indices[1]]

        for iter in range(0,max_iter):
            #print("loss is", self.loss(parameters ,data))
            grads = grad(self.loss,0)(parameters ,train_data)
            #print(grads[1].shape)
            parameters[0] += -alpha*grads[0]
            parameters[1] += -alpha*grads[1]

        if latent_indices is not None:
            self.parameters[0][latent_indices[0] , :] = parameters[0]
            self.parameters[1][: , latent_indices[1]] = parameters[1]