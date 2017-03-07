import Model
from sklearn.decomposition import NMF as NMFSCIKIT
import autograd.numpy as np
from autograd import grad
import inspect
param_rowLatents = 0
param_colLatents = 1

class NMF():

    def __init__(self,n_components=0, row_size = 0, col_size = 0, data = None):
        #TODO: Extract row, col from data
        self.components = n_components
        self.data = data
        self.parameters = [np.zeros((row_size,n_components)), np.zeros((n_components,row_size))]
        self.loss = self.defaultLoss
        self.inference = self.defaultInference
        self.model = NMFSCIKIT(n_components=n_components, init='random', random_state=0, max_iter = 1000, alpha=.00, l1_ratio=.0)



    def defaultLoss(self,parameters,data):
        #Squared error term
        error = np.square(data-self.inference(parameters)).sum()
        return error

    def defaultInference(self, parameters, indices=None):
        rowLatents = parameters[0]
        colLatents = parameters[1]
        pred = np.dot(rowLatents,colLatents)

        if (indices == None):
            return pred
        else:
            return pred[indices]

    def train(self):
        global param_rowLatents, param_colLatents
        self.parameters[param_rowLatents] = self.model.fit_transform(self.data)
        self.parameters[param_colLatents] = self.model.components_
        print("loss is", self.loss(self.parameters,self.data))
        print(grad(self.loss,0)(self.parameters,self.data))