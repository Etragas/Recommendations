import Model
from sklearn.decomposition import NMF as NMFSCIKIT
import numpy as np

rowLatents = "rowLatents"
colLatents = "colLatents"

class NMF():


    def __init__(self,n_components=0, row_size = 0, col_size = 0, data = None):
        #TODO: Extract row, col from data
        global rowLatents, colLatents
        self.components = n_components
        self.data = data
        self.parameters = {rowLatents : np.zeros((row_size,n_components)), colLatents : np.zeros((n_components,row_size))}
        self.loss = self.defaultLoss
        self.inference = self.defaultInference
        self.model = NMFSCIKIT(n_components=n_components, init='random', random_state=0, max_iter = 1000, alpha=.001, l1_ratio=.1)



    def defaultLoss(self):
        return 0

    def do_inference(self,indices=None):
        return self.inference(indices)


    def defaultInference(self,indices=None):
        global rowLatents, colLatents
        pred = np.dot(self.parameters.get(rowLatents),self.parameters.get(colLatents))

        if (indices == None):
            return pred
        else:
            return pred[indices]

    def train(self):
        global rowLatents, colLatents
        self.parameters[rowLatents] = self.model.fit_transform(self.data)
        self.parameters[colLatents] = self.model.components_
