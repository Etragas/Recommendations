
# A collection of utility methods for loading data
import numpy as np
from scipy.sparse import *
class DataLoader:

    MOVIELENS = "Movielens"
    NETFLIX = "Netflix"


    def LoadData(self,file_path,data_type):
        """
        Entrance method for this class.
        It's responsible for loading in arbitrary files.

        :param path: The location of the file containing the data
        :param type: The type of data (e.g Netflix ...)
        :return: A 2D numpy array containing real numbered entries
        """
        return {self.MOVIELENS : self.LoadMovieLens(file_path), self.NETFLIX : 2}.get(data_type,None)



    def LoadMovieLens(self,file_path):

        f = open(file_path,'r')
        #Determine length later
        X = np.zeros((100001,100001))                                   #TODO: FIX THIS MAGIC
        for elem in f.readlines():
            user, item, rating, _ = [int(x) for x in elem.split()]
            X[user,item] = rating

        return X

