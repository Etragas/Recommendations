
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
        X = np.zeros((10000,10000))                                   #TODO: FIX THIS MAGIC

        for elem in f.readlines():
            user, item, rating, _ = [int(x) for x in elem.split()]
            if (user < 10000) and (item < 10000):
                X[user-1,item-1] = rating
        return X


    def getBatch(self, data, indices, type):
        """
        In order to construct a batch for training, the method must do the following:
        For each user, collect every movie they've rated along with the rating.
        Then it constructs a matrix where every user has their own row, and the spanning set of movies is the columns
        :param data:
        :param indices: These are the indices for the items
        :param axis: Decides whether the user or the movie is the basis element
        :return: See above
        """

        #First, aggregate all necessary elements of opposite type
        other_type_indices = []
        if type == 'u':
            for user in indices:
                for movie in range(len(data[user,:])):
                    if data[user,movie] != 0:
                        other_type_indices.append(movie)
        other_type_indices = list(set(other_type_indices))
        #Now all movie indices are aggregated
        #Next build out array
        batch = np.zeros((len(indices),len(other_type_indices)))
        for user_ind in range(len(indices)):
            for movie_ind in range(len(other_type_indices)):
                batch[user_ind,movie_ind] = data[indices[user_ind],other_type_indices[movie_ind]]

        return batch, indices, other_type_indices