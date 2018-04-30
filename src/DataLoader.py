# A collection of utility methods for loading data
from os import listdir
from os.path import join

import gc
import numpy as np
from scipy.sparse import *
import os
import os.path
import pickle
import sys


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def linesToDict(lines):
    full_data = dok_matrix((490000,18000),
                           dtype=int)
    print("lines", lines)
    for elem in lines:
        print("elem", elem)
        user, item, rating = elem.split(",")
        full_data[int(user), int(item)] = int(rating)
    return full_data

class DataLoader:
    MOVIELENS = "Movielens"
    NETFLIX = "Netflix"
    pklExtension = "pickled"

    def LoadData(self, file_path, data_type, size):
        """
        Entrance method for this class.
        It's responsible for loading in arbitrary files.

        :param path: The location of the file containing the data
        :param type: The type of data (e.g Netflix ...)
        :return: A 2D numpy array containing real numbered entries
        """
        return {self.MOVIELENS: self.LoadMovieLens, self.NETFLIX: self.LoadNetflix}.get(data_type, None)(file_path,size)

    def fixMovelens100m(self, file_path):
        encountered = {}
        idx = 1
        f = open(file_path, 'r')
        fixed = open(file_path + "better", 'w+')
        # Determine length later
        X = [list() for x in range(72000)]

        for elem in f.readlines():
            user, item, rating, _ = [x for x in elem.split('::')]
            if item not in encountered:
                encountered[item] = idx
                idx += 1
            X[int(user)].append((encountered[item], float(rating)))
        for user_id in range(72000):
            out = ""
            for rating_info in X[user_id]:
                movie_id, rating = rating_info
                out += "{} {} {} \n".format(user_id, movie_id, rating)
            fixed.write(out)
        return X




    def LoadNetflix(self,file_path,size):
        #We know how many users, we know how many movies
        #
        pickledArray = self.loadCachedIfPossible(filePath=file_path)
        fullDataDok = dok_matrix((size),
                     dtype=np.byte)
        if type(pickledArray) != type(None):
            fullDataDok.update(pickledArray)
            return fullDataDok

        full_data = {}
        f = open(file_path,'r')
        iter = 0
        for elem in f:
            iter+=1
            user, item, rating = [int(x) for x in elem.split(",")]
            if iter%1000000==0:
                gc.collect()
            full_data[user, item] = rating

        fullDataDok.update(full_data)
        self.savePickledVersion(file_path, full_data, protocol=2)
        full_data = None
        return fullDataDok

    def loadCachedIfPossible(self,filePath):
        pickleFile = filePath+self.pklExtension
        if os.path.isfile(pickleFile):
            print("Loading pickled version at {}".format(pickleFile))
            with open(pickleFile, 'rb') as f:
                return pickle.load(f)
        return None

    def savePickledVersion(self, filePath, array, protocol = 0):
        pickleFile = filePath+self.pklExtension
        print("Saving pickled version at {}".format(pickleFile))
        with open(pickleFile, 'wb') as f:
            pickle.dump(array, f, protocol=protocol)

    def LoadMovieLens(self, file_path, size):
        encountered = {}
        idx = 1
        pickledArray = self.loadCachedIfPossible(filePath=file_path)
        if type(pickledArray) != type(None): return pickledArray
        f = open(file_path, 'r')
        # Determine length later
        full_data_dok = dok_matrix(size,#np.zeros((6050, 3910),
                     dtype=int)
        full_data = dict()
        for elem in f.readlines():
            user, item, rating = [int(float(x)) for x in elem.split()][:3]
            if item not in encountered:
                encountered[item] = idx
                idx += 1

            full_data[(user - 1, item - 1)] = rating
        f.close()
        full_data_dok.update(full_data)
        self.savePickledVersion(file_path,full_data_dok)
        return full_data_dok

    def parseNetflixMovieData(self, file_path, user_arr, movie_arr, seen_id, counter):
        f = open(file_path, 'r')
        movie_id = int(f.readline().split(':')[0])
        for elem in f.readlines():
            user_id, rating = [int(x) for x in elem.split(',')[:2]]

            if str(user_id) not in seen_id:
                seen_id[str(user_id)] = counter
                counter += 1
            # user_arr[seen_id[str(user_id)]].append((movie_id,rating))
            movie_arr[movie_id].append((seen_id.get(str(user_id)), rating))

    def genNetflixRatingCounts(self,
                               out_path='/Users/EliasApple/Data/',
                               folder_path='/Users/EliasApple/PycharmProjects/Recommendations/Data/download/training_set/'):
        # print onlyfiles
        seen_id = {}
        user_arr = [list() for x in range(600000)]
        movie_arr = [list() for x in range(18000)]
        fil = 0
        for file in listdir(folder_path):
            self.parseNetflixMovieData(join(folder_path, file), user_arr, movie_arr, seen_id, len(seen_id.keys()) + 1)
            fil += 1
        # user_first = open(join(out_path,'user_first.txt'),"w+")
        movie_first = open(join(out_path, 'movie_first.txt'), "w+")
        # Print format: user_id, movie_id, rating, original_id
        # for user_id in range(len(user_arr)):
        #     print user_id
        #     out = ""
        #     for rating_info in user_arr[user_id]:
        #         movie_id, rating = rating_info
        #         out += "{},{},{} \n".format(user_id,movie_id,rating)
        #     user_first.write(out)
        #     user_arr[user_id] = 0
        # user_first.close()

        for movie_id in range(len(movie_arr)):
            out = ""
            for rating_info in movie_arr[movie_id]:
                user_id, rating = rating_info
                out += ("{},{},{} \n".format(movie_id, user_id, rating))
            movie_first.write(out)
            movie_arr[movie_id] = 0
        movie_first.close()

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

        # First, aggregate all necessary elements of opposite type
        other_type_indices = []
        if type == 'u':
            for user in indices:
                for movie in range(len(data[user, :])):
                    if data[user, movie] != 0:
                        other_type_indices.append(movie)
        other_type_indices = list(set(other_type_indices))
        # Now all movie indices are aggregated
        # Next build out array
        batch = np.zeros((len(indices), len(other_type_indices)))
        for user_ind in range(len(indices)):
            for movie_ind in range(len(other_type_indices)):
                batch[user_ind, movie_ind] = data[indices[user_ind], other_type_indices[movie_ind]]

        return batch, indices, other_type_indices
