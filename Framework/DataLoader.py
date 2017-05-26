# A collection of utility methods for loading data
from os import listdir
from os.path import join
import numpy as np
from scipy.sparse import *
from utils import keys_row_first,keys_col_first

class DataLoader:
    MOVIELENS = "Movielens"
    NETFLIX = "Netflix"

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
            print user
            X[int(user)].append((encountered[item], float(rating)))
        for user_id in range(72000):
            out = ""
            print user_id
            for rating_info in X[user_id]:
                movie_id, rating = rating_info
                out += "{} {} {} \n".format(user_id, movie_id, rating)
            fixed.write(out)
        return X

    def LoadNetflix(self,file_path,size):
        #We know how many users, we know how many movies
        #
        row_size, col_size = size
        row_count = [0]*row_size
        col_count = [0]*col_size
        row_first = [[list(),list()] for lol in range(row_size)]
        col_first = [[list(),list()] for lol in range(col_size)]
        print "HERE"
        f = open(file_path,'r')
        for elem in f.readlines():
            user, item, rating = [x for x in elem.split()[:3]]
            user, item, rating = int(user)-1, int(item)-1, float(rating)
            row_count[user] += 1
            col_count[item] += 1
            row_first[user][0].append(item)
            row_first[user][1].append(rating)
            col_first[item][0].append(user)
            col_first[item][1].append(rating)
        u_idx = 0
        m_idx = 0
        while u_idx < (row_size):
            items_ratings = zip(*row_first[u_idx])
            if not items_ratings:
                print "EMPTY Row"
                del row_first[u_idx]
                row_size -= 1
                u_idx += 1
                continue
            row_first[u_idx] = zip(*sorted(items_ratings))
            u_idx += 1

        while m_idx < (col_size):
            items_ratings = zip(*col_first[m_idx])
            if not items_ratings:
                print "EMPTY Col"
                del col_first[m_idx]
                col_size -= 1
                m_idx += 1
                continue
            col_first[m_idx] = zip(*sorted(items_ratings))
            m_idx += 1
        row_first = [r for r in row_first if sum(r[0]) > 0]
        col_first = [c for c in col_first if sum(c[0]) > 0]
        return {keys_row_first: row_first,keys_col_first:col_first}

    def LoadMovieLens(self, file_path, size):
        encountered = {}
        idx = 1
        f = open(file_path, 'r')
        # Determine length later
        full_data = np.zeros((size),#np.zeros((6050, 3910),
                     dtype=int)  #                                   ##TODO: FIX THIS MAGIC

        for elem in f.readlines():
            user, item, rating = [x for x in elem.split()][:3]
            if item not in encountered:
                encountered[item] = idx
                idx += 1

            user, item, rating = [int(user), int(item), float(rating)]
            full_data[user - 1, item - 1] = rating
        f.close()
        rows = [x for x in range((full_data.shape[0])) if full_data[x,:].sum() > 0]
        cols = [x for x in range((full_data.shape[1])) if full_data[:,x].sum() > 0]
        full_data = full_data[rows,:]
        full_data = full_data[:,cols]

        return full_data

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
            print file
            self.parseNetflixMovieData(join(folder_path, file), user_arr, movie_arr, seen_id, len(seen_id.keys()) + 1)
            fil += 1
        print len(seen_id.keys())
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
