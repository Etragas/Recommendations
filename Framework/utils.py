import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

movie_latent_size = 160
user_latent_size = 160
hyp_user_network_sizes = [movie_latent_size + 1, 200, 200, user_latent_size]
hyp_movie_network_sizes = [user_latent_size + 1, 200, 200, movie_latent_size]
rating_network_sizes = [movie_latent_size + user_latent_size, 200, 200, 5, 1]
scale = .1

dtype = torch.FloatTensor
if (torch.cuda.device_count() > 0):
    dtype = torch.cuda.FloatTensor  # Uncomment this to run on GPU


class GeneratorNet(nn.Module):
    def __init__(self, latent_size=None):
        super(GeneratorNet, self).__init__()
        self.fc1 = nn.Linear(latent_size + 1, 2000)
        # self.bn1 = torch.nn.BatchNorm1d(2000)
        self.fc2 = nn.Linear(2000, 2000)
        # self.bn2 = torch.nn.BatchNorm1d(2000)
        self.fc3 = nn.Linear(2000, 2000)
        # self.bn3 = torch.nn.BatchNorm1d(2000)
        self.fc4 = nn.Linear(2000, 1000)
        # self.bn4 = torch.nn.BatchNorm1d(1000)
        self.fc5 = nn.Linear(1000, 1000)
        # self.bn5 = torch.nn.BatchNorm1d(1000)
        self.fc6 = nn.Linear(1000, latent_size)
        initParams(self)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # self.bn1(
        x = F.relu(self.fc2(x))  # self.bn2(
        x = F.relu(self.fc3(x))  # self.bn3(
        x = F.relu(self.fc4(x))  # self.bn4(
        x = F.relu(self.fc5(x))  # self.bn5(
        x = self.fc6(x)
        return x


class RatingGeneratorNet(nn.Module):
    def __init__(self):
        super(RatingGeneratorNet, self).__init__()
        self.fc1 = nn.Linear(user_latent_size + movie_latent_size, 4000)
        # self.bn1 = torch.nn.BatchNorm1d(4000)
        self.fc2 = nn.Linear(4000, 2000)
        # self.bn2 = torch.nn.BatchNorm1d(2000)
        self.fc3 = nn.Linear(2000, 2000)
        # self.bn3 = torch.nn.BatchNorm1d(2000)
        self.fc4 = nn.Linear(2000, 2000)
        # self.bn4 = torch.nn.BatchNorm1d(2000)
        self.fc5 = nn.Linear(2000, 2000)
        # self.bn5 = torch.nn.BatchNorm1d(2000)
        self.fc6 = nn.Linear(2000, 1)
        initParams(self)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # self.bn1(
        x = F.relu(self.fc2(x))  # self.bn2(
        x = F.relu(self.fc3(x))  # self.bn3(
        x = F.relu(self.fc4(x))  # self.bn4(
        x = F.relu(self.fc5(x))  # self.bn5(
        x = self.fc6(x)  # self.bn6(
        return x


def build_params(num_user_latents=20, num_movie_latents=20):
    parameters = {}
    parameters[keys_movie_to_user_net] = GeneratorNet(latent_size=movie_latent_size).type(dtype)
    parameters[keys_user_to_movie_net] = GeneratorNet(latent_size=user_latent_size).type(dtype)
    parameters[keys_col_latents] = Variable(
        torch.from_numpy(scale * np.random.normal(size=(movie_latent_size, num_movie_latents))).float().type(dtype),
        requires_grad=True)  # Column Latents
    parameters[keys_row_latents] = Variable(
        torch.from_numpy((scale * np.random.normal(size=(num_user_latents, user_latent_size)))).float().type(dtype),
        requires_grad=True)  # Row Latents
    parameters[keys_rating_net] = RatingGeneratorNet().type(dtype)
    return parameters


def initParams(net):
    for param in net.parameters():
        if (param.data.dim() > 1):
            param.data = torch.nn.init.xavier_uniform(param.data)


# Credit to David Duvenaud for sleek init code
def init_random_params(scale, layer_sizes, rs=np.random.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [[scale * rs.randn(m, n),  # weight matrix
             scale * rs.randn(n)]  # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]


def get_canonical_indices(data, latent_sizes):
    indicators = data > 0
    user_rating_counts = indicators.sum(axis=1)  # Bug one found
    movie_rating_counts = indicators.sum(axis=0)  # Bug one found
    user_indices = list(get_top_n(user_rating_counts, latent_sizes[0]))
    movie_indices = list(get_top_n(movie_rating_counts, latent_sizes[1]))
    for val in range(data.shape[0]):
        if val not in user_indices:
            user_indices.append(val)
    for val in range(data.shape[1]):
        if val not in movie_indices:
            movie_indices.append(val)
    return np.array(user_indices), np.array(movie_indices)


def fill_in_gaps(canonical_indices, new_indices, full_data):
    can_sizes = map(lambda x: x.size, canonical_indices)
    # Sort the indices before.
    new_axis_sizes = tuple([can_sizes[x] + new_indices[x].size for x in range(len(canonical_indices))])
    print(new_axis_sizes)
    training_block = np.zeros(new_axis_sizes)
    # To load data properly, cross product every axis with every other of new indices.
    # Fill in gaps along diagonal
    training_block[:can_sizes[0], :can_sizes[1]] = full_data[np.ix_(*canonical_indices)]
    training_block[can_sizes[0]:, can_sizes[1]:] = full_data[np.ix_(*new_indices)]
    # Fill in gaps in between.
    training_block[:can_sizes[0], can_sizes[1]:] = full_data[np.ix_(canonical_indices[0], new_indices[1])]
    training_block[can_sizes[0]:, :can_sizes[1]] = full_data[np.ix_(new_indices[0], canonical_indices[1])]
    return training_block


def get_top_n(data, n):
    indices = np.ravel((data.astype(int)).flatten().argsort())[-n:]
    return indices


def splitData(data, train_ratio=.8):
    np.random.seed(0)  # Debugging line
    data_bool = data > 0
    data_ind = data_bool * (np.random.rand(*data.shape))
    train = data_ind <= train_ratio
    test = data_ind > train_ratio
    train = data * train
    test = data * test
    return train, test  # [row_indices[:row_split], col_indices[:col_split]], [row_indices[row_split:], col_indices[col_split:]]


def listify(data):
    """
    Returns a dict of two lists, one row-first the other column-first.
    Each of the two lists contains a tuple of lists, where tuple[0] is the indices of the rated movies and tuple[1] the ratings.
    :param data:
    :return:
    """
    row_size, col_size = data.shape
    row_first = []
    col_first = []
    for usr_idx in range(row_size):
        rav = np.ravel(np.nonzero(data[usr_idx, :]))
        # if len(rav) == 0:
        #     continue
        movie_indices = list(rav)
        ratings = list(data[usr_idx, movie_indices])
        row_first.append((movie_indices, ratings))

    for movie_idx in range(col_size):
        rav = np.ravel(np.nonzero(data[:, movie_idx]))
        # if len(rav) == 0:
        #     continue
        user_indices = list(rav)
        ratings = list(data[user_indices, movie_idx])
        col_first.append((user_indices, ratings))
    print("done")
    return {keys_row_first: row_first, keys_col_first: col_first}


def getXinCanonical(data, len_can):
    num_here = 0
    for x in range(data.shape[0]):
        if (data[x, :len_can] > 0).sum() > 0:
            num_here += 1
    print("wat, ", num_here)
    return num_here


def getNeighbours(full_data, percentiles=[.01, .01, .02, .03, .04, .05, .10, .20]):
    user_results = []
    for percent in percentiles:
        num_canonicals = int(np.ceil(full_data.shape[1] * percent))
        can_idx = get_canonical_indices(full_data,
                                        [num_canonicals, num_canonicals])  # The one we aren't testing doesn't amtter

        # Resort data so that canonical users and movies are in top left
        full_data = full_data[:, can_idx[1]]
        full_data = full_data[can_idx[0], :]
        user_results.append(getXinCanonical(full_data, num_canonicals) / float(full_data.shape[0]))
    plt.plot(percentiles, user_results)
    plt.show()
    return user_results

keys_row_first = "row"
keys_col_first = "column"
keys_movie_to_user_net = "MovieToUser"
keys_user_to_movie_net = "UserToMovie"
keys_row_latents = "RowLatents"
keys_col_latents = "ColLatents"
keys_rating_net = "PredNet"
