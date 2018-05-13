import random
import scipy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy.sparse import dok_matrix
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch import Tensor
from itertools import chain
from sortedcontainers import SortedList
from NonZeroHero import non_zero_hero

movie_latent_size = 100
user_latent_size = 100
hyp_user_network_sizes = [movie_latent_size + 1, 200, 200, user_latent_size]
hyp_movie_network_sizes = [user_latent_size + 1, 200, 200, movie_latent_size]
rating_network_sizes = [movie_latent_size + user_latent_size, 200, 200, 200, 1]
scale = .1

dtype = torch.FloatTensor
if (torch.cuda.device_count() > 0):
    dtype = torch.cuda.FloatTensor  # Uncomment this to run on GPU

# The if statement checks if the code is running on gpu or cpu
if (dtype == torch.FloatTensor):
    class GeneratorNet(nn.Module):
        def __init__(self, latent_size=None):
            super(GeneratorNet, self).__init__()
            self.fc1 = nn.Linear(latent_size + 1, 200)
            # self.bn1 = torch.nn.BatchNorm1d(2000)
            self.fc2 = nn.Linear(200, 200)
            # self.bn2 = torch.nn.BatchNorm1d(2000)
            self.fc3 = nn.Linear(200, latent_size)
            # self.bn3 = torch.nn.BatchNorm1d(2000)
            initParams(self)

        def forward(self, x):
            x = F.relu(self.fc1(x))  # self.bn1(
            x = F.relu(self.fc2(x))  # self.bn2(
            x = self.fc3(x)  # self.bn3(
            return x


    class RatingGeneratorNet(nn.Module):
        def __init__(self):
            super(RatingGeneratorNet, self).__init__()
            self.fc1 = nn.Linear(user_latent_size + movie_latent_size, 200)
            # self.bn1 = torch.nn.BatchNorm1d(4000)
            self.fc2 = nn.Linear(200, 200)
            # self.bn2 = torch.nn.BatchNorm1d(2000
            self.fc3 = nn.Linear(200, 1)
            initParams(self)

        def forward(self, x):
            x = F.relu(self.fc1(x))  # self.bn1(
            x = F.relu(self.fc2(x))  # self.bn2(
            x = self.fc3(x)  # self.bn3(
            return x
else:
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

def getDictOfParams(parameters: dict):
    paramsToOptDict = {}
    for k, v in parameters.items():
        if type(v) == Tensor:
            paramsToOptDict[(k, k)] = v
        else:
            for subkey, param in v.named_parameters():
                paramsToOptDict[(k, subkey)] = param
    return paramsToOptDict

def initParams(net):
    for param in net.parameters():
        if (param.data.dim() > 1):
            param.data = torch.nn.init.xavier_uniform_(param.data)

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

def shuffleNonPrototypeEntries(entries: SortedList, prototypeThreshold):
    breakIdx = 0
    for entry in entries:
        if entry <= prototypeThreshold:
            breakIdx += 1
        else:
            break
    return chain(random.sample(range(0, breakIdx), breakIdx),
                 (random.randint(breakIdx, len(entries) - 1) for x in range(breakIdx, len(entries))))

def removeZeroRows(M):
    M = scipy.sparse.csr_matrix(M)
    M = M[M.getnnz(1) > 0][:, M.getnnz(0) > 0]
    return M.todok()


def splitDOK(data, trainPercentage):
    nonZero = shuffle(list(zip(*data.nonzero())))
    stop = int(trainPercentage * len(nonZero))
    testIdx = nonZero[stop:]
    testData = dok_matrix(data.shape)
    for idx in testIdx:
        testData[idx] = data[idx]
        del data[idx]
    return data, testData

def get_top_n(data, n):
    indices = np.ravel((data.astype(int)).flatten().argsort())[-n:]
    return indices

class RowIter():
    def __init__(self, itemIdx, data : non_zero_hero, numEmbeddings):
        self.itemIdx = itemIdx
        self.entries = data.get_non_zero(col=itemIdx).rows
        self.idx = shuffleNonPrototypeEntries(entries=self.entries, prototypeThreshold=numEmbeddings)

    def __iter__(self):
        for i in self.idx:
            yield self.entries[i]

class ColIter():
    def __init__(self, rowIdx, data : non_zero_hero, numEmbeddings):
        self.rowIdx = rowIdx
        self.entries = data.get_non_zero(row=rowIdx).cols
        self.idx = shuffleNonPrototypeEntries(entries=self.entries, prototypeThreshold=numEmbeddings)

    def __iter__(self):
        for j in self.idx:
            yield self.entries[j]

def clip_grads(params, clip=5):
    for k, v in params.items():
        if (v.grad is None):
            continue
        v.grad.data.clamp_(-clip, clip)

def mask_grad(params, keysToMask):
    for keys, value in params.items():
        superKey, subKey = keys
        if superKey in keysToMask:
            if (value.grad is None):
                continue
            value.grad.data.zero_()
    return

# TODO: Following functions are currently unused, please review
def getXinCanonical(data, len_can):
    num_here = 0
    for x in range(data.shape[0]):
        if (data[x, :len_can] > 0).sum() > 0:
            num_here += 1
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

# Credit to David Duvenaud for sleek init code
def init_random_params(scale, layer_sizes, rs=np.random.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [[scale * rs.randn(m, n),  # weight matrix
             scale * rs.randn(n)]  # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]


keys_row_first = "row"
keys_col_first = "column"
keys_movie_to_user_net = "MovieToUser"
keys_user_to_movie_net = "UserToMovie"
keys_row_latents = "RowLatents"
keys_col_latents = "ColLatents"
keys_rating_net = "PredNet"
