import argparse
import pickle

parser = argparse.ArgumentParser(description='Handle preivous files')
parser.add_argument('file', metavar='File path', nargs='?',
                    help='Use  a file for weights')
args = parser.parse_args()
print(args.file)
import numpy as np
# mkl.set_num_threads(4)
# os.environ['OMP_NUM_THREADS'] = '{:d}'.format(4)

import torch
from DataLoader import *
from train import *
from utils import build_params, get_canonical_indices, \
    splitData

# Load the data using DataLoader
# full_data = DataLoader().LoadData(file_path="../Data/download/user_first.txt", data_type=DataLoader.NETFLIX, size= (490000,18000))
# full_data = DataLoader().LoadData(file_path="../Data/ml-10m/ratingsbetter.dat", data_type=DataLoader.MOVIELENS, size= (72000,11000))
full_data = DataLoader().LoadData(file_path="../Data/ml-1m/ratingsbetter.dat", data_type=DataLoader.MOVIELENS,
                                  size=(6100, 4000))

# full_data = DataLoader().LoadData(file_path="../Data/ml-100k/u.data", data_type=DataLoader.MOVIELENS, size=(1200, 2000))
print(full_data.shape)

# print("Threads",torch.set_num_threads(4))
print("Threads", torch.get_num_threads())
print(np.mean(np.sum(full_data > 0, axis=1)))  # Check average ratings per user
# Reduce the matrix to toy size
num_user_latents = 150  # int(np.ceil(full_data.shape[0]))
num_movie_latents = 150  # int(np.ceil(full_data.shape[1]))

# full_data = full_data[:100,:100]
# num_user_latents = 20  # int(np.ceil(full_data.shape[0]))
# num_movie_latents = 20 # int(np.ceil(full_data.shape[1]))
rows = [x for x in range((full_data.shape[0])) if full_data[x, :].sum() > 0]
cols = [x for x in range((full_data.shape[1])) if full_data[:, x].sum() > 0]
full_data = full_data[rows, :]
full_data = full_data[:, cols]
d = np.add(1, 2)
# Determine number of latents for movie/user
print(full_data.shape)
nrows, ncols = full_data.shape

print(num_user_latents, num_movie_latents)
can_idx = get_canonical_indices(full_data, [num_user_latents, num_movie_latents])

# Resort data so that canonical users and movies are in top left
full_data = full_data[:, can_idx[1]]
full_data = full_data[can_idx[0], :]

train_data, test_data = splitData(full_data)
train_idx = test_idx = np.array([np.array(range(nrows)), np.array(range(ncols))])

# Training Parameters
step_size = 0.001
num_users_per_batch = 100

num_epochs = 40
# hypert = [step_size, num_epochs, batches_per_epoch]
hypert = [1, 1, 2]
optimizer = None
epoch = 0
# Build the dictionary of parameters for the nets, etc.
if args.file:
    filename = args.file
    state_dict = torch.load(filename)
    parameters = state_dict['params']
    optimizer = state_dict['optimizer']
    epoch = state_dict['epoch']
    "pre-trained epoch number: {}".format(epoch)
else:
    parameters = build_params(num_user_latents, num_movie_latents)
# y = parameters[keys_rating_net].forward(inputlatent)
# print(y)
# Train the parameters.  Pretraining the nets and canon latents are optional.
parameters = train(train_data, test_data, parameters=parameters, optimizer=optimizer, epoch=epoch)
filename = "final_trained_parameters.pkl"
pickle.dump(parameters, open(filename, 'wb'))
