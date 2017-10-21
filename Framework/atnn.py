import numpy as np
import torch
import utils
from DataLoader import *
from train import *
from utils import build_params, keys_rating_net, keys_row_latents, keys_col_latents, get_canonical_indices, \
    splitData

# Load the data using DataLoader
# full_data = DataLoader().LoadData(file_path="../Data/download/user_first.txt", data_type=DataLoader.NETFLIX, size= (490000,18000))
# full_data = DataLoader().LoadData(file_path="../Data/ml-10m/ratingsbetter.dat", data_type=DataLoader.MOVIELENS, size= (72000,11000))
# full_data = DataLoader().LoadData(file_path="../Data/ml-1m/ratingsbetter.dat", data_type=DataLoader.MOVIELENS, size= (6100,4000))

full_data = DataLoader().LoadData(file_path="../Data/ml-100k/u.data", data_type=DataLoader.MOVIELENS, size=(1200, 2000))
print(full_data.shape)
print(np.mean(np.sum(full_data > 0, axis=1)))  # Check average ratings per user
# Reduce the matrix to toy size
# full_data = full_data[:100,:100]
rows = [x for x in range((full_data.shape[0])) if full_data[x, :].sum() > 0]
cols = [x for x in range((full_data.shape[1])) if full_data[:, x].sum() > 0]
full_data = full_data[rows, :]
full_data = full_data[:, cols]

# Determine number of latents for movie/user
print(full_data.shape)
nrows, ncols = full_data.shape

num_user_latents = 20  # int(np.ceil(full_data.shape[0]))
num_movie_latents = 20  # int(np.ceil(full_data.shape[1]))

print(utils.num_user_latents, utils.num_movie_latents)
can_idx = get_canonical_indices(full_data, [utils.num_user_latents, utils.num_movie_latents])

# Resort data so that canonical users and movies are in top left
full_data = full_data[:, can_idx[1]]
full_data = full_data[can_idx[0], :]

train_data, test_data = splitData(full_data)
train_idx = test_idx = np.array([np.array(range(nrows)), np.array(range(ncols))])

# Training Parameters
step_size = 0.001
num_users_per_batch = 100
batches_per_epoch = int(np.ceil(float(nrows) / num_users_per_batch))
batches_per_can_epoch = int(np.ceil(float(utils.num_user_latents) / num_users_per_batch))

num_epochs = 40
hyperp1 = [step_size * 10, 40, batches_per_can_epoch]
hyperp2 = [step_size * 10, 40, batches_per_can_epoch]
hypert = [step_size, num_epochs, batches_per_epoch]

# Build the dictionary of parameters for the nets, etc.
parameters = build_params(num_user_latents, num_movie_latents)
collatent = parameters[keys_col_latents][:, 0]
rowllatent = parameters[keys_row_latents][0, :]
print(collatent)
print(rowllatent)
inputlatent = torch.cat((collatent, rowllatent), 0)
print(inputlatent)
y = parameters[keys_rating_net].forward(inputlatent)
print(y)
# Train the parameters.  Pretraining the nets and canon latents are optional.
parameters = train(train_data, test_data, can_idx, train_idx, test_idx, parameters,
                   p1=False, p1Args=hyperp1, p2=False, p2Args=hyperp2, trainArgs=hypert, use_cache=False)
