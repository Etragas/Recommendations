import argparse
import numpy as np
import pickle
import torch
from DataLoader import *
from train import *
from utils import build_params, get_canonical_indices, splitData

parser = argparse.ArgumentParser(description='Handle previous files')
parser.add_argument('file', metavar='File path', nargs='?',
                   help='Use a file for weights')
args = parser.parse_args()
print("Args File ", args.file)

# mkl.set_num_threads(4)
# os.environ['OMP_NUM_THREADS'] = '{:d}'.format(4)

# Parameters to set
num_user_canonicals = 150  # int(np.ceil(full_data.shape[0]))
num_movie_canonicals = 150 # int(np.ceil(full_data.shape[1]))
optimizer = None
epoch = 0


# Load the data using DataLoader
# full_data = DataLoader().LoadData(file_path="../Data/download/user_first.txt", data_type=DataLoader.NETFLIX, size= (490000,18000))
# full_data = DataLoader().LoadData(file_path="../Data/ml-10m/ratingsbetter.dat", data_type=DataLoader.MOVIELENS, size= (72000,11000))
# full_data = DataLoader().LoadData(file_path="../Data/ml-1m/ratingsbetter.dat", data_type=DataLoader.MOVIELENS, size= (6100,4000))
full_data = DataLoader().LoadData(file_path="../Data/ml-100k/u.data", data_type=DataLoader.MOVIELENS, size=(1200, 2000))

# Reduce the matrix to toy size
# full_data = full_data[:100,:100]
print("Data shape: ", full_data.shape)

# Print some statistics
print("Threads ",torch.set_num_threads(4))
print("Threads ",torch.get_num_threads())
print("Average Ratings per User: ", np.mean(np.sum(full_data > 0, axis=1)))

# Clean empty rows from the dataset
rows = [x for x in range((full_data.shape[0])) if full_data[x, :].sum() > 0]
cols = [x for x in range((full_data.shape[1])) if full_data[:, x].sum() > 0]
full_data = full_data[rows, :]
full_data = full_data[:, cols]

# Determine number of latents for movie/user
print("Cleaned Data Shape: ", full_data.shape)
nrows, ncols = full_data.shape

print("Number of User Canonicals: ", num_user_canonicals)
print("Number of Movie Canonicals: ", num_movie_canonicals)
# can_idx holds two arrays - they are of canonical indices for users and movies respectively.
can_idx = get_canonical_indices(full_data, [num_user_canonicals, num_movie_canonicals])
# Resort data so that canonical users and movies are in top left
full_data = full_data[:, can_idx[1]]
full_data = full_data[can_idx[0], :]

# Split full dataset into train and test sets.  Ratio is 0.8.
train_data, test_data = splitData(full_data)
train_idx = test_idx = np.array([np.array(range(nrows)), np.array(range(ncols))])

# If there is an arguments file, load our parameters from it.
# Otherwise build the dictionary of parameters for our nets and latents.
if args.file:
    filename = args.file
    state_dict = torch.load(filename)
    parameters = state_dict['params']
    optimizer = state_dict['optimizer']
    epoch = state_dict['epoch']
    "pre-trained epoch number: {}".format(epoch)
else:
    parameters = build_params(num_user_canonicals, num_movie_canonicals)

# Train the parameters.
parameters = train(train_data, test_data, parameters=parameters, optimizer = optimizer, epoch = epoch)

# Store the trained parameters for future use.
filename = "final_trained_parameters.pkl"
pickle.dump(parameters, open(filename, 'wb'))
