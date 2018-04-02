import argparse
import pickle

import numpy as np
import torch
from DataLoader import *
from scipy.sparse import dok_matrix
from train import *

from Framework.DataLoader import DataLoader
from Framework.utils import stripEmptyRowAndCols, get_canonical_indices, splitData


def parseArgs():
    # Handle loading in previously trained weights.
    parser = argparse.ArgumentParser(description='Handle previous files')
    parser.add_argument('file', metavar='File path', nargs='?', help='Use a file for weights')
    args = parser.parse_args()
    print("Args File: ", args.file)
    return args


if __name__ == "__main__":

    args = parseArgs()
    numUserProto = 25
    numItemProto = 25
    optimizer = None
    epoch = 0

    # Load the data using DataLoader
    # full_data = DataLoader().LoadData(file_path="../Data/download/user_first.txt", data_type=DataLoader.NETFLIX, size= (490000,18000))
    # full_data = DataLoader().LoadData(file_path="../Data/ml-10m/ratingsbetter.dat", data_type=DataLoader.MOVIELENS, size= (72000,11000))
    # DataLoader().fixMovelens100m('../Data/ml-1m/ratings.dat')
    # full_data = DataLoader().LoadData(file_path="../Data/ml-1m/ratingsbetter.dat", data_type=DataLoader.MOVIELENS, size= (6100,4000))
    full_data = DataLoader().LoadData(file_path="../Data/ml-100k/u.data", data_type=DataLoader.MOVIELENS,
                                       size=(1200, 2000))

    # Reduce the matrix to toy size
    # full_data = full_data[:100,:100]
    print("Data shape: ", full_data.shape)

    # Print some statistics
    print("Average Ratings per User: ", np.mean(np.sum(full_data > 0, axis=1)))

    # Clean empty rows from the dataset
    full_data = stripEmptyRowAndCols(full_data)
    # Determine number of latents for movie/user
    print("Cleaned Data Shape: ", full_data.shape)
    nrows, ncols = full_data.shape

    print("Number of User Prototypes: {} \n Number of Movie Prototypes: {}".format(numUserProto, numItemProto))
    # can_idx holds two arrays - they are of canonical indices for users and movies respectively.
    can_idx = get_canonical_indices(full_data, [numUserProto, numItemProto])
    # Resort data so that canonical users and movies are in top left
    full_data = full_data[:, can_idx[1]]
    full_data = full_data[can_idx[0], :]
    print(full_data)
    # Split full dataset into train and test sets.
    train_data, test_data = splitData(full_data, train_ratio=.9)
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
        parameters = build_params(numUserProto, numItemProto)

    # Train the parameters.
    parameters = train(train_data, test_data, parameters=parameters, optimizer=optimizer, initialIteration=epoch)

    # Store the trained parameters for future use.
    filename = "final_trained_parameters.pkl"
    pickle.dump(parameters, open(filename, 'wb'))
