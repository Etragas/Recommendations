import argparse
import sklearn.utils
import math

import torch
import torch.nn as nn

from NonZeroHero import non_zero_hero
from dl import *
from train import train
from utils import get_canonical_indices, splitDOK, removeZeroRows, build_params, dropDataFromRows
from losses import rmse
from model import get_predictions_tensor, dataCallback, get_predictions

def parseArgs():
    # Handle loading in previously trained weights.
    parser = argparse.ArgumentParser(description='Handle previous files')
    parser.add_argument('file', metavar='File path', nargs='?', help='Use a file for weights')
    args = parser.parse_args()
    print("Args File: ", args.file)
    return args


if __name__ == "__main__":

    # Set model parameters
    args = parseArgs()
    numUserProto = 50
    numItemProto = 50
    num_epochs = 10
    batch_size = 1000
    optimizer = None
    epoch = 0

    # Load the data using DataLoader
    # full_data = dl().LoadData(file_path="Data/netflix/user_first.txt", data_type=dl.NETFLIX, size= (490000,18000))
    # full_data = dl().LoadData(file_path="Data/ml-10m/ratingsbetter.dat", data_type=dl.MOVIELENS, size= (72000,11000))
    # DataLoader().fixMovelens100m('../Data/ml-1m/ratings.dat')
    # full_data = dl().LoadData(file_path="Data/ml-1m/ratingsbetter.dat", data_type=dl.MOVIELENS, size=(6100, 4000))
    full_data = dl().LoadData(file_path="../Data/ml-100k/u.data", data_type=dl.MOVIELENS,
                              size=(1200, 2000))

    # Reduce the matrix to toy size
    # full_data = full_data[:100, :100]
    print("Data shape: ", full_data.shape)

    # Print some statistics
    print("Average Ratings per User: ", np.mean(np.sum(full_data > 0, axis=1)))

    # Clean empty rows from the dataset
    full_data = removeZeroRows(full_data)
    randomRows = sklearn.utils.shuffle(range(full_data.shape[0]))
    randomCols = sklearn.utils.shuffle(range(full_data.shape[1]))
    full_data = full_data[randomRows, :]
    full_data = full_data[:, randomCols]

    # Scalability experiments
    decay = 0.0000001
    scalability = False
    if scalability:
        percent_keep = 1
        batch_size = int(batch_size * percent_keep)
        keep_rows = np.random.randint(0, full_data.shape[0], int(percent_keep * full_data.shape[0]))
        full_data = full_data[keep_rows, :]
    pmf = False
    if pmf:
        numUserProto, numItemProto = full_data.shape
        decay = .1
    print(full_data.shape)

    # Determine number of latents for movie/user
    print("Cleaned Data Shape: ", full_data.shape)
    nrows, ncols = full_data.shape

    print("Number of User Prototypes: {} \nNumber of Movie Prototypes: {}".format(numUserProto, numItemProto))
    # can_idx holds two arrays - they are of canonical indices for users and movies respectively.
    can_idx = get_canonical_indices(full_data, [numUserProto, numItemProto])

    # Resort data so that canonical users and movies are in top left
    print("Mean of prototype block pre sorting {}".format(np.mean(full_data[:numUserProto, :numItemProto])))
    full_data = full_data.tocsc()[:, can_idx[1]]
    full_data = full_data.tocsr()[can_idx[0], :]
    full_data = full_data.todok()
    print("Mean of prototype block post sorting {}".format(np.mean(full_data[:numUserProto, :numItemProto])))

    density_drop = False
    if density_drop:
        percent_keep = .2
        print("Densities pre drop")
        print("Region 1 {}".format(np.mean(np.sum(full_data[numUserProto:, numItemProto:] > 0, axis=1))))
        print("Region 2 {}".format(np.mean(np.sum(full_data[numUserProto:, :numItemProto] > 0, axis=1))))
        print("Region 3 {}".format(np.mean(np.sum(full_data[:numUserProto, numItemProto:] > 0, axis=0))))
        print("Region 4 {}".format(np.mean(np.sum(full_data[:numUserProto, :numItemProto] > 0, axis=1))))
        full_data[:numUserProto, numItemProto:], _ = splitDOK(full_data[:numUserProto, numItemProto:],
                                                              trainPercentage=percent_keep)
        full_data[numUserProto:, :numItemProto], _ = splitDOK(full_data[numUserProto:, :numItemProto],
                                                              trainPercentage=percent_keep)
        print("Desnities post drop")
        print("Region 1 {}".format(np.mean(np.sum(full_data[numUserProto:, numItemProto:] > 0, axis=1))))
        print("Region 2 {}".format(np.mean(np.sum(full_data[numUserProto:, :numItemProto] > 0, axis=1))))
        print("Region 3 {}".format(np.mean(np.sum(full_data[:numUserProto, numItemProto:] > 0, axis=0))))
        print("Region 4 {}".format(np.mean(np.sum(full_data[:numUserProto, :numItemProto] > 0, axis=1))))

    # Split full dataset into train and test sets.
    train_data, test_data = splitDOK(full_data, trainPercentage=.8)
    train_data = non_zero_hero(train_data)
    test_data = non_zero_hero(test_data)
    train_data.freeze_dataset()
    test_data.freeze_dataset()

    num_rows = train_data.shape[0]
    num_cols = train_data.shape[1]
    train_data_online = non_zero_hero(train_data[:int(num_rows/5), :int(num_cols/5)])
    test_data_online = non_zero_hero(test_data[:int(num_rows/5), :int(num_cols/5)])
    train_data_online.freeze_dataset()
    test_data_online.freeze_dataset()
    #print("train data online: ", train_data_online, train_data_online.shape)
    #print("test data online: ", test_data_online, test_data_online.shape)
    #input()

    cold_start = False
    drop_rows = None
    if cold_start:
        print("Pre drop matrix sum", np.sum(full_data))
        num_drop_rows = 150
        keep_rows = np.random.randint(0, train_data.shape[0], num_drop_rows)
        dropDataFromRows(data=train_data, rows=keep_rows)
        print("Post drop matrix sum", np.sum(full_data))
    # plt.imshow(full_data.todense(), cmap='hot', interpolation='nearest')
    # plt.show()

    print(
        "Mean of prototype block post sorting after split {}".format(np.mean(train_data[:numUserProto, :numItemProto])))

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
    parameters = train(train_data_online, test_data_online, parameters=parameters, optimizer=optimizer, dropped_rows=drop_rows,
                       initialIteration=epoch, num_epochs=num_epochs, weight_decay=decay, batch_size=batch_size,
                       pretrain=not pmf)
    
    loss_function = nn.MSELoss(size_average=False)
    for i in range(1,6):
      num_online_rows = int(num_rows/5 * i)
      num_online_cols = int(num_cols/5 * i)
      test_data_online = test_data[:num_online_rows, :num_online_cols]
      idxData = np.array([(k[0], k[1], float(v)) for k, v in test_data_online.items()])
      online_rows = idxData[:, 0].astype(int)
      online_cols = idxData[:, 1].astype(int)
      online_values = torch.FloatTensor(idxData[:, 2]).unsqueeze(1)
      online_indices = list(zip(online_rows, online_cols))

      train_data_online = non_zero_hero(train_data[:num_online_rows, :num_online_cols])
      train_data_online.freeze_dataset()
      pred = get_predictions_tensor(parameters, data=train_data_online, indices=online_indices)
      data_loss = loss_function(pred, online_values)
      print("Data loss is: ", math.sqrt(data_loss/len(pred)), " for percent of data: ", len(test_data_online) / len(test_data))

    # Store the trained parameters for future use.
    filename = "final_trained_parameters.pkl"
    pickle.dump(parameters, open(filename, 'wb'))
