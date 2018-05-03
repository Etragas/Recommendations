import argparse
import cProfile

from dl import *
from train import *
from train import train
from utils import get_canonical_indices, splitDOK

from src.NonZeroHero import non_zero_hero


def parseArgs():
    # Handle loading in previously trained weights.
    parser = argparse.ArgumentParser(description='Handle previous files')
    parser.add_argument('file', metavar='File path', nargs='?', help='Use a file for weights')
    args = parser.parse_args()
    print("Args File: ", args.file)
    return args


def main():
    args = parseArgs()
    optimizer = None
    epoch = 0

    # Load the data using DataLoader
    # full_data = dl().LoadData(file_path="Data/netflix/user_first.txt", data_type=dl.NETFLIX, size= (490000,18000))
    # full_data = dl().LoadData(file_path="Data/ml-10m/ratingsbetter.dat", data_type=dl.MOVIELENS, size= (72000,11000))
    # DataLoader().fixMovelens100m('../Data/ml-1m/ratings.dat')
    # full_data = dl().LoadData(file_path="Data/ml-1m/ratingsbetter.dat", data_type=dl.MOVIELENS, size= (6100,4000))
    full_data = dl().LoadData(file_path="Data/ml-100k/u.data", data_type=dl.MOVIELENS,size=(1200, 2000))

    numUserProto = int((full_data.shape[0]) / 943) * 40
    numItemProto = int((full_data.shape[1]) / 1650) * 40

    # Reduce the matrix to toy size
    # full_data = full_data[:100,:100]
    print("Data shape: ", full_data.shape)

    # Print some statistics
    print("Median Ratings Ratings per User: ", np.median(list(np.sum(full_data > 0, axis=1))))
    # Determine number of latents for movie/user
    print("Cleaned Data Shape: ", full_data.shape)

    print("Number of User Prototypes: {} \n Number of Movie Prototypes: {}".format(numUserProto, numItemProto))
    # can_idx holds two arrays - they are of canonical indices for users and movies respectively.
    can_idx = get_canonical_indices(full_data, [numUserProto, numItemProto])
    # Resort data so that canonical users and movies are in top left
    print("Mean of prototype block p1 sorting {}".format(np.mean(full_data[:numUserProto, :numItemProto])))
    full_data = full_data.tocsc()[:, can_idx[1]]
    full_data = full_data.tocsr()[can_idx[0], :]
    full_data = full_data.todok()

    # plt.imshow(full_data.todense(), cmap='hot', interpolation='nearest')
    # plt.show()

    print("Mean of prototype block post sorting {}".format(np.mean(full_data[:numUserProto, :numItemProto])))
    # Split full dataset into train and test sets.

    train_data, test_data = splitDOK(full_data, trainPercentage=.8)
    train_data = non_zero_hero(train_data)
    test_data = non_zero_hero(test_data)
    train_data.freeze_dataset()
    test_data.freeze_dataset()
    print("Mean of prototype block post sorting {}".format(np.mean(train_data[:numUserProto, :numItemProto])))

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
    return lambda: train(train_data, test_data, parameters=parameters, optimizer=optimizer, initialIteration=epoch)
    # Train the parameters.
    # Store the trained parameters for future use.
    filename = "final_trained_parameters.pkl"
    pickle.dump(parameters, open(filename, 'wb'))


if __name__ == "__main__":
    train_profile = main()
    # train_profile()
    cProfile.run('train_profile()', 'ml-1m-stats')
