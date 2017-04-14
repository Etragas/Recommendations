from autograd.optimizers import adam

import NMF_ATNN
from NMF_ATNN import *
from utils import *


def pretrain_canon_and_rating(full_data, can_idx, parameters, step_size, num_epochs, batches_per_epoch):
    '''
    Pretrains the canonical latents and the weights of the combiner net.

    :param full_data: the entire dataset, which we need to reference in order to create canon set
    :param can_usr_idx, can_mov_idx: the indices of the canonical users and latents in the full data
    :param parameters: the initial parameter configuration
    :param step_size, num_epochs: hyperparamters for our training

    :return: updated canonical latent parameters and combiner net parameters.
    '''

    # Create our canonical from given indices and the full data
    train = listify(full_data[np.ix_(*can_idx)])
    grads = NMF_ATNN.lossGrad(train)
    # Optimize our parameters using adam
    parameters = adam(grads, parameters, step_size=step_size, num_iters=batches_per_epoch*num_epochs,
                      callback=NMF_ATNN.dataCallback(train))

    return parameters


def pretrain_combiners(full_data, can_idx, parameters, step_size, num_epochs, batches_per_epoch):
    '''
    Pretrains the weights of the rowless and columnless nets.

    :param full_data: the entire dataset, which we need to reference in order to create canon set
    :param can_usr_idx, can_mov_idx: the indices of the canonical users and latents in the full data
    :param parameters: the initial parameter configuration
    :param step_size, num_epochs: hyperparamters for our training

    :return: updated rowless net and columnless net parameters
    '''

    # Create a quadrupled canonical set using a clever fill_in_gaps call
    train = fill_in_gaps(can_idx, can_idx, full_data)
    # Initialize a zeroed array of equal size to our canonical set
    zeros = np.zeros((num_user_latents, num_movie_latents))
    # Set the first and third quadrants of the quadrupled canonical graph to zero.  Set up for clever trickery.
    train[:num_user_latents, :num_movie_latents] = np.array(0)
    train[num_user_latents:, num_movie_latents:] = np.array(0)
    train = listify(train)
    grads = NMF_ATNN.lossGrad(train)
    # Optimize our parameters using adam
    parameters = adam(grads, parameters, step_size=step_size, num_iters=num_epochs*batches_per_epoch,
                      callback=NMF_ATNN.dataCallback(train))

    return parameters


def train(train_data, test_data, can_idx=None, train_idx=None, test_idx=None, parameters=None, p1=False, p1Args=[.001, 2,1],
          p2=False, p2Args=[.001, 2, 1], trainArgs=[.001, 2, 1]):
    '''
    Trains ALL THE THINGS.  Also optionally performs pretraining on the canonicals, rating net weights, rowless net weights, and
    columnless weights.  Prints out the train and test results upon terminination.

    :param train_data: the entire dataset, which we need to reference in order to create canon set
    :param sizes: an array that determines the size of the train set that takes shape [user_size, movie_size]
    :param parameters: the initial parameter configuration
    :param p1, p2: Indicators representing whether or not we perform the corresponding pretraining
    :param p1Args, p2Args, trainArgs: hyperparamters for the training of each net

    :return: final trained parameters
    '''

    # Generate the indices for the canonical users and canonical movies

    if p1:
        # Perform pretraining on the canonicals and rating net
        parameters = pretrain_canon_and_rating(train_data, can_idx, parameters, *p1Args)

    if p2:
        # Perform pretraining on the columnless and rowless nets
        parameters = pretrain_combiners(train_data, can_idx, parameters, *p2Args)

    # Create our training matrix with canonicals using fill_in_gaps
    train_data = listify(fill_in_gaps(can_idx, train_idx, train_data))
    # Create our test matrix with canonicals using fill_in_gaps
    test_data = listify(fill_in_gaps(can_idx, test_idx, test_data))
    # Define the loss for our train
    grads = lossGrad(train_data,num_batches=trainArgs[2])

    # Optimize our parameters using adam
    parameters = adam(grads, parameters, step_size=trainArgs[0], num_iters=trainArgs[1],
                      callback=dataCallback(train_data, test_data), b1=.8)

    # Generate our rating predictions on the train set from the trained parameters and print performance and comparison
    invtrans = getInferredMatrix(parameters, train_data)
    print "\n".join([str(x) for x in ["Train", print_perf(parameters, train=train_data), train_data, np.round(invtrans)]])

    # Generate our rating predictions on the test set from the trained parameters and print performance and comparison
    invtrans = getInferredMatrix(parameters, test_data)
    print "\n".join([str(x) for x in ["Test", print_perf(parameters, train=test_data), test_data, np.round(invtrans)]])

    return parameters
