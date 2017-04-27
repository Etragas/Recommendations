from autograd.optimizers import adam,sgd, rmsprop

import NMF_ATNN
from NMF_ATNN import *
import utils
import cPickle as pickle

def pretrain_canon_and_rating(full_data, parameters, step_size, num_epochs, batches_per_epoch):
    '''
    Pretrains the canonical latents and the weights of the combiner net.

    :param full_data: the entire dataset, which we need to reference in order to create canon set
    :param can_usr_idx, can_mov_idx: the indices of the canonical users and latents in the full data
    :param parameters: the initial parameter configuration
    :param step_size, num_epochs: hyperparamters for our training

    :return: updated canonical latent parameters and combiner net parameters.
    '''

    # Create our canonical from given indices and the full data
    train = full_data[:utils.num_user_latents,:utils.num_movie_latents].copy()
    train = listify(train)
    print "in p1 wtf", num_epochs, batches_per_epoch
    grads = NMF_ATNN.lossGrad(train, num_batches=batches_per_epoch, fixed_params=parameters, params_to_opt=[keys_col_latents,keys_row_latents,keys_rating_net], reg_alpha=1)
    # Optimize our parameters using adam
    parameters = adam(grads, parameters, step_size=step_size, num_iters=batches_per_epoch*num_epochs,
                      callback=NMF_ATNN.dataCallback(train), b1=0.5)
    print "training"
    return parameters


def pretrain_combiners(full_data, parameters, step_size, num_epochs, batches_per_epoch):
    '''
    Pretrains the weights of the rowless and columnless nets.

    :param full_data: the entire dataset, which we need to reference in order to create canon set
    :param can_usr_idx, can_mov_idx: the indices of the canonical users and latents in the full data
    :param parameters: the initial parameter configuration
    :param step_size, num_epochs: hyperparamters for our training

    :return: updated rowless net and columnless net parameters
    '''

    # Create a quadrupled canonical set using a clever fill_in_gaps call
    idx = [np.array(range(utils.num_user_latents)),np.array(range(utils.num_movie_latents))]
    train = fill_in_gaps(np.array(idx),np.array(idx),full_data)
    # Initialize a zeroed array of equal size to our canonical set
    zeros = np.zeros((utils.num_user_latents, utils.num_movie_latents))
    # Set the first and third quadrants of the quadrupled canonical graph to zero.  Set up for clever trickery.
    train[:utils.num_user_latents, :utils.num_movie_latents] = np.array(0)
    train[utils.num_user_latents:, utils.num_movie_latents:] = np.array(0)
    train = listify(train)
    grads = NMF_ATNN.lossGrad(train, num_batches=batches_per_epoch, fixed_params=parameters, params_to_opt=[keys_user_to_movie_net,keys_movie_to_user_net], reg_alpha=1)
    # Optimize our parameters using adam
    parameters = adam(grads, parameters, step_size=step_size, num_iters=num_epochs*batches_per_epoch,b1 = 0.5,
                      callback=NMF_ATNN.dataCallback(train))

    return parameters

def pretrain_all(full_data, parameters, step_size, num_epochs, batches_per_epoch):
    idx = [np.array(range(utils.num_user_latents)),np.array(range(utils.num_movie_latents))]
    train = fill_in_gaps(np.array(idx),np.array(idx),full_data)
    # Initialize a zeroed array of equal size to our canonical set
    # Set the first and third quadrants of the quadrupled canonical graph to zero.  Set up for clever trickery.
    train = listify(train)
    grads = NMF_ATNN.lossGrad(train, num_batches=batches_per_epoch, reg_alpha=1)
    # Optimize our parameters using adam
    parameters = adam(grads, parameters, step_size=step_size, num_iters=20,b1 = 0.5,
                      callback=NMF_ATNN.dataCallback(train), iter_val=2)

    return parameters

def train(train_data, test_data, can_idx=None, train_idx=None, test_idx=None, parameters=None, p1=False, p1Args=[.001, 2,1],
          p2=False, p2Args=[.001, 2, 1], trainArgs=[.001, 2, 1], use_cache = False):
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

    if not use_cache:
        # Generate the indices for the canonical users and canonical movies
        if p1:
            # Perform pretraining on the canonicals and rating net
            parameters = pretrain_canon_and_rating(train_data, parameters.copy(), *p1Args)

        if p2:
            # Perform pretraining on the columnless and rowless nets
            parameters = pretrain_combiners(train_data, parameters.copy(), *p2Args)

        parameters = pretrain_all(train_data, parameters.copy(), *p2Args)

        pickle.dump(parameters, open("parameters", "wb"))
    else:
        parameters = pickle.load( open( "parameters", "rb" ) )

    # Create our training matrix with canonicals using fill_in_gaps
    train_data = listify(train_data)
    # Create our test matrix with canonicals using fill_in_gaps
    test_data = listify(test_data)
    # Define the loss for our train

    # param_to_opt = [[keys_col_latents,keys_row_latents,keys_movie_to_user_net,keys_user_to_movie_net,keys_rating_net]]
    # #print raw_input("So it begins")
    # num_opt_passes = 100
    # for iter in range(num_opt_passes):
    #     batch_indices = (disseminate_values(shuffle(range(len(train_data[keys_row_first]))),trainArgs[2]))
    #     for param_keys in (param_to_opt):
    grads = NMF_ATNN.lossGrad(train_data, num_batches=trainArgs[2], reg_alpha=.01)
        # Optimize our parameters using adam
    parameters = adam(grads, parameters, step_size=trainArgs[0], num_iters=trainArgs[1]*trainArgs[2],
              callback=dataCallback(train_data, test_data), b1 = 0.9,iter_val=4)

    # Generate our rating predictions on the train set from the trained parameters and print performance and comparison
    invtrans = getInferredMatrix(parameters, train_data)
    print "\n".join([str(x) for x in ["Train", print_perf(parameters, train=train_data), train_data, np.round(invtrans)]])

    # Generate our rating predictions on the test set from the trained parameters and print performance and comparison
    invtrans = getInferredMatrix(parameters, test_data)
    print "\n".join([str(x) for x in ["Test", print_perf(parameters, train=test_data), test_data, np.round(invtrans)]])

    return parameters
