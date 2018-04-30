import datetime
import shutil
import time

import torch
from torch import Tensor

from src.NonZeroHero import non_zero_hero
from src.losses import rmse, mae, standard_loss, regularization_loss
from utils import *
from datetime import datetime

from src.utils import shuffleNonPrototypeEntries, RowIter, ColIter

"""
Initialize all non-mode-specific parameters
"""

curtime = time.time()
MAX_RECURSION = 4
trainingMode = True
EVIDENCELIMIT = 80

train_mse_iters = []
train_mse = []

filename = "intermediate_trained_parameters.pkl"
param_dict = {}

userDistances = {}
itemDistances = {}

rowLatents = 0
colLatents = 0
userNonZeroCache = {}
itemNonZeroCache = {}
hitcount = [0] * (MAX_RECURSION + 1)
NUM_USERS, NUM_MOVIES = 0, 0
userLatentCache, itemLatentCache = [None] * NUM_USERS, [None] * NUM_MOVIES
VOLATILE = False
BESTPREC = 0


def get_predictions(parameters, data: non_zero_hero, indices=None):
    """
    Computes the predictions for the specified users and movie pairs

    :param parameters: all the parameters in our model
    :param data: dictionary of the data in row form and column form
    :param indices: user and movie indices for which we generate predictions

    :return: rating predictions for all user/movie combinations specified.  If unspecified,
             computes all rating predictions.
    """
    setup_caches(data, parameters)
    full_predictions = {}

    if indices is None:
        print("Generating indices...")
        indices = data.get_random_indices(1000)

    for userIdx, itemIdx in indices:
        key = (userIdx, itemIdx)
        itemLatent = getItemEmbedding(parameters, data, itemIdx)[0]
        userLatent = getUserEmbedding(parameters, data, userIdx)[0]
        if (userLatent is None or itemLatent is None):
            full_predictions[key] = Variable(torch.FloatTensor([float(3.4)])).type(
                dtype)  # Assign an average rating
        else:
            full_predictions[key] = parameters[keys_rating_net].forward(torch.cat((userLatent, itemLatent), 0))
            # full_predictions[key] = Variable(torch.FloatTensor([float(3.4)]), volatile=VOLATILE).type(
            #   dtype)  # Assign an average rating
    ## torch.dot(userLatent, itemLatent)

    return full_predictions


def getUserEmbedding(parameters, data, userIdx, recursionStepsRemaining=MAX_RECURSION, caller_id=[[], []],
                     dist=MAX_RECURSION + 1):
    """
    Generate or retrieve the user latent.
    If it is a canonical, we retrieve it.
    If it is not a canonical, we generate it as a function of their ratings and the latents of their rated movies

    :param parameters: dictionary of all the parameters in our model
    :param data: The dataset
    :param userIdx: index of the user for which we want to generate a latent
    :param recursionStepsRemaining: the max recursion depth which we can generate latents from.
    :param caller_id: All the [user, movie] ancestors logged, which we check to avoid cycles

    :return: the predicted user latent
    """

    global userLatentCache, hitcount, trainingMode, EVIDENCELIMIT, numUserEmbeddings, VOLATILE, userDistanceCache

    # Get our necessary parameters from the parameters dictionary
    movie_to_user_net = parameters[keys_movie_to_user_net]

    # If user is canonical, return their latent immediately and cache it.
    if userIdx < numUserEmbeddings:
        rowLatents = parameters[keys_row_latents]
        userDistances[0].add(userIdx)
        return rowLatents[userIdx, :], 0

    # If user latent is cached at a height greater than or equal to current, return their latent immediately
    if userLatentCache[userIdx] is not None and userLatentCache[userIdx][1] >= recursionStepsRemaining:
        hitcount[userLatentCache[userIdx][1]] += 1
        return userLatentCache[userIdx][0], 1

    # If we reached our recursion depth, return None
    if recursionStepsRemaining < 1:
        return None, MAX_RECURSION

    # Must generate latent

    # Initialize lists for our dense ratings and latents
    itemEmbbeddings = []
    # update the current caller_id with this user index appended
    callier_id_with_self = [caller_id[0] + [userIdx], caller_id[1]]
    # Retrieve latents for every user who watched the movie
    # Retrieve latents for every movie watched by user
    evidenceCount = 0
    evidenceLimit = EVIDENCELIMIT / (2 ** (MAX_RECURSION - recursionStepsRemaining))

    itemColumns = ColIter(userIdx,data,numUserEmbeddings)
    for itemIdx in itemColumns:
        itemRating = data[userIdx, itemIdx]
        # When it is training mode we use evidence count.
        # When we go over the evidence limit, we no longer need to look for latents
        if trainingMode and evidenceCount > evidenceLimit:
            break

            # If the item latent is valid, and does not produce a cycle, append it
        if itemIdx not in callier_id_with_self[1]:
            movie_latent, curr_depth = getItemEmbedding(parameters, data, itemIdx, recursionStepsRemaining - 1,
                                                        caller_id=callier_id_with_self)

            if movie_latent is not None:
                latentWithRating = torch.cat(
                    (movie_latent, Variable(torch.FloatTensor([float(itemRating)]).type(dtype))),
                    dim=0)
                itemEmbbeddings.append(latentWithRating)  # We got another movie latent
                evidenceCount += 1
                dist = min(dist, curr_depth)

    if (len(itemEmbbeddings) < 2):
        return None, MAX_RECURSION
    # Now have all latents, prepare for concatenations
    embeddingConversions = movie_to_user_net.forward(torch.stack(itemEmbbeddings, 0))  # Feed through NN
    row_latent = torch.mean(embeddingConversions, dim=0)
    userLatentCache[userIdx] = (row_latent, recursionStepsRemaining)
    # print "user: ", user_index, " has depth: ", recursion_depth, "and caller id: ", internal_caller, " and all callers are: ", zip(entries, data[user_index, entries]), " and ", data[user_index, :], " and dist is: ", dist + 1

    return row_latent, 1


def getItemEmbedding(parameters, data, itemIdx, recursionStepsRemaining=MAX_RECURSION, caller_id=[[], []],
                     dist=MAX_RECURSION + 1):
    """
    Generate or retrieve the movie latent.
    If it is a canonical, we retrieve it.
    If it is not a canonical, we generate it as a function of the ratings and the latents of its viewers

    :param parameters: dictionary of all the parameters in our model
    :param data: The dataset
    :param itemIdx: index of the movie for which we want to generate a latent
    :param recursionStepsRemaining: the max recursion depth which we can generate latents from.
    :param caller_id: All the [user, movie] ancestors logged, which we check to avoid cycles

    :return: the predicted movie latent
    """

    global itemLatentCache, hitcount, trainingMode, EVIDENCELIMIT, numItemEmbeddings, itemDistanceCache

    # Get our necessary parameters from the parameters dictionary
    user_to_movie_net_parameters = parameters[keys_user_to_movie_net]
    colLatents = parameters[keys_col_latents]


    # If movie is canonical, return their latent immediately and cache it.
    if itemIdx < numItemEmbeddings:
        itemDistances[0].add(itemIdx)
        return colLatents[:, itemIdx], 0

    # If movie latent is cached, return their latent immediately
    if itemLatentCache[itemIdx] is not None and itemLatentCache[itemIdx][1] <= recursionStepsRemaining:
        hitcount[itemLatentCache[itemIdx][1]] += 1
        return itemLatentCache[itemIdx][0], 1

    # If we reached our recursion depth, return None
    if recursionStepsRemaining < 1:
        return None, MAX_RECURSION

    # Must Generate Latent

    userEmbeddings = []
    # update the current caller_id with this movie index appended
    internal_caller = [caller_id[0], caller_id[1] + [itemIdx]]

    # Retrieve latents for every user who watched the movie
    evidenceCount = 0
    evidenceLimit = EVIDENCELIMIT / (2 ** (MAX_RECURSION - recursionStepsRemaining))

    userRows = RowIter(itemIdx,data,numItemEmbeddings)
    # for i in candidateRows:
    #         if len(userRows) > 2*evidenceLimit:
    #             break
    #         if data.get((i,itemIdx)):
    #             userRows.append(i)
    # userRows = []

    for userIdx in userRows:
        userRating = data[userIdx, itemIdx]
        # When it is training mode we use evidence count.
        # When we go over the evidence limit, we no longer need to look for latents
        if trainingMode and evidenceCount > evidenceLimit:

            break

        # If the user latent is valid, and does not produce a cycle, append it
        if userIdx not in internal_caller[0]:
            user_latent, curr_depth = getUserEmbedding(parameters, data, userIdx, recursionStepsRemaining - 1,
                                                       caller_id=internal_caller)
            if user_latent is not None:
                latentWithRating = torch.cat(
                    (user_latent, Variable(torch.FloatTensor([float(userRating)]).type(dtype))),
                    dim=0)
                userEmbeddings.append(latentWithRating)  # We got another movie latent
                evidenceCount += 1
                dist = min(dist, curr_depth)

    if (len(userEmbeddings) < 2):
        return None, MAX_RECURSION
    prediction = user_to_movie_net_parameters.forward(torch.stack(userEmbeddings, 0))  # Feed through NN
    targetItemEmbedding = torch.mean(prediction, dim=0)

    # Now we have all latents, prepare for concatenation
    itemLatentCache[itemIdx] = (
        targetItemEmbedding, recursionStepsRemaining)  # Cache the movie latent with the current recursion depth

    return targetItemEmbedding, 1


# Stolen from Mamy Ratsimbazafy
def save_checkpoint(state, is_best, filename='Weights/checkpoint{}.pth.tar'):
    """
    For the best results currently achieved, save the epoch, parameters,
    rmse_result, and optimizer

    :param state: a dictionary that stores the relevant parameters of the model to be saved
    :param is_best: a flag indicating whether or not this is the current best performance
    :param filename: the name of the file we store our checkpoint in
    """
    filename = filename.format(0)
    torch.save(state, filename)  # TODO: Clarify whether or not this should go inside the if statement
    if is_best:
        shutil.copyfile(filename, 'Weights/model_best.pth.tar')


def print_perf(params, iter=0, gradient={}, train: non_zero_hero=None, test:non_zero_hero=None, userDistances={}, itemDistances={}, optimizer=None):
    """
    Prints the performance of the model every ten iterations, in terms of MAE, RMSE, and Loss.
    Also includes graphing functionalities.

    :param params: the dictionary of the parameters of the model
    :param iter: the current iteration number
    :param gradient: the calculated gradient, for sanity checking purposes
    :param train: the training dataset, None by default
    :param test: the test dataset, None by default
    :param optimizer: the optimizer we are using for the model, None by default
    """
    global curtime, hitcount, trainingMode, filename, param_dict, VOLATILE, BESTPREC

    print("Iteration ", iter)

    # Print gradients for testing.
    # print("Gradients are: ", gradient)
    if (iter % 4 != 0):
        return
    torch.set_grad_enabled(False)
    trainingMode = True

    print("It took: {} seconds".format(time.time() - curtime))
    print("The time is {}".format(datetime.now()))
    pred = get_predictions(params, data=train, indices=train.get_random_indices(500))
    mae_result = mae(gt=train, pred=pred)
    rmse_result = rmse(gt=train, pred=pred)
    loss_result = standard_loss(parameters=params, data=train, predictions=pred) + regularization_loss(
        parameters=params)
    print("MAE is", mae_result.data[0])
    print("RMSE is ", rmse_result)
    print("Loss is ", loss_result)
    if (test is not None):
        print("Printing performance for test:")
        test_indices = test.get_random_indices(5000)
        test_pred = get_predictions(params, train, indices=test_indices)
        test_rmse_result = rmse(gt=test, pred=test_pred)
        print("Test RMSE is ", test_rmse_result)
    for k, v in params.items():
        print("Key is: ", k)
        if type(v) == Tensor:
            print("Latent Variable Gradient Analytics")
            if (v.grad is None):
                print("ERROR - GRADIENT MISSING")
                continue
            flattened = v.grad.view(v.grad.nelement())
            avg_square = torch.sum(torch.pow(v.grad, 2)) / flattened.size()[0]
            median = torch.median(torch.abs(flattened))
            print("\t average of squares is: ", avg_square)
            print("\t median is: ", median)
        else:
            print("Neural Net Variable Gradient Analytics")
            for param in v.parameters():
                if param.grad is None:
                    print("ERROR - GRADIENT MISSING")
                    continue
                flattened = param.grad.view(param.grad.nelement())
                avg_square = torch.sum(torch.pow(param.grad, 2)) / flattened.size()[0]
                median = torch.median(torch.abs(flattened))
                print("\t average of squares is: ", avg_square)
                print("\t median is: ", median)

    print("Hitcount is: ", hitcount, sum(hitcount))
    print("Number of users per distance", {key: len(value) for (key, value) in userDistances.items()})
    print("User average distance to prototypes: ",
          np.mean(list(map(lambda keyValue: len(keyValue[1]) * keyValue[0], userDistances.items()))))
    print("Number of movies per distance: ", {key: len(value) for (key, value) in itemDistances.items()})
    print("Movie average distance to prototypes: ",
          np.mean(list(map(lambda keyValue: len(keyValue[1]) * keyValue[0], itemDistances.items()))))
    if (iter % 20 == 0):
        is_best = False
        if (test_rmse_result < BESTPREC):
            BESTPREC = test_rmse_result
            is_best = True
        save_checkpoint({
            'epoch': iter + 1,
            'params': params,
            'best_prec1': test_rmse_result,
            'optimizer': optimizer,
        }, is_best)

    torch.set_grad_enabled(True)
    trainingMode = True
    curtime = time.time()
    # train_mse.append(rmse_result.data[0])
    # train_mse_iters.append(iter)
    # if len(train_mse) % 10 == 0:
    #     print("Performance Update (every 10 iters): ", train_mse)

    # plt.scatter(train_mse_iters, train_mse, color='black')

    # plt.plot(train_mse_iters, train_mse)
    # plt.title('MovieLens 100K Performance (with pretraining)')
    # plt.xlabel('Iterations')
    # plt.ylabel('RMSE')
    # plt.draw()
    # plt.pause(0.001)
    # if len(train_mse)%10 == 0:
    #  #End the plotting with a raw input
    #  plt.savefig('finalgraph.png')
    #  print("Final Total Performance: ", train_mse)


def dataCallback(data, test=None):
    return lambda params, iter, grad, optimizer: print_perf(params, iter, grad, train=data,
                                                            test=test,
                                                            userDistances=userDistances,
                                                            itemDistances=itemDistances,
                                                            optimizer=optimizer)


def setup_caches(data, parameters):
    """
    Initializes model attributes and the cache for user and movie latents

    :param data: the dataset
    :param parameters: the dictionary of parameters of our model
    """
    global NUM_USERS, NUM_MOVIES, numUserEmbeddings, numItemEmbeddings
    # NUM_USERS, NUM_MOVIES = list(map(lambda x: len(set(x)), data.nonzero()))
    NUM_USERS, NUM_MOVIES = data.shape
    numUserEmbeddings = parameters[keys_row_latents].size()[0]
    numItemEmbeddings = parameters[keys_col_latents].size()[1]

    wipe_caches()


def wipe_caches():
    """
    Initializes caches for user and movie latents
    """
    global userLatentCache, itemLatentCache, userDistanceCache, itemDistanceCache
    global hitcount
    hitcount = [0] * (MAX_RECURSION + 1)
    userLatentCache = [None] * NUM_USERS
    itemLatentCache = [None] * NUM_MOVIES
    userDistanceCache = [None] * NUM_USERS
    itemDistanceCache = [None] * NUM_MOVIES
    for i in range(MAX_RECURSION + 1):
        userDistances[i] = set()
        itemDistances[i] = set()
