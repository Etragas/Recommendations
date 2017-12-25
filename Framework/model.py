import pickle
import numpy as np
import shutil
import torch
import time

from functools import reduce
from sklearn.utils import shuffle
from torch.autograd import Variable
from utils import *


"""
Initialize all non-mode-specific parameters
"""

curtime = time.time()
MAX_RECURSION = 4
TRAININGMODE = True
EVIDENCELIMIT = 80
RATINGLIMIT = 50

train_mse_iters = []
train_mse = []

filename = "intermediate_trained_parameters.pkl"
param_dict = {}

user_distances = {}
movie_distances = {}

def standard_loss(parameters, iter=0, data=None, indices=None, num_proc=1, num_batches=1, predictions=None):
    """
    Compute simplified version of squared loss

    :param parameters: Same as class parameter, here for autograd
    :param iter: CURRENTLY UNUSED.  0 by default
    :param data: The dataset.  None by default
    :param indices: The indices we compute our loss on.  None by default
    :param num_proc: CURRENTLY UNUSED.  Purpose unknown. 1 by default
    :param num_batches: CURRENTLY UNUSED. Number of batches, 1 by default

    :return: A scalar denoting the non-regularized squared loss
    """
    # Frobenius Norm squared error term
    # Regularization Terms

    global hitcount
    print("Hitcount is: ", hitcount)

    # generate predictions on specified indices with given parameters
    if predictions is None:
        predictions = inference(parameters, data=data, indices=indices)
    numel = len(predictions.keys())
    data_loss = numel * torch.pow(rmse(data, predictions), 2)

    return data_loss

def regularization_loss(parameters=None, paramsToOpt=None, reg_alpha=.01):
    """
    Computes the regularization loss, penalizing vector norms

    :param parameters: the parameters in our model
    :param paramsToOpt: the dictionary of parameters to optimize
    :param reg_alpha: the regularization term, .01 by default

    :return: A scalar denoting the regularization loss
    """
    reg_loss = reg_alpha * momentDiff(parameters, torch.mean)
    reg_loss += reg_alpha * momentDiff(parameters, torch.var)
    reg_loss += reg_alpha * computeWeightLoss(parameters)

    return reg_loss

def get_predictions(parameters, data, indices=None):
    """
    Computes the predictions for the specified users and movie pairs

    :param parameters: all the parameters in our model
    :param data: dictionary of the data in row form and column form
    :param indices: user and movie indices for which we generate predictions

    :return: rating predictions for all user/movie combinations specified.  If unspecified,
             computes all rating predictions.
    """
    global VOLATILE
    full_predictions = {}
    fail_keys = [] # Stores invalid pairs of user and movie latents
    good_keys = [] # Stores valid pairs of user and movie latents
    input_vectors = [] # Stores concatenated user and movie latents as input to the rating net

    setup_caches(data,parameters)
    for i in range(MAX_RECURSION + 1):
        user_distances[i] = set() 
        movie_distances[i] = set() 

    if indices is None:
        indices = shuffle(list(zip(*data.nonzero())))[:]
        print("Generating indices...")

    # Generates user and movie latents for every pair of rows and columns in indices
    for user_index, movie_index in indices:
        movieLatent = getMovieLatent(parameters, data, movie_index)[0]
        userLatent = getUserLatent(parameters, data, user_index)[0]
        if (userLatent is None or movieLatent is None):
            fail_keys.append((user_index, movie_index))
        else:
            good_keys.append((user_index, movie_index))
            input_vectors.append(torch.cat((movieLatent, userLatent), 0))
        # TODO: Refactor this using perform_inference
        # full_predictions[(user_index, movie_index)] = perform_inference(parameters, data, user_index, movie_index)
    # Feed through the rating net
    predictions = parameters[keys_rating_net].forward(torch.stack(input_vectors, 0))
    for idx, key in enumerate(good_keys):
        full_predictions[key] = predictions[idx]
    for key in fail_keys:
        full_predictions[key] = Variable(torch.FloatTensor([float(3.5)]),volatile=VOLATILE).type(dtype) # Assign an average rating

    return full_predictions


def perform_inference(parameters, data=None, user_index=0, movie_index=0):
    """
    Perform inference on the specifed user and movie.

    :param parameters: all the parameters in our model
    :param data: The dataset, None by default.
    :param user_index: The index of the user we want to generate a movie for
    :param movie_index: The index of the movie we want to generate a rating for

    :return: The predicted rating value for the specified user and movie
    """
    global VOLATILE
    # Generate user and movie latents
    movieLatent = getMovieLatent(parameters, data, movie_index)
    userLatent = getUserLatent(parameters, data, user_index)

    # Default value for the latents is arbitrarily chosen to be 3.5
    if movieLatent is None or userLatent is None:
        return Variable(torch.FloatTensor([float(3.5)]),volatile=VOLATILE).type(dtype)
    # Run through the rating net, passing in rating net parameters and the concatenated latents
    val = parameters[keys_rating_net].forward((torch.cat((movieLatent, userLatent), 0)))
    return val  # np.dot(np.array([1,2,3,4,5]),softmax())


def getUserLatent(parameters, data, user_index, recursion_depth=MAX_RECURSION, caller_id=[[], []], curr_dist=0):
    """
    Generate or retrieve the user latent.
    If it is a canonical, we retrieve it.
    If it is not a canonical, we generate it as a function of their ratings and the latents of their rated movies

    :param parameters: dictionary of all the parameters in our model
    :param data: The dataset
    :param user_index: index of the user for which we want to generate a latent
    :param recursion_depth: the max recursion depth which we can generate latents from.
    :param caller_id: All the [user, movie] ancestors logged, which we check to avoid cycles

    :return: the predicted user latent
    """

    global USERLATENTCACHE, hitcount, USERCACHELOCK, TRAININGMODE, EVIDENCELIMIT, UCANHIT, NUM_USER_LATENTS, VOLATILE, USERDISTANCECACHE

    # Get our necessary parameters from the parameters dictionary
    movie_to_user_net_parameters = parameters[keys_movie_to_user_net]
    rowLatents = parameters[keys_row_latents]

    dist = MAX_RECURSION + 1

    # If user is canonical, return their latent immediately and cache it.
    if user_index < NUM_USER_LATENTS:
        USERLATENTCACHE[user_index] = (rowLatents[user_index, :], 1)
        user_distances[0].add(user_index)
        return rowLatents[user_index, :], 0

    # If user latent is cached, return their latent immediately
    if USERLATENTCACHE[user_index] is not None and USERLATENTCACHE[user_index][1] >= recursion_depth:
        hitcount[USERLATENTCACHE[user_index][1]] += 1
        return USERLATENTCACHE[user_index][0], USERDISTANCECACHE[user_index]

    # If we reached our recursion depth, return None
    if recursion_depth < 1:
        return None, MAX_RECURSION

    # Must generate latent
    evidence_count = 0
    evidence_limit = EVIDENCELIMIT / (2 ** (MAX_RECURSION - recursion_depth))

    # items, ratings = get_candidate_latents(data[keys_row_first][user_index][get_items], data[keys_row_first][user_index][get_ratings], split=num_movie_latents)

    # Initialize lists for our dense ratings and latents
    input_latents = []
    # update the current caller_id with this user index appended
    internal_caller = [caller_id[0] + [user_index], caller_id[1]]
    # Retrieve latents for every user who watched the movie
    entries = data[user_index, :].nonzero()[0]
    can_entries = [x for x in entries if x < NUM_USER_LATENTS]
    uncan_entries = shuffle(list(set(entries)-set(can_entries)))
    entries = can_entries+uncan_entries
    # Retrieve latents for every movie watched by user
    for movie_index, rating in zip(entries, data[user_index, entries]):

        # When it is training mode we use evidence count.
        # When we go over the evidence limit, we no longer need to look for latents
        if TRAININGMODE and evidence_count > evidence_limit:
            break

            # If the movie latent is valid, and does not produce a cycle, append it
        if movie_index not in internal_caller[1]:
            movie_latent, curr_depth = getMovieLatent(parameters, data, movie_index, recursion_depth - 1, caller_id=internal_caller)

            if movie_latent is not None:
                latentWithRating = torch.cat((movie_latent, Variable(torch.FloatTensor([float(rating)]).type(dtype),volatile = VOLATILE)), dim=0)
                input_latents.append(latentWithRating)  # We got another movie latent
                evidence_count += 1
                dist = min(dist, curr_depth)

    if (len(input_latents) < 2):
        return None, MAX_RECURSION
    # Now have all latents, prepare for concatenations
    prediction = movie_to_user_net_parameters.forward(torch.stack(input_latents, 0))  # Feed through NN
    row_latent = torch.mean(prediction, dim=0)
    USERLATENTCACHE[user_index] = (row_latent, recursion_depth)
    # print "user: ", user_index, " has depth: ", recursion_depth, "and caller id: ", internal_caller, " and all callers are: ", zip(entries, data[user_index, entries]), " and ", data[user_index, :], " and dist is: ", dist + 1
    if USERDISTANCECACHE[user_index] is not None:
        user_distances[USERDISTANCECACHE[user_index]].remove(user_index)
        USERDISTANCECACHE[user_index] = min(dist + 1, USERDISTANCECACHE[user_index])
        user_distances[USERDISTANCECACHE[user_index]].add(user_index)
    else:
        user_distances[dist + 1].add(user_index)
        USERDISTANCECACHE[user_index] = dist + 1

    return row_latent, USERDISTANCECACHE[user_index]


def getMovieLatent(parameters, data, movie_index, recursion_depth=MAX_RECURSION, caller_id=[[], []], curr_dist = 0):
    """
    Generate or retrieve the movie latent.
    If it is a canonical, we retrieve it.
    If it is not a canonical, we generate it as a function of the ratings and the latents of its viewers

    :param parameters: dictionary of all the parameters in our model
    :param data: The dataset
    :param movie_index: index of the movie for which we want to generate a latent
    :param recursion_depth: the max recursion depth which we can generate latents from.
    :param caller_id: All the [user, movie] ancestors logged, which we check to avoid cycles

    :return: the predicted movie latent
    """

    global MOVIELATENTCACHE, hitcount, MOVIECACHELOCK, TRAININGMODE, EVIDENCELIMIT, MCANHIT, NUM_MOVIE_LATENTS, MOVIEDISTANCECACHE

    # Get our necessary parameters from the parameters dictionary
    user_to_movie_net_parameters = parameters[keys_user_to_movie_net]
    colLatents = parameters[keys_col_latents]

    dist = MAX_RECURSION + 1

    # If movie is canonical, return their latent immediately and cache it.
    if movie_index < NUM_MOVIE_LATENTS:
        MOVIELATENTCACHE[movie_index] = (colLatents[:, movie_index], 1)
        movie_distances[0].add(movie_index)
        return colLatents[:, movie_index], 0

    # If movie latent is cached, return their latent immediately
    if MOVIELATENTCACHE[movie_index] is not None and MOVIELATENTCACHE[movie_index][1] <= recursion_depth:
        hitcount[MOVIELATENTCACHE[movie_index][1]] += 1
        return MOVIELATENTCACHE[movie_index][0], MOVIEDISTANCECACHE[movie_index]

    # If we reached our recursion depth, return None
    if recursion_depth < 1:
        return None, MAX_RECURSION

    # Must Generate Latent
    evidence_count = 0
    evidence_limit = EVIDENCELIMIT / (2 ** (MAX_RECURSION - recursion_depth))
    # print evidence_limit
    # items, ratings = get_candidate_latents(data[keys_col_first][movie_index][get_items], data[keys_col_first][movie_index][get_ratings], split = num_user_latents)

    # Initialize lists for our dense ratings and latents
    dense_ratings, input_latents = [], []
    # update the current caller_id with this movie index appended
    internal_caller = [caller_id[0], caller_id[1] + [movie_index]]
    # Retrieve latents for every user who watched the movie
    entries = data[:, movie_index].nonzero()[0]
    can_entries = [x for x in entries if x < NUM_MOVIE_LATENTS]
    uncan_entries = shuffle(list(set(entries) - set(can_entries)))
    entries = can_entries + uncan_entries

    for user_index, rating in zip(entries, data[entries, movie_index]):
        # When it is training mode we use evidence count.
        # When we go over the evidence limit, we no longer need to look for latents
        if TRAININGMODE and evidence_count > evidence_limit:
            break

        # If the user latent is valid, and does not produce a cycle, append it
        if user_index not in internal_caller[0]:
            user_latent, curr_depth = getUserLatent(parameters, data, user_index, recursion_depth - 1, caller_id=internal_caller)
            if user_latent is not None:
                latentWithRating = torch.cat((user_latent, Variable(torch.FloatTensor([float(rating)]).type(dtype),volatile = VOLATILE)), dim=0)
                input_latents.append(latentWithRating)  # We got another movie latent
                evidence_count += 1
                dist = min(dist, curr_depth)

    if (len(input_latents) < 2):
        return None, MAX_RECURSION
    prediction = user_to_movie_net_parameters.forward(torch.stack(input_latents, 0))  # Feed through NN
    column_latent = torch.mean(prediction, dim=0)
    # print("Row lat")
    # # USERLATENTCACHE[user_index] = (row_latent, recursion_depth)
    #
    # return row_latent
    # Now we have all latents, prepare for concatenation
    MOVIELATENTCACHE[movie_index] = (column_latent, recursion_depth)  # Cache the movie latent with the current recursion depth
    # movie_distances[recursion_depth].add(movie_index)
    if MOVIEDISTANCECACHE[movie_index] is not None:
        movie_distances[MOVIEDISTANCECACHE[movie_index]].remove(movie_index)
        MOVIEDISTANCECACHE[movie_index] = min(dist + 1, MOVIEDISTANCECACHE[movie_index])
        movie_distances[MOVIEDISTANCECACHE[movie_index]].add(movie_index)
    else:
        movie_distances[dist + 1].add(movie_index)
        MOVIEDISTANCECACHE[movie_index] = dist + 1
    return column_latent, MOVIEDISTANCECACHE[movie_index]


def dataCallback(data, test=None):
    return lambda params, iter, grad, optimizer: print_perf(params, iter, grad, train=data, test=test, optimizer = optimizer)


def print_perf(params, iter=0, gradient={}, train=None, test=None, optimizer=None):
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
    global curtime, hitcount, TRAININGMODE, filename, param_dict, VOLATILE, BESTPREC

    print("Iteration ", iter)

    # Print gradients for testing.
    # print("Gradients are: ", gradient)
    if (iter % 10 != 0):
        return
    VOLATILE = True
    TRAININGMODE = True

    print("It took: {} seconds".format(time.time() - curtime))
    pred = inference(params, data=train, indices=shuffle(list(zip(*train.nonzero())))[:5000])
    mae_result = mae(gt=train, pred=pred)
    rmse_result = rmse(gt=train, pred=pred)
    loss_result = loss(parameters=params, data=train, predictions=pred)
    print("MAE is", mae_result.data[0])
    print("RMSE is ", rmse_result.data[0])
    print("Loss is ", loss_result.data[0])
    if (test is not None):
        print("Printing performance for test:")
        test_indices = shuffle(list(zip(*test.nonzero())))[:5000]
        test_pred = inference(params, train, indices=test_indices)
        test_rmse_result = rmse(gt=test, pred=test_pred)
        print("Test RMSE is ", (test_rmse_result.data)[0])
    for k, v in params.items():
        print("Key is: ", k)
        if type(v) == Variable:
            print("Latent Variable Gradient Analytics")
            if (v.grad is None):
                print("ERROR - GRADIENT MISSING")
                continue
            flattened = v.grad.view(v.grad.nelement())
            avg_square = torch.sum(torch.pow(v.grad, 2)) / flattened.size()[0]
            median = torch.median(torch.abs(flattened))
            print("\t average of squares is: ", avg_square.data[0])
            print("\t median is: ", median.data[0])
        else:
            print("Neural Net Variable Gradient Analytics")
            for param in v.parameters():
                if param.grad is None:
                    print("ERROR - GRADIENT MISSING")
                    continue
                flattened = param.grad.view(param.grad.nelement())
                avg_square = torch.sum(torch.pow(param.grad, 2)) / flattened.size()[0]
                median = torch.median(torch.abs(flattened))
                print("\t average of squares is: ", avg_square.data[0])
                print("\t median is: ", median.data[0])

    print("Hitcount is: ", hitcount, sum(hitcount))
    print("Number of movies per distance", {key: len(value) for (key, value) in user_distances.items()})
    print("User average distance to prototypes: ", sum(x for x in USERDISTANCECACHE if x is not None) / (1.0 * sum(x is not None for x in USERDISTANCECACHE)))
    print("Number of movies per distance: ", {key: len(value) for (key, value) in movie_distances.items()})
    print("Movie average distance to prototypes: ", sum(x for x in MOVIEDISTANCECACHE if x is not None) / (1.0 * sum(x is not None for x in MOVIEDISTANCECACHE)))
    if (iter % 20 == 0):
        is_best = False
        if (test_rmse_result.data[0] < BESTPREC):
            BESTPREC = test_rmse_result.data[0]
            is_best = True
        save_checkpoint({
            'epoch': iter+ 1,
            'params': params,
            'best_prec1': test_rmse_result,
            'optimizer' : optimizer,
        }, is_best)

    VOLATILE = False
    TRAININGMODE = True
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

#Stolen from Mamy Ratsimbazafy
def save_checkpoint(state, is_best, filename='Weights/checkpoint{}.pth.tar'):
    """
    For the best results currently achieved, save the epoch, parameters,
    rmse_result, and optimizer

    :param state: a dictionary that stores the relevant parameters of the model to be saved
    :param is_best: a flag indicating whether or not this is the current best performance
    :param filename: the name of the file we store our checkpoint in
    """
    filename = filename.format(0)
    torch.save(state, filename) # TODO: Clarify whether or not this should go inside the if statement
    if is_best:
        shutil.copyfile(filename, 'Weights/model_best.pth.tar')

def setup_caches(data,parameters):
    """
    Initializes model attributes and the cache for user and movie latents

    :param data: the dataset
    :param parameters: the dictionary of parameters of our model
    """
    global NUM_USERS, NUM_MOVIES, NUM_USER_LATENTS, NUM_MOVIE_LATENTS
    #NUM_USERS, NUM_MOVIES = list(map(lambda x: len(set(x)), data.nonzero()))
    NUM_USERS, NUM_MOVIES = data.shape
    NUM_USER_LATENTS = parameters[keys_row_latents].size()[0]
    NUM_MOVIE_LATENTS = parameters[keys_col_latents].size()[1]
    wipe_caches()


def wipe_caches():
    """
    Initializes caches for user and movie latents
    """
    global USERLATENTCACHE, MOVIELATENTCACHE, UCANHIT, MCANHIT, USERDISTANCECACHE, MOVIEDISTANCECACHE
    global hitcount
    hitcount = [0] * (MAX_RECURSION + 1)
    USERLATENTCACHE = [None] * NUM_USERS
    MOVIELATENTCACHE = [None] * NUM_MOVIES
    USERDISTANCECACHE = [None] * NUM_USERS
    MOVIEDISTANCECACHE = [None] * NUM_MOVIES


def rmse(gt, pred):
    """
    Computes the rmse given a ground truth and a prediction

    :param gt: the ground truth, a.k.a. the dataset
    :param pred: the predicted ratings

    :return: the root mean squared error between ground truth and prediction
    """
    diff = 0
    lens = (len(pred.keys()))
    mean = []
    for key in pred.keys():
        mean.append(pred[key])
        diff = diff + torch.pow(float(gt[key]) - pred[key], 2)
    print("Num of items is {} average pred value is {}".format(lens, np.mean(mean)))
    return torch.sqrt((diff / len(pred.keys())))

def mae(gt, pred):
    """
    Computes the mean absolute error given a ground truth and prediction

    :param gt: the ground truth, a.k.a. the dataset
    :param pred: the predicted ratings

    :return: the mean absolute error value between ground truth and prediction
    """
    val = 0
    for key in pred.keys():
        val = val + torch.abs(pred[key] - float(gt[key]))
    val = val / len(pred.keys())
    return val

def computeWeightLoss(parameters):
    global USERLATENTCACHE, MOVIELATENTCACHE
    regLoss = 0
    print("Reg loss")
    for k, v in parameters.items():
        if type(v) == Variable:
            print("Key is: ", k, "Value is: ", v.size())
            if k == keys_row_latents:
                regLoss += torch.sum(torch.pow(v.data,2))
                useMask = [1 if x is not None else 0 for x in USERLATENTCACHE[:v.size()[0]]]
                print(useMask)
                print(np.sum(useMask))
            else:
                useMask = [1 if x is not None else 0 for x in MOVIELATENTCACHE[:v.size()[1]]]
                print(useMask)
                print(np.sum(useMask))
                regLoss += torch.sum(torch.pow(v.data,2))
        else:
            for subkey, param in v.named_parameters():
                regLoss += torch.sum(torch.pow(param.data,2))
    return regLoss

def momentDiff(parameters,momentFn):
    colLatents = parameters[keys_col_latents]
    rowLatents = parameters[keys_row_latents]
    colLatntWithRating = torch.cat(
        (colLatents, Variable(3.3 * torch.FloatTensor(torch.ones((1, colLatents.size()[1]))).type(dtype))), dim=0)
    rowLatentsWithRating = torch.cat(
        (rowLatents, Variable(3.3 * torch.FloatTensor(torch.ones((rowLatents.size()[0], 1))).type(dtype))), dim=1)
    averagePredRow = momentFn(parameters[keys_movie_to_user_net].forward(torch.t(colLatntWithRating)), dim=1)
    averagePredCol = momentFn(parameters[keys_user_to_movie_net].forward(rowLatentsWithRating), dim=1)
    averageRow = momentFn(rowLatents, dim=1)
    averageCol = momentFn(colLatents, dim=0)
    meanDiffCol = torch.sum(torch.pow(averageCol - averagePredCol, 2))
    meanDiffRow = torch.sum(torch.pow(averageRow - averagePredRow, 2))
    print("Mean Diff Col {} and Mean Diff Row {}".format(meanDiffCol,meanDiffRow))
    meanDiff = (meanDiffCol + meanDiffRow)

    return meanDiff

rowLatents = 0
colLatents = 0

inference = get_predictions
loss = standard_loss
hitcount = [0] * (MAX_RECURSION + 1)
NUM_USERS = 0
NUM_MOVIES = 0
USERLATENTCACHE = [None] * NUM_USERS
MOVIELATENTCACHE = [None] * NUM_MOVIES
USERLATENTCACHEPRIME = [None] * NUM_USERS
MOVIELATENTCACHEPRIME = [None] * NUM_MOVIES
BESTPREC = 100
VOLATILE = False
