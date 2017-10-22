import os
import pickle
import time
from functools import reduce

import numpy as np
import shutil
import torch
from sklearn.utils import shuffle
from utils import *
import copy

from utils import keys_row_latents, keys_col_latents
from multiprocessing import Pool

from utils import keys_rating_net

"""
Initialize all non-mode-specific parameters
"""

curtime = 0
MAX_RECURSION = 4
TRAININGMODE = True
EVIDENCELIMIT = 80
RATINGLIMIT = 50

train_mse_iters = []
train_mse = []

filename = "intermediate_trained_parameters.pkl"
param_dict = {}


def standard_loss(parameters, iter=0, data=None, indices=None, num_proc=1, num_batches=1, reg_alpha=.01,
                  predictions=None):
    """
    Compute simplified version of squared loss with penalty on vector norms

    :param parameters: Same as class parameter, here for autograd
    :param iter: Currently Unused.  0 by default.
    :param data: The dataset.  None by default
    :param indices: The indices we compute our loss on.  None by default
    :param num_proc: UNKNOWN, APPARENTLY SOME SCALAR? 1 by default
    :param num_batches: Number of batches, IS THIS USED? 1 by default
    :param reg_alpha: The regularization term.  .01 by default

    :return: A scalar denoting the regularized squared loss
    """
    # Frobenius Norm squared error term
    # Regularization Terms

    # generate predictions on specified indices with given parameters
    if predictions is None:
        predictions = inference(parameters, data=data, indices=indices)
    global hitcount
    print(hitcount)
    numel = len(predictions.keys())

    data_loss = numel * torch.pow(rmse(data, predictions, indices), 2)
    # print("RMSE is",rmse(data,predictions,indices))
    # canonicals = [parameters[keys_col_latents],parameters[keys_row_latents]]
    # combiners = [parameters[keys_movie_to_user_net], parameters[keys_user_to_movie_net]]
    # rating_net = parameters[keys_rating_net]
    # reg_loss = 0

    reg_loss = reg_alpha*momentDiff(parameters,torch.mean)
    reg_loss += reg_alpha*momentDiff(parameters,torch.var)
    # for reg,params in zip([.00001,.0001,.001],[canonicals,combiners,rating_net]):
    #     reg_loss = reg_loss + reg*np.square(flatten(params)[0]).sum() / float(num_proc)
    # # reg_loss = .0001 * canonicals
    # reg_loss = reg_alpha * np.abs(flatten(parameters)[0]).sum() / float(num_proc)
    return reg_loss + data_loss


def get_pred_for_users(parameters, data, indices=None, training_mode = False):
    """
    Computes the predictions for the specified users and movie pairs

    :param parameters: all the parameters in our model
    :param data: dictionary of our data in row form and column form
    :param indices: user and movie indices for which we generate predictions

    :return: rating predictions for all user/movie combinations specified.  If unspecified,
             computes all rating predictions.
    """
    diff = 0
    global VOLATILE
    setup_caches(data,parameters)

    row_size, col_size = data.shape
    # print(row_size,col_size)
    if indices is None:
        indices = shuffle(list(zip(*data.nonzero())))[:300]
        print("Shuffling")
    # Generate predictions over each row
    full_predictions = {}
    fail_keys = []
    good_keys = []
    input_vectors = []
    for user_index, movie_index in indices:
        movieLatent = getMovieLatent(parameters, data, movie_index)
        userLatent = getUserLatent(parameters, data, user_index)
        if (userLatent is None or movieLatent is None):
            fail_keys.append((user_index,movie_index))
        else:
            good_keys.append((user_index,movie_index))
            input_vectors.append(torch.cat((movieLatent, userLatent), 0))
        # full_predictions[user_index, movie_index] = recurrent_inference(parameters, data, user_index, movie_index)
    predictions = parameters[keys_rating_net].forward(torch.stack(input_vectors, 0))  # Feed through NN
    for idx, key in enumerate(good_keys):
        full_predictions[key] = predictions[idx]
    for key in fail_keys:
        full_predictions[key] = Variable(torch.FloatTensor([float(3.5)]),volatile=VOLATILE).type(dtype)
    return full_predictions


def recurrent_inference(parameters, data=None, user_index=0, movie_index=0):
    """
    Using our recurrent structure, perform inference on the specifed user and movie.

    :param parameters: all the parameters in our model
    :param iters: The current iteration number, 0 by default. IS THIS USED?
    :param data: The dataset, None by default.
    :param user_index: The index of the user we want to generate a movie for
    :param movie_index: The index of the movie we want to generate a rating for

    :return val: The predicted rating value for the specified user and movie
    """
    # Generate user and movie latents
    movieLatent = getMovieLatent(parameters, data, movie_index)
    userLatent = getUserLatent(parameters, data, user_index)

    # Default value for the latents is arbitrarily chosen to be 2.5
    if movieLatent is None or userLatent is None:
        return
    # Run through the rating net, passing in rating net parameters and the concatenated latents
    val = parameters[keys_rating_net].forward((torch.cat((movieLatent, userLatent), 0)))
    return val  # np.dot(np.array([1,2,3,4,5]),softmax())


def getUserLatent(parameters, data, user_index, recursion_depth=MAX_RECURSION, caller_id=[[], []]):
    """
    Generate or retrieve the user latent.
    If it is a canonical, we retrieve it.
    If it is not a canonical, we generate it as a function of their ratings and the latents of their rated movies

    :params parameters: dictionary of all the parameters in our model
    :params data: The dataset
    :params user_index: index of the user for which we want to generate a latent
    :params recursion_depth: the max recursion depth which we can generate latents from.
    :params caller_id: All the [user, movie] ancestors logged, which we check to avoid cycles

    :return: the predicted user latent
    """

    global USERLATENTCACHE, hitcount, USERCACHELOCK, TRAININGMODE, EVIDENCELIMIT, UCANHIT, NUM_USER_LATENTS, VOLATILE

    # Get our necessary parameters from the parameters dictionary
    movie_to_user_net_parameters = parameters[keys_movie_to_user_net]
    rowLatents = parameters[keys_row_latents]

    # If user is canonical, return their latent immediately and cache it.
    if user_index < NUM_USER_LATENTS:
        # USERLATENTCACHE[user_index] = (rowLatents[user_index, :], recursion_depth)
        return rowLatents[user_index, :]

    # If user latent is cached, return their latent immediately
    if USERLATENTCACHE[user_index] is not None and USERLATENTCACHE[user_index][1] >= recursion_depth:
        hitcount[USERLATENTCACHE[user_index][1]] += 1
        return USERLATENTCACHE[user_index][0]

    # If we reached our recursion depth, return None
    if recursion_depth < 1:
        return None

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
            movie_latent = getMovieLatent(parameters, data, movie_index, recursion_depth - 1, caller_id=internal_caller)

            if movie_latent is not None:
                latentWithRating = torch.cat((movie_latent, Variable(torch.FloatTensor([float(rating)]).type(dtype),volatile = VOLATILE)), dim=0)
                input_latents.append(latentWithRating)  # We got another movie latent
                evidence_count += 1

    if (len(input_latents) < 2):
        return None
    # Now have all latents, prepare for concatenations
    prediction = movie_to_user_net_parameters.forward(torch.stack(input_latents, 0))  # Feed through NN
    row_latent = torch.mean(prediction, dim=0)
    USERLATENTCACHE[user_index] = (row_latent, recursion_depth)

    return row_latent


def getMovieLatent(parameters, data, movie_index, recursion_depth=MAX_RECURSION, caller_id=[[], []]):
    """
    Generate or retrieve the movie latent.
    If it is a canonical, we retrieve it.
    If it is not a canonical, we generate it as a function of the ratings and the latents of its viewers

    :params parameters: dictionary of all the parameters in our model
    :params data: The dataset
    :params movie_index: index of the movie for which we want to generate a latent
    :params recursion_depth: the max recursion depth which we can generate latents from.
    :params caller_id: All the [user, movie] ancestors logged, which we check to avoid cycles

    :return: the predicted movie latent
    """

    global MOVIELATENTCACHE, hitcount, MOVIECACHELOCK, TRAININGMODE, EVIDENCELIMIT, MCANHIT, NUM_MOVIE_LATENTS

    # Get our necessary parameters from the parameters dictionary
    user_to_movie_net_parameters = parameters[keys_user_to_movie_net]
    colLatents = parameters[keys_col_latents]

    # If movie is canonical, return their latent immediately and cache it.
    if movie_index < NUM_MOVIE_LATENTS:
        # MOVIELATENTCACHE[movie_index] = (colLatents[:, movie_index], recursion_depth)
        return colLatents[:, movie_index]

    # If movie latent is cached, return their latent immediately
    if MOVIELATENTCACHE[movie_index] is not None and MOVIELATENTCACHE[movie_index][1] >= recursion_depth:
        hitcount[MOVIELATENTCACHE[movie_index][1]] += 1
        return MOVIELATENTCACHE[movie_index][0]

    # If we reached our recursion depth, return None
    if recursion_depth < 1:
        return None

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
            user_latent = getUserLatent(parameters, data, user_index, recursion_depth - 1, caller_id=internal_caller)
            if user_latent is not None:
                latentWithRating = torch.cat((user_latent, Variable(torch.FloatTensor([float(rating)]).type(dtype),volatile = VOLATILE)), dim=0)
                input_latents.append(latentWithRating)  # We got another movie latent
                evidence_count += 1

    if (len(input_latents) < 2):
        return None
    prediction = user_to_movie_net_parameters.forward(torch.stack(input_latents, 0))  # Feed through NN
    column_latent = torch.mean(prediction, dim=0)
    # print("Row lat")
    # # USERLATENTCACHE[user_index] = (row_latent, recursion_depth)
    #
    # return row_latent
    # Now we have all latents, prepare for concatenation
    MOVIELATENTCACHE[movie_index] = (
        column_latent, recursion_depth)  # Cache the movie latent with the current recursion depth

    return column_latent


#
# def lossGrad(data, num_batches=1, fixed_params =500 None, params_to_opt = None, batch_indices = None, reg_alpha=.01, num_aggregates = 1):
#     if not batch_indices:
#         batch_indices = disseminate_values(shuffle(range(len(data[keys_row_first]))),num_batches)
#     fparams = None500500
#
#     if fixed_params:
#         fparams = {key:fixed_params[key] if key not in params_to_opt else None for key in list(set(fixed_params.keys())-set(params_to_opt))}
#
#     def training(params,iter, data=None, indices = None,fixed_params = None, param_keys = None):
#         global TRAININGMODE, RATINGLIMIT
#         TRAININGMODE = True
#         print(batch_indices[iter%num_batches])
#         indices = get_indices_from_range(batch_indices[iter%num_batches],data[keys_row_first], rating_limit=RATINGLIMIT)
#         #print indices
#         if fixed_params:
#             new_params = {key:fixed_params[key] if key in fixed_params else params[key] for key in params}
#             params = new_params
#
#         loss = standard_loss(params,iter,data=data,indices=indices,num_batches=num_batches, reg_alpha=reg_alpha, num_proc=num_aggregates)
#         TRAININGMODE = False
#         return loss
#
#     return grad(lambda params, iter: training(params, iter,data=data,indices = batch_indices, fixed_params = fparams, param_keys = params_to_opt))


def dataCallback(data, test=None):
    return lambda params, iter, grad, optimizer: print_perf(params, iter, grad, train=data, test=test, optimizer = optimizer)


def print_perf(params, iter=0, gradient={}, train=None, test=None, optimizer=None):
    """
    Prints the performance of the model
    """
    global curtime, hitcount, TRAININGMODE, filename, param_dict, VOLATILE, BESTPREC
    print("iter is ", iter)
    if (iter % 10 != 0):
        return
    VOLATILE = True
    TRAININGMODE = True

    print("It took: {} s".format(time.time() - curtime))
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
        test_pred = pred=inference(params, train, indices=test_indices)
        test_rmse_result = rmse(gt=test, pred=test_pred, indices=test_indices)
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
    if (iter % 20 == 0):
        is_best = False
        if (test_rmse_result.data[0] < BESTPREC).any():
            BESTPREC = test_rmse_result
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


def get_candidate_latents(all_items, all_ratings, split=None):
    can_items, can_ratings = all_items[:split], all_ratings[:split]
    rec_items, rec_ratings = all_items[split:], all_ratings[split:]
    rec_items, rec_ratings = shuffle(rec_items, rec_ratings)
    items, ratings = list(can_items) + list(rec_items), list(can_ratings) + list(rec_ratings)
    return items, ratings

#Stolen from Mamy Ratsimbazafy
def save_checkpoint(state, is_best, filename='Weights/checkpoint{}.pth.tar'):
    iter = state["epoch"]
    filename = filename.format(0)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'Weights/model_best.pth.tar')

def setup_caches(data,parameters):
    global NUM_USERS, NUM_MOVIES, NUM_USER_LATENTS, NUM_MOVIE_LATENTS
    NUM_USERS, NUM_MOVIES = list(map(lambda x: len(x), data.nonzero()))
    NUM_USER_LATENTS = parameters[keys_row_latents].size()[0]
    NUM_MOVIE_LATENTS = parameters[keys_col_latents].size()[1]
    wipe_caches()


def wipe_caches():
    global USERLATENTCACHE, MOVIELATENTCACHE, UCANHIT, MCANHIT
    global hitcount
    hitcount = [0] * (MAX_RECURSION + 1)
    USERLATENTCACHE = [None] * NUM_USERS
    MOVIELATENTCACHE = [None] * NUM_MOVIES


def rmse(gt, pred, indices=None):
    diff = 0
    lens = (len(pred.keys()))
    mean = []
    for key in pred.keys():
        mean.append(pred[key])
        diff = diff + torch.pow(float(gt[key]) - pred[key], 2)
    print("Num of items is {} average pred value is {}".format(lens, np.mean(mean)))
    return torch.sqrt((diff / len(pred.keys())))

    row_first = gt[keys_row_first]

    numel = reduce(lambda x, y: x + len(pred[y]), range(len(pred)), 0)
    if numel == 0:
        return 0

    if not indices:
        indices = get_indices_from_range(range(len(pred)), row_first)

    if type(indices) is int:
        print("UH OH")
        print(indices)
        print(pred)
        input("WHY")
        return 0

    val = raw_idx = 0
    for user_index, movie_indices in indices:
        valid_gt_ratings = []
        used_idx = []
        items, ratings = row_first[user_index][get_items], row_first[user_index][get_ratings]
        for idx in range(len(items)):
            if items[idx] in np.sort(movie_indices):
                used_idx.append(items[idx])
                valid_gt_ratings.append(ratings[idx])
        if used_idx != list(movie_indices):
            input("OH SHIT")
        # valid_gt_ratings = row_first[user_index][get_ratings]
        valid_pred_ratings = pred[raw_idx]
        val = val + (np.square(valid_gt_ratings - valid_pred_ratings)).sum()
        raw_idx += 1

    return np.sqrt(val / numel)


def mae(gt, pred):
    val = 0
    for key in pred.keys():
        val = val + torch.abs(pred[key] - float(gt[key]))
    val = val / len(pred.keys())
    return val


def getInferredMatrix(parameters, data):
    """
    Uses the network's predictions to generate a full matrix for comparison.
    """
    row_len, col_len = len(data[keys_row_first]), len(data[keys_col_first])
    inferred = inference(parameters, data=data, indices=range(row_len))
    # inferred = inference(parameters, data=data, indices = get_indices_from_range(range(row_len),data[keys_row_first]))
    newarray = np.zeros((len(data[keys_row_first]), len(data[keys_col_first])))

    for i in range(row_len):
        ratings_high = data[keys_row_first][i][get_items]
        newarray[i, ratings_high] = inferred[i]
    return newarray


def iterateParams(params):
    for k, v in parameters.items():
        if type(v) == Variable:
            paramsToOpt.append(v)
        else:
            for param in v.parameters():
                paramsToOpt.append(param)

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

caches_done = False
ret_list = [[]]
inference = get_pred_for_users
loss = standard_loss
hitcount = [0] * (MAX_RECURSION + 1)
NUM_USERS = 0
NUM_MOVIES = 0
USERLATENTCACHE = [None] * NUM_USERS
MOVIELATENTCACHE = [None] * NUM_MOVIES
USERLATENTCACHEPRIME = [None] * NUM_USERS
MOVIELATENTCACHEPRIME = [None] * NUM_MOVIES
BESTPREC = torch.LongTensor([100]).type(dtype)
VOLATILE = False
