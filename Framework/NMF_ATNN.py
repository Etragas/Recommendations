import autograd.numpy as np
from autograd import grad
from NMF import NMF
from autograd.util import flatten
import time
from utils import *
from autograd.core import primitive
from autograd.scipy.misc import logsumexp
from autograd.optimizers import adam
from joblib import Parallel, delayed
from multiprocessing.pool import ThreadPool
import threading
import time
from copy import deepcopy

import inspect
import Queue
import multiprocessing as mp

curtime = 0
MAX_RECURSION = 64

movie_cache_lock = threading.Lock()
user_cache_lock = threading.Lock()
rowLatents = 0
colLatents = 0
hitcount = 0

def lossGrad(data):
    return grad(lambda params,_: nnLoss(params,data=data))


def dataCallback(data):
    return lambda x,iter,grad: print_perf(x,iter,grad,data=data)

# def lossGradWithFixed(fixed_params,data):
#     return grad(lambda params,_: nnLoss_with_injection(params,fixed_params,data=data))

# def nnLoss_with_injection(params,fixed_params,data):
#     fixed_params.update(params)
#     print total_params.keys()
#     return nnLoss(parameters=total_params,data=data)

def nnLoss(parameters,iter=0,data=None):
    """
    Compute simplified version of squared loss with penalty on vector norms
    :param parameters: Same as class parameter, here for autograd
    :param data:
    :return: A scalar denoting the loss
    """
    #Frobenius Norm squared error term
    keep = data > 0

    # Regularization Terms
    loss = reg_alpha*np.square(flatten(parameters)[0]).sum()

    #Generate predictions
    inferred = inference(parameters,data=data)

    #Squared error between
    for usr_ind in range(data.shape[0]):
        user_ratings = data[usr_ind,keep[usr_ind,:]]
        prediction = inferred[usr_ind]
        loss = loss + np.square(user_ratings - prediction).sum()
    return loss

def getInferredMatrix(parameters,data):
    """
    Uses the network's predictions to generate a full matrix for comparison.
    """

    inferred = inference(parameters,data=data)
    newarray = np.zeros((data.shape))

    for i in range(data.shape[0]):
        ratings_high = data[i,:]>0
        newarray[i,ratings_high] = inferred[i]
    return newarray


def print_perf(params, iter=0, gradient={}, data = None):
    """
    Prints the performance of the model
    """
    global curtime, hitcount
    predicted_data = getInferredMatrix(params,data)
    print "It took: {} s".format(time.time()- curtime)
    print("iter is ", iter)
    print("MSE is ",(abs(data-predicted_data).sum())/((data>0).sum()))
    print "Hitcount is: ", hitcount
    for key in gradient.keys():
        x = gradient[key]
        print key
        print np.square(flatten(x)[0]).sum()/flatten(x)[0].size
    print(loss(parameters=params,data=data))
    curtime = time.time()

def neural_net_inference(parameters,iter = 0, data = None):
    """
    Generates predictions for each user.
    :param parameters: Parameters of model
    :param iter: Placeholder for training
    :param data: data to work on
    :return: A list of numpy arrays, each of which corresponds to a dense prediction of user ratings.
    """
    rating_predictions = [0]*data.shape[0]
    global hitcount
    hitcount = 0

    num_rows, num_columns = data.shape
    half1 = get_pred_for_users(parameters,data,np.array(range(num_rows/2)))
    half2 = get_pred_for_users(parameters,data,np.array(range(num_rows/2,num_rows)))
    wipe_caches()
    return half1+half2

def get_pred_for_users(parameters,data,user_indices):
    rating_predictions = []

    for user_index in user_indices:
        current_row = data[user_index,:]
        rating_indices = flatten(current_row > 0)[0] #Only keep indices where the ratings are non-zero
        if rating_indices.sum() == 0:
            rating_predictions.append(np.array(0)) #In the case where no ratings exist, predict 0.
            continue

        predictions = []
        for rating_index in np.flatnonzero(current_row):
            #For each index where a rating exists, generate it.
            predictions.append(recurrent_inference(parameters,iter,data,user_index,rating_index))

        rating_predictions.append(np.array(predictions).reshape((rating_indices.sum())))
    return rating_predictions

def recurrent_inference(parameters,iter=0,data = None,user_index = 0,movie_index =0):
    #Predict full matrix
    rating_net_parameters = parameters[keys_rating_net]

    userLatent = getUserLatent(parameters,data,user_index)
    movieLatent = getMovieLatent(parameters,data,movie_index)
    return neural_net_predict(
            rating_net_parameters,
            np.concatenate((userLatent
                            ,movieLatent)))

def getUserLatent(parameters,data,user_index,recursion_depth = MAX_RECURSION, caller_id = [[],[]]):
    movie_to_user_net_parameters = parameters[keys_movie_to_user_net]
    rowLatents = parameters[keys_row_latents]

    #Check if latent is cached
    if any(USERLATENTCACHE[user_index]):
        global hitcount
        hitcount += 1

        return USERLATENTCACHE[user_index]

    #Check if latent is canonical
    if user_index < rowLatents.shape[0]:
        return rowLatents[user_index,:]

    #Exit
    if recursion_depth < 0:
        return None


    #Must generate latent
    current_row = data[user_index,:]
    dense_ratings = []
    dense_latents = []
    internal_caller = [caller_id[0]+[user_index], caller_id[1]]

    for movie_index in np.flatnonzero(current_row):
        if movie_index not in internal_caller[1]:
            movie_latent = getMovieLatent(parameters,data,movie_index,recursion_depth-1,internal_caller)
            if movie_latent is not None:
                dense_latents.append(movie_latent) #We got another movie latent
                dense_ratings.append(current_row[movie_index]) #Add its corresponding rating
    #Now have all latents
    #Prepare for concatenations
    dense_latents = np.transpose(np.array(dense_latents))
    dense_ratings = np.array(dense_ratings).reshape((1,len(dense_ratings)))

    if (dense_ratings.sum() > 0):
        latents_with_ratings = np.concatenate((dense_latents,dense_ratings),axis = 0 ) #Append ratings to latents
        prediction = neural_net_predict(movie_to_user_net_parameters,np.transpose(latents_with_ratings)) #Feed through NN
        row_latent = np.mean(prediction, axis = 0)
    else:
        return None

    USERLATENTCACHE[user_index] = row_latent
    return row_latent

def getMovieLatent(parameters,data,movie_index,recursion_depth = MAX_RECURSION, caller_id = [[],[]]):
    #print "movie", (MAX_RECURSION-recursion_depth)

    user_to_movie_net_parameters = parameters[keys_user_to_movie_net]
    row_size, col_size = data.shape

    colLatents = parameters[keys_col_latents]

    if movie_index<colLatents.shape[1]:
        return colLatents[:,movie_index]

    if any(MOVIELATENTCACHE[movie_index]):
        global hitcount
        hitcount += 1
        return MOVIELATENTCACHE[movie_index]

    if recursion_depth < 0:
        return None

    dense_ratings = []
    dense_latents = []
    #Must Generate Latent
    current_column = data[:,movie_index]
    internal_caller = [caller_id[0], caller_id[1] + [movie_index]]

    for user_index in np.flatnonzero(current_column):
        if user_index not in internal_caller[0]:
            user_latent = getUserLatent(parameters,data,user_index,recursion_depth-1,internal_caller)
            if user_latent is not None:
                dense_latents.append(user_latent)
                dense_ratings.append(current_column[user_index])

    dense_latents = np.transpose(np.array(dense_latents))
    dense_ratings = np.array(dense_ratings).reshape((1,len(dense_ratings)))

    if (dense_ratings.sum() > 0):
        latents_with_ratings = np.concatenate((dense_latents,dense_ratings),axis = 0 ) #Append ratings to latents
        prediction = neural_net_predict(user_to_movie_net_parameters,np.transpose(latents_with_ratings)) #Feed through NN
        column_latent = np.mean(prediction, axis = 0)
    else:
        return None

    MOVIELATENTCACHE[movie_index] = column_latent

    return column_latent


def neural_net_predict(params, inputs):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = relu(outputs)
    return outputs

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def relu(data):
    return data * (data > 0)

def wipe_caches():
    global USERLATENTCACHE
    global MOVIELATENTCACHE
    USERLATENTCACHE = [np.array((0,0))]*2000
    MOVIELATENTCACHE = [np.array((0,0))]*2000

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

inference = neural_net_inference
loss = nnLoss

reg_alpha = .1
USERLATENTCACHE = [np.array((0,0))]*2000
MOVIELATENTCACHE = [np.array((0,0))]*2000
