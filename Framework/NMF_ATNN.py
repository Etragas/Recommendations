import autograd.numpy as np
from autograd import grad
from NMF import NMF
from autograd.util import flatten
import time
from utils import *
from autograd.core import primitive
from autograd.scipy.misc import logsumexp
from autograd.optimizers import adam
from multiprocessing.pool import ThreadPool
import threading
import time
import Queue
from copy import deepcopy
import sys
import dill
import threading
from autograd.util import flatten,flatten_func
import inspect
import multiprocess as mp
#import pathos.helpers as mp_vanilla
from threading import Lock
from multiprocessing.queues import Queue
curtime = 0
MAX_RECURSION = 4

movie_cache_lock = Lock()
user_cache_lock = Lock()
rowLatents = 0
colLatents = 0
hitcount = 0
import os

ret_list = [[]]
def lossGrad(data):
    return grad(lambda params,_: nnLoss(params,data=data))

def tricky_wrapper(grad_fun,params,iter,queue):
    print "we're in ", iter
    queue.put(grad_fun(params,iter))
    print "we're out ", iter

def lossGradMultiCore(data,user_range, num_proc = 1):
    return grad(lambda params,_:nnLoss(params,data=data,indices=user_range, num_proc = num_proc))

def gen_grads(data,num_proc):
    num_rows, num_col = data.shape
    gradfuns = []
    ranges = disseminate_values(num_rows,num_proc)
    prev_pos = 0
    for x in range(num_proc) :
        user_range = range(prev_pos,prev_pos+ranges[x])
        prev_pos += ranges[x]
        gradfuns.append(lossGradMultiCore(data,user_range,num_proc))

    return tuple(gradfuns)

def dataCallback(data):
    return lambda params,iter,grad: print_perf(params,iter,grad,data=data)

def nnLoss(parameters,iter=0,data=None,indices = None, num_proc = 1):
    """
    Compute simplified version of squared loss with penalty on vector norms
    :param parameters: Same as class parameter, here for autograd
    :param data:
    :return: A scalar denoting the loss
    """
    #Frobenius Norm squared error term
    print"ab"
    keep = data > 0
    if not indices:
        indices = range(data.shape[0])
    # Regularization Terms
    loss = reg_alpha*np.square(flatten(parameters)[0]).sum()/float(num_proc)
    #Generate predictions
    inferred = inference(parameters,data=data, indices = indices)
    #Squared error between
    for i in range(len(indices)):
        usr_ind = indices[i]
        user_ratings = data[usr_ind,keep[usr_ind,:]]
        prediction = inferred[i]
        loss = loss + np.square(user_ratings - prediction).sum()
    return loss

def black_adam(grad_funs, init_params, callback=None, num_iters=100,
               step_size=0.001, b1=0.9, b2=0.999, eps=10**-8, num_proc = 1):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    multi_grad = []
    for proc_num in range(num_proc):
        flattened_grad, _, _ = flatten_func(grad_funs[proc_num], init_params)
        multi_grad.append(flattened_grad)
    x, unflatten = flatten(init_params)
    multi_grad = tuple(multi_grad)
    manage = mp.Manager()
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    result_queue = manage.Queue()
    pool = mp.Pool(num_proc)
    print num_iters
    for i in range(num_iters):
        print "here we are"
        for proc in range(0,num_proc) :
            pool.apply_async(tricky_wrapper, args=(multi_grad[proc],x,proc,result_queue))
        print "pool please"
        print "and here"
        results = []
        for answer in range(num_proc):
            print "t"
            results.append(result_queue.get())
        g = np.array(reduce(lambda a,b : a+b,results,np.array(0)))

        if callback: callback(unflatten(x), i, unflatten(g))

        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        print multi_grad
        #multi_grad = [multi_grad[2]]*4
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
    return unflatten(x)


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

def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)

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

def neural_net_inference(parameters,iter = 0, data = None, indices = None):
    """
    Generates predictions for each user.
    :param parameters: Parameters of model
    :param iter: Placeholder for training
    :param data: data to work on
    :return: A list of numpy arrays, each of which corresponds to a dense prediction of user ratings.
    """
    global hitcount
    hitcount = 0
    rating_predictions = [0]*data.shape[0]
    print"bc"
    num_rows, num_columns = data.shape
    if not indices:
        indices = np.array(range(num_rows))
    half1 = get_pred_for_users(parameters,data,indices)
    wipe_caches()
    return half1

def disseminate_values(num_items,num_bins):
    chunk_size = num_items/num_bins
    ranges = [chunk_size]*num_bins
    remainder = num_items-num_bins*chunk_size
    #Remainder < num_bins
    for x in range(remainder):
        ranges[x] += 1
    return ranges

def get_pred_for_users(parameters,data,user_indices, queue = None):
    print "cd"
    rating_predictions = []

    for user_index in user_indices:
        #print user_index
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
    with user_cache_lock:
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

    with movie_cache_lock:
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

NUM_USERS = 70000
NUM_MOVIES = 11000
def wipe_caches():
    global USERLATENTCACHE
    global MOVIELATENTCACHE
    USERLATENTCACHE = [np.array((0,0))]*NUM_USERS
    MOVIELATENTCACHE = [np.array((0,0))]*NUM_MOVIES

inference = neural_net_inference
loss = nnLoss

reg_alpha = .1
USERLATENTCACHE = [np.array((0,0))]*NUM_USERS
MOVIELATENTCACHE = [np.array((0,0))]*NUM_MOVIES
