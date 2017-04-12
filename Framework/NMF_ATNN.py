import time
from threading import Lock

from autograd import grad
from autograd.util import flatten
from autograd.util import flatten_func
import multiprocess as mp
from utils import *
from sklearn.utils import shuffle
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

rowLatents = 0
colLatents = 0
hitcount = [0]*(MAX_RECURSION+1)

caches_done = False
ret_list = [[]]
TRAININGMODE = False

def standard_loss(parameters, iter=0, data=None, indices=None, num_proc=1):
    """
    Compute simplified version of squared loss with penalty on vector norms
    :param parameters: Same as class parameter, here for autograd
    :param data:
    :return: A scalar denoting the loss
    """
    # Frobenius Norm squared error term
    # Regularization Terms

    predictions = inference(parameters, data=data, indices=indices)
    data_loss = np.square((rmse(data,predictions,indices)))
    reg_loss = reg_alpha * np.square(flatten(parameters)[0]).sum() / float(num_proc)
    return reg_loss+data_loss



def get_pred_for_users(parameters, data, indices=None, queue=None):
    setup_caches(data)

    if not indices:
        indices = range(len(data[keys_row_first]))
    row_first = data[keys_row_first]
    full_predictions = []

    #Generate predictions over each row
    for user_index in indices:
        user_predictions = []
        #Generate prediction for some user and some movie
        for movie_index in row_first[user_index][get_items]:
            # For each index where a rating exists, generate it.
            user_predictions.append(recurrent_inference(parameters, iter, data, user_index, movie_index))

        full_predictions.append(np.array(user_predictions).reshape((len(user_predictions))))
    return full_predictions


def recurrent_inference(parameters, iter=0, data=None, user_index=0, movie_index=0):
    # Predict full matrix
    movieLatent = getMovieLatent(parameters, data, movie_index)
    userLatent = getUserLatent(parameters, data, user_index)
    if movieLatent is None or userLatent is None:
        return 2.5

    return neural_net_predict(
      parameters=parameters[keys_rating_net],
      inputs=np.concatenate((userLatent, movieLatent)))


def getUserLatent(parameters, data, user_index, recursion_depth=MAX_RECURSION, caller_id=[[], []]):
    global USERLATENTCACHE, hitcount, USERCACHELOCK, TRAININGMODE
    movie_to_user_net_parameters = parameters[keys_movie_to_user_net]
    rowLatents = parameters[keys_row_latents]
    # Check if latent is canonical
    if user_index < rowLatents.shape[0]:
        return rowLatents[user_index, :]

    # Check if latent is cached
    if  USERLATENTCACHE[user_index] is not None:
        hitcount[USERLATENTCACHE[user_index][1]] += 1
        return USERLATENTCACHE[user_index][0]

    # Exit
    if recursion_depth < 1:
        return None

    # Must generate latent
    items, ratings = shuffle(data[keys_row_first][user_index][get_items], data[keys_row_first][user_index][get_ratings])
    dense_ratings, dense_latents = [], []
    internal_caller = [caller_id[0] + [user_index], caller_id[1]]

    evidence_count = 0
    raw_idx = 0
    for movie_index in items:
        if TRAININGMODE and evidence_count > 10:
             break
        if movie_index not in internal_caller[1]:
            movie_latent = getMovieLatent(parameters, data, movie_index, recursion_depth - 1, internal_caller)

            if movie_latent is not None:
                dense_latents.append(movie_latent)  # We got another movie latent
                dense_ratings.append(ratings[raw_idx])  # Add its corresponding rating
                evidence_count += 1
    raw_idx += 1
    if dense_ratings == []:
        return None
    # Now have all latents
    # Prepare for concatenations
    dense_latents = (np.array(dense_latents))
    dense_ratings = np.transpose(np.array(dense_ratings).reshape((1, len(dense_ratings))))


    latents_with_ratings = np.concatenate((dense_latents, dense_ratings), axis=1)  # Append ratings to latents
    prediction = neural_net_predict(movie_to_user_net_parameters, (latents_with_ratings))  # Feed through NN
    row_latent = np.mean(prediction, axis=0)
    USERLATENTCACHE[user_index] = (row_latent, recursion_depth)

    return row_latent


def getMovieLatent(parameters, data, movie_index, recursion_depth=MAX_RECURSION, caller_id=[[], []]):
    global MOVIELATENTCACHE, hitcount, MOVIECACHELOCK, TRAININGMODE
    user_to_movie_net_parameters = parameters[keys_user_to_movie_net]
    colLatents = parameters[keys_col_latents]
    if movie_index < colLatents.shape[1]:
        return colLatents[:, movie_index]

    if  MOVIELATENTCACHE[movie_index] is not None:
        hitcount[MOVIELATENTCACHE[movie_index][1]] += 1
        return MOVIELATENTCACHE[movie_index][0]

    if recursion_depth < 1:
        return None

    dense_ratings, dense_latents = [], []

    # Must Generate Latent
    evidence_count = 0
    raw_idx = 0
    items, ratings = shuffle(data[keys_col_first][movie_index][get_items], data[keys_col_first][movie_index][get_ratings])

    internal_caller = [caller_id[0], caller_id[1] + [movie_index]]  # [[],[]]#
    for user_index in items:
        if TRAININGMODE and evidence_count > 10:
             break
        if user_index not in internal_caller[0]:
            user_latent = getUserLatent(parameters, data, user_index, recursion_depth - 1, internal_caller)
            if user_latent is not None:
                dense_latents.append(user_latent)
                dense_ratings.append(ratings[raw_idx])
                evidence_count+=1
        raw_idx+=1
    if dense_ratings == []:
        return None
    dense_latents = np.array(dense_latents)
    dense_ratings = np.transpose(np.array(dense_ratings).reshape((1, len(dense_ratings))))


    latents_with_ratings = np.concatenate((dense_latents, dense_ratings), axis=1)  # Append ratings to latents
    prediction = neural_net_predict(user_to_movie_net_parameters, latents_with_ratings)  # Feed through NN
    column_latent = np.mean(prediction, axis=0)
    MOVIELATENTCACHE[movie_index] = (column_latent, recursion_depth)

    return column_latent


def neural_net_predict(parameters=None, inputs=None):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    for W, b in parameters:
        outputs = np.dot(inputs, W) + b
        inputs = relu(outputs)
    return outputs


def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def relu(data):
    return data * (data > 0)




def dataCallback(data,test=None):
    return lambda params, iter, grad: print_perf(params, iter, grad, train=data, test=test)


def print_perf(params, iter=0, gradient={}, train = None, test = None):
    """
    Prints the performance of the model
    """
    global curtime, hitcount, U_HITS, M_HITS
    print "It took: {} s".format(time.time() - curtime)
    print("iter is ", iter)
    print("MAE is", mae(gt=train, pred=inference(params, train)))
    print("RMSE is ", rmse(gt=train, pred=inference(params, train)))
    print("Loss is ", loss(parameters=params, data=train))
    if (test):
        print"Test RMSE is ", rmse(gt=test,pred=inference(params,test))
    for key in gradient.keys():
        x = gradient[key]
        print key
        print np.square(flatten(x)[0]).sum() / flatten(x)[0].size
    print "Hitcount is: ", hitcount, sum(hitcount)
    # print U_HITS
    # print M_HITS
    curtime = time.time()


NUM_USERS = 1000
NUM_MOVIES = 1800

def setup_caches(data):
    global NUM_USERS, NUM_MOVIES
    NUM_USERS, NUM_MOVIES = len(data[keys_row_first]), len(data[keys_col_first])
    wipe_caches()


def wipe_caches():
    global USERLATENTCACHE, USERCACHELOCK, UPERMACACHE
    global MOVIELATENTCACHE, MOVIECACHELOCK, MPERMACACHE
    global U_HITS
    global M_HITS
    global hitcount
    hitcount = [0]*(MAX_RECURSION+1)
    USERLATENTCACHE = [None] * NUM_USERS
    MOVIELATENTCACHE = [None] * NUM_MOVIES
    U_HITS = [0] * NUM_USERS
    M_HITS = [0] * NUM_MOVIES
    USERCACHELOCK = [Lock() for x in range(NUM_USERS)]
    MOVIECACHELOCK = [Lock() for x in range(NUM_MOVIES)]

def rmse(gt,pred, indices = None):
    if not indices:
        indices = range(len(pred))
    val = 0
    raw_idx = 0
    row_first = gt[keys_row_first]
    for i in indices:
        val = val + (np.square(row_first[i][get_ratings] - pred[raw_idx])).sum()
        raw_idx+=1
    val = np.sqrt(val / sum([len(row[get_ratings]) for row in row_first]))
    return val

def mae(gt,pred):
    val = 0
    row_first = gt[keys_row_first]
    for i in range(len(pred)):
        val = val + abs(row_first[i][get_ratings] - pred[i]).sum()
    val = val / sum([len(row[get_ratings]) for row in row_first])
    return val

def getInferredMatrix(parameters, data):
    """
    Uses the network's predictions to generate a full matrix for comparison.
    """
    row_len, col_len = len(data[keys_row_first]), len(data[keys_col_first])
    inferred = inference(parameters, data=data, indices=range(row_len))
    newarray = np.zeros((len(data[keys_row_first]),len(data[keys_col_first])))

    for i in range(row_len):
        ratings_high = data[keys_row_first][i][get_items]
        newarray[i, ratings_high] = inferred[i]
    return newarray


def black_adam(grad_funs, init_params, callback=None, num_iters=100,
               step_size=0.001, b1=0.9, b2=0.999, eps=10 ** -8, num_proc=1):
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
        for proc in range(0, num_proc):
            pool.apply_async(tricky_wrapper, args=(multi_grad[proc], x, proc, result_queue))
        results = []
        for answer in range(num_proc):
            results.append(result_queue.get())
        g = np.array(reduce(lambda a, b: a + b, results, np.array(0)))

        if callback: callback(unflatten(x), i, unflatten(g))

        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g ** 2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
        vhat = v / (1 - b2 ** (i + 1))
        #print multi_grad
        # multi_grad = [multi_grad[2]]*4
        x = x - step_size * mhat / (np.sqrt(vhat) + eps)
    return unflatten(x)


def gen_grads(data, num_proc):
    num_rows = len(data[keys_row_first])

    gradfuns = []
    ranges = disseminate_values(num_rows, num_proc)
    prev_pos = 0
    for x in range(num_proc):
        user_range = range(prev_pos, prev_pos + ranges[x])
        prev_pos += ranges[x]
        gradfuns.append(lossGradMultiCore(data, range(num_rows), num_proc))

    return tuple(gradfuns)


def tricky_wrapper(grad_fun, params, iter, queue):
    print "we're in ", iter
    queue.put(grad_fun(params, iter))
    print "we're out ", iter


def lossGradMultiCore(data, user_range, num_proc=1):

    def training(params):
        global TRAININGMODE
        TRAININGMODE = True
        loss = standard_loss(params, data=data, indices=user_range, num_proc=num_proc)
        TRAININGMODE = False

        return loss
    return grad(lambda params, _: training(params) )

def lossGrad(data):
    def training(params):
        global TRAININGMODE
        TRAININGMODE = True
        loss = standard_loss(params,data=data,)
        TRAININGMODE = False
        return loss
    return grad(lambda params, _: training(params))

def disseminate_values(num_items, num_bins):
    chunk_size = num_items / num_bins
    ranges = [chunk_size] * num_bins
    remainder = num_items - num_bins * chunk_size
    # Remainder < num_bins
    for x in range(remainder):
        ranges[x] += 1
    return ranges

inference = get_pred_for_users
loss = standard_loss

reg_alpha = .01
USERLATENTCACHE = [None] * NUM_USERS
MOVIELATENTCACHE = [None] * NUM_MOVIES
U_HITS = [0] * NUM_USERS
M_HITS = [0] * NUM_MOVIES
USERCACHELOCK = [Lock() for x in range(NUM_USERS)]
MOVIECACHELOCK = [Lock() for x in range(NUM_MOVIES)]