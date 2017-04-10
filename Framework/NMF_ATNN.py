import time
from threading import Lock

from autograd import grad
from autograd.util import flatten

from utils import *

curtime = 0
MAX_RECURSION = 4

rowLatents = 0
colLatents = 0
hitcount = 0

caches_done = False
ret_list = [[]]


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

    data_loss = np.square((rmse(data,predictions)))
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

    return neural_net_predict(
      parameters=parameters[keys_rating_net],
      inputs=np.concatenate((userLatent, movieLatent)))


def getUserLatent(parameters, data, user_index, recursion_depth=MAX_RECURSION, caller_id=[[], []]):
    global USERLATENTCACHE
    movie_to_user_net_parameters = parameters[keys_movie_to_user_net]
    rowLatents = parameters[keys_row_latents]

    # Check if latent is canonical
    if user_index < rowLatents.shape[0]:
        return rowLatents[user_index, :]

    # Check if latent is cached
    if any(USERLATENTCACHE[user_index]):
        return USERLATENTCACHE[user_index]

    # Exit
    if recursion_depth < 1:
        return None

    # Must generate latent
    current_row = data[keys_row_first][user_index]
    dense_ratings, dense_latents = [], []
    internal_caller = [caller_id[0] + [user_index], caller_id[1]]

    raw_idx = 0
    for movie_index in current_row[get_items]:
        if movie_index not in internal_caller[1]:
            movie_latent = getMovieLatent(parameters, data, movie_index, recursion_depth - 1, internal_caller)

            if movie_latent is not None:
                dense_latents.append(movie_latent)  # We got another movie latent
                dense_ratings.append(current_row[get_ratings][raw_idx])  # Add its corresponding rating
    raw_idx += 1
    # Now have all latents
    # Prepare for concatenations
    dense_latents = (np.array(dense_latents))
    dense_ratings = np.transpose(np.array(dense_ratings).reshape((1, len(dense_ratings))))

    if not any(dense_ratings):
        return None

    latents_with_ratings = np.concatenate((dense_latents, dense_ratings), axis=1)  # Append ratings to latents
    prediction = neural_net_predict(movie_to_user_net_parameters, (latents_with_ratings))  # Feed through NN
    row_latent = np.mean(prediction, axis=0)
    USERLATENTCACHE[user_index] = row_latent

    return row_latent


def getMovieLatent(parameters, data, movie_index, recursion_depth=MAX_RECURSION, caller_id=[[], []]):
    global MOVIELATENTCACHE
    user_to_movie_net_parameters = parameters[keys_user_to_movie_net]
    colLatents = parameters[keys_col_latents]

    if movie_index < colLatents.shape[1]:
        return colLatents[:, movie_index]

    if any(MOVIELATENTCACHE[movie_index]):
        return MOVIELATENTCACHE[movie_index]

    if recursion_depth < 1:
        return None

    dense_ratings, dense_latents = [], []

    # Must Generate Latent
    raw_idx = 0
    current_column = data[keys_col_first][movie_index]
    internal_caller = [caller_id[0], caller_id[1] + [movie_index]]  # [[],[]]#
    for user_index in current_column[get_items]:
        if user_index not in internal_caller[0]:
            user_latent = getUserLatent(parameters, data, user_index, recursion_depth - 1, internal_caller)
            if user_latent is not None:
                dense_latents.append(user_latent)
                dense_ratings.append(current_column[get_ratings][raw_idx])
        raw_idx+=1
    dense_latents = np.array(dense_latents)
    dense_ratings = np.transpose(np.array(dense_ratings).reshape((1, len(dense_ratings))))

    if not any(dense_ratings):
        return None

    latents_with_ratings = np.concatenate((dense_latents, dense_ratings), axis=1)  # Append ratings to latents
    prediction = neural_net_predict(user_to_movie_net_parameters, latents_with_ratings)  # Feed through NN
    column_latent = np.mean(prediction, axis=0)
    MOVIELATENTCACHE[movie_index] = column_latent

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


def lossGrad(data):
    return grad(lambda params, _: standard_loss(params, data=data))


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
    print "Hitcount is: ", hitcount
    print U_HITS
    print M_HITS
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
    hitcount = 0
    USERLATENTCACHE = [np.array((0, 0))] * NUM_USERS
    MOVIELATENTCACHE = [np.array((0, 0))] * NUM_MOVIES
    U_HITS = [0] * NUM_USERS
    M_HITS = [0] * NUM_MOVIES
    USERCACHELOCK = [Lock() for x in range(NUM_USERS)]
    MOVIECACHELOCK = [Lock() for x in range(NUM_MOVIES)]

def rmse(gt,pred):
    val = 0
    row_first = gt[keys_row_first]
    for i in range(len(pred)):
        val = val + (np.square(row_first[i][get_ratings] - pred[i])).sum()
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


inference = get_pred_for_users
loss = standard_loss

reg_alpha = .01
USERLATENTCACHE = [np.array((0, 0))] * NUM_USERS
MOVIELATENTCACHE = [np.array((0, 0))] * NUM_MOVIES
U_HITS = [0] * NUM_USERS
M_HITS = [0] * NUM_MOVIES
