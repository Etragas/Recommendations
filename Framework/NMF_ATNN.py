import time
from threading import Lock

from autograd import grad
from autograd.util import flatten

from utils import *
from sklearn.utils import shuffle
from MultiCore import disseminate_values
from autograd.util import flatten_func

curtime = 0
MAX_RECURSION = 4
TRAININGMODE = False
EVIDENCELIMIT = 80
RATINGLIMIT = 50
glob = np.array([1,2,3,4,5])
def standard_loss(parameters, iter=0, data=None, indices=None, num_proc=1, num_batches = 1, reg_alpha = .01):
    """
    Compute simplified version of squared loss with penalty on vector norms
    :param parameters: Same as class parameter, here for autograd
    :param data:
    :return: A scalar denoting the loss
    """
    # Frobenius Norm squared error term
    # Regularization Terms
    predictions = inference(parameters, data=data, indices=indices)
    numel = reduce(lambda x,y:x+len(predictions[y]),range(len(predictions)),0)
    data_loss = numel*np.square(rmse(data,predictions,indices))
    reg_loss = reg_alpha * np.square(flatten(parameters)[0]).sum() / float(num_proc)

    return reg_loss+data_loss


def get_pred_for_users(parameters, data, indices=None):
    setup_caches(data)
    row_first = data[keys_row_first]
    full_predictions = []

    if not indices:
        indices = get_indices_from_range(range(len(row_first)),data[keys_row_first])

    #Generate predictions over each row
    for user_index,movie_indices in indices:
        user_predictions = []
        for movie_index in movie_indices:
            # For each index where a rating exists, generate it.
            user_predictions.append(recurrent_inference(parameters, iter, data, user_index, movie_index))

        full_predictions.append(np.array(user_predictions).reshape((len(user_predictions))))

    return full_predictions


def recurrent_inference(parameters, iter=0, data=None, user_index=0, movie_index=0):
    # Predict full matrix
    userLatent = getUserLatent(parameters, data, user_index)
    movieLatent = getMovieLatent(parameters, data, movie_index)

    if movieLatent is None or userLatent is None:
        return 2.5
    val = neural_net_predict(
      parameters=parameters[keys_rating_net],
      inputs=np.concatenate((userLatent, movieLatent)))
    return val#np.dot(np.array([1,2,3,4,5]),softmax())


def getUserLatent(parameters, data, user_index, recursion_depth=MAX_RECURSION, caller_id=[[], []]):

    global USERLATENTCACHE, hitcount, USERCACHELOCK, TRAININGMODE, EVIDENCELIMIT, UCANHIT
    movie_to_user_net_parameters = parameters[keys_movie_to_user_net]
    rowLatents = parameters[keys_row_latents]

    # Check if latent is canonical
    if user_index < rowLatents.shape[0]:
        USERLATENTCACHE[user_index] = (rowLatents[user_index, :], recursion_depth)
        return rowLatents[user_index, :]

    # Check if latent is cached
    if  USERLATENTCACHE[user_index] is not None :#and USERLATENTCACHE[user_index][1] >= recursion_depth:
        hitcount[USERLATENTCACHE[user_index][1]] += 1
        return USERLATENTCACHE[user_index][0]

    # Exit
    if recursion_depth < 1:
        return None

    # Must generate latent
    evidence_count = raw_idx = 0
    items, ratings = get_candidate_latents(data[keys_row_first][user_index][get_items], data[keys_row_first][user_index][get_ratings], split=num_movie_latents)

    dense_ratings, dense_latents = [], []
    internal_caller = [caller_id[0] + [user_index], caller_id[1]]

    for movie_index in items:
        if TRAININGMODE and evidence_count > EVIDENCELIMIT:
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
    global MOVIELATENTCACHE, hitcount, MOVIECACHELOCK, TRAININGMODE, EVIDENCELIMIT, MCANHIT
    user_to_movie_net_parameters = parameters[keys_user_to_movie_net]
    colLatents = parameters[keys_col_latents]

    # Check if latent is canonical
    if movie_index < colLatents.shape[1]:
        MOVIELATENTCACHE[movie_index] = (colLatents[:, movie_index], recursion_depth)
        return colLatents[:, movie_index]

    # Check if latent is cached
    if MOVIELATENTCACHE[movie_index] is not None :#and MOVIELATENTCACHE[movie_index][1] >= recursion_depth:
        hitcount[MOVIELATENTCACHE[movie_index][1]] += 1
        return MOVIELATENTCACHE[movie_index][0]

    if recursion_depth < 1:
        return None


    # Must Generate Latent
    evidence_count = raw_idx = 0
    items, ratings = get_candidate_latents(data[keys_col_first][movie_index][get_items], data[keys_col_first][movie_index][get_ratings], split = num_user_latents)

    dense_ratings, dense_latents = [], []
    internal_caller = [caller_id[0], caller_id[1] + [movie_index]]  # [[],[]]#

    for user_index in items:
        if TRAININGMODE and evidence_count > EVIDENCELIMIT:
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
    return np.array(out)


def relu(data):
    return data * (data > 0)


def lossGrad(data, num_batches=1, fixed_params = None, params_to_opt = None, batch_indices = None, reg_alpha=.01, num_aggregates = 1):
    if not batch_indices:
        batch_indices = disseminate_values(shuffle(range(len(data[keys_row_first]))),num_batches)
    fparams = None

    if fixed_params:
        fparams = {key:fixed_params[key] if key not in params_to_opt else None for key in list(set(fixed_params.keys())-set(params_to_opt))}

    def training(params,iter, data=None, indices = None,fixed_params = None, param_keys = None):
        global TRAININGMODE, RATINGLIMIT
        TRAININGMODE = True
        print batch_indices[iter%num_batches]
        indices = get_indices_from_range(batch_indices[iter%num_batches],data[keys_row_first], rating_limit=RATINGLIMIT)
        #print indices
        if fixed_params:
            new_params = {key:fixed_params[key] if key in fixed_params else params[key] for key in params}
            params = new_params

        loss = standard_loss(params,iter,data=data,indices=indices,num_batches=num_batches, reg_alpha=reg_alpha, num_proc=num_aggregates)
        TRAININGMODE = False
        return loss

    return grad(lambda params, iter: training(params, iter,data=data,indices = range(len(data[keys_row_first])), fixed_params = fparams, param_keys = params_to_opt))


def dataCallback(data,test=None):
    return lambda params, iter, grad: print_perf(params, iter, grad, train=data, test=test)


def get_indices_from_range(range,row_first,rating_limit =None):
    #return map(lambda x: (x,row_first[x][get_items])[:rating_limit],range)
    return map(lambda x: (x,np.sort(shuffle(row_first[x][get_items])[:rating_limit])),range)

def print_perf(params, iter=0, gradient={}, train = None, test = None):
    """
    Prints the performance of the model
    """
    global curtime, hitcount
    print("iter is ", iter)
    #if (iter%10 != 0):
    #    return
    print "It took: {} s".format(time.time() - curtime)
    print("iter is ", iter)
    print("MAE is", mae(gt=train, pred=inference(params, train)))
    print("RMSE is ", rmse(gt=train, pred=inference(params, train)))
    print("Loss is ", loss(parameters=params, data=train))
    if (test):
        print "TEST"
        test_idx = get_indices_from_range(range(len(test[keys_row_first])),test[keys_row_first])
        print"Test RMSE is ", rmse(gt=test,pred=inference(params,train,indices=test_idx), indices=test_idx)
    for key in gradient.keys():
        x = gradient[key]
        print key
        print np.square(flatten(x)[0]).sum() / flatten(x)[0].size
        print np.median(abs(flatten(x)[0]))
    print "Hitcount is: ", hitcount, sum(hitcount)
    curtime = time.time()

def get_candidate_latents(all_items, all_ratings, split = None):
    can_items, can_ratings = all_items[:split], all_ratings[:split]
    rec_items, rec_ratings = all_items[split:], all_ratings[split:]
    rec_items, rec_ratings=shuffle(rec_items,rec_ratings)
    items, ratings = can_items+rec_items, can_ratings+rec_ratings
    return items, ratings

def setup_caches(data):
    global NUM_USERS, NUM_MOVIES
    NUM_USERS, NUM_MOVIES = len(data[keys_row_first]), len(data[keys_col_first])
    wipe_caches()


def wipe_caches():
    global USERLATENTCACHE, MOVIELATENTCACHE, UCANHIT, MCANHIT
    global hitcount
    hitcount = [0]*(MAX_RECURSION+1)
    USERLATENTCACHE = [None] * NUM_USERS
    MOVIELATENTCACHE = [None] * NUM_MOVIES


def rmse(gt,pred, indices = None):
    row_first = gt[keys_row_first]

    numel = reduce(lambda x,y:x+len(pred[y]),range(len(pred)),0)
    if numel == 0:
        return 0

    if not indices:
        indices = get_indices_from_range(range(len(pred)),row_first)


    if type(indices) is int:
        print "UH OH"
        print indices
        print pred
        raw_input()
        return 0

    val = raw_idx = 0
    for user_index, movie_indices in indices:
        valid_gt_ratings = []
        used_idx = []
        items, ratings = row_first[user_index][get_items],row_first[user_index][get_ratings]
        for idx in range(len(items)):
            if items[idx] in np.sort(movie_indices):
                used_idx.append(items[idx])
                valid_gt_ratings.append(ratings[idx])
        if used_idx != list(movie_indices):
            raw_input("OH SHIT")
        #valid_gt_ratings = row_first[user_index][get_ratings]
        valid_pred_ratings = pred[raw_idx]
        val = val + (np.square(valid_gt_ratings-valid_pred_ratings)).sum()
        raw_idx+=1

    return np.sqrt(val / numel)

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



rowLatents = 0
colLatents = 0

caches_done = False
ret_list = [[]]
inference = get_pred_for_users
loss = standard_loss
hitcount = [0]*(MAX_RECURSION+1)
NUM_USERS = 0
NUM_MOVIES = 0
USERLATENTCACHE = [None] * NUM_USERS
MOVIELATENTCACHE = [None] * NUM_MOVIES
