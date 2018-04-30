import datetime
import shutil
import time

from losses import rmse, mae, standard_loss, regularization_loss
from utils import *
from datetime import datetime

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

hitcount = [0] * (MAX_RECURSION + 1)
NUM_USERS, NUM_MOVIES = 0, 0
userLatentCache, itemLatentCache = [None] * NUM_USERS, [None] * NUM_MOVIES
VOLATILE = False
BESTPREC = 0


def get_predictions(parameters, data, num_flops=0, indices=None):
    """
    Computes the predictions for the specified users and movie pairs

    :param parameters: all the parameters in our model
    :param data: dictionary of the data in row form and column form
    :param indices: user and movie indices for which we generate predictions

    :return: rating predictions for all user/movie combinations specified.  If unspecified,
             computes all rating predictions.
    """
    global VOLATILE
    setup_caches(data, parameters)
    full_predictions = {}
    paramDict = getDictOfParams(parameters)
    
    num_flops += 1 # One comparison flop
    if indices is None:
        print("Generating indices...")
        indices = shuffle(list(zip(*data.nonzero())))[:1000]

    for userIdx, itemIdx in indices:
        key = (userIdx, itemIdx)
        itemLatent, num_flops = getItemEmbedding(parameters, data, itemIdx, num_flops=num_flops)[0:2]
        userLatent, num_flops = getUserEmbedding(parameters, data, userIdx, num_flops=num_flops)[0:2]

        num_flops += 2 # Two comparison flops
        if (userLatent is None or itemLatent is None):
            full_predictions[key] = Variable(torch.FloatTensor([float(3.4)]), volatile=VOLATILE).type(
                dtype)  # Assign an average rating
            # Assignments do not count as flops
        else:
            full_predictions[key] = parameters[keys_rating_net].forward(torch.cat((userLatent, itemLatent), 0))
            # full_predictions[key] = Variable(torch.FloatTensor([float(3.4)]), volatile=VOLATILE).type(
            #   dtype)  # Assign an average rating
    ## torch.dot(userLatent, itemLatent)
            #handle flop count for pushing through the rating net
            num_flops += 2 * rating_network_sizes[0] * rating_network_sizes[1] + 1 #Calculate matrix multiplication, then summation with bias
            num_flops += 2 * rating_network_sizes[1] #Each ReLu takes 2 flop, one comparison and one multiply
            num_flops += 2 * rating_network_sizes[1] * rating_network_sizes[2] + 1 #Calculate matrix multiplication, then summation with bias
            num_flops += 2 * rating_network_sizes[2] #Each ReLu takes 2 flop, one comparison and one multiply
            num_flops += 2 * rating_network_sizes[2] * rating_network_sizes[3] + 1 #Calculate matrix multiplication, then summation with bias

    return full_predictions, num_flops


def getUserEmbedding(parameters, data, userIdx, num_flops=0, recursionStepsRemaining=MAX_RECURSION, caller_id=[[], []],
                     dist=MAX_RECURSION + 1):
    """
    Generate or retrieve the user latent.
    If it is a canonical, we retrieve it.
    If it is not a canonical, we generate it as a function of their ratings and the latents of their rated movies

    :param parameters: dictionary of all the parameters in our model
    :param data: The dataset
    :param userIdx: index of the user for which we want to generate a latent
    :param recursionStepsRemaining: the amount of recursion steps remaining until termination
    :param caller_id: All the [user, movie] ancestors logged, which we check to avoid cycles

    :return: the predicted user latent
    """

    global userLatentCache, hitcount, trainingMode, EVIDENCELIMIT, numUserEmbeddings, VOLATILE, userDistanceCache

    # Get our necessary parameters from the parameters dictionary
    movie_to_user_net = parameters[keys_movie_to_user_net]

    # If user is canonical, return their latent immediately and cache it.
    num_flops +=  1 # One comparison is a flop
    if userIdx < numUserEmbeddings:
        rowLatents = parameters[keys_row_latents]
        userDistances[0].add(userIdx)
        return rowLatents[userIdx, :], num_flops, 0

    # If user latent is cached at a height greater than or equal to current, return their latent immediately
    num_flops += 2 # Two comparison flops
    if userLatentCache[userIdx] is not None and userLatentCache[userIdx][1] >= recursionStepsRemaining:
        hitcount[userLatentCache[userIdx][1]] += 1
        return userLatentCache[userIdx][0], num_flops, userDistanceCache[userIdx]
  
    # If we reached our recursion depth, return None
    num_flops += 1 # One comparison flop
    if recursionStepsRemaining < 1:
        return None, num_flops, MAX_RECURSION

    # Must generate latent

    # Initialize lists for our dense ratings and latents
    itemEmbeddings = []
    # update the current caller_id with this user index appended
    callier_id_with_self = [caller_id[0] + [userIdx], caller_id[1]]
    num_flops += 1 # One addition flop
    # Retrieve latents for every user who watched the movie
    nonZeroColumns = data[userIdx, :].nonzero()[1]
    itemColumns = shuffleNonPrototypeEntries(entries=nonZeroColumns, prototypeThreshold=numUserEmbeddings)
    # Retrieve latents for every movie watched by user
    evidenceCount = 0
    evidenceLimit = EVIDENCELIMIT / (2 ** (MAX_RECURSION - recursionStepsRemaining))
    num_flops += (MAX_RECURSION - recursionStepsRemaining) + 1 + 1 #exponent is difference of recursions, subtraction and division is one flop each

    for itemIdx in itemColumns:
        itemRating = data[userIdx, itemIdx]
        # When it is training mode we use evidence count.
        # When we go over the evidence limit, we no longer need to look for latents
        num_flops += 2 # Two comparison flops
        if trainingMode and evidenceCount > evidenceLimit:
            break

            # If the item latent is valid, and does not produce a cycle, append it
        num_flops += 1 # One comparison flop
        if itemIdx not in callier_id_with_self[1]:
            movie_latent, num_flops, curr_depth = getItemEmbedding(parameters, data, itemIdx, num_flops=num_flops, recursionStepsRemaining=recursionStepsRemaining-1,
                                                        caller_id=callier_id_with_self)

            num_flops += 1 # One comparison flop
            if movie_latent is not None:
                latentWithRating = torch.cat(
                    (movie_latent, Variable(torch.FloatTensor([float(itemRating)]).type(dtype), volatile=VOLATILE)),
                    dim=0)
                itemEmbeddings.append(latentWithRating)  # We got another movie latent
                evidenceCount += 1
                num_flops += 1 # One addition flop
                dist = min(dist, curr_depth)
                num_flops += 1 # One comparison flop

    num_flops += 1 # One comparison flop
    if (len(itemEmbeddings) < 2):
        return None, num_flops, MAX_RECURSION

    # Now have all latents, prepare for concatenations
    embeddingConversions = movie_to_user_net.forward(torch.stack(itemEmbeddings, 0))  # Feed through NN
    num_flops += hyp_user_network_sizes[0] * hyp_user_network_sizes[1] * 2 # number of flops to calculate weighted first layer
    num_flops += hyp_user_network_sizes[1] * 2 # number of flops to perform relu (one comparison one multiplication) on first layer
    num_flops += hyp_user_network_sizes[1] * hyp_user_network_sizes[2] * 2 + 1 # number of flops to calculate weighted second layer, including bias term
    num_flops += hyp_user_network_sizes[2] * 2 # number of flops to perform relu (one comparison one multiplication) on second layer
    num_flops += hyp_user_network_sizes[2] * hyp_user_network_sizes[3] * 2 + 1 # number of flops to calculate weighted third layer, including bias term
    
    row_latent = torch.mean(embeddingConversions, dim=0)
    num_flops += embeddingConversions.size()[0] * embeddingConversions.size()[1] # An average over n elements is performed using n flops (n-1 additions, 1 division).  We perform average for each attribute of latent
    
    userLatentCache[userIdx] = (row_latent, recursionStepsRemaining)
    # print "user: ", user_index, " has depth: ", recursionStepsRemaining, "and caller id: ", internal_caller, " and all callers are: ", zip(entries, data[user_index, entries]), " and ", data[user_index, :], " and dist is: ", dist + 1
    num_flops += 1 # One comparison flop
    if userDistanceCache[userIdx] is not None:
        userDistances[userDistanceCache[userIdx]].remove(userIdx)
        userDistanceCache[userIdx] = min(dist + 1, userDistanceCache[userIdx])
        num_flops += 1 # One comparison flop
        userDistances[userDistanceCache[userIdx]].add(userIdx)
    else:
        userDistances[dist + 1].add(userIdx)
        userDistanceCache[userIdx] = dist + 1
        num_flops += 1 # One addition flop

    return row_latent, num_flops, userDistanceCache[userIdx]


def getItemEmbedding(parameters, data, itemIdx, num_flops=0, recursionStepsRemaining=MAX_RECURSION, caller_id=[[], []],
                     dist=MAX_RECURSION + 1):
    """
    Generate or retrieve the movie latent.
    If it is a canonical, we retrieve it.
    If it is not a canonical, we generate it as a function of the ratings and the latents of its viewers

    :param parameters: dictionary of all the parameters in our model
    :param data: The dataset
    :param itemIdx: index of the movie for which we want to generate a latent
    :param recursionStepsRemaining: the amount of recursion steps remaining until termination
    :param caller_id: All the [user, movie] ancestors logged, which we check to avoid cycles

    :return: the predicted movie latent
    """

    global itemLatentCache, hitcount, trainingMode, EVIDENCELIMIT, numItemEmbeddings, itemDistanceCache

    # Get our necessary parameters from the parameters dictionary
    user_to_movie_net_parameters = parameters[keys_user_to_movie_net]

    # If movie is canonical, return their latent immediately and cache it.
    num_flops += 1 # One comparison flop
    if itemIdx < numItemEmbeddings:
        colLatents = parameters[keys_col_latents]
        itemDistances[0].add(itemIdx)
        return colLatents[:, itemIdx], num_flops, 0

    # If movie latent is cached, return their latent immediately
    num_flops += 2 # Two comparison flop
    if itemLatentCache[itemIdx] is not None and itemLatentCache[itemIdx][1] <= recursionStepsRemaining:
        hitcount[itemLatentCache[itemIdx][1]] += 1
        return itemLatentCache[itemIdx][0], num_flops, itemDistanceCache[itemIdx]

    # If we reached our recursion depth, return None
    num_flops += 1 # One comparison flop
    if recursionStepsRemaining < 1:
        return None, num_flops, MAX_RECURSION

    # Must Generate Latent

    userEmbeddings = []
    # update the current caller_id with this movie index appended
    internal_caller = [caller_id[0], caller_id[1] + [itemIdx]]
    num_flops += 1 # One addition flop

    # Retrieve latents for every user who watched the movie
    nonZeroRows = data[:, itemIdx].nonzero()[0]
    userRows = shuffleNonPrototypeEntries(entries=nonZeroRows, prototypeThreshold=numUserEmbeddings)
    evidenceCount = 0
    evidenceLimit = EVIDENCELIMIT / (2 ** (MAX_RECURSION - recursionStepsRemaining))
    num_flops += (MAX_RECURSION - recursionStepsRemaining) + 1 + 1 #exponent is difference of recursions, subtraction and division is one flop each

    for userIdx in userRows:
        userRating = data[userIdx, itemIdx]
        # When it is training mode we use evidence count.
        # When we go over the evidence limit, we no longer need to look for latents
        num_flops += 2 # Two comparison flops
        if trainingMode and evidenceCount > evidenceLimit:
            break

        # If the user latent is valid, and does not produce a cycle, append it
        num_flops += 1 # One comparison flop
        if userIdx not in internal_caller[0]:
            user_latent, num_flops, curr_depth = getUserEmbedding(parameters, data, userIdx, num_flops=num_flops, recursionStepsRemaining=recursionStepsRemaining - 1,
                                                       caller_id=internal_caller)
            num_flops += 1 # One comparison flop
            if user_latent is not None:
                latentWithRating = torch.cat(
                    (user_latent, Variable(torch.FloatTensor([float(userRating)]).type(dtype), volatile=VOLATILE)),
                    dim=0)
                userEmbeddings.append(latentWithRating)  # We got another movie latent
                evidenceCount += 1
                num_flops += 1 # One addition flop
                dist = min(dist, curr_depth)
                num_flops += 1 # One comparison flop

    num_flops += 1 # One comparison flop
    if (len(userEmbeddings) < 2):
        return None, num_flops, MAX_RECURSION

    # Now have all latents, prepare for concatenations
    embeddingConversions = user_to_movie_net_parameters.forward(torch.stack(userEmbeddings, 0))  # Feed through NN
    num_flops += hyp_movie_network_sizes[0] * hyp_movie_network_sizes[1] * 2 # number of flops to calculate weighted first layer
    num_flops += hyp_movie_network_sizes[1] * 2 # number of flops to perform relu (one comparison one multiplication) on first layer
    num_flops += hyp_movie_network_sizes[1] * hyp_movie_network_sizes[2] * 2 + 1 # number of flops to calculate weighted second layer, including bias term
    num_flops += hyp_movie_network_sizes[2] * 2 # number of flops to perform relu (one comparison one multiplication) on second layer
    num_flops += hyp_movie_network_sizes[2] * hyp_movie_network_sizes[3] * 2 + 1 # number of flops to calculate weighted third layer, including bias term
    
    col_latent = torch.mean(embeddingConversions, dim=0)
    num_flops += embeddingConversions.size()[0] * embeddingConversions.size()[1] # An average over n elements is performed using n flops (n-1 additions, 1 division).  We perform average for each attribute of latent

    # Now we have all latents, prepare for concatenation
    itemLatentCache[itemIdx] = (
        col_latent, recursionStepsRemaining)  # Cache the movie latent with the current recursion depth

    num_flops += 1 # One comparison flop
    if itemDistanceCache[itemIdx] is not None:
        itemDistances[itemDistanceCache[itemIdx]].remove(itemIdx)
        itemDistanceCache[itemIdx] = min(dist + 1, itemDistanceCache[itemIdx])
        num_flops += 1 # One comparison flop
        itemDistances[itemDistanceCache[itemIdx]].add(itemIdx)
    else:
        itemDistances[dist + 1].add(itemIdx)
        itemDistanceCache[itemIdx] = dist + 1
        num_flops += 1 # One addition flop

    return col_latent, num_flops, itemDistanceCache[itemIdx]


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


def print_perf(params, iter=0, gradient={}, train=None, test=None, userDistances={}, itemDistances={}, optimizer=None, num_flops=0):
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
        print("Number of flops is ", num_flops)
        return
    VOLATILE = True
    trainingMode = True

    print("It took: {} seconds".format(time.time() - curtime))
    print("The time is {}".format(datetime.now()))
    pred, trash_flops = get_predictions(params, data=train, indices=shuffle(list(zip(*train.nonzero())))[:500])
    mae_result = mae(gt=train, pred=pred)
    rmse_result = rmse(gt=train, pred=pred)
    loss_result = standard_loss(parameters=params, data=train, predictions=pred)[0] + regularization_loss(
        parameters=params)[0]
    print("MAE is", mae_result.data[0])
    print("RMSE is ", rmse_result.data[0])
    print("Loss is ", loss_result.data[0])
    #print("Number of flops is ", num_flops)
    if (test is not None):
        print("Printing performance for test:")
        test_indices = shuffle(list(zip(*test.nonzero())))[:5000]
        test_pred, num_flops = get_predictions(params, train, indices=test_indices)
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
    print("Number of users per distance", {key: len(value) for (key, value) in userDistances.items()})
    print("User average distance to prototypes: ",
          np.mean(list(map(lambda keyValue: len(keyValue[1]) * keyValue[0], userDistances.items()))))
    print("Number of movies per distance: ", {key: len(value) for (key, value) in itemDistances.items()})
    print("Movie average distance to prototypes: ",
          np.mean(list(map(lambda keyValue: len(keyValue[1]) * keyValue[0], itemDistances.items()))))
    if (iter % 20 == 0):
        is_best = False
        if (test_rmse_result.data[0] < BESTPREC):
            BESTPREC = test_rmse_result.data[0]
            is_best = True
        save_checkpoint({
            'epoch': iter + 1,
            'params': params,
            'best_prec1': test_rmse_result,
            'optimizer': optimizer,
        }, is_best)

    VOLATILE = False
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
    return lambda params, iter, grad, optimizer, num_flops: print_perf(params, iter, grad, train=data,
                                                            test=test,
                                                            userDistances=userDistances,
                                                            itemDistances=itemDistances,
                                                            optimizer=optimizer, num_flops=num_flops)


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
