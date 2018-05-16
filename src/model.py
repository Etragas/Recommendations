import math
import shutil
import time

from losses import rmse
from utils import *

"""
Initialize all non-mode-specific parameters
"""
curtime = time.time()
MAX_RECURSION = 4
EVIDENCELIMIT = 80
VOLATILE = False
BESTPREC = 0

filename = "intermediate_trained_parameters.pkl"

"""
Initialize dictionaries and lists for storing metrics
"""
userDistances = {}
itemDistances = {}

hitcount = [0] * (MAX_RECURSION + 1)
NUM_USERS, NUM_MOVIES = 0, 0
userLatentCache, itemLatentCache = [None] * NUM_USERS, [None] * NUM_MOVIES

train_mse_iters = []
train_mse = []


def get_predictions(parameters, data, indices=None):
    """
    Computes the predictions for the specified users and item pairs

    :param parameters: all the parameters in our model
    :param data: dictionary of the data in row form and column form
    :param indices: user and item indices for which we generate predictions

    :return: rating predictions for all user/item combinations specified.  If unspecified,
             computes all rating predictions.
    """
    global VOLATILE
    setup_caches(data, parameters)
    full_predictions = {}

    if indices is None:
        print("Generating indices...")
        indices = shuffle(list(zip(*data.nonzero())))[:1000]

    for userIdx, itemIdx in indices:
        key = (userIdx, itemIdx)
        itemLatent = getItemEmbedding(parameters, data, itemIdx)
        userLatent = getUserEmbedding(parameters, data, userIdx)
        if (userLatent is None or itemLatent is None):
            full_predictions[key] = torch.tensor([float(3.4)], requires_grad=True).type(
                dtype).view(1, 1)  # Assign an average rating
        else:
            # NNREC
            # full_predictions[key] = parameters[keys_rating_net].forward(torch.cat((userLatent, itemLatent), 0)).type(dtype)
            # LREC
            full_predictions[key] = torch.dot(userLatent, itemLatent).type(dtype)
    ## torch.dot(userLatent, itemLatent)

    return full_predictions


def get_predictions_tensor(parameters, data, indices=None):
    """
    Computes the predictions for the specified users and item pairs

    :param parameters: all the parameters in our model
    :param data: dictionary of the data in row form and column form
    :param indices: user and item indices for which we generate predictions

    :return: rating predictions for all user/item combinations specified.  If unspecified,
             computes all rating predictions.
    """
    setup_caches(data, parameters)
    full_predictions = torch.FloatTensor()

    if indices is None:
        print("Generating indices...")
        indices = data.get_random_indices(1024)
    # For each user and item pair, generate their latents and compute a prediction if possible.
    for userIdx, itemIdx in indices:
        itemLatent = getItemEmbedding(parameters, data, itemIdx)
        userLatent = getUserEmbedding(parameters, data, userIdx)
        if (userLatent is None or itemLatent is None):
            full_predictions = torch.cat(
                (full_predictions, Variable(torch.tensor([float(3.4)], requires_grad=True)).type(
                    dtype).view(1, 1)), dim=0)  # Assign an average rating
        else:
            # NNREC
            full_predictions = torch.cat((full_predictions, parameters[keys_rating_net].forward(torch.cat((userLatent, itemLatent), 0)).type(dtype).view(1,1)), dim=0)
            # LREC
            # full_predictions = torch.cat((full_predictions, torch.dot(userLatent, itemLatent).type(dtype).view(1, 1)),
            #                             dim=0)
    return full_predictions


def getUserEmbedding(parameters, data, userIdx, recursionStepsRemaining=MAX_RECURSION, ancestor_ids=[[], []],
                     dist=MAX_RECURSION + 1):
    """
    Generate or retrieve the user latent.
    If it is a canonical, we retrieve it.
    If it is not a canonical, we generate it as a function of their ratings and the latents of their rated items

    :param parameters: dictionary of all the parameters in our model
    :param data: The dataset
    :param userIdx: index of the user for which we want to generate a latent
    :param recursionStepsRemaining: the max recursion depth which we can generate latents from.
    :param ancestor_ids: All the [user, item] ancestors logged, which we check to avoid cycles

    :return: the predicted user latent
    """

    global VOLATILE, EVIDENCELIMIT, numUserEmbeddings, userLatentCache, userDistanceCache, hitcount

    # If user is canonical, return their latent immediately and cache it.
    if userIdx < numUserEmbeddings:
        rowLatents = parameters[keys_row_latents]
        if not userLatentCache[userIdx]:
            userLatentCache[userIdx] = (None,recursionStepsRemaining)
        hitcount[userLatentCache[userIdx][1]] += 1
        userDistances[0].add(userIdx)
        return rowLatents[userIdx, :]

    # If user latent is cached at a higher recursion level, return their latent immediately
    if userLatentCache[userIdx] is not None and userLatentCache[userIdx][1] >= recursionStepsRemaining:
        hitcount[userLatentCache[userIdx][1]] += 1
        return userLatentCache[userIdx][0]

    # If we reached our recursion depth, return None
    if recursionStepsRemaining < 1:
        return None

    # Must generate latent
    # Exponentially decrease evidence count according to recursion depth
    evidenceCount = 0
    evidenceLimit = EVIDENCELIMIT / (2 ** (MAX_RECURSION - recursionStepsRemaining))
    # Initialize lists for our dense ratings and latents
    itemEmbeddings = []
    # update the current ancestor tracker with this user index appended
    ancestor_ids = [ancestor_ids[0] + [userIdx], ancestor_ids[1]]
    # Retrieve indices of every item rated by the user
    itemColumns = ColIter(userIdx, data, numUserEmbeddings)
    # Generate latents for each item rated by user
    for itemIdx in itemColumns:
        # When we go over the evidence limit, we no longer need to look for latents
        if evidenceCount > evidenceLimit:
            break

        # If the item latent is valid, and does not produce a cycle, store it and its rating
        if itemIdx not in ancestor_ids[1]:
            item_latent = getItemEmbedding(parameters, data, itemIdx, recursionStepsRemaining - 1,
                                                       ancestor_ids=ancestor_ids)

            if item_latent is not None:
                itemRating = data[userIdx, itemIdx]
                with torch.set_grad_enabled(not VOLATILE):
                    latentWithRating = torch.cat(
                        (item_latent, Variable(torch.FloatTensor([float(itemRating)]).type(dtype))),
                        dim=0)
                itemEmbeddings.append(latentWithRating)  # We got another item latent
                evidenceCount += 1

    # Not enough item embeddings to generate a user embedding
    if (len(itemEmbeddings) < 2):
        return None

    # Get our necessary parameters from the parameters dictionary
    item_to_user_net = parameters[keys_movie_to_user_net]
    # Now have all latents, concatenation and pass through user latent generator net
    embeddingConversions = item_to_user_net.forward(torch.stack(itemEmbeddings, 0))
    row_latent = torch.mean(embeddingConversions, dim=0)  # Final user latent is calculated as the mean.
    userLatentCache[userIdx] = (row_latent, recursionStepsRemaining)
    hitcount[userLatentCache[userIdx][1]] += 1

    return row_latent


def getItemEmbedding(parameters, data, itemIdx, recursionStepsRemaining=MAX_RECURSION, ancestor_ids=[[], []],
                     dist=MAX_RECURSION + 1):
    """
    Generate or retrieve the item latent.
    If it is a canonical, we retrieve it.
    If it is not a canonical, we generate it as a function of the ratings and the latents of its viewers

    :param parameters: dictionary of all the parameters in our model
    :param data: The dataset
    :param itemIdx: index of the item for which we want to generate a latent
    :param recursionStepsRemaining: the max recursion depth which we can generate latents from.
    :param ancestor_ids: All the [user, item] ancestors logged, which we check to avoid cycles

    :return: the predicted item latent
    """

    global VOLATILE, EVIDENCELIMIT, numItemEmbeddings, itemLatentCache, itemDistanceCache, hitcount

    # If item is canonical, return its latent immediately and cache it.
    if itemIdx < numItemEmbeddings:
        colLatents = parameters[keys_col_latents]
        if not itemLatentCache[itemIdx]:
            itemLatentCache[itemIdx] = (None,recursionStepsRemaining)
        hitcount[itemLatentCache[itemIdx][1]] += 1
        itemDistances[0].add(itemIdx)
        return colLatents[:, itemIdx]

    # If item latent is cached at a higher recursion level, return their latent immediately
    if itemLatentCache[itemIdx] is not None and itemLatentCache[itemIdx][1] >= recursionStepsRemaining:
        hitcount[itemLatentCache[itemIdx][1]] += 1
        return itemLatentCache[itemIdx][0]

    # If we reached our recursion depth, return None
    if recursionStepsRemaining < 1:
        return None

    # Must generate latent
    # Exponentially decrease evidence count according to recursion depth
    evidenceCount = 0
    evidenceLimit = EVIDENCELIMIT / (2 ** (MAX_RECURSION - recursionStepsRemaining))
    # Initialize lists for our dense ratings and latents
    userEmbeddings = []
    # update the current ancestor tracker with this item index appended
    ancestor_ids = [ancestor_ids[0], ancestor_ids[1] + [itemIdx]]
    # Retrieve indices for every user who rated the item
    userRows = RowIter(itemIdx, data, numItemEmbeddings)
    # Generate latents for each user who rated the item
    for userIdx in userRows:
        # When we go over the evidence limit, we no longer need to look for latents
        if evidenceCount > evidenceLimit:
            break

        # If the user latent is valid, and does not produce a cycle, store it and its rating
        if userIdx not in ancestor_ids[0]:
            user_latent = getUserEmbedding(parameters, data, userIdx, recursionStepsRemaining - 1,
                                                       ancestor_ids=ancestor_ids)
            if user_latent is not None:
                userRating = data[userIdx, itemIdx]
                with torch.set_grad_enabled(not VOLATILE):
                    latentWithRating = torch.cat(
                        (user_latent, Variable(torch.FloatTensor([float(userRating)]).type(dtype))),
                        dim=0)
                userEmbeddings.append(latentWithRating)  # We got another item latent
                evidenceCount += 1

    # Not enough user embeddings to generate an item embedding
    if (len(userEmbeddings) < 2):
        return None

    # Get our necessary parameters from the parameters dictionary
    user_to_movie_net_parameters = parameters[keys_user_to_movie_net]
    # Now have all latents, perform concatenation and pass through user latent generator net
    prediction = user_to_movie_net_parameters.forward(torch.stack(userEmbeddings, 0))  # Feed through NN
    targetItemEmbedding = torch.mean(prediction, dim=0)  # Final item latent is calculated as the mean
    itemLatentCache[itemIdx] = (targetItemEmbedding, recursionStepsRemaining)
    hitcount[itemLatentCache[itemIdx][1]] += 1

    return targetItemEmbedding


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


def print_perf(params, iter=0, train=None, test=None, predictions=None, loss=None,
               dropped_rows=None, userDistances={}, itemDistances={}, optimizer=None):
    """
    Prints the performance of the model every ten iterations, in terms of MAE, RMSE, and Loss.
    Also includes graphing functionalities.

    :param params: the dictionary of the parameters of the model
    :param iter: the current iteration number
    :param train: the training dataset, None by default
    :param test: the test dataset, None by default
    :param optimizer: the optimizer we are using for the model, None by default
    """
    global curtime, hitcount, filename, VOLATILE, BESTPREC

    print("Iteration ", iter)
    print("RMSE is", math.sqrt(loss / len(predictions)))
    print("Loss is", loss)

    if (iter % 4 != 0):
        return
    VOLATILE = True

    print("It took: {} seconds".format(time.time() - curtime))
    # pred = get_predictions(params, data=train, indices=shuffle(list(zip(*train.nonzero())))[:500])
    # mae_result = mae(gt=train, pred=pred)
    # rmse_result = rmse(gt=train, pred=pred)
    # loss_result = standard_loss(parameters=params, data=train, predictions=pred)# + regularization_loss(
    #    #parameters=params)
    # print("MAE is", mae_result.item())
    # print("RMSE is ", rmse_result.item())
    # print("Loss is ", loss_result.item())
    if (test is not None):
        print("Printing performance for test:")
        test_indices = shuffle(list(zip(*test.nonzero())))#[:5000]
        #split into hot and cold indices
        test_cold_indices = [x for x in test_indices if x[0] in dropped_rows]
        test_hot_indices = list(set(test_indices) - set(test_cold_indices))
        test_pred = get_predictions(params, train, indices=test_indices)
        test_rmse_result = rmse(gt=test, pred=test_pred)
        print("Test Total RMSE is ", test_rmse_result.item())
        with open("REC-1000-total.txt", "a+") as f:
          f.write(str(test_rmse_result.item()))
        if (len(test_cold_indices) > 0):
          test_cold_pred = get_predictions(params, train, indices=test_cold_indices)
          test_cold_rmse_result = rmse(gt=test, pred=test_cold_pred)
          print("Test Cold RMSE ", test_cold_rmse_result.item())
          with open("REC-1000-cold.txt", "a+") as f:
            f.write(str(test_cold_rmse_result.item()))
        else:
          print("Test Cold RMSE is N/A - No dropped users were selected for Test")
        if (len(test_hot_indices) > 0):
          test_hot_pred = get_predictions(params, train, indices=test_hot_indices)
          test_hot_rmse_result = rmse(gt=test, pred=test_hot_pred)
          print("Test Hot RMSE is ", test_hot_rmse_result.item())
          with open("REC-1000-hot.txt", "a+") as f:
            f.write(str(test_hot_rmse_result.item()))
        else:
          print("Test Hot RMSE is N/A - All dropped users were selected for Test")
          
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
            print("\t average of squares is: ", avg_square.item())
            print("\t median is: ", median.item())
        else:
            print("Neural Net Variable Gradient Analytics")
            for param in v.parameters():
                if param.grad is None:
                    print("ERROR - GRADIENT MISSING")
                    continue
                flattened = param.grad.view(param.grad.nelement())
                avg_square = torch.sum(torch.pow(param.grad, 2)) / flattened.size()[0]
                median = torch.median(torch.abs(flattened))
                print("\t average of squares is: ", avg_square.item())
                print("\t median is: ", median.item())

    print("Hitcount is: ", hitcount, sum(hitcount))
    print("Number of users per distance", {key: len(value) for (key, value) in userDistances.items()})
    print("User average distance to prototypes: ",
          np.mean(list(map(lambda keyValue: len(keyValue[1]) * keyValue[0], userDistances.items()))))
    print("Number of items per distance: ", {key: len(value) for (key, value) in itemDistances.items()})
    print("Movie average distance to prototypes: ",
          np.mean(list(map(lambda keyValue: len(keyValue[1]) * keyValue[0], itemDistances.items()))))
    '''
    if (iter % 20 == 0):
        is_best = False
        if (test_rmse_result.item() < BESTPREC):
            BESTPREC = test_rmse_result.item()
            is_best = True
        save_checkpoint({
            'epoch': iter + 1,
            'params': params,
            'best_prec1': test_rmse_result,
            'optimizer': optimizer,
        }, is_best)
    '''
    VOLATILE = False
    curtime = time.time()
    '''
    # Stuff for plots
    train_mse.append(rmse_result.data[0])
    train_mse_iters.append(iter)
    if len(train_mse) % 10 == 0:
      print("Performance Update (every 10 iters): ", train_mse)

    plt.scatter(train_mse_iters, train_mse, color='black')

    plt.plot(train_mse_iters, train_mse)
    plt.title('MovieLens 100K Performance (with pretraining)')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.draw()
    plt.pause(0.001)
    if len(train_mse)%10 == 0:
      #End the plotting with a raw input
      plt.savefig('finalgraph.png')
      print("Final Total Performance: ", train_mse)
    '''


def dataCallback(data, test=None):
    return lambda params, iter, prediction, loss, dropped_rows, optimizer: print_perf(params, iter, train=data,
                                                                        test=test,
                                                                        predictions=prediction,
                                                                        loss=loss,
                                                                        dropped_rows=dropped_rows,
                                                                        userDistances=userDistances,
                                                                        itemDistances=itemDistances,
                                                                        optimizer=optimizer)


def setup_caches(data, parameters):
    """
    Initializes model attributes and the cache for user and item latents

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
    Initializes caches for user and item latents
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
