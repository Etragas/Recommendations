import autograd.numpy as np
from autograd.optimizers import adam
import NMF_ATNN
from NMF_ATNN import *
import utils
from utils import *

def pretrain_canon_and_rating(full_data, can_usr_idx, can_mov_idx, parameters, step_size, num_iters):
  '''
  Pretrains the canonical latents and the weights of the combiner net.

  :param full_data: the entire dataset, which we need to reference in order to create canon set
  :param can_usr_idx, can_mov_idx: the indices of the canonical users and latents in the full data
  :param parameters: the initial parameter configuration
  :param step_size, num_iters: hyperparamters for our training

  :return: updated canonical latent parameters and combiner net parameters.
  '''
  
  #Create our canonical from given indices and the full data
  train = full_data[np.ix_(can_usr_idx, can_mov_idx)]
  #Define the loss for our train
  grads = NMF_ATNN.lossGrad(train)
  #Optimize our parameters using adam
  parameters = adam(grads, parameters, step_size = step_size, num_iters = num_iters, callback=NMF_ATNN.dataCallback(train))

  return parameters

def pretrain_combiners(full_data, can_usr_idx, can_mov_idx, parameters, step_size, num_iters):
  '''
  Pretrains the weights of the rowless and columnless nets.

  :param full_data: the entire dataset, which we need to reference in order to create canon set
  :param can_usr_idx, can_mov_idx: the indices of the canonical users and latents in the full data
  :param parameters: the initial parameter configuration
  :param step_size, num_iters: hyperparamters for our training

  :return: updated rowless net and columnless net parameters
  '''

  #Create a quadrupled canonical set using a clever fill_in_gaps call
  train = fill_in_gaps([can_usr_idx, can_mov_idx], [can_usr_idx, can_mov_idx], full_data)
  #Initialize a zeroed array of equal size to our canonical set
  zeros = np.zeros((num_user_latents,num_movie_latents))
  #Set the first and third quadrants of the quadrupled canonical graph to zero.  Set up for clever trickery.
  train[:num_user_latents,:num_movie_latents] = np.array(0)
  train[num_user_latents:,num_movie_latents:] = np.array(0)

  #Define the loss for our train
  grads = NMF_ATNN.lossGrad(train)
  #Optimize our parameters using adam
  parameters = adam(grads, parameters, step_size = step_size, num_iters = num_iters, callback = NMF_ATNN.dataCallback(train))

  return parameters

def train(full_data, sizes, parameters, p1 = False, p1Args = [.005, 2], p2 = False, p2Args = [.005, 2], trainArgs = [.005, 2]):
  '''
  Trains ALL THE THINGS.  Also optionally performs pretraining on the canonicals, rating net weights, rowless net weights, and
  columnless weights.  Prints out the train and test results upon terminination.

  :param full_data: the entire dataset, which we need to reference in order to create canon set
  :param sizes: an array that determines the size of the train set that takes shape [user_size, movie_size]
  :param parameters: the initial parameter configuration
  :param p1, p2: Indicators representing whether or not we perform the corresponding pretraining
  :param p1Args, p2Args, trainArgs: hyperparamters for the training of each net

  :return: final trained parameters
  '''

  #Generate the indices for the canonical users and canonical movies
  can_usr_idx, can_mov_idx = get_canonical_indices(full_data, [utils.num_user_latents, utils.num_movie_latents])
  #Initialize our train matrix with given size
  train_indx = [(np.array(range(sizes[0]))),(np.array(range(sizes[1])))]
  #Create our test matrix with given size
  #TODO: make it not a magic number / part of sizes
  test_indx = [np.random.choice(range(1000),40),np.random.choice(range(1700),20)]
   
  if p1:
    #Perform pretraining on the canonicals and rating net
    parameters = pretrain_canon_and_rating(full_data, can_usr_idx, can_mov_idx, parameters, *p1Args)

  if p2:
    #Perform pretraining on the columnless and rowless nets
    parameters = pretrain_combiners(full_data, can_usr_idx, can_mov_idx, parameters, *p2Args)
  
  #Create our training matrix with canonicals using fill_in_gaps
  train = fill_in_gaps([can_usr_idx, can_mov_idx], train_indx,full_data)
  #Create our test matrix with canonicals using fill_in_gaps
  test = fill_in_gaps([can_usr_idx, can_mov_idx], test_indx, full_data)

  #Define the loss for our train
  grads = lossGrad(train)
  #Optimize our parameters using adam
  parameters = adam(grads,parameters, step_size=trainArgs[0], num_iters=trainArgs[1], callback=dataCallback(train))

  #TODO: Make an inference function that calls the below
  #Generate our rating predictions on the train set from the trained parameters and print performance and comparison
  invtrans = getInferredMatrix(parameters,train)
  print "\n".join([str(x) for x in ["Train", print_perf(parameters,data=train), train, np.round(invtrans)]])

  #Generate our rating predictions on the test set from the trained parameters and print performance and comparison
  invtrans = getInferredMatrix(parameters,test)
  print "\n".join([str(x) for x in ["Test", print_perf(parameters,data=test), test, np.round(invtrans)]])

  return parameters
