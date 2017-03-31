import autograd.numpy as np
from autograd.optimizers import adam
import NMF_ATNN
from NMF_ATNN import *
import utils
from utils import *

def pretrain_canon_and_rating(full_data, can_usr_idx, can_mov_idx, parameters, step_size, num_iters):
  #Pretrain rating net and latents
  train = full_data[np.ix_(can_usr_idx, can_mov_idx)]

  grads = NMF_ATNN.lossGrad(train)
  parameters = adam(grads, parameters, step_size = step_size, num_iters = num_iters, callback=NMF_ATNN.dataCallback(train))
  return parameters

def pretrain_combiners(full_data, can_usr_idx, can_mov_idx, parameters, step_size, num_iters):
	#Pretrain rowless and columness combiner weights
  train = fill_in_gaps([can_usr_idx, can_mov_idx], [can_usr_idx, can_mov_idx], full_data)
  zeros = np.zeros((num_user_latents,num_movie_latents))
  train[:num_user_latents,:num_movie_latents] = np.array(0)
  train[num_user_latents:,num_movie_latents:] = np.array(0)

  grads = NMF_ATNN.lossGrad(train)
  parameters = adam(grads, parameters, step_size = step_size, num_iters = num_iters, callback = NMF_ATNN.dataCallback(train))
  return parameters

def train(full_data, sizes, parameters, p1 = False, p1Args = [.005, 20], p2 = False, p2Args = [.005, 20], trainArgs = None):
  #Train ALL THE THINGS
  can_usr_idx, can_mov_idx = get_canonical_indices(full_data, [utils.num_user_latents, utils.num_movie_latents])
  train_indx = [(np.array(range(sizes[0]))),(np.array(range(sizes[1])))]
  #TODO: make it not a magic number / part of sizes
  test_indx = [np.random.choice(range(1000),40),np.random.choice(range(1700),20)]
   
  if p1:
    parameters = pretrain_canon_and_rating(full_data, can_usr_idx, can_mov_idx, parameters, *p1Args)

  if p2:
    parameters = pretrain_combiners(full_data, can_usr_idx, can_mov_idx, parameters, *p2Args)
  
  train = fill_in_gaps([can_usr_idx, can_mov_idx], train_indx,full_data)
  test = fill_in_gaps([can_usr_idx, can_mov_idx], test_indx, full_data)

  grads = lossGrad(train)
  parameters = adam(grads,parameters, step_size=trainArgs[0], num_iters=trainArgs[1], callback=dataCallback(train))

  #TODO: Make an inference function that gets called
  invtrans = getInferredMatrix(parameters,train)
  print "\n".join([str(x) for x in ["Train", print_perf(parameters,data=train), train, np.round(invtrans)]])

  invtrans = getInferredMatrix(parameters,test)
  print "\n".join([str(x) for x in ["Test", print_perf(parameters,data=test), test, np.round(invtrans)]])

  return parameters
