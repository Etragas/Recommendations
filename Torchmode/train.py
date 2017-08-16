from torch import optim

from autograd.optimizers import sgd, rmsprop
from NMF_ATNN import *
from autograd.util import flatten_func
import numpy as np
import utils
import cPickle as pickle
import gc

def train(train_data, test_data, can_idx=None, train_idx=None, test_idx=None, param=None, p1=False, p1Args=[.001, 2, 1],
          p2=False, p2Args=[.001, 2, 1], trainArgs=[.001, 2, 1], use_cache = False):
    '''
    Trains ALL THE THINGS.  Also optionally performs pretraining on the canonicals, rating net weights, rowless net weights, and
    columnless weights.  Prints out the train and test results upon terminination.

    :param train_data: the entire dataset, which we need to reference in order to create canon set
    :param sizes: an array that determines the size of the train set that takes shape [user_size, movie_size]
    :param param: the initial parameter configuration
    :param p1, p2: Indicators representing whether or not we perform the corresponding pretraining
    :param p1Args, p2Args, trainArgs: hyperparamters for the training of each net

    :return: final trained parameters
    '''
    print(param[keys_rating_net].parameters())
    gen = [param[keys_rating_net].parameters(),param[keys_col_latents],param[keys_row_latents]]#,param[keys_user_to_movie_net].parameters(),param[keys_movie_to_user_net].parameters()]
    optimizer = optim.SGD(gen, lr=0.01)
    batch = train_data
    optimizer.zero_grad()   # zero the gradient buffers
    output = get_pred_for_users(param, batch)
    loss = nn.MSELoss(output, batch)
    loss.backward()
    optimizer.step()    # Does the update

    # Generate our rating predictions on the train set from the trained parameters and print performance and comparison
    invtrans = getInferredMatrix(param, train_data)
    print "\n".join([str(x) for x in ["Train", print_perf(param, train=train_data), train_data, np.round(invtrans)]])

    # Generate our rating predictions on the test set from the trained parameters and print performance and comparison
    invtrans = getInferredMatrix(param, test_data)
    print "\n".join([str(x) for x in ["Test", print_perf(param, train=test_data), test_data, np.round(invtrans)]])

    return param



def adam(grad, init_params, callback=None, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8, iter_val = 1):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    flattened_grad, unflatten, x = flatten_func(grad, init_params)
    test_x = x + eps
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(0,num_iters,iter_val):
        g = 0
        print "i ,", i
        for next_batch in range(iter_val):
            #print "aggregating grad, ", next_batch
            g += flattened_grad(x,i+next_batch)
            #print "g is 0 sum, ", (g == 0).sum()
        #g = clip(g,-.2,.2)
        #clip
        if callback: callback(unflatten(x), i, unflatten(g))

        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
    return unflatten(x)
