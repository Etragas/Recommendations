from autograd.optimizers import sgd, rmsprop
from numpy.ma import clip, dot
from scipy import linalg
from scipy.sparse import *
from NMF_ATNN import *
from autograd.util import flatten_func
import numpy as np
import utils
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.decomposition import NMF
import gc
def pretrain_canon_and_rating(full_data, parameters, step_size, num_epochs, batches_per_epoch):
    '''
    Pretrains the canonical latents and the weights of the combiner net.

    :param full_data: the entire dataset, which we need to reference in order to create canon set
    :param can_usr_idx, can_mov_idx: the indices of the canonical users and latents in the full data
    :param parameters: the initial parameter configuration
    :param step_size, num_epochs: hyperparamters for our training

    :return: updated canonical latent parameters and combiner net parameters.
    '''

    # Create our canonical from given indices and the full data
    train = full_data[:utils.num_user_latents,:utils.num_movie_latents].copy()
    train = listify(train)
    print "in p1 wtf", num_epochs, batches_per_epoch
    grads = lossGrad(train, num_batches=batches_per_epoch, fixed_params=parameters, params_to_opt=[keys_col_latents,keys_row_latents,keys_rating_net], reg_alpha=.001,num_aggregates=1)
    # Optimize our parameters using adam
    parameters = adam(grads, parameters, step_size=step_size, num_iters=batches_per_epoch*num_epochs,
                      callback=dataCallback(train), b1=0.5, iter_val=1)
    print "training"
    return parameters


def pretrain_combiners(full_data, parameters, step_size, num_epochs, batches_per_epoch):
    '''
    Pretrains the weights of the rowless and columnless nets.

    :param full_data: the entire dataset, which we need to reference in order to create canon set
    :param can_usr_idx, can_mov_idx: the indices of the canonical users and latents in the full data
    :param parameters: the initial parameter configuration
    :param step_size, num_epochs: hyperparamters for our training

    :return: updated rowless net and columnless net parameters
    '''

    # Create a quadrupled canonical set using a clever fill_in_gaps call
    idx = [np.array(range(utils.num_user_latents)),np.array(range(utils.num_movie_latents))]
    train = fill_in_gaps(np.array(idx),np.array(idx),full_data)
    # Initialize a zeroed array of equal size to our canonical set
    zeros = np.zeros((utils.num_user_latents, utils.num_movie_latents))
    # Set the first and third quadrants of the quadrupled canonical graph to zero.  Set up for clever trickery.
    train[:utils.num_user_latents, :utils.num_movie_latents] = np.array(0)
    train[utils.num_user_latents:, utils.num_movie_latents:] = np.array(0)
    train = listify(train)
    grads = lossGrad(train, num_batches=batches_per_epoch, fixed_params=parameters, params_to_opt=[keys_user_to_movie_net,keys_movie_to_user_net], reg_alpha=.001, num_aggregates=1)
    # Optimize our parameters using adam
    parameters = adam(grads, parameters, step_size=step_size, num_iters=num_epochs*batches_per_epoch,b1 = 0.5,
                      callback=dataCallback(train),iter_val=1)

    return parameters

def pretrain_all(full_data, parameters, step_size, num_epochs, batches_per_epoch):
    idx = [np.array(range(utils.num_user_latents)),np.array(range(utils.num_movie_latents))]
    train = fill_in_gaps(np.array(idx),np.array(idx),full_data)
    # Initialize a zeroed array of equal size to our canonical set
    # Set the first and third quadrants of the quadrupled canonical graph to zero.  Set up for clever trickery.
    train = listify(train)
    grads = lossGrad(train, num_batches=batches_per_epoch, reg_alpha=.001, num_aggregates=1)
    # Optimize our parameters using adam
    parameters = adam(grads, parameters, step_size=step_size, num_iters=20,b1 = 0.5,
                      callback=dataCallback(train), iter_val=1)

    return parameters

def train(train_data, test_data, can_idx=None, train_idx=None, test_idx=None, parameters=None, p1=False, p1Args=[.001, 2,1],
          p2=False, p2Args=[.001, 2, 1], trainArgs=[.001, 2, 1], use_cache = False):
    '''
    Trains ALL THE THINGS.  Also optionally performs pretraining on the canonicals, rating net weights, rowless net weights, and
    columnless weights.  Prints out the train and test results upon terminination.

    :param train_data: the entire dataset, which we need to reference in order to create canon set
    :param sizes: an array that determines the size of the train set that takes shape [user_size, movie_size]
    :param parameters: the initial parameter configuration
    :param p1, p2: Indicators representing whether or not we perform the corresponding pretraining
    :param p1Args, p2Args, trainArgs: hyperparamters for the training of each net

    :return: final trained parameters
    '''
    nrows,ncols = train_data.shape
    if use_cache:
        print("Using cache")
        parameters = pickle.load( open( "parameters", "rb" ) )
    eps=10**-8
    U = np.random.rand(nrows, 80)
    #U = np.maximum(U, eps)
    V = np.random.rand(80,ncols)
    #V = np.maximum(V, eps)
    #U,V= nmf(coo_matrix(train_data),40,max_iter=1000)
    #print(np.dot(U,V))

    # Create our training matrix with canonicals using fill_in_gaps
    # Define the loss for our train
    recLosses = []
    svdLosess = []
    num_opt_passes = 1000
    ltrain_data = listify(train_data)
    ltest_data = listify(test_data)
    mask = np.sign(train_data)
    masked_X = mask * train_data
    for iteration in range(num_opt_passes):
        #First select random rows for minibatch
        mini_batch_users_gen = iter(np.random.choice(range(nrows),100,replace=False))
        num_ratings = 0
        print(np.sum(hitMatrix))
        mini_batch_users = []
        while (num_ratings < 100):
            mini_batch_users+=[mini_batch_users_gen.next()]
            num_ratings += np.sum(train_data[mini_batch_users[-1],:] > 0)
        print(mini_batch_users)
        print("--------------------")
        print("Mini_batch_users",mini_batch_users)
        data_est = np.dot(U,V)
        prevLoss = np.sqrt(np.sum(mask * (test_data - data_est)**2)/np.sum(test_data > 0))
        print 'prev loss SVD', np.round(prevLoss, 4)
        prevLossRec = rmse(ltest_data,get_pred_for_users(parameters,ltest_data))#(rmse(ltest_data,get_pred_for_users(parameters,ltest_data,indices=get_indices_from_range(mini_batch_users,ltest_data[keys_row_first])),indices=get_indices_from_range(mini_batch_users,ltest_data[keys_row_first])))
        print 'prev loss REC', np.round(prevLossRec, 4)
        U[mini_batch_users,:], V = nmf(coo_matrix(train_data[mini_batch_users,:]),80,max_iter=1,initU=U[mini_batch_users,:],initV=V)
        data_est = np.dot(U,V)
        curLossSVD = np.sqrt(np.sum(mask * (test_data - data_est)**2)/np.sum(test_data > 0))
        print("--------------------")
        grads = lossGrad(ltrain_data, num_batches=trainArgs[2], reg_alpha=.001, num_aggregates=1,batch_indices=[mini_batch_users])
        parameters = adam(grads, parameters, step_size=trainArgs[0], num_iters=1,callback=dataCallback(ltrain_data, ltest_data), b1 = 0.5,iter_val=1)
        print("--------------------")
        curLossRec= rmse(ltest_data,get_pred_for_users(parameters,ltest_data))
        print(np.sum(hitMatrix))
        print 'post loss SVD', np.round(curLossSVD, 4)
        print 'post loss REC', np.round(curLossRec, 4)
        print("Percent reductions are {} for REC and {} for SVD".format(curLossRec/prevLossRec,curLossSVD/prevLoss))
        recLosses.append(curLossRec)
        svdLosess.append(curLossSVD)
        plt.plot(range(len(recLosses)), recLosses, 'bs',range(len(svdLosess)),svdLosess,'g^')
        plt.show()
    # Generate our rating predictions on the train set from the trained parameters and print performance and comparison
    invtrans = getInferredMatrix(parameters, train_data)
    print "\n".join([str(x) for x in ["Train", print_perf(parameters, train=train_data), train_data, np.round(invtrans)]])

    # Generate our rating predictions on the test set from the trained parameters and print performance and comparison
    invtrans = getInferredMatrix(parameters, test_data)
    print "\n".join([str(x) for x in ["Test", print_perf(parameters, train=test_data), test_data, np.round(invtrans)]])

    return parameters



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
        g = clip(g,-2,2)
        #clip
        if callback: callback(unflatten(x), i, unflatten(g))

        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
    return unflatten(x)

def nmf(X, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6, initU = None, initV = None):
    """
    Decompose X to A*Y
    """
    eps = 1e-5
    print 'Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter)
    X = X.toarray()  # I am passing in a scipy sparse matrix

    # mask
    mask = np.sign(X)

    # initial matrices. A is random [0,1] and Y is A\X.
    rows, columns = X.shape
    U,V  = initU, initV
    if(U is None):
        U = np.random.rand(rows, latent_features)
        U = np.maximum(U, eps)
    if (V is None):
        V = np.random.rand(latent_features,columns)
        V = np.maximum(V, eps)

    masked_X = mask * X
    X_est_prev = dot(U, V)
    for i in range(1, max_iter + 1):
        # ===== updates =====
        # Matlab: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        top = dot(masked_X, V.T)
        bottom = (dot((mask * dot(U, V)), V.T)) + eps
        U *= top / bottom

        U = np.maximum(U, eps)
        # print 'A',  np.round(A, 2)

        # Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
        top = dot(U.T, masked_X)
        bottom = dot(U.T, mask * dot(U, V)) + eps
        V *= top / bottom
        V = np.maximum(V, eps)
        # print 'Y', np.round(Y, 2)


        # ==== evaluation ====
        if i % 5 == 0 or i == 1 or i == max_iter:
            print 'Iteration {}:'.format(i),
            X_est = dot(U, V)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est

            curRes = linalg.norm(mask * (X - X_est), ord='fro')
            print 'fit residual', np.round(fit_residual, 4),
            print 'total residual', np.round(curRes, 4)
            if curRes < error_limit or fit_residual < fit_error_limit:
                break

    return U, V


