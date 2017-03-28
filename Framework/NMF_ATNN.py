import autograd.numpy as np
from autograd import grad
from NMF import NMF
from autograd.util import flatten
import time
from autograd.core import primitive
from autograd.scipy.misc import logsumexp
from autograd.optimizers import adam


def lossGrad(data):
    return grad(lambda params,_: nnLoss(params,data=data))

def dataCallback(data):
    return lambda x,iter,grad: print_perf(x,iter,grad,data=data)

def nnLoss(parameters,iter=0,data=None):
    """
    Compute simplified version of squared loss with penalty on vector norms
    :param parameters: Same as class parameter, here for autograd
    :param data:
    :return: A scalar denoting the loss
    """
    #Frobenius Norm squared error term
    print NET_DEPTH
    keep = data > 0

    # Regularization Terms
    loss = .001*np.square(flatten(parameters)[0]).sum()

    #Generate predictions
    inferred = inference(parameters,data=data)

    #Squared error between
    for usr_ind in range(data.shape[0]):
        user_ratings = data[usr_ind,keep[usr_ind,:]]
        prediction = inferred[usr_ind]
        loss = loss + np.square(user_ratings - prediction).sum()
    return loss

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


def print_perf(params, iter=0, gradient=0, data = None):
    """
    Prints the performance of the model
    """

    predicted_data = getInferredMatrix(params,data)
    print("iter is ", iter)
    print("MSE is ",(abs(data-predicted_data).sum())/((data>0).sum()))
    print(loss(parameters=params,data=data))


def neural_net_inference(parameters,iter = 0, data = None):
    """
    Generates predictions for each user.
    :param parameters: Parameters of model
    :param iter: Placeholder for training
    :param data: data to work on
    :return: A list of numpy arrays, each of which corresponds to a dense prediction of user ratings.
    """
    net_parameters = parameters[:NET_DEPTH]
    colLatents = parameters[NET_DEPTH]
    attention_weight = parameters[NET_DEPTH+1]
    rating_predictions = [0]*data.shape[0]

    num_rows, num_columns = data.shape
    for i in range(num_rows):
        current_row = data[i,:]
        rating_indices = flatten(current_row > 0)[0] #Only keep indices where the ratings are non-zero

        if rating_indices.sum() == 0:
            rating_predictions[i] = np.array(0)
            continue

        dense_latents = colLatents[:,rating_indices] #Grab latents for movied rated
        dense_ratings = current_row[rating_indices].reshape((1,rating_indices.sum())) #Grab corresponding ratings

        latents_with_ratings = np.concatenate((dense_latents,dense_ratings),axis = 0 ) #Append ratings to latents
        prediction = neural_net_predict(net_parameters,np.transpose(latents_with_ratings)) #Feed through NN
        latent_weights = softmax(np.dot(prediction,attention_weight))#Multiply NN outputs by shared weight a_w

        row_latent = np.transpose(np.dot(dense_latents, latent_weights))
        row_predictions = np.dot(row_latent,dense_latents)

        rating_predictions[i] = row_predictions

    return rating_predictions #Actual inference

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

inference = neural_net_inference
loss = nnLoss
NET_DEPTH = 0