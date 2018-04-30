import torch.optim as optim
from model import *
from utils import keys_rating_net, getDictOfParams

from losses import standard_loss, regularization_loss
from model import get_predictions


def train(train_data, test_data, parameters=None, optimizer=None, numIter=10000, initialIteration=0,
          alternatingOptimization=False,
          gradientClipping=False):
    
    # Prepare dictionary of parameters to optimize
    paramsToOptDict = getDictOfParams(parameters)
    paramsToOptList = paramsToOptDict.values()
    print("These are the parameters to optimize:", paramsToOptDict.keys())
    flops_per_gradient_update = 0 # Count the total number of parameters to optimize, to count flops for each gradient step
    for k, v in parameters.items(): 
        if type(v) == Variable:
            flops_per_gradient_update += v.size()[0] * v.size()[1]
        else:
            for subkey, param in v.named_parameters():
                if (len(param.size()) > 1):
                  flops_per_gradient_update += param.size()[0] * param.size()[1]
                else:
                  flops_per_gradient_update += param.size()[0] 

    # If optimizer is not specified, use the default Adam optimizer
    if optimizer is None:
        optimizer = optim.Adam(paramsToOptList, lr=.001, weight_decay=0)
    # print(optimizer.__getstate__())

    # Set callback function for reporting performance
    callback = dataCallback(train_data, test_data)

    # Mask parameters if necessary
    if alternatingOptimization:
        paramSet1 = [keys_rating_net]
        paramSet2 = [x for x in parameters.keys() if x not in [keys_rating_net]]
        maskParams = [paramSet1, paramSet2]
        print("Masked parameters are: ", maskParams)

    # Perform the optimization
    num_flops = 0
    for iter in range(numIter):
        iter = iter + initialIteration
        optimizer.zero_grad()  # zero the gradient buffers
        #num_flops += flops_per_gradient_update # Clearing gradients is an assignment, which is not a flop
        predictions, num_flops = get_predictions(parameters, data=train_data, num_flops=num_flops, indices=None)
        data_loss, num_flops = standard_loss(parameters, data=train_data, num_flops=num_flops, indices=None, predictions=predictions)
        reg_loss, num_flops = regularization_loss(parameters, paramsToOptDict, num_flops=num_flops, reg_alpha=0.00001)
        loss = data_loss + reg_loss
        num_flops += 1 # One addition flop
        loss.backward()
        num_flops +=  3 * flops_per_gradient_update # Each derivative takes 3 operations - 2 subtractions and 1 division, and we do this for each param to optimize
        if alternatingOptimization:
            mask_grad(paramsToOptDict, maskParams[iter % 2])
        if gradientClipping:
            clip_grads(paramsToOptDict, clip=1)
        optimizer.step()  # Does the update
        #num_flops += flops_per_gradient_update # Updating parameters is an assignment, which is not a flop
        callback(parameters, iter, None, optimizer=optimizer, num_flops=num_flops)

    return parameters


def clip_grads(params, clip=5):
    for v in params:
        if (v.grad is None):
            continue
        v.grad.data.clamp_(-clip, clip)


def mask_grad(params, keysToMask):
    for keys, value in params.items():
        superKey, subKey = keys
        if superKey in keysToMask:
            if (value.grad is None):
                continue
            value.grad.data.zero_()
    return
