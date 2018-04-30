import torch.optim as optim
from model import *
from torch import Tensor
from torch.autograd import Variable
from utils import keys_rating_net

from src.losses import standard_loss, regularization_loss
from src.model import get_predictions


def train(train_data, test_data, parameters=None, optimizer=None, numIter=10000, initialIteration=0,
          alternatingOptimization=False,
          gradientClipping=False):
    # Prepare dictionary of parameters to optimize

    paramsToOptDict = getDictOfParams(parameters)
    paramsToOptList = paramsToOptDict.values()
    print("These are the parameters to optimize:", paramsToOptDict.keys())

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
    for iter in range(numIter):
        if iter == 400:
            break
        iter = iter + initialIteration
        optimizer.zero_grad()  # zero the gradient buffers
        predictions = get_predictions(parameters, data=train_data, indices=None)
        data_loss = standard_loss(parameters, data=train_data, indices=None, predictions=predictions)
        reg_loss = regularization_loss(parameters, paramsToOptDict, reg_alpha=0.00001)
        loss = data_loss + reg_loss
        loss.backward()
        if alternatingOptimization:
            mask_grad(paramsToOptDict, maskParams[iter % 2])
        if gradientClipping:
            clip_grads(paramsToOptDict, clip=1)
        optimizer.step()  # Does the update
        callback(parameters, iter, None, optimizer=optimizer)

    return parameters


def getDictOfParams(parameters: dict):
    paramsToOptDict = {}
    for k, v in parameters.items():
        if type(v) == Tensor:
            paramsToOptDict[(k, k)] = v
        else:
            print(dir(v))
            for subkey, param in v.named_parameters():
                paramsToOptDict[(k, subkey)] = param
    return paramsToOptDict


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
