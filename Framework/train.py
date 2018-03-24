import torch.optim as optim
from model import *
from utils import keys_rating_net


def train(train_data, test_data, parameters=None, optimizer=None, numIter=10000, epoch=0, alternatingOptimization=False,
          gradientClipping=False):
    # cudnn.benchmark = True
    # torch.backends.cudnn.benchmark = True
    print("Shape of the training data:", train_data.shape)

    # Prepare dictionary of parameters to optimize

    paramsToOptDict = getDictOfParams(parameters)
    paramsToOptList = paramsToOptDict.values()
    print("These are the parameters to optimize:", paramsToOptDict)

    # If optimizer is not specified, use the default Adam optimizer
    if optimizer is None:
        optimizer = optim.Adam(paramsToOptList, lr=.0001, weight_decay=0.0000)
    # print(optimizer.__getstate__())

    # Set callback function
    callback = dataCallback(train_data, test_data)

    # Mask parameters if necessary
    if alternatingOptimization:
        paramSet1 = [keys_rating_net]
        paramSet2 = [x for x in parameters.keys() if x not in [keys_rating_net]]
        maskParams = [paramSet1, paramSet2]
        print("Masked parameters are: ", maskParams)

    # Perform the optimization
    for iter in range(numIter):
        iter = iter + epoch
        optimizer.zero_grad()  # zero the gradient buffers
        data_loss = standard_loss(parameters, data=train_data, indices=None)
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

def getDictOfParams(parameters:dict):
    paramsToOptDict = {}
    for k, v in parameters.items():
        if type(v) == Variable:
            paramsToOptDict[(k, k)] = v
        else:
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
