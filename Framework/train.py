import torch.optim as optim
from NMF_ATNN import *
from utils import keys_rating_net

def train(train_data, test_data, parameters=None, optimizer=None, epoch=0):
    '''
    Trains ALL THE THINGS.  Also optionally performs pretraining on the canonicals, rating net weights, rowless net weights, and
    columnless weights.  Prints out the train and test results upon terminination.

    :param train_data: the train dataset, which we need to calculate the train loss.
    :param test_data: the test dataset, which we need to calculate the test loss.
    :param parameters: the initial parameter configuration.
    :param optimizer: the optionally passed in optimizer, defaults to Adam if not set.
    :param epoch: the current epoch.

    :return: final trained parameters.
    '''
    # Training parameters
    num_opt_passes = 10000
    num_accumul = 1
    # cudnn.benchmark = True
    # torch.backends.cudnn.benchmark = True
    print("Shape of the training data:", train_data.shape)
    
    # Prepare dictionary of parameters to optimize
    paramsToOpt = {}
    for k, v in parameters.items():
        if type(v) == Variable:
            paramsToOpt[(k, k)] = v
        else:
            for subkey, param in v.named_parameters():
                paramsToOpt[(k, subkey)] = param
    paramList = paramsToOpt.values()
    print("These are the parameters to optimize:", paramsToOpt)

    # If optimizer is not specified, use the default Adam optimizer
    if optimizer is None:
        optimizer = optim.Adam(paramList, lr=.0001, weight_decay=0.0000)
    # print(optimizer.__getstate__())

    # Set callback function
    callback = dataCallback(train_data, test_data)

    # Mask parameters if necessary
    # maskParams = [[keys_rating_net], [x for x in parameters.keys() if x not in [keys_rating_net]]]
    # print("Masked parameters are: ", maskParams)

    # Perform the optimization
    for iter in range(num_opt_passes):
        iter = iter + epoch
        # pred = inference(parameters, data=train_data, indices=shuffle(list(zip(*train_data.nonzero())))[:100])
        optimizer.zero_grad()  # zero the gradient buffers
        for i in range(num_accumul):
            data_loss = standard_loss(parameters, iter, data=train_data, indices=None, reg_alpha=.00001, num_proc=num_accumul)
            reg_loss = regularization_loss(parameters, paramsToOpt)
            loss = data_loss + reg_loss
            loss.backward()
        # mask_grad(paramsToOpt, maskParams[iter % 2])
        # clip_grads(paramsToOpt,clip=1)
        optimizer.step()  # Does the update
        callback(parameters, iter, None, optimizer=optimizer)

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
