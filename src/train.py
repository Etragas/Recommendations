import torch.optim as optim
from torch import cuda, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import *
from utils import keys_rating_net

from src.model import get_predictions, get_predictions_tensor, dataCallback


def train(train_data, test_data, parameters=None, optimizer=None, numIter=10000, initialIteration=0,
          alternatingOptimization=False,
          gradientClipping=False):
    # Prepare dictionary of parameters to optimize

    paramsToOptDict = getDictOfParams(parameters)
    paramsToOptList = paramsToOptDict.values()
    print("These are the parameters to optimize:", paramsToOptDict.keys())

    # If optimizer is not specified, use the default Adam optimizer
    if optimizer is None:
        optimizer = optim.Adam(paramsToOptList, lr=1e-3, weight_decay=0.0001)
    # print(optimizer.__getstate__())

    # Set callback function for reporting performance
    callback = dataCallback(train_data, test_data)
    # parameters = {keys_row_latents: parameters[keys_row_latents], keys_col_latents: parameters[keys_col_latents]}
    # Mask parameters if necessary
    # Perform the optimization
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    idxData = np.array([(k[0], k[1], float(v)) for k, v in train_data.items()])
    print(idxData.shape)
    batch_size = 10000
    batch_per_epoch = int(idxData.shape[0]/batch_size)
    for epoch in range(100):
        idx_loader = DataLoader(dataset=idxData, batch_size=batch_size, shuffle=True,
                                **kwargs)
        loss_function = nn.MSELoss(size_average=False)

        for iter, batch in enumerate(idx_loader):
            iter = iter + initialIteration
            optimizer.zero_grad()  # zero the gradient buffers
            row = batch[:, 0]
            col = batch[:, 1]
            val = batch[:, 2]
            row = Variable(row.long())
            col = Variable(col.long())
            val = Variable(val.float())
            indices = list(zip(row,col))

            predictions = get_predictions_tensor(parameters, data=train_data, indices=indices)
            data_loss = loss_function(predictions, val.view(len(indices), 1))
            # data_loss = standard_loss(parameters, data=train_data, indices=None, predictions=predictions)
            reg_loss = 0  # regularization_loss(parameters, paramsToOptDict, reg_alpha=0.00000)
            loss = data_loss + reg_loss
            loss.backward()
            optimizer.step()  # Does the update
            if iter % batch_per_epoch == 0:
                callback(parameters, batch_per_epoch*epoch+iter, None, optimizer=optimizer)

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
