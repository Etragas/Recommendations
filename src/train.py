import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from torch import cuda
from torch.utils.data import DataLoader

from losses import rmse
from model import get_predictions_tensor, dataCallback, get_predictions
from utils import keys_rating_net, getDictOfParams, clip_grads, mask_grad, keys_col_latents, keys_row_latents


def train(train_data, test_data, parameters=None, optimizer=None, num_epochs=100,
          batch_size=1024, initialIteration=0, alternatingOptimization=False,
          gradientClipping=False, weight_decay=0.0001, dropped_rows=None):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    # Prepare dictionary of parameters to optimize
    paramsToOptDict = getDictOfParams(parameters)

    print("These are the parameters to optimize:", paramsToOptDict.keys())
    numUserProtos = parameters['RowLatents'].size()[0]
    numItemProtos = parameters['ColLatents'].size()[1]

    # If optimizer is not specified, use the default Adam optimizer
    if optimizer is None:
        optimizer = optim.Adam(paramsToOptDict.values(), lr=.001, weight_decay=weight_decay)
    # print(optimizer.__getstate__())

    # Mask parameters if necessary
    if alternatingOptimization:
        paramSet1 = [keys_rating_net]
        paramSet2 = [x for x in parameters.keys() if x not in [keys_rating_net]]
        maskParams = [paramSet1, paramSet2]
        print("Masked parameters are: ", maskParams)

    # Define loss function
    loss_function = nn.MSELoss(size_average=False)

    # Set callback function for reporting performance
    callback = dataCallback(train_data, test_data)

    idxPretraining = np.array([(k[0], k[1], float(v)) for k, v in train_data[:numUserProtos, :numItemProtos].items()])
    pretrain_rows = idxPretraining[:, 0].astype(int)
    pretrain_cols = idxPretraining[:, 1].astype(int)
    pretrain_values = torch.FloatTensor(idxPretraining[:, 2]).unsqueeze(1)
    pretrain_indices = list(zip(pretrain_rows, pretrain_cols))
    for pretraining in range(250):
        predictions = get_predictions_tensor(parameters, data=train_data, indices=pretrain_indices)
        data_loss = loss_function(predictions, pretrain_values)
        loss = data_loss
        print("Pretraining iteration: ", pretraining)
        optimizer.zero_grad()  # zero the gradient buffers
        loss.backward()
        optimizer.step()  # Does the update

    # Get the indices of our training data ratings
    idxData = np.array([(k[0], k[1], float(v)) for k, v in train_data.items()])

    # Perform the optimization
    for epoch in range(num_epochs):
        # Split up the training data according to the batch size
        idx_loader = DataLoader(dataset=idxData, batch_size=batch_size, shuffle=True, **kwargs)
        iters_per_epoch = len(idx_loader)  # Number of iterations per epoch.
        for iter, batch in enumerate(idx_loader):
            iter = iter + epoch * iters_per_epoch
            row = batch[:, 0].long().tolist()
            col = batch[:, 1].long().tolist()
            val = batch[:, 2].float()
            indices = list(zip(row, col))
            predictions = get_predictions_tensor(parameters, data=train_data, indices=indices)
            data_loss = loss_function(predictions, val.view(len(indices), 1))
            loss = data_loss
            callback(parameters, iter, predictions, loss=data_loss.item(), optimizer=optimizer, dropped_rows=dropped_rows)
            optimizer.zero_grad()  # zero the gradient buffers
            loss.backward()
            if alternatingOptimization:
                paramsToOptDict[(keys_col_latents, keys_col_latents)].grad.zero_()
                paramsToOptDict[(keys_row_latents, keys_row_latents)].grad.zero_()
            if gradientClipping:
                clip_grads(paramsToOptDict, clip=1)
            optimizer.step()  # Does the update

    return parameters
