import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
from torch import cuda
from torch.utils.data import DataLoader
from model import get_predictions_tensor, dataCallback
from utils import keys_rating_net, getDictOfParams, clip_grads, mask_grad
from losses import standard_loss, regularization_loss

def train(train_data, test_data, parameters=None, optimizer=None, num_epochs=100, 
          batch_size=1024, initialIteration=0, alternatingOptimization=False,
          dropped_rows=None, gradientClipping=False):
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    # Prepare dictionary of parameters to optimize
    paramsToOptDict = getDictOfParams(parameters)
    print("These are the parameters to optimize:", paramsToOptDict.keys())

    # If optimizer is not specified, use the default Adam optimizer
    if optimizer is None:
        optimizer = optim.Adam(paramsToOptDict.values(), lr=.001, weight_decay=0.0001)
    # print(optimizer.__getstate__())

    # Mask parameters if necessary
    if alternatingOptimization:
        paramSet1 = [keys_rating_net]
        paramSet2 = [x for x in parameters.keys() if x not in [keys_rating_net]]
        maskParams = [paramSet1, paramSet2]
        print("Masked parameters are: ", maskParams)
    
    # Get the indices of our training data ratings
    idxData = np.array([(k[0], k[1], float(v)) for k, v in train_data.items()])

    # Define loss function
    loss_function = nn.MSELoss(size_average=False)

    # Set callback function for reporting performance
    callback = dataCallback(train_data, test_data)

    # Perform the optimization
    for epoch in range(num_epochs):
      # Split up the training data according to the batch size
      idx_loader = DataLoader(dataset=idxData, batch_size=batch_size, shuffle=True, **kwargs)
      iters_per_epoch = len(idx_loader) # Number of iterations per epoch.
      for iter, batch in enumerate(idx_loader):
        iter = iter + epoch * iters_per_epoch
        row = batch[:, 0].long().tolist()
        col = batch[:, 1].long().tolist()
        val = batch[:, 2].float()
        indices = list(zip(row, col))
        predictions = get_predictions_tensor(parameters, data=train_data, indices=indices)
        data_loss = loss_function(predictions, val.view(len(indices), 1))
        reg_loss = regularization_loss(parameters, paramsToOptDict, reg_alpha=0.00001)
        loss = data_loss + reg_loss
        callback(parameters, iter, predictions, loss=data_loss.item(), dropped_rows=dropped_rows, optimizer=optimizer)
        optimizer.zero_grad()  # zero the gradient buffers
        loss.backward()
        if alternatingOptimization:
            mask_grad(paramsToOptDict, maskParams[iter % 2])
        if gradientClipping:
            clip_grads(paramsToOptDict, clip=1)
        optimizer.step()  # Does the update

    return parameters
