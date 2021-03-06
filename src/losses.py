from torch import Tensor

from utils import *
from utils import keys_col_latents, keys_row_latents, keys_movie_to_user_net, keys_user_to_movie_net


def standard_loss(parameters, data=None, indices=None, predictions={}):
    """
    Compute simplified version of squared loss

    :param parameters: Same as class parameter, here for autograd
    :param data: The dataset.  None by default
    :param indices: The indices we compute our loss on.  None by default

    :return: A scalar denoting the non-regularized squared loss
    """

    # generate predictions on specified indices with given parameters
    numel = len(predictions.keys())
    data_loss = numel * torch.pow(rmse(data, predictions), 2)

    return data_loss


def regularization_loss(parameters=None, paramsToOpt=None, reg_alpha=.01):
    """
    Computes the regularization loss, penalizing vector norms

    :param parameters: the parameters in our model
    :param paramsToOpt: the dictionary of parameters to optimize
    :param reg_alpha: the regularization term, .01 by default

    :return: A scalar denoting the regularization loss
    """
    reg_loss = 0
    # reg_loss = reg_alpha * momentDiffV2(parameters, torch.mean)
    # reg_loss += reg_alpha * momentDiff(parameters, torch.var)
    reg_loss += reg_alpha * computeWeightLoss(parameters)

    return reg_loss


def rmse_logging(gt, pred, num_user_protos, num_item_protos):
    """
    Computes the rmse given a ground truth and a prediction

    :param gt: the ground truth, a.k.a. the dataset
    :param pred: the predicted ratings

    :return: the root mean squared error between ground truth and prediction
    """
    diff = 0
    numItems = (len(pred.keys()))
    protoRMSE = torch.FloatTensor()
    partialRMSE = torch.FloatTensor()
    recursiveRMSE = torch.FloatTensor()
    for key in pred.keys():
        row, col = key
        if row < num_user_protos and col < num_item_protos:
            protoRMSE = torch.cat([protoRMSE, torch.pow(float(gt[key]) - pred[key], 2)], dim=0)
        elif row < num_user_protos or col < num_item_protos:
            partialRMSE = torch.cat([partialRMSE, torch.pow(float(gt[key]) - pred[key], 2)], dim=0)
        else:
            recursiveRMSE = torch.cat([recursiveRMSE, torch.pow(float(gt[key]) - pred[key], 2)], dim=0)
    errors = torch.cat((protoRMSE, partialRMSE, recursiveRMSE), 0)
    diff = torch.sum(errors)
    print(diff)
    print("Num of items is {} mean is {}, variance is {}".format(numItems, torch.mean(errors), torch.var(errors)))
    try:
        print("Proto RMSE is {}".format(torch.sqrt(torch.mean(protoRMSE))))
        print("Partial RMSE is {}".format(torch.sqrt(torch.mean(partialRMSE))))
        print("Recursive RMSE is {}".format(torch.sqrt(torch.mean(recursiveRMSE))))
    except:
        pass
    return torch.sqrt((diff / len(pred.keys())))


def rmse(gt, pred):
    """
    Computes the rmse given a ground truth and a prediction

    :param gt: the ground truth, a.k.a. the dataset
    :param pred: the predicted ratings

    :return: the root mean squared error between ground truth and prediction
    """
    diff = 0
    for key in pred.keys():
        diff += (float(gt[key]) - pred[key]) ** 2
    rmse = torch.sqrt((diff / len(pred.keys())))
    print("Rmse is {}".format(rmse))
    return rmse


def mae(gt, pred):
    """
    Computes the mean absolute error given a ground truth and prediction

    :param gt: the ground truth, a.k.a. the dataset
    :param pred: the predicted ratings

    :return: the mean absolute error value between ground truth and prediction
    """
    val = 0
    for key in pred.keys():
        val = val + torch.abs(pred[key] - float(gt[key]))
    val = val / len(pred.keys())
    return val


def computeWeightLoss(parameters):
    regLoss = 0
    for k, v in parameters.items():
        if type(v) == Tensor:
            regLoss += torch.sum(torch.pow(v.data, 2))
        else:
            for subkey, param in v.named_parameters():
                regLoss += torch.sum(torch.pow(param.data, 2))
    return regLoss


def momentDiff(parameters, momentFn):
    colLatents = parameters[keys_col_latents]
    rowLatents = parameters[keys_row_latents]
    colLatntWithRating = torch.cat(
        (colLatents, Variable(3.3 * torch.FloatTensor(torch.ones((1, colLatents.size()[1]))).type(dtype))), dim=0)
    rowLatentsWithRating = torch.cat(
        (rowLatents, Variable(3.3 * torch.FloatTensor(torch.ones((rowLatents.size()[0], 1))).type(dtype))), dim=1)
    averagePredRow = momentFn(parameters[keys_movie_to_user_net].forward(torch.t(colLatntWithRating)), dim=1)
    averagePredCol = momentFn(parameters[keys_user_to_movie_net].forward(rowLatentsWithRating), dim=1)
    averageRow = momentFn(rowLatents, dim=1)
    averageCol = momentFn(colLatents, dim=0)
    meanDiffCol = torch.sum(torch.pow(averageCol - averagePredCol, 2))
    meanDiffRow = torch.sum(torch.pow(averageRow - averagePredRow, 2))
    print("Mean Diff Col {} and Mean Diff Row {}".format(meanDiffCol, meanDiffRow))
    meanDiff = (meanDiffCol + meanDiffRow)

    return meanDiff


def momentDiffV2(parameters, momentFn):
    colLatents = parameters[keys_col_latents]
    rowLatents = parameters[keys_row_latents]
    colLatntWithRating = torch.cat(
        (colLatents, Variable(3.3 * torch.FloatTensor(torch.ones((1, colLatents.size()[1]))).type(dtype))), dim=0)
    rowLatentsWithRating = torch.cat(
        (rowLatents, Variable(3.3 * torch.FloatTensor(torch.ones((rowLatents.size()[0], 1))).type(dtype))), dim=1)
    print(colLatntWithRating.shape)
    print(rowLatentsWithRating.shape)
    averagePredRow = momentFn(parameters[keys_movie_to_user_net].forward(torch.t(colLatntWithRating)), dim=1)
    averagePredCol = momentFn(parameters[keys_user_to_movie_net].forward(rowLatentsWithRating), dim=1)
    averageRow = momentFn(rowLatents, dim=1)
    averageCol = momentFn(colLatents, dim=0)
    meanDiffCol = torch.sum(torch.pow(averagePredCol, 2))
    meanDiffRow = torch.sum(torch.pow(averagePredRow, 2))
    print("Mean Diff Col {} and Mean Diff Row {}".format(meanDiffCol, meanDiffRow))
    meanDiff = (meanDiffCol + meanDiffRow)

    return meanDiff
