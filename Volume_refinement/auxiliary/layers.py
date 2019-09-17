# -*- coding: utf-8 -*-
import numpy as np
import collections

import torch
import torch.nn as nn
from torch.autograd import Variable


###############################################################################
#                                                                             #
#                unpooling layer using PyTorch                                #
#                                                                             #
###############################################################################
class Unpool3DLayer(nn.Module):
    def __init__(self, unpool_size=2, padding=0):
        print("\ninitializing \"Unpool3DLayer\"")
        super(Unpool3DLayer, self).__init__()
        self.unpool_size = unpool_size
        self.padding = padding

    def forward(self, x):
        n = self.unpool_size
        p = self.padding
        # x.size() is (batch_size, channels, depth, height, width)
        output_size = (x.size(0), x.size(1), n * x.size(2), n * x.size(3), n * x.size(4))

        out_tensor = torch.FloatTensor(*output_size).zero_()

        if torch.cuda.is_available():
            out_tensor = out_tensor.type(torch.cuda.FloatTensor)

        out = Variable(out_tensor)

        out[:, \
        :, \
        p: p + output_size[2] + 1: n, \
        p: p + output_size[3] + 1: n, \
        p: p + output_size[4] + 1: n] = x
        return out


###############################################################################
#                                                                             #
#                        softmax with loss 3D                                 #
#                                                                             #
###############################################################################
class SoftmaxWithLoss3D(nn.Module):
    def __init__(self):
        print("\ninitializing \"SoftmaxWithLoss3D\"")
        super(SoftmaxWithLoss3D, self).__init__()

    def forward(self, inputs, y=None, test=False):

        if type(test) is not bool:
            raise Exception("keyword argument \"test\" needs to be a bool type")
        if (test == False) and (y is None):
            raise Exception("\"y is None\" and \"test is False\" cannot happen at the same time")

        """
        Before actually compute the loss, we need to address the possible numberical instability.
        If some elements of inputs are very large, and we compute their exponential value, then we
        might encounter some infinity. So we need to subtract them by the largest value along the
        "channels" dimension to avoid very large exponential.
        """
        # the size of inputs and y is (batch_size, channels, depth, height, width)
        # torch.max return a tuple of (max_value, index_of_max_value)
        max_channel = torch.max(inputs, dim=1)[0]
        adj_inputs = inputs - max_channel.repeat(1, 2, 1, 1, 1)

        exp_x = torch.exp(adj_inputs)
        sum_exp_x = torch.sum(exp_x, dim=1)

        # if the ground truth is provided the loss will be computed
        if y is not None:
            loss = torch.mean(
                torch.sum(-y * adj_inputs, dim=1) + \
                torch.log(sum_exp_x))

        # if this is in the test mode, then the prediction and loss need to be returned
        if test:
            prediction = exp_x / sum_exp_x
            if y is not None:
                return [prediction, loss]
            else:
                return [prediction]
        return loss


class WeightSoftmaxWithLoss3D(nn.Module):
    def __init__(self):
        print("\ninitializing \"SoftmaxWithLoss3D\"")
        super(WeightSoftmaxWithLoss3D, self).__init__()

    def forward(self, inputs, y=None, test=False):

        if type(test) is not bool:
            raise Exception("keyword argument \"test\" needs to be a bool type")
        if (test == False) and (y is None):
            raise Exception("\"y is None\" and \"test is False\" cannot happen at the same time")

        """
        Before actually compute the loss, we need to address the possible numberical instability.
        If some elements of inputs are very large, and we compute their exponential value, then we
        might encounter some infinity. So we need to subtract them by the largest value along the
        "channels" dimension to avoid very large exponential.
        """
        # the size of inputs and y is (batch_size, channels, depth, height, width)
        # torch.max return a tuple of (max_value, index_of_max_value)
        max_channel = torch.max(inputs, dim=1)[0]
        adj_inputs = inputs - max_channel.repeat(1, 2, 1, 1, 1)

        exp_x = torch.exp(adj_inputs)
        sum_exp_x = torch.sum(exp_x, dim=1)

        # if the ground truth is provided the loss will be computed
        if y is not None:
            loss = torch.mean(
                0.2 * (y[:, 0, :, :, :]) * (-adj_inputs[:, 0, :, :, :] + torch.log(sum_exp_x)) + \
                0.8 * (y[:, 1, :, :, :]) * (-adj_inputs[:, 1, :, :, :] + torch.log(sum_exp_x))
            )

        # if this is in the test mode, then the prediction and loss need to be returned
        if test:
            prediction = exp_x / sum_exp_x
            if y is not None:
                return [prediction, loss]
            else:
                return [prediction]
        return loss
