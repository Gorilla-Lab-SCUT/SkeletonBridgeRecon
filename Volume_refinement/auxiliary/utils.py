import os
import math
import random
import numpy as np

class_name = {
            'plane': '02691156',
            'bench': '02828884',
            'cabinet': '02933112',
            'car': '02958343',
            'chair': '03001627',
            'lamp': '03636649',
            'monitor': '03211117',
            'speaker': '03691459',
            'firearm': '04090263',
            'couch': '04256520',
            'cellphone': '04401088',
            'table': '04379243',
            'watercraft': '04530566'}

# initialize the weighs of the network for Convolutional layers and batchnorm layers
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.02)
        # m.weight.data.normal_(0,math.sqrt(2.0/n))
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def adjust_learning_rate(optimizer, epoch, phase):
    if (epoch % phase == (phase - 1)):
        for para_group in optimizer.para_groups():
            para_group['lr'] = para_group['lr'] / 10.0


class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count