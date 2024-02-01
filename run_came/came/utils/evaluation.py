# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:43:10 2021

@author: Xingyan Liu
"""

import numpy as np
from sklearn import metrics
import torch
from torch import Tensor
from typing import Sequence
from ..model import detach2numpy


def accuracy(logits: Tensor, labels: Tensor):
    labels = labels.to(logits.device)
    if len(logits.shape) >= 2:
        _, preds = torch.max(logits, dim=1)
    else:
        preds = logits
    if len(labels.shape) >= 2:
        _, labels = torch.max(labels, dim=1)
    else:
        labels = labels
    correct = torch.sum(preds == labels)
    return correct.item() * 1.0 / len(labels)


def get_AMI(y_true, y_pred, **kwds):
    y_true, y_pred = list(map(detach2numpy, (y_true, y_pred)))
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred, **kwds)
    return ami


def get_F1_score(y_true, y_pred, average='macro', **kwds):
    y_true, y_pred = list(map(detach2numpy, (y_true, y_pred)))
    f1 = metrics.f1_score(y_true, y_pred, average=average, **kwds)
    return f1


