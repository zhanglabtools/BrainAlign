# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 18:46:19 2021

@author: Xingyan Liu
"""
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        if reduction not in {'sum', 'mean'}:
            raise ValueError('`reduction` should be either "sum" or "mean",'
                             f'got {reduction}')
        self.reduction = reduction

    def forward(self, output, target, weight=None):
        c = output.size()[-1]
        # F.cross_entropy() combines `log_softmax` and `nll_loss` in a single function.
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = - log_preds.sum()
        else:
            loss = - log_preds.sum(dim=-1)
            loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=weight)


def multilabel_binary_cross_entropy(
        logits: torch.Tensor,
        labels: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
):
    """ multi-label binary cross-entropy

    Parameters
    ----------
    logits
        model output logits, without softmax
    labels
        two-dimensional one-hot labels
    weight
        class weights
    reduction
        'mean' or 'sum'

    Returns
    -------
    loss
    """
    # probas = F.sigmoid(logits)
    # loss = F.binary_cross_entropy(probas, labels, weight=weight)
    loss = F.binary_cross_entropy_with_logits(
        logits, labels, weight=weight, reduction=reduction)
    return loss


def cross_entropy_loss(
        logits, labels, weight=None,
        smooth_eps=0.1,
        reduction='mean',
):
    # take out representations of nodes that have labels
    # F.cross_entropy() combines `log_softmax` and `nll_loss` in a single function
    if smooth_eps > 0.:
        loss = LabelSmoothingCrossEntropy(eps=smooth_eps, reduction=reduction)(
            logits, labels, weight=weight
        )
    else:
        loss = F.cross_entropy(logits, labels, weight=weight, reduction=reduction)
    return loss


def classification_loss(
        logits, labels,
        labels_1hot=None,
        weight=None,
        smooth_eps=0.1,
        reduction='mean',
        beta=1.,  # balance factor
        **ignored
):
    loss = cross_entropy_loss(
        logits,
        labels,
        weight=weight,
        smooth_eps=smooth_eps,
        reduction=reduction,
    )
    # multi-label loss
    if labels_1hot is not None and beta > 0.:
        loss += multilabel_binary_cross_entropy(
            logits,
            labels_1hot,
            weight=weight,
            reduction=reduction,
        ) * beta
    return loss


def compute_kl_loss(
        p, q,
        reduction='mean',
        pad_mask=None,
):

    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
                      # reduction=reduction)  # reduction='none'
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
                      # reduction=reduction)  # reduction='none'

    # pad_mask is for seq-level tasks
    # if pad_mask is not None:
    #     p_loss.masked_fill_(pad_mask, 0.)
    #     q_loss.masked_fill_(pad_mask, 0.)
    # choose whether to use function "sum" or "mean" depending on your task
    if reduction == 'mean':
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()
    elif reduction == 'sum':
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
    else:
        raise ValueError("`reduction` should be either 'sum' or 'mean'")

    loss = (p_loss + q_loss) / 2
    return loss


def ce_loss_with_rdrop(
        logits1, logits2,
        labels,
        train_idx=None,
        alpha=0.5,
        reduction='mean',
        weight=None,
        loss_fn: Callable = cross_entropy_loss,
        **kwargs
):
    # keep dropout and forward twice
    # logits = model(x)
    # logits2 = model(x)

    # cross entropy loss for classifier
    if train_idx is None:
        logits1_tr, logits2_tr = logits1, logits2
    else:
        logits1_tr, logits2_tr = logits1[train_idx], logits2[train_idx]

    loss = loss_fn(logits1_tr, labels, reduction=reduction, weight=weight,
                   **kwargs) + \
           loss_fn(logits2_tr, labels, reduction=reduction, weight=weight,
                   **kwargs)

    if alpha > 0.:
        kl_loss = compute_kl_loss(logits1, logits2, reduction=reduction)
        # carefully choose hyper-parameters
        loss += alpha * kl_loss
    return loss


