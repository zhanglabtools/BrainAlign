# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:15:45 2021

@author: Xingyan Liu
"""
from pathlib import Path
import os
from typing import Sequence, Union, Mapping, Optional

import time
import random
import numpy as np
from pandas import value_counts
import torch
from torch import Tensor, LongTensor
import dgl
import tqdm
from ..datapair.aligned import AlignedDataPair
from ..datapair.unaligned import DataPair

from came.model.model_v0 import (
    to_device, onehot_encode,
    ce_loss_with_rdrop,
    classification_loss
)
from came.model.model_v0._minibatch import create_batch, create_blocks
from .evaluation import accuracy, get_AMI, get_F1_score, detach2numpy
from .plot import plot_records_for_trainer
from ._base_trainer import BaseTrainer


def seed_everything(seed=123):
    """ not works well """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_class_weights(labels, astensor=True, foo=np.sqrt, n_add=0):
    if isinstance(labels, Tensor):
        labels = labels.cpu().clone().detach().numpy()

    counts = value_counts(labels).sort_index()  # sort for alignment
    n_cls = len(counts) + n_add
    w = counts.apply(lambda x: 1 / foo(x + 1) if x > 0 else 0)  
    w = (w / w.sum() * (1 - n_add / n_cls)).values
    w = np.array(list(w) + [1 / n_cls] * int(n_add))

    if astensor:
        return Tensor(w)
    return w


def prepare4train(
        dpair: Union[DataPair, AlignedDataPair],
        key_class,
        key_clust='clust_lbs',
        scale_within=True,
        unit_var=True,
        clip=False,
        clip_range=(-3, 3.5),
        categories: Optional[Sequence] = None,
        cluster_labels: Optional[Sequence] = None,
        test_idx: Optional[Sequence] = None,
        ground_truth: bool = True,
        **kwds  # for code compatibility (not raising error)
) -> dict:
    """
    dpair: DataPair
    test_idx: if provided, should be an index-sequence of the same length as 
        `cluster_labels`
    """
    if key_clust and cluster_labels is None:
        cluster_labels = dpair.obs_dfs[1][key_clust].values

    feat_dict = dpair.get_feature_dict(scale=scale_within,
                                       unit_var=unit_var,
                                       clip=clip,
                                       clip_range=clip_range)
    labels, classes = dpair.get_obs_labels(
        key_class, categories=categories, add_unknown_force=False)
    if 'unknown' in classes:
        classes.remove('unknown')

    if test_idx is None:
        train_idx = dpair.get_obs_ids(0, astensor=True)
        test_idx = dpair.get_obs_ids(1, astensor=True)
    else:
        train_idx = LongTensor([i for i in range(dpair.n_obs) if i not in test_idx])
        test_idx = LongTensor(test_idx)

    g = dpair.get_whole_net(rebuild=False, )

    ENV_VARs = dict(
        classes=classes,
        g=g,
        feat_dict=feat_dict,
        train_labels=labels[train_idx],
        test_labels=labels[test_idx] if ground_truth else None,
        train_idx=train_idx,
        test_idx=test_idx,
        cluster_labels=cluster_labels,
    )
    return ENV_VARs


class Trainer(BaseTrainer):
    """
    
    
    """

    def __init__(self,
                 model,
                 feat_dict: Mapping,
                 g: dgl.DGLGraph,
                 train_idx: Tensor,
                 test_idx: Tensor,
                 train_labels: Tensor,
                 test_labels: Optional[Tensor] = None,
                 cluster_labels: Optional[Sequence] = None,
                 lr=1e-3,
                 l2norm=1e-2,  # 1e-2 is tested for all datasets
                 use_cuda=True,
                 dir_main=Path('.'),
                 **kwds  # for code compatibility (not raising error)
                 ):
        super(Trainer, self).__init__(
            model,
            feat_dict=feat_dict,
            train_labels=train_labels,
            test_labels=test_labels,
            train_idx=train_idx,
            test_idx=test_idx,
            cluster_labels=cluster_labels,
            lr=lr,
            l2norm=l2norm,  # 1e-2 is tested for all datasets
            use_cuda=use_cuda,
            dir_main=dir_main,
            **kwds  # for code compatibility (not raising error)
        )

        self.g = g
        self.set_class_weights()
        # infer n_classes
        self.n_classes = len(self.class_weights)
        # for multi-label loss calculation
        self.train_labels_1hot = onehot_encode(
            self.train_labels, sparse_output=False, astensor=True)

        _record_names = (
            'dur',
            'train_loss',
            'test_loss',
            'train_acc',
            'test_acc',
            'AMI',
            'microF1',
            'macroF1',
            'weightedF1',
        )
        self.set_recorder(*_record_names)

    def set_class_weights(self, class_weights=None, foo=np.sqrt, n_add=0):
        if class_weights is None:
            self.class_weights = make_class_weights(
                self.train_labels, foo=foo, n_add=n_add)
        else:
            self.class_weights = Tensor(class_weights)

    def plot_cluster_index(self, start=0, end=None, fp=None):
        plot_records_for_trainer(
            self,
            record_names=['test_acc', 'AMI'],
            start=start, end=end,
            lbs=['test accuracy', 'AMI'],
            tt='test accuracy and cluster index',
            fp=fp)

    def plot_class_losses(self, start=0, end=None, fp=None):
        plot_records_for_trainer(
            self,
            record_names=['train_loss', 'test_loss'],
            start=start, end=end,
            lbs=['training loss', 'testing loss'],
            tt='classification losses',
            fp=fp)

    def plot_class_accs(self, start=0, end=None, fp=None):
        plot_records_for_trainer(
            self,
            record_names=['train_acc', 'test_acc'],
            start=start, end=end,
            lbs=['training acc', 'testing acc'],
            tt='classification accuracy',
            fp=fp)

    # In[]
    def train(self, n_epochs=350,
              use_class_weights=True,
              params_lossfunc={},
              n_pass=100,
              eps=1e-4,
              cat_class='cell',
              device=None,
              backup_stride: int = 43,
              **other_inputs):
        """ Main function for model training (whole-graph based)

        Parameters
        ----------
        n_epochs: int
            The total number of epochs to train the model
        use_class_weights: bool
            whether to use the class-weights, useful for unbalanced
            sample numbers for each class
        params_lossfunc: dict
            parameters for loss-functions
        n_pass: int
            The number of epochs to be skipped (not backup model checkpoint)
        eps:
            tolerance for cluster-index
        cat_class: str
            node type for classification
        device:
            one of {'cpu', 'gpu', None}
        backup_stride: int
            saving checkpoint after `backup_stride` epochs
        other_inputs:
            other inputs for `model.forward()`

        Returns
        -------
        None
        """
        # setting device to train
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.g = self.g.to(device)
        _, feat_dict, train_labels, test_labels, train_idx, test_idx = \
            self.all_to_device(device)
        train_labels_1hot = self.train_labels_1hot.to(device)

        if use_class_weights:
            class_weights = to_device(self.class_weights, device)
        else:
            class_weights = None
        if not hasattr(self, 'ami_max'):
            self.ami_max = 0.
        if not hasattr(self, 'test_acc_max'):
            self.test_acc_max = 0.

        print(f" start training (device='{device}') ".center(60, '='))
        self.model.train()

        for epoch in range(n_epochs):
            self._cur_epoch += 1

            self.optimizer.zero_grad()
            t0 = time.time()
            logits = self.model(feat_dict, self.g, **other_inputs)[cat_class]
            logits2 = self.model(feat_dict, self.g, **other_inputs)[cat_class]
            loss = ce_loss_with_rdrop(
                logits, logits2, labels=train_labels,
                labels_1hot=train_labels_1hot,
                train_idx=train_idx, weight=class_weights,
                loss_fn=classification_loss,
                **params_lossfunc
            )
            # loss = classification_loss(
            #     logits[train_idx],
            #     train_labels, labels_1hot=train_labels_1hot,
            #     weight=class_weights,
            #     **params_lossfunc
            # )

            # prediction 
            _, y_pred = torch.max(logits, dim=1)
            y_pred_test = y_pred[test_idx]

            # ========== evaluation (Acc.) ==========
            train_acc = accuracy(y_pred[train_idx], train_labels)
            if test_labels is None:
                # ground-truth unavailable
                test_acc = microF1 = macroF1 = weightedF1 = -1.
            else:
                test_acc = accuracy(y_pred_test, test_labels)
                # F1-scores
                microF1 = get_F1_score(test_labels, y_pred_test, 'micro')
                macroF1 = get_F1_score(test_labels, y_pred_test, 'macro')
                weightedF1 = get_F1_score(test_labels, y_pred_test, 'weighted')

            # unsupervised cluster index
            if self.cluster_labels is not None:
                ami = get_AMI(self.cluster_labels, y_pred_test)
            else:
                ami = -1.

            backup = False
            if self._cur_epoch >= n_pass - 1:
                self.ami_max = max(self.ami_max, ami)
                if ami >= self.ami_max - eps > 0:
                    self._cur_epoch_best = self._cur_epoch
                    self.save_model_weights()
                    backup = True
                    print('[current best] model weights backup')
                elif self._cur_epoch % backup_stride == 0:
                    self.save_model_weights()
                    backup = True
                    print('model weights backup')

            # backward AFTER the model being saved
            loss.backward()
            self.optimizer.step()
            t1 = time.time()

            #  recording
            self._record(dur=t1 - t0,
                         train_loss=loss.item(),
                         train_acc=train_acc,
                         test_acc=test_acc,
                         AMI=ami,
                         microF1=microF1,
                         macroF1=macroF1,
                         weightedF1=weightedF1,
                         )

            dur_avg = np.average(self.dur)
            self.test_acc_max = max(test_acc, self.test_acc_max)

            if self.with_ground_truth:
                logfmt = "Epoch {:04d} | Train Acc: {:.4f} | Test: {:.4f} (max={:.4f}) | AMI={:.4f} | Time: {:.4f}"
                self._cur_log = logfmt.format(
                    self._cur_epoch, train_acc, test_acc, self.test_acc_max,
                    ami, dur_avg)
            else:
                logfmt = "Epoch {:04d} | Train Acc: {:.4f} | AMI: {:.4f} (max={:.4f}) | Time: {:.4f}"
                self._cur_log = logfmt.format(
                    self._cur_epoch, train_acc, ami, self.ami_max,
                    ami, dur_avg)

            if self._cur_epoch % 5 == 0 or backup:
                print(self._cur_log)

        self._cur_epoch_adopted = self._cur_epoch
        self.save_checkpoint_record()

    def train_minibatch(self, n_epochs=100,
                        use_class_weights=True,
                        params_lossfunc={},
                        n_pass=100,
                        eps=1e-4,
                        cat_class='cell',
                        batch_size=128,
                        device=None,
                        backup_stride: int = 43,
                        **other_inputs):
        """ Main function for model training (based on mini-batches)
        """
        # setting device to train
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_idx, test_idx, train_labels, test_labels = self.train_idx, self.test_idx, self.train_labels, self.test_labels
        labels = torch.cat((train_labels, test_labels), 0)
        self.g.nodes['cell'].data['ids'] = torch.arange(self.g.num_nodes('cell'))  # track the random shuffle

        if use_class_weights:
            class_weights = to_device(self.class_weights, device)
        else:
            class_weights = None

        if not hasattr(self, 'ami_max'):
            self.ami_max = 0.
        if not hasattr(self, 'test_acc_max'):
            self.test_acc_max = 0.

        print("start training".center(50, '='))
        model = self.model.to(device)
        model.train()
        feat_dict = {}
        train_labels, test_labels, batch_list, shuffled_idx = create_batch(
            train_idx=train_idx, test_idx=test_idx, batch_size=batch_size,
            labels=labels, shuffle=True)
        shuffled_test_idx = detach2numpy(
            shuffled_idx[shuffled_idx >= len(train_idx)]
        ) - len(train_idx)
        cluster_labels = self.cluster_labels[shuffled_test_idx]
        blocks = []
        for output_nodes in tqdm.tqdm(batch_list):
            block = create_blocks(g=self.g, output_nodes=output_nodes)
            blocks.append(block)

        for epoch in range(n_epochs):
            self._cur_epoch += 1
            all_train_preds = to_device(torch.tensor([]), device)
            all_test_preds = to_device(torch.tensor([]), device)
            t0 = time.time()
            batch_rank = 0
            for output_nodes in tqdm.tqdm(batch_list):
                self.optimizer.zero_grad()
                block = blocks[batch_rank]
                batch_rank += 1
                feat_dict['cell'] = self.feat_dict['cell'][block.nodes['cell'].data['ids'], :]
                # TODO: here might raise error if the data are shuffled!
                batch_train_idx = output_nodes.clone().detach() < len(train_idx)
                batch_test_idx = output_nodes.clone().detach() >= len(train_idx)
                feat_dict, block = to_device(feat_dict, device), to_device(block, device)
                logits = model(feat_dict, block, **other_inputs)[cat_class]
                logits2 = model(feat_dict, block, **other_inputs)[cat_class]

                # out_cell = logits[cat_class]
                # output_labels = labels[output_nodes]
                out_train_labels = to_device(
                    labels[output_nodes][batch_train_idx].clone().detach(), device)
                out_train_lbs1hot = to_device(
                    self.train_labels_1hot[output_nodes[batch_train_idx]].clone().detach(),
                    device,
                )

                loss = ce_loss_with_rdrop(
                    logits, logits2, labels=out_train_labels,
                    labels_1hot=out_train_lbs1hot,
                    train_idx=batch_train_idx, weight=class_weights,
                    loss_fn=classification_loss,
                    **params_lossfunc
                )
                # loss = classification_loss(
                #     logits[batch_train_idx],
                #     to_device(out_train_labels, device),
                #     labels_1hot=to_device(out_train_lbs1hot, device),
                #     weight=class_weights,
                #     **params_lossfunc
                # )
                loss.backward()
                self.optimizer.step()

                _, y_pred = torch.max(logits, dim=1)
                y_pred_train = y_pred[batch_train_idx]
                y_pred_test = y_pred[batch_test_idx]
                all_train_preds = torch.cat((all_train_preds, y_pred_train), 0)
                all_test_preds = torch.cat((all_test_preds, y_pred_test), 0)

            # ========== evaluation (Acc.) ==========
            all_train_preds = all_train_preds.cpu()
            all_test_preds = all_test_preds.cpu()
            with torch.no_grad():
                train_acc = accuracy(train_labels, all_train_preds)
                if self.with_ground_truth:
                    test_acc = accuracy(test_labels, all_test_preds)
                    # F1-scores
                    microF1 = get_F1_score(test_labels, all_test_preds, average='micro')
                    macroF1 = get_F1_score(test_labels, all_test_preds, average='macro')
                    weightedF1 = get_F1_score(test_labels, all_test_preds, average='weighted')
                else:
                    test_acc = microF1 = macroF1 = weightedF1 = -1.
                # unsupervised cluster index
                if self.cluster_labels is not None:
                    ami = get_AMI(cluster_labels, all_test_preds)
                else:
                    ami = -1.
                backup = False
                if self._cur_epoch >= n_pass - 1:
                    self.ami_max = max(self.ami_max, ami)
                    if ami > self.ami_max - eps > 0:
                        self._cur_epoch_best = self._cur_epoch
                        self.save_model_weights()
                        backup = True
                        print('[current best] model weights backup')
                    elif self._cur_epoch % backup_stride == 0:
                        self.save_model_weights()
                        backup = True
                        print('model weights backup')

                t1 = time.time()

                # recording
                self._record(dur=t1 - t0,
                             train_loss=loss.item(),
                             train_acc=train_acc,
                             test_acc=test_acc,
                             AMI=ami,
                             microF1=microF1,
                             macroF1=macroF1,
                             weightedF1=weightedF1,
                             )

                dur_avg = np.average(self.dur)
                self.test_acc_max = max(test_acc, self.test_acc_max)

                if self.with_ground_truth:
                    logfmt = "Epoch {:04d} | Train Acc: {:.4f} | Test: {:.4f} (max={:.4f}) | AMI={:.4f} | Time: {:.4f}"
                    self._cur_log = logfmt.format(
                        self._cur_epoch, train_acc, test_acc, self.test_acc_max,
                        ami, dur_avg)
                else:
                    logfmt = "Epoch {:04d} | Train Acc: {:.4f} | AMI: {:.4f} (max={:.4f}) | Time: {:.4f}"
                    self._cur_log = logfmt.format(
                        self._cur_epoch, train_acc, ami, self.ami_max,
                        ami, dur_avg)

                if self._cur_epoch % 5 == 0 or backup:
                    print(self._cur_log)

            self._cur_epoch_adopted = self._cur_epoch

    def get_current_outputs(self,
                            feat_dict=None,
                            g=None,
                            batch_size=None,
                            **other_inputs):
        """ get the current states of the model output
        """
        if feat_dict is None:
            feat_dict = self.feat_dict
        if g is None:
            g = self.g
        from ..model import get_model_outputs
        outputs = get_model_outputs(
            self.model, feat_dict, g,
            batch_size=batch_size,
            device=self.device,
            **other_inputs
        )
        return outputs


