# -*- coding: UTF-8 -*-
"""
@Author: Xingyan Liu
@File: _base_trainer.py
@Date: 2021-08-22
@Project: CAME
"""
import os
import json
from pathlib import Path
from typing import Union, Optional, Sequence, Mapping, List
import logging
import numpy as np
import pandas as pd
import torch
from torch import Tensor
# from .base import check_dirs, save_json_dict
# from ..model import to_device, onehot_encode, multilabel_binary_cross_entropy

SUBDIR_MODEL = '_models'


def save_json_dict(dct, fname='test_json.json', encoding='utf-8'):
    with open(fname, 'w', encoding=encoding) as jsfile:
        json.dump(dct, jsfile, ensure_ascii=False)
    logging.info(fname)


def load_json_dict(fname, encoding='utf-8'):
    with open(fname, encoding=encoding) as f:
        dct = json.load(f)
    return dct


def get_checkpoint_list(dirname):
    """Get all saved checkpoint numbers (in the given directory)"""
    all_ckpts = [
        int(_fn.strip('weights_epoch.pt'))
        for _fn in os.listdir(dirname) if _fn.endswith('.pt')
    ]
    return all_ckpts


def plot_records_for_trainer(
        trainer, record_names, start=0, end=None,
        lbs=None, tt='training logs', fp=None,
        **kwds):
    if lbs is None:
        lbs = record_names
    if end is not None:
        end = int(min([trainer._cur_epoch + 1, end]))
    line_list = [getattr(trainer, nm)[start: end] for nm in record_names]

    return plot_line_list(line_list, lbs=lbs, tt=tt, fp=fp, **kwds)


def plot_line_list(ys, lbs=None,
                   ax=None, figsize=(4.5, 3.5),
                   tt=None, fp=None,
                   legend_loc=(1.05, 0),
                   **kwds):
    """
    ys: a list of lists, each sub-list is a set of curve-points to be plotted.
    """
    from matplotlib import pyplot as plt
    if lbs is None:
        lbs = list(map(str, range(len(ys))))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    for i, y in enumerate(ys):
        ax.plot(y, label=lbs[i], **kwds)
    ax.legend(loc=legend_loc)
    if tt is not None:
        ax.set_title(tt)
    ax.figure.savefig(fp, bbox_inches='tight',)

    return ax


class BaseTrainer(object):
    """
    DO NOT use it directly!

    """

    def __init__(self,
                 model,
                 lr: float = 1e-3,
                 l2norm: float = 1e-2,  # 1e-2 is tested for all datasets
                 dir_main: Union[str, Path] = Path('.'),
                 # **kwds  # for code compatibility (not raising error)
                 ):
        self.model = model
        self.device = None
        self.dir_main = None
        self.dir_model = None
        self.lr = None
        self.l2norm = None
        self._record_names = None
        self._train_logs = None
        self.with_ground_truth = None

        self.set_dir(dir_main)
        # self.set_inputs(
        #     model, feat_dict,
        #     train_labels, test_labels,
        #     train_idx, test_idx,
        # )

        # self.cluster_labels = cluster_labels
        self._set_train_params(
            lr=lr,
            l2norm=l2norm,
        )

        self._cur_epoch = -1
        self._cur_epoch_best = -1
        self._cur_epoch_adopted = 0

    def set_dir(self, dir_main=Path('.')):
        self.dir_main = Path(dir_main)
        self.dir_model = self.dir_main / SUBDIR_MODEL
        os.makedirs(self.dir_model, exist_ok=True)

        print('main directory:', self.dir_main)
        print('model directory:', self.dir_model)

    def _set_train_params(self,
                          lr: float = 1e-3,
                          l2norm: float = 1e-2,
                          ):
        """
        setting parameters for model training

        Parameters
        ----------
        lr: float
            the learning rate (default=1e-3)
        l2norm:
            the ``weight_decay``, 1e-2 is tested for all datasets
        """
        self.lr = lr
        self.l2norm = l2norm
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=l2norm)

    def set_recorder(self, *names):
        self._record_names = tuple(names)
        for nm in names:
            setattr(self, nm, [])

    def _record(self, **kwds):
        for nm in self._record_names:
            if nm in kwds:
                getattr(self, nm).append(kwds[nm])

    def make_train_logs(self, ):
        dct = {}
        for nm in self._record_names:
            v = getattr(self, nm)
            if len(v) > 0:
                dct[nm] = v
        self._train_logs = pd.DataFrame(dct)

    def write_train_logs(self, fn='train_logs.csv', fp=None):
        if fp is None:
            fp = self.dir_main / fn
        self.make_train_logs()
        self._train_logs.to_csv(fp, index_label='epoch')

    def save_whole_model(self, fn='model.pt', fp=None, **kwds):
        if fp is None:
            fp = self.dir_model / fn
        torch.save(self.model, fp, **kwds)
        print('model saved into:', fp)

    def save_model_weights(self, fp=None, **kwds):
        """ better NOT set the path `fp` manually~
        """
        n_epoch = self._cur_epoch
        if fp is None:
            fp = self.dir_model / f'weights_epoch{n_epoch}.pt'
        torch.save(self.model.state_dict(), fp, **kwds)

    def load_model_weights(self, n_epoch=None, fp=None, **kwds):
        if fp is None:
            if n_epoch is None:
                if self._cur_epoch_best > 0:
                    n_epoch = self._cur_epoch_best
                else:
                    n_epoch = self._cur_epoch
            fp = self.dir_model / f'weights_epoch{n_epoch}.pt'
        sdct = torch.load(fp, **kwds)
        self.model.load_state_dict(sdct)
        self._cur_epoch_adopted = n_epoch
        print('states loaded from:', fp)

    def save_checkpoint_record(self):
        cur_epoch = self._cur_epoch
        if self._cur_epoch_best > 0:
            cur_epoch_rec = self._cur_epoch_best
        else:
            cur_epoch_rec = self._cur_epoch
        all_ckpts = get_checkpoint_list(self.dir_model)
        all_ckpts = [x for x in all_ckpts if
                     x not in {cur_epoch, cur_epoch_rec}]
        ckpt_dict = {
            'recommended': cur_epoch_rec,
            'last': cur_epoch,
            'others': all_ckpts
        }
        save_json_dict(
            ckpt_dict, self.dir_model / 'checkpoint_dict.json')
        # load_json_dict(self.dir_model / 'checkpoint_dict.json')

    def plot_record(self,
                    record_names, start=0, end=None,
                    lbs=None, tt=None, fp=None,
                    **kwargs):
        # lbs: the line labels (corresponding to the `record_names`)
        return plot_records_for_trainer(
            self, record_names, start, end=end, lbs=lbs,
            tt=tt, fp=fp, **kwargs)

    def get_current_outputs(self, **other_inputs):
        """ get the current states of the model output
        """
        raise NotImplementedError

    def train(self, ):
        """ abstract method to be re-defined
        """
        raise NotImplementedError

    def train_minibatch(self, **kwargs):
        """ abstract method to be re-defined
        """
        raise NotImplementedError


def __test__():
    pass


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s-%(lineno)d-%(funcName)s(): '
               '%(levelname)s\n %(message)s')
    __test__()
