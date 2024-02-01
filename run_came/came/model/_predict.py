# -*- coding: UTF-8 -*-
"""
@CreateDate: 2021/06/21
@Author: Xingyan Liu
@File: _predict.py
@Project: CAME
"""
import collections
import logging
from typing import Union, Sequence, Optional
import numpy as np
# from copy import deepcopy
import pandas as pd
import torch
from scipy.special import softmax, expit
from scipy import stats
from . import detach2numpy, onehot_encode
from ..utils.analyze import load_dpair_and_model
from ..utils.base import save_json_dict, load_json_dict


def sigmoid(x: np.ndarray):
    sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig


def as_probabilities(
        logits,
        mode: str = 'softmax',
):
    """
    Transform model-output logits into probabilities.

    Parameters
    ----------
    logits: np.ndarrary
    mode: str, should either be 'softmax' or 'sigmoid'.
        If 'sigmoid', make multi-label prediction. i.e., predict each class
        independently and each sample could be assigned by more than one class.
        If 'softmax', assume that the classes are mutually exclusive so that
        each sample will be assigned to only one class, with the maximal
        probability.
    Returns
    -------
    np.ndarrary
    """
    x = detach2numpy(logits)
    if mode.lower() == 'softmax':
        return softmax(x, axis=1)
    elif mode.lower() == 'sigmoid':
        return sigmoid(x,)


def predict_from_logits(logits, classes=None):
    """ transform `logits` to predictions
    logits: shape=(n_sample, n_classes)
    classes: list-like, unique categories
    """
    logits = detach2numpy(logits)
    preds = np.argmax(logits, axis=1)
    if classes is not None:
        preds = np.take(classes, preds)
    return preds


def predict(model: torch.nn.Module,
            feat_dict: dict,
            g=None,
            classes: Sequence = None,
            key: str = 'cell',
            **other_inputs):
    logits = model.forward(feat_dict, g, **other_inputs)[key]
    return predict_from_logits(logits, classes=classes)


def translate_binary_labels(
        labels,
        trans_mode: Union[str, int] = 'multi-label',
        classes: Optional[Sequence] = None,
):
    """
    Parameters
    ----------
    labels: np.ndarray or sparse matrix
        two-dimensional binary labels, of shape (n_samples, n_classes)
    trans_mode: str
        {0, 'ml', 'multi-label'}: return an array of multi-label tuples
        {1, 'unc', 'uncertain'}: "0"or">2" -> "uncertain"
        {2, 'unk', 'unknown'}: "0"->"unknown", ">2"->"multi-type"
    classes

    Returns
    -------
    depends on `trans_mode`
    """
    from sklearn.preprocessing import MultiLabelBinarizer
    if classes is None:
        classes = np.arange(labels.shape[1])
    else:
        classes = np.asarray(classes)
    binarizer = MultiLabelBinarizer(classes=classes)
    binarizer.fit([classes])
    label_tuples = binarizer.inverse_transform(labels)
    if trans_mode in {0, 'ml', 'multi-label', 'multilabel'}:
        return label_tuples
    elif trans_mode in {1, 'unc', 'uncertain'}:
        func = lambda x: x[0] if len(x) == 1 else 'uncertain'
        return list(map(func, label_tuples))
    elif trans_mode in {2, 'unk', 'unknown'}:
        def func(x):
            if len(x) <= 0:
                return 'unknown'
            elif len(x) == 1:
                return x[0]
            elif len(x) >= 2:
                return 'multi-type'
    else:
        raise ValueError("invalid `trans_mode`")
    return list(map(func, label_tuples))


def uncertainty_entropy(p, axis=1):
    """ normalized entropy """
    p = np.asarray(p)
    from scipy.stats import entropy
    # the default basis is `e`
    # normalize automatically
    return entropy(p, axis=axis) / np.log(p.shape[axis])


def uncertainty_gini(p, axis=1):
    """ normalized Gini-index """
    p = np.asarray(p)
    sums2 = np.square(p.sum(axis=axis))
    p2sum = np.square(p).sum(axis=axis)
    gini = 1. - p2sum / sums2
    return gini / (1 - 1 / p.shape[axis])


class Predictor(object):
    """ Class predictor (help identify the potential unknown classes)

    Input:
    * output logits of testing samples
    * output logits of training samples

    Output:
    * multi-label probabilities
    * binarized multi-labels
    * uncertainty (p-values)

    Steps:
    0. multi-label probabilities for each sample
    1. fit:
        Compute the background distributions for each class
    2. predict:
        * computing thresholds for each given classes
        * thresholding, reject
        * label the samples with uncertainty
    """

    def __init__(
            self,
            classes: Union[Sequence, int],
            mode: str = 'sigmoid',
            mean: Optional[Sequence] = None,
            std: Optional[Sequence] = None,
            mean_fg: Optional[Sequence] = None,
            std_fg: Optional[Sequence] = None,
    ):
        if isinstance(classes, int):
            self._classes = list(range(classes))
        else:
            _unitype = lambda x: int(x) if isinstance(x, np.integer) else x
            classes = [_unitype(c) for c in classes]
            self._classes = classes
        self._mode = mode
        self._background_mean_std = None
        self._foreground_mean_std = None

        if not (mean is None or std is None):
            self.set_backgrounds(mean, std)

        if not (mean_fg is None or std_fg is None):
            self.set_foregrounds(mean_fg, std_fg)

    @property
    def is_fitted(self) -> bool:
        return self._background_mean_std is not None

    @property
    def classes(self) -> list:
        return self._classes

    @property
    def n_classes(self) -> int:
        return len(self._classes)

    @classmethod
    def load(cls, json_path, encoding='utf-8'):
        dct = load_json_dict(json_path, encoding=encoding)
        predictor = Predictor(**dct)
        return predictor

    def save(self, json_path, encoding='utf-8'):
        mean, std = list(zip(*self._background_mean_std))
        
        dct = {
            'classes': self.classes,
            'mode': self._mode,
            'mean': list(map(float, mean)),
            'std': list(map(float, std)),
        }
        
        if self._foreground_mean_std:
            mean_fg, std_fg = list(zip(*self._foreground_mean_std))
            dct.update({
                'mean_fg': list(map(float, mean_fg)),
                'std_fg': list(map(float, std_fg))
            })
        # logging.info(dct)
        save_json_dict(dct, json_path, encoding=encoding)
        logging.info(json_path)

    def set_backgrounds(
            self,
            mean: Sequence,
            std: Sequence
    ):
        # assert len(self.classes) == len(mean)
        self._background_mean_std = tuple(zip(mean, std))

    def set_foregrounds(
            self,
            mean: Sequence,
            std: Sequence
    ):
        # assert len(self.classes) == len(mean)
        self._foreground_mean_std = tuple(zip(mean, std))

    def fit(self,
            logits: np.ndarray,
            labels: np.ndarray,
            label_encoded: bool = True,
            ):
        """
        Compute the background distributions for each class, i.e., mean and
        standard deviation.

        Parameters
        ----------
        logits
        labels
        label_encoded: bool
            whether the labels are encoded as integers, corresponding to each
            column index

        Returns
        -------
        self, i.e. fitted predictor
        """
        if len(labels.shape) == 1:
            if label_encoded:
                _classes = np.arange(self.n_classes)
            else:
                _classes = self.classes
            labels = onehot_encode(
                labels, classes=_classes, astensor=False, sparse_output=False
            ).astype(bool)
        assert logits.shape == labels.shape

        probas = as_probabilities(logits, mode=self._mode)
        # step 1: fit Gaussian distribution of background (negative samples)
        means_fg, stds_fg = [], []
        means, stds = [], []
        for i in range(self.n_classes):
            is_i = labels[:, i].flatten()#.astype(bool)
            m_i, std_i = stats.norm.fit(probas[is_i, i])
            means_fg.append(m_i)
            stds_fg.append(std_i)

            # p_bg = probas[~ is_i, i] # TODO: may be empty
            # m, std = stats.norm.fit(p_bg)
            m = - np.inf
            std = 0.25
            for j in range(self.n_classes):
                if j == i:
                    continue
                is_j = labels[:, j].flatten()#.astype(bool)
                p_ji = probas[is_j, i]
                _m, _std = stats.norm.fit(p_ji)
                if _m > m:
                    m, std = _m, _std
            means.append(m)
            stds.append(std)
        self.set_foregrounds(means_fg, stds_fg)
        self.set_backgrounds(means, stds)
        return self

    def decide_thresholds(self, p: float = 0.001, map_class: bool = False):
        if not self.is_fitted:
            raise AttributeError("predictor un-fitted!")
        thresholds = []
        for m, std in self._background_mean_std:
            thresholds.append(stats.norm(m, std).isf(p))

        # for m, std in self._foreground_mean_std:
            # thresholds.append(stats.norm(m, std).ppf(p))


        thresholds = []
        for (mb, _), (mf, _) in zip(self._background_mean_std,
                                    self._foreground_mean_std):
            thresholds.append((mb + mf) / 2)

        if map_class:
            thresholds = dict(zip(self.classes, thresholds))

        logging.info(f"thresholds: {thresholds}")
        return thresholds

    def predict(
            self,
            logits,
            p: float = 1e-4,
            trans_mode: Union[int, str, None] = 'top1-or-unknown',
    ) -> np.ndarray:
        probas = as_probabilities(logits, mode=self._mode)
        # decide thresholds and cut-off
        thresholds = self.decide_thresholds(p)
        binary_labels = np.vstack([
            (probas[:, i] > thresholds[i])
            for i in range(self.n_classes)
        ]).astype(int).T
        if trans_mode is not None:
            if trans_mode in {3, 'top', 'top1-or-unknown'}:
                preds = np.take(self.classes, np.argmax(probas, axis=1))
                preds[binary_labels.sum(1) <= 0] = 'unknown'
                return preds
            else:
                return self.translate_binary_labels(binary_labels, trans_mode)
        else:
            return binary_labels

    def predict_pvalues(
            self,
            logits,
    ) -> np.ndarray:
        probas = as_probabilities(logits, mode=self._mode)
        pvalues = np.vstack([
            stats.norm(m, std).sf(probas[:, i])
            for i, (m, std) in enumerate(self._background_mean_std)
        ]).T
        return pvalues

    def translate_binary_labels(
            self, labels,
            trans_mode: Union[str, int] = 'multi-label',
    ):
        """
        Parameters
        ----------
        labels: two-dimensional binary labels, of shape (n_samples, n_classes)
        trans_mode: str
            {0, 'ml', 'multi-label'}: return an array of multi-label tuples
            {1, 'unc', 'uncertain'}: "0"or">2" -> "uncertain"
            {2, 'unk', 'unknown'}: "0"->"unknown", ">2"->"multi-type"
        """
        return translate_binary_labels(labels, trans_mode, self.classes)

    def __str__(self):
        desc = f'''
        Predictor
        - is fitted: {self.is_fitted}
        - classes: {self.classes}
        - backgrounds: {self._background_mean_std}
        - foregrounds: {self._foreground_mean_std}
        '''
        return desc


def __test__():
    test_datadir = "./_temp/('Baron_human', 'Baron_mouse')-(06-20 19.49.07)"
    dpair, model = load_dpair_and_model(test_datadir)
    labels, classes = dpair.get_obs_labels(
        "cell_ontology_class", add_unknown_force=False)
    obs_ids1, obs_ids2 = dpair.obs_ids1, dpair.obs_ids2
    df_logits = pd.read_csv(f'{test_datadir}/df_logits.csv', index_col=0)
    classes = df_logits.columns

    proba = as_probabilities(df_logits.values, mode='sigmoid')

    predictor = Predictor(classes=classes)
    predictor.fit(
        df_logits.values[obs_ids1, :],
        labels[obs_ids1],
    )
    predictor.save(f'{test_datadir}/predictor.json')
    # predictor = Predictor.load(f'{test_datadir}/predictor.json')
    pred_test = predictor.predict(
        df_logits.values[obs_ids1, :], p=5e-2, trans_mode=3)
    logging.info(f"pred_test {len(pred_test)}:\n{pred_test}")
    logging.debug(collections.Counter(pred_test))

    pval_test = predictor.predict_pvalues(df_logits.values[obs_ids1, :])
    # logging.info(f"pval_test {pval_test.shape}:\n{pval_test}")

    logging.info(predictor)
    return predictor

