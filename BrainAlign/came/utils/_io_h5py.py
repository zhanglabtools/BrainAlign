# -*- coding: UTF-8 -*-
"""
@Author: Xingyan Liu
@File: _tmp_h5py.py
@Date: 2021-08-03
@Project: CAME
"""
import os
from pathlib import Path
from typing import Union, Optional, List, Mapping
import logging
import numpy as np
import h5py


def save_hidden_states(data_list: list, path: Union[Path, str]):
    """ Save hidden states into .h5 file

    Parameters
    ----------
    data_list
        a list of data matrix, or a list of dicts whose values are matrices
    path
        file-path ends with .h5, if not, '.h5' will be appended to it.

    Returns
    -------
    None
    """
    if not str(path).endswith('.h5'):
        path = str(path) + '.h5'
    f = h5py.File(path, 'w')
    if isinstance(data_list[0], dict):
        for i, dct in enumerate(data_list):
            for key, _data in dct.items():
                f.create_dataset(f'/layer{i}/{key}', data=_data)
    else:
        for i, _data in enumerate(data_list):
            f.create_dataset(f'/layer{i}', data=_data)

    f.close()


def load_hidden_states(path) -> List[dict]:
    """ Load hidden states from .h5 file
    the data structure should be like
        [
        'layer0/cell', 'layer0/gene',
        'layer1/cell', 'layer1/gene',
        'layer2/cell', 'layer2/gene'
        ]

    Parameters
    ----------
    path
        .h5 file path

    Returns
    -------
    values: a list of dicts
    """
    f = h5py.File(path, 'r')
    prefix = 'layer'
    keys = sorted(f.keys(), key=lambda x: int(x.strip(prefix)))
    # print(keys)
    values = [_unfold_to_dict(f[key]) for key in keys]
    return values


def _unfold_to_dict(d: h5py.Group) -> dict:
    dct = {}
    for key, val in d.items():
        if isinstance(val, h5py.Dataset):
            dct[key] = np.array(val)
    return dct


def _visit(f: h5py.File):
    tree = []

    def foo(_name, _obj):
        if isinstance(_obj, h5py.Dataset):
            tree.append(_name)
    f.visititems(foo)
    logging.info(f'tree={tree}')
    return tree


def __test__():
    n_cells = 100
    n_genes = 114
    n_dims = 64
    hidden_data = [
        {'cell': np.random.randn(n_cells, n_dims),
         'gene': np.random.randn(n_genes, n_dims)}
        for i in range(3)
    ]
    hidden_data.append({'cell': np.random.randn(n_cells, n_dims)})

    # logging.debug(hidden_data)
    save_hidden_states(hidden_data, '_tmp_data')
    f1 = h5py.File('_tmp_data.h5', 'r')
    h_list = load_hidden_states('../../_tmp_data.h5')
    # logging.info(values)
    for k, d in zip(f1.keys(), h_list):
        print(f'{k}: {list(d.keys())}')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s-%(lineno)d-%(funcName)s(): '
               '%(levelname)s\n %(message)s')
    __test__()
