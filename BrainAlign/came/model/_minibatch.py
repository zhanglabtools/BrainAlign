# -*- coding: UTF-8 -*-
"""
@CreateDate: 2021/07/15
@Author: Qunlun Shen
@File: _minibatch.py
@Project: CAME
"""
from pathlib import Path
from typing import Sequence, Union, Mapping, Optional
import time
import numpy as np
import torch
from torch import Tensor
import dgl
import tqdm


def make_fanouts(etypes, etypes_each_layers, k_each_etype: Union[int, dict]):
    if isinstance(k_each_etype, int):
        k_each_etype = dict.fromkeys(etypes, k_each_etype)

    fanouts = []
    for _etypes in etypes_each_layers:
        _fanout = dict.fromkeys(etypes, 0)
        _fanout.update({e: k_each_etype[e] for e in _etypes})
        fanouts.append(_fanout)
    return fanouts


def involved_nodes(g,) -> dict:
    """ collect all the involved nodes from the edges on g
    (a heterogeneous graph)

    Examples
    --------

    >>> input_nodes, output_nodes, mfgs = next(iter(train_dataloader))
    >>> g.subgraph(involved_nodes(mfgs[0]))

    """
    from collections import defaultdict
    nodes = defaultdict(set)
    for stype, etype, dtype in g.canonical_etypes:
        src, dst = g.edges(etype=etype)
        nodes[stype].update(src.numpy())
        nodes[dtype].update(dst.numpy())

    nodes = {k: sorted(v) for k, v in nodes.items()}
    return nodes

