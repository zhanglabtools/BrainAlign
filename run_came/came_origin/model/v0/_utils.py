# -*- coding: UTF-8 -*-
"""
@author: Xingyan Liu
@file: utils.py
@time: 2021-06-12
"""

from typing import Union, Sequence, Optional, Mapping, Dict, Any, List, Callable
import logging

import numpy as np
import torch
from scipy import sparse
import torch as th
from torch import Tensor
import torch.nn as nn
import dgl
import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from .cggc import CGGCNet
from .cgc import CGCNet
from ._minibatch import create_blocks, create_batch


def detach2numpy(x):
    if isinstance(x, Tensor):
        x = x.cpu().clone().detach().numpy()
    elif isinstance(x, Mapping):
        x = {k: detach2numpy(v) for k, v in x.items()}
    elif isinstance(x, List):
        x = [detach2numpy(v) for v in x]
    return x


def as_torch_tensor(x, dtype: Callable = Tensor):
    if isinstance(x, np.ndarray):
        x = dtype(x, )
    elif isinstance(x, Mapping):
        x = {k: as_torch_tensor(v) for k, v in x.items()}
    elif isinstance(x, List):
        x = [as_torch_tensor(v) for v in x]
    return x


def to_device(
        x: Union[Tensor, List[Tensor], Mapping[Any, Tensor], dgl.DGLGraph],
        device='cuda'):
    if not th.cuda.is_available():
        if 'cuda' in str(device):
            logging.warning("`to_device(x)`: CUDA is not available")
        device = 'cpu'
    if isinstance(x, Tensor):
        return x.to(device)
    elif isinstance(x, List) and isinstance(x[0], Tensor):
        return [xx.to(device) for xx in x]
    elif isinstance(x, Mapping):
        return {k: v.to(device) for k, v in x.items()}
    elif isinstance(x, dgl.DGLGraph):
        return [x.to(device)]
    else:
        raise NotImplementedError('Unresolved input type')


def concat_tensor_dicts(dicts: Sequence[Mapping], dim=0) -> Dict:
    """Helper function for merging feature_dicts from multiple batches"""
    keys = set()
    for d in dicts:
        keys.update(d.keys())
    result = {}
    for _key in list(keys):
        result[_key] = th.cat([d[_key] for d in dicts], dim=dim)
    return result


def onehot_encode(
        x: Sequence,
        classes=None,
        sparse_output: bool = True,
        astensor: bool = True,
        **kwargs
):
    x = detach2numpy(x)
    if not isinstance(x[0], Sequence):
        x = [[_x] for _x in x]
    binarizer = MultiLabelBinarizer(
        classes=classes,
        sparse_output=sparse_output and (not astensor),
    )
    x_onehot = binarizer.fit_transform(x)
    logging.info("classes = %s", binarizer.classes)
    if astensor:
        return Tensor(x_onehot)
    else:
        return x_onehot


def get_all_hidden_states(
        model: Union[CGGCNet, CGCNet, nn.Module],
        feat_dict: Mapping[Any, Tensor],
        g: dgl.DGLGraph,
        detach2np: bool = True,
        batch_size: Optional[int] = None,
        device=None,
        **other_inputs
):
    """ Get the embeddings on ALL the hidden layers
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    if batch_size is None:
        g = g.to(device)
        feat_dict = to_device(feat_dict, device=device)
        with torch.no_grad():
            model.train()
            # embedding layer
            h_embed = model.embed_layer(g, feat_dict)
            # hidden layers
            _ = model.rgcn(g, h_embed, **other_inputs)
            h_list = [h_embed] + model.rgcn.hidden_states
            # out_cls_dict = model.cell_classifier(g, h_list[-1], **other_inputs)
            # h_list.append(out_cls_dict)
    else:
        batch_list, all_idx, _, _ = create_batch(
            sample_size=feat_dict['cell'].shape[0],
            batch_size=batch_size, shuffle=False, label=False
        )
        batch_h_lists = []  # n_batches x n_layers x {'cell': .., 'gene': ..}
        with th.no_grad():
            for output_nodes in tqdm.tqdm(batch_list):
                model.train()  # semi-supervised learning
                block = create_blocks(g=g, output_nodes=output_nodes)
                _feat_dict = {
                    'cell': feat_dict['cell'][block.nodes['cell'].data['ids'], :]
                }
                _out_h_list = get_all_hidden_states(
                    model, _feat_dict, block, detach2np=False, device=device)
                batch_h_lists.append(_out_h_list)
        h_list = [concat_tensor_dicts(lyr) for lyr in zip(batch_h_lists)]

    if detach2np:
        h_list = [detach2numpy(h) for h in h_list]
    return h_list


def get_attentions(
        model: nn.Module,
        feat_dict: Mapping[str, Tensor],
        g: dgl.DGLGraph,
        fuse='mean',
        from_scratch: bool = True,
) -> sparse.spmatrix:
    """
    compute cell-by-gene attention matrix from model

    Returns
    -------
    sparse.spmatrix
        cell-by-gene attention matrix (sparse)
    """
    if from_scratch:
        h_dict = model.get_hidden_states(
            feat_dict=feat_dict, g=g, detach2np=False)
    else:
        h_dict = feat_dict

    # getting subgraph and the hidden states
    g_sub = g['gene', 'expressed_by', 'cell']

    # getting heterogeneous attention (convolutional) classifier
    HAC = model.cell_classifier.conv.mods['expressed_by']
    feats = (h_dict['gene'], h_dict['cell'])
    HAC.train(False)  # semi-supervised
    _out_dict, attn0 = HAC(g_sub, feats, return_attn=True)

    # constructing attention matrix
    if fuse == 'max':
        attn, _ = th.max(attn0, dim=1)
    elif fuse == 'mean':
        attn = th.mean(attn0, dim=1)
    else:
        raise ValueError("`fuse` should be either 'max' or 'mean'")

    attn = detach2numpy(attn).flatten()
    ig, ic = list(map(detach2numpy, g_sub.edges()))
    n_vnodes, n_obs = g.num_nodes('gene'), g.num_nodes('cell')
    attn_mat = sparse.coo_matrix(
        (attn, (ig, ic)), shape=(n_vnodes, n_obs)).tocsc().T

    return attn_mat


def get_model_outputs(
        model: nn.Module,
        feat_dict: Mapping[Any, Tensor],
        g: Union[dgl.DGLGraph, List[dgl.DGLGraph]],
        batch_size: Optional[int] = None,
        device=None,
        **other_inputs
):
    """
    Function facilitate to make mini-batch-wise forward pass

    Parameters
    ----------
    model:
        heterogeneous graph-neural-network model
    feat_dict:
        dict of feature matrices (Tensors)
    g:
        graph or a list or graph (blocks)
    batch_size: int or None
        the batch-size
    device: {'cpu', 'gpu', None}
    other_inputs:
        other inputs for model.forward function

    Returns
    -------
    Tensor or a dict of Tensor
    depends on the model, if batch_size is not None, results will be
    merged by batch.
    """
    if device is not None:
        model.to(device)

    if batch_size is None:
        if device is not None:
            feat_dict = to_device(feat_dict, device)
            g = g.to(device)
        with th.no_grad():
            model.train()  # semi-supervised learning
            outputs = model.forward(feat_dict, g, **other_inputs)
            # outputs = self.model.get_out_logits(feat_dict, g, **other_inputs)
        return outputs
    else:
        batch_list, all_idx, _, _ = create_batch(
            sample_size=feat_dict['cell'].shape[0],
            batch_size=batch_size, shuffle=False, label=False
        )
        batch_output_list = []
        with th.no_grad():
            model.train()  # semi-supervised learning
            for output_nodes in tqdm.tqdm(batch_list):
                block = create_blocks(g=g, output_nodes=output_nodes)
                _feat_dict = {
                    'cell': feat_dict['cell'][block.nodes['cell'].data['ids'], :]
                    # 'cell': feat_dict['cell'][block.nodes['cell'].data[dgl.NID], :]
                }
                print("TEST:", dgl.NID)
                if device is not None:
                    _feat_dict = to_device(_feat_dict, device)
                    block = to_device(block, device)
                # logging.debug('DEBUG', _feat_dict, block,)
                # logging.debug(other_inputs)
                _out = model.forward(_feat_dict, block, **other_inputs)
                batch_output_list.append(_out)
        outputs = concat_tensor_dicts(batch_output_list)

    return outputs


