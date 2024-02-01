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
# from ._minibatch import create_blocks, create_batch


def idx_hetero(feat_dict, id_dict):
    sub_feat_dict = {}
    for k, ids in id_dict.items():
        if k in feat_dict:
            sub_feat_dict[k] = feat_dict[k][ids]
        else:
            # logging.warning(f'key "{k}" does not exist in {feat_dict.keys()}')
            pass
    return sub_feat_dict


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
    elif isinstance(x, List) and isinstance(x[0], (Tensor, dgl.DGLGraph)):
        return [xx.to(device) for xx in x]
    elif isinstance(x, Mapping):
        return {k: v.to(device) for k, v in x.items()}
    elif isinstance(x, dgl.DGLGraph):
        return x.to(device)
    else:
        raise NotImplementedError('Unresolved input type')


def concat_tensor_dicts(dicts: Sequence[Mapping], dim=0) -> Dict:
    """Helper function for merging feature_dicts from multiple batches"""
    keys = set()
    for d in dicts:
        keys.update(d.keys())
    result = {}
    for _key in list(keys):
        result[_key] = th.cat([d[_key] for d in dicts if _key in d], dim=dim)
    return result


def infer_classes(y: Sequence or Sequence[Sequence]):
    """ infer all the classes from the given label sequence """
    if isinstance(y[0], Sequence):
        # multi-label scenario
        import itertools
        return sorted(set(itertools.chain.from_iterable(y)))
    else:
        return sorted(set(y))


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
    # logging.debug("classes = %s", binarizer.classes)
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
        sampler=None,
        device=None,
        **other_inputs
):
    """ Get the embeddings on ALL the hidden layers

    NOTE: Heterogeneous graph mini-batch sampling: first sample batch for nodes of one type, and then
        sample batch for the next type.
        For example, the nodes of the first few batches are all cells, followed by
        gene-nodes


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
            # TODO: get ride of storing hidden_states!
            h_list = [h_embed] + model.rgcn.hidden_states
            # out_cls_dict = model.cell_classifier(g, h_list[-1], **other_inputs)
            # h_list.append(out_cls_dict)
    else:
        ######################################
        if sampler is None:
            sampler = model.get_sampler(g.canonical_etypes, k_each_etype=50, )
            # remove the last layer
            sampler = dgl.dataloading.MultiLayerNeighborSampler(
                sampler.fanouts[:-1])

        dataloader = dgl.dataloading.DataLoader(
            g, {'cell': g.nodes('cell'), 'gene': g.nodes('gene')},
            sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0
        )
        batch_output_list = []
        with tqdm.tqdm(dataloader) as tq, torch.no_grad():
            for input_nodes, output_nodes, mfgs in tq:
                # inputs = idx_hetero(feat_dict, input_nodes)
                inputs = to_device(idx_hetero(feat_dict, input_nodes), device)
                mfgs = to_device(mfgs, device)
                # DGL默认前面的节点ID代表输出节点
                output_subids = {_k: torch.arange(len(_v)) for _k, _v in
                                 output_nodes.items()}
                _h_list = []
                h = model.embed_layer(mfgs[0], inputs)
                _h_list.append(idx_hetero(h, output_subids))

                h = model.rgcn(mfgs[1:], h, **other_inputs)
                # h_list.append(idx_hetero(h, output_subids))
                # TODO: not storing hidden states
                _h_list.extend([idx_hetero(_h, output_subids)
                               for _h in model.rgcn.hidden_states])
                batch_output_list.append(_h_list)

        h_list = [concat_tensor_dicts(lyr) for lyr in zip(*batch_output_list)]
        # for x in h_list:
        #     print("TEST-came.model._utils.py:", {k: v.shape for k, v in x.items()})
    if detach2np:
        h_list = [detach2numpy(h) for h in h_list]
    return h_list


def get_attentions(
        model: nn.Module,
        feat_dict: Mapping[str, Tensor],
        g: dgl.DGLGraph,
        fuse='mean',
        from_scratch: bool = True,
        is_train: bool = False,
        return_logits: bool = False,
        device=None,
):
    """
    compute cell-by-gene attention matrix from model

    Returns
    -------
    attn_mat: sparse.spmatrix
        cell-by-gene attention matrix (sparse)
    out_cell: Tensor (if return_logits is True)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # if batch_size is None: TODO: consider batch_size
    feat_dict = to_device(feat_dict, device=device)

    if from_scratch:
        model = model.to(device)
        h_dict = model.get_hidden_states(
            feat_dict=feat_dict,
            g=g.to(device), detach2np=False)
    else:
        h_dict = feat_dict

    # getting subgraph and the hidden states
    g_sub = g['gene', 'expressed_by', 'cell'].to(device)

    # getting heterogeneous attention (convolutional) classifier
    cell_classifier = model.cell_classifier.to(device)
    HAC = cell_classifier.conv.mods['expressed_by']
    feats = (h_dict['gene'], h_dict['cell'])
    HAC.train(is_train)
    _out_cell, attn0 = HAC(g_sub, feats, return_attn=True)

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
    if return_logits:
        out_cell = cell_classifier.apply_out({'cell': _out_cell})['cell']
        return attn_mat, out_cell
    return attn_mat


def get_model_outputs(
        model: nn.Module,
        feat_dict: Mapping[Any, Tensor],
        g: Union[dgl.DGLGraph, List[dgl.DGLGraph]],
        batch_size: Optional[int] = None,
        sampler=None,
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
        ######################################
        if sampler is None:
            sampler = model.get_sampler(g.canonical_etypes, 50)
        dataloader = dgl.dataloading.DataLoader(
            g, {'cell': g.nodes('cell')},
            sampler, device=device,
            batch_size=batch_size,
            shuffle=False, drop_last=False, num_workers=0
        )
        batch_output_list = []
        with tqdm.tqdm(dataloader) as tq, torch.no_grad():
            for input_nodes, output_nodes, mfgs in tq:
                inputs = to_device(idx_hetero(feat_dict, input_nodes), device)
                mfgs = to_device(mfgs, device)
                # mfgs = [blk.to(device) for blk in mfgs]
                batch_output_list.append(model(inputs, mfgs, **other_inputs))
        outputs = concat_tensor_dicts(batch_output_list)

    return outputs


