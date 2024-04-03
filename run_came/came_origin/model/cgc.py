# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:11:14 2021

@author: Xingyan Liu
"""

from typing import Union, Sequence, Optional
import logging

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .base_layers import GeneralRGCLayer  # , HeteroLayerNorm
from .base_layers import BaseMixConvLayer

from .loss import LabelSmoothingCrossEntropy
from .hidden import HiddenRRGCN, HiddenRGCN
from .cggc import CGGCNet


# In[]
def detach2numpy(x):
    if isinstance(x, th.Tensor):
        x = x.cpu().clone().detach().numpy()
    return x


# In[]
class CGCNet(CGGCNet):
    """
    Cell-Gene-Cell graph neural network (used when features are 1-to-1 aligned)

    Graph Convolutional Network for cell-gene Heterogeneous graph,
    with edges named as:

    * ('cell', 'express', 'gene'):        ov_adj
    * ('gene', 'expressed_by', 'cell'):   ov_adj.T
    * ('cell', 'self_loop_cell', 'cell'): sparse.eye(n_cells)

    Notes
    -----
    * gene embeddings are computed from cells;
    * weight sharing across hidden layers is allowed by setting
      ``share_hidden_weights`` as ``True``.
    * attention can be applied on the last layer (`self.cell_classifier`);
    * the graph for the embedding layer and the hidden layers can be different.

    Parameters
    ----------

    g_or_canonical_etypes: dgl.DGLGraph or a list of 3-length-tuples
        if provide a list of tuples, each of the tuples should be like
        ``(node_type_source, edge_type, node_type_destination)``.

    in_dim_dict: Dict[str, int]
        Input dimensions for each node-type
    h_dim: int
        number of dimensions of the hidden states
    h_dim_add: Optional[int or Tuple]
        if provided, an extra hidden layer will be add before the classifier
    out_dim: int
        number of classes (e.g., cell types)
    num_hidden_layers: int
        number of hidden layers
    norm: str
        normalization method for message aggregation, should be one of
        {'none', 'both', 'right', 'left'} (Default: 'right')
    use_weight: bool
        True if a linear layer is applied after message passing. Default: True
    dropout_feat: float
        dropout-rate for the input layer
    dropout: float
        dropout-rate for the hidden layer
    negative_slope: float
        negative slope for ``LeakyReLU``
    batchnorm_ntypes: List[str]
        specify the node types to apply BatchNorm (Default: None)
    layernorm_ntypes: List[str]
        specify the node types to apply ``LayerNorm``
    out_bias: bool
        whether to use the bias on the output classifier
    rel_names_out: a list of tuples or strings
        names of the output relations; if not provided, use all the relations
        of the graph.
    share_hidden_weights: bool
        whether to share the graph-convolutional weights across hidden layers
    attn_out: bool
        whether to use attentions on the output layer
    kwdict_outgat: Dict
        a dict of key-word parameters for the output graph-attention layers
    share_layernorm: bool
        whether to share the LayerNorm across hidden layers
    residual: bool
        whether to use the residual connection between the embedding layer and
        the last hidden layer. This operation may NOT be helpful in
        transfer-learning scenario. (Default: False)

    See Also
    --------
    CGGCNet

    """

    def __init__(self, **kwargs):
        super(CGCNet, self).__init__(**kwargs)


# In[]
# class CGCNet(nn.Module):
#     """
#     cell-gene-cell network
#     =======================
#     (used when features are 1-to-1 aligned)
#
#     Graph Convolutional Network for cell-gene Heterogenous graph,
#     with edges named as:
#         ('cell', 'express', 'gene'):        cg_net,
#         ('gene', 'expressed_by', 'cell'):   cg_net.T,
#         ('cell', 'self_loop_cell', 'cell'):      sparse.eye(n_cells),
#
#     * gene embeddings are computed from cells;
#     * weight sharing across hidden layers is allowed by setting
#         `share_hidden_weights=True`
#     * attention can be applied on the last layer (`self.cell_classifier`);
#     * the graph for the embedding layer and the hidden layers can be different;
#     * allow expression values as static edge weights. (but seems not work...)
#     """
#
#     def __init__(self,
#                  g_or_canonical_etypes,
#                  in_dim_dict={},
#                  h_dim=32,
#                  h_dim_add=None,  # None --> rgcn2
#                  out_dim=32,  # number of classes
#                  num_hidden_layers=1,
#                  norm='right',
#                  use_weight=True,
#                  dropout_feat=0.,
#                  dropout=0,
#                  negative_slope=0.2,
#                  batchnorm_ntypes=None,
#                  layernorm_ntypes=None,
#                  out_bias=False,
#                  rel_names_out=None,
#                  share_hidden_weights=False,
#                  attn_out=False,
#                  kwdict_outgat={},
#                  share_layernorm=True,
#                  residual=False,
#                  **kwds):  # ignored
#         super(CGCNet, self).__init__()
#         if isinstance(g_or_canonical_etypes, dgl.DGLGraph):
#             canonical_etypes = g_or_canonical_etypes.canonical_etypes
#         else:
#             canonical_etypes = g_or_canonical_etypes
#         self.in_dim_dict = in_dim_dict
#         if h_dim_add is not None:
#             if isinstance(h_dim_add, int):
#                 self.h_dims = (h_dim, h_dim_add)
#             elif isinstance(h_dim_add, Sequence):
#                 self.h_dims = (h_dim,) + tuple(h_dim_add)
#         else:
#             self.h_dims = (h_dim, h_dim)
#         self.out_dim = out_dim
#         self.rel_names_out = canonical_etypes if rel_names_out is None else rel_names_out
#         self.gcn_norm = norm
#         self.batchnorm_ntypes = batchnorm_ntypes
#         self.layernorm_ntypes = layernorm_ntypes
#         self.out_bias = out_bias
#         self.attn_out = attn_out
#
#         self._build_embed_layer(activation=nn.LeakyReLU(negative_slope),
#                                dropout_feat=dropout_feat,
#                                dropout=dropout)
#
#         hidden_model = HiddenRRGCN if share_hidden_weights else HiddenRGCN
#         self.rgcn = hidden_model(canonical_etypes,
#                                  h_dim=h_dim,
#                                  out_dim=h_dim_add,  # additional hidden layer if not None
#                                  num_hidden_layers=num_hidden_layers,
#                                  norm=self.gcn_norm,
#                                  use_weight=use_weight,
#                                  dropout=dropout,
#                                  negative_slope=negative_slope,
#                                  batchnorm_ntypes=batchnorm_ntypes,
#                                  layernorm_ntypes=layernorm_ntypes,
#                                  share_layernorm=share_layernorm,
#                                  )
#
#         self._build_cell_classifier(kwdict_gat=kwdict_outgat)
#         self.residual = residual
#
#     def forward(self,
#                 feat_dict, g,
#                 **kwds):
#
#         if self.residual:
#             h_dict0 = self.embed_layer(g, feat_dict, )
#             h_dict = self.rgcn.forward(g, h_dict0, norm=True, activate=False, **kwds)
#             relu = self.rgcn.leaky_relu
#             # residual connection
#             # h_dict = {'cell': relu(h_dict0['cell'] + h_dict['cell']),
#             #           'gene': relu(h_dict['gene'])}
#             h_dict['cell'] = relu(h_dict0['cell'] + h_dict['cell'])
#             h_dict['gene'] = relu(h_dict0['gene'] + h_dict['gene'])
#         else:
#             h_dict = self.embed_layer(g, feat_dict, )
#             h_dict = self.rgcn.forward(g, h_dict, **kwds).copy()
#
#         h_dict['cell'] = self.cell_classifier.forward(g, h_dict, **kwds)['cell']
#
#         return h_dict
#
#     def get_hidden_states(self,
#                           feat_dict=None, g=None,
#                           i_layer=-1,
#                           detach2np: bool = True,
#                           **kwds):
#         """
#         detach2np: whether tensors be detached and transformed into numpy.ndarray
#
#         """
#         if (feat_dict is not None) and (g is not None):
#             logging.info('Forward passing...')
#             # activate the random dropouts, which gives a better integrated embedding
#             self.train()
#             _ = self.forward(feat_dict, g=g, **kwds)
#         else:
#             logging.warning('No inputs were given for the forward passing, so the '
#                   'returned are the current hidden states of the model.')
#
#         h_dict = self.rgcn.hidden_states[i_layer]
#         if detach2np:
#             h_dict = {
#                 k: h.cpu().clone().detach().numpy()
#                 for k, h in h_dict.items()
#             }
#         return h_dict
#
#     def get_attentions(self, feat_dict, g, fuse='mean'):
#         """
#         output:
#             cell-by-gene attention matrix
#         """
#         # getting subgraph and the hidden states
#         g_sub = g.to('cuda')['gene', 'expressed_by', 'cell']
#         h_dict = self.get_hidden_states(feat_dict=feat_dict, g=g, detach2np=False)
#         # getting heterogenous attention (convolutional) classifier
#         HAC = self.cell_classifier.conv.mods['expressed_by']
#         feats = (h_dict['gene'], h_dict['cell'])
#         HAC.train(False)
#         _out_dict, attn0 = HAC(g_sub, feats, return_attn=True)
#
#         # constructing attention matrix
#         if fuse == 'max':
#             attn, _ = th.max(attn0, dim=1)
#         else:  # fuse == 'mean':
#             attn = th.mean(attn0, dim=1)
#         #        else:
#
#         attn = detach2numpy(attn).flatten()
#         ig, ic = list(map(detach2numpy, g_sub.edges()))
#         n_vnodes, n_obs = g.num_nodes('gene'), g.num_nodes('cell')
#
#         from scipy import sparse
#         attn_mat = sparse.coo_matrix(
#             (attn, (ig, ic)), shape=(n_vnodes, n_obs)).tocsc().T
#
#         return attn_mat
#
#     @staticmethod
#     def get_classification_loss(
#             out_cell, labels, weight=None,
#             smooth=True, smooth_eps=0.1, smooth_reduction='mean'):
#         # take out representations of nodes that have labels
#         # F.cross_entropy() combines `log_softmax` and `nll_loss` in a single function.
#         if smooth:
#             criterion = LabelSmoothingCrossEntropy(
#                 eps=smooth_eps, reduction=smooth_reduction)
#         else:
#             criterion = F.cross_entropy
#         class_loss = criterion(out_cell, labels, weight=weight)
#
#         return class_loss
#
#     def _build_embed_layer(self, activation=None, **kwds):
#         embed_params = dict(
#             in_dim_dict=self.in_dim_dict,
#             out_dim_dict=self.h_dims[0],
#             canonical_etypes=[('cell', 'express', 'gene'),
#                               ('cell', 'self_loop_cell', 'cell')],
#             norm='right',
#             use_weight=True,
#             bias=True,  #
#             activation=activation,  # None #nn.LeakyReLU(0.2)
#             batchnorm_ntypes=self.batchnorm_ntypes,
#             layernorm_ntypes=self.layernorm_ntypes,  # None, #
#             dropout_feat=0.0,
#             dropout=0.2,
#             aggregate='sum',
#         )
#         if len(kwds) > 0:
#             embed_params.update(**kwds)
#         self.embed_layer = GeneralRGCLayer(**embed_params)
#
#     def _build_cell_classifier(self, kwdict_gat={}):
#         if self.attn_out:
#             self.cell_classifier = self._make_out_gat(kwdict_gat)
#         else:
#             self.cell_classifier = self._make_out_gcn()
#
#     def _make_out_gat(self, kwdict_gat={}):
#         mod_params = {
#             "gat": dict(
#                 h_dim=self.out_dim,
#                 n_heads=8,  # 16, # 16 seems no better
#                 feat_drop=0.01,
#                 attn_drop=0.6,
#                 negative_slope=0.2,
#                 residual=False,
#                 activation=None,
#                 attn_type='add',  # 'mul' or 'add' as the original paper
#                 heads_fuse='mean',  # 'flat' or 'mean'
#             ),
#             "gcn": dict(
#                 norm='right',
#                 weight=True,
#                 bias=False,
#                 activation=None,
#             ),
#         }
#         if len(kwdict_gat) > 0:
#             mod_params["gat"].update(kwdict_gat)
#
#         kwdicts = [
#             (('gene', 'expressed_by', 'cell'), 'gat', mod_params['gat']),
#         ]
#         if ('cell', 'self_loop_cell', 'cell') in self.rel_names_out:
#             kwdicts.append(
#                 (('cell', 'self_loop_cell', 'cell'), 'gcn', mod_params['gcn'])
#             )
#
#         return BaseMixConvLayer(
#             self.h_dims[-1], self.out_dim,
#             mod_kwdicts=kwdicts,
#             bias=self.out_bias,
#             activation=None,
#             #                    dropout=self.dropout,
#             layernorm_ntypes=['cell'] if self.layernorm_ntypes is not None else None
#         )
#
#     def _make_out_gcn(self, ):
#         return GeneralRGCLayer(
#             self.h_dims[0], self.out_dim,
#             canonical_etypes=self.rel_names_out,
#             norm=self.gcn_norm,
#             use_weight=True,
#             activation=None,
#             bias=self.out_bias,  # if False, No bias in the last layer
#             self_loop=False,
#             layernorm_ntypes=['cell'])
