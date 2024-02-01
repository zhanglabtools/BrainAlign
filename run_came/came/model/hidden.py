# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 18:42:39 2021

@author: Xingyan Liu

Hidden layers (not including the embedding layer)

"""
from typing import Union, Sequence, Optional, List
# from torch.autograd import Variable, Function
# from collections import defaultdict
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from .base_layers import GeneralRGCLayer, HeteroLayerNorm


# In[]

class HiddenRGCN(nn.Module):
    """
    NOT sharing parameters across hidden layers
    
    """

    def __init__(self,
                 canonical_etypes,
                 h_dim: Union[Sequence[int], int] = 32,  # integer or (in_dim, h1, ..., out_dim)
                 num_hidden_layers: int = 2,
                 norm: str = 'right',
                 use_weight: bool = True,
                 dropout: Union[float, int] = 0.,
                 negative_slope: Union[float, int] = 0.2,
                 batchnorm_ntypes: Optional[Sequence[str]] = None,
                 layernorm_ntypes: Optional[Sequence[str]] = None,  # g.ntypes
                 share_layernorm: bool = False,  # ignored
                 bias: bool = True,
                 activate_out: bool = True,
                 **kwds):
        super(HiddenRGCN, self).__init__()
        self.canonical_etypes = canonical_etypes  # list(set(g.etypes))

        if isinstance(h_dim, Sequence):
            self.dims = tuple(h_dim)
            print('the parameter `num_hidden_layers` will be ignored')
        else:  # integer
            self.dims = tuple([h_dim] * (num_hidden_layers + 1))

        self.num_hidden_layers = len(self.dims) - 1
        self.dropout = dropout
#        self.use_self_loop = use_self_loop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        #        self.hidden_states = []

        self.layers = nn.ModuleList()
        # h2h
        for i, d in enumerate(self.dims[: -1]):
            if not activate_out and i == self.num_hidden_layers:
                activation = None
            else:
                activation = self.leaky_relu
            self.layers.append(GeneralRGCLayer(
                d, self.dims[i + 1], self.canonical_etypes,
                norm=norm,
                use_weight=use_weight,
                activation=activation,
                bias=bias,
                dropout=self.dropout,
                batchnorm_ntypes=batchnorm_ntypes,
                layernorm_ntypes=layernorm_ntypes,  # g.ntypes
                aggregate='sum',
            ))

    def forward(self, g, h_dict,
                norm=True, bias=True, activate=True,
                **kwds, ):
        """
        No copies made for h_dict, so make sure the forward functions do not
        make any changes directly on the h_dict !
        """
        self.hidden_states = []
        for layer in self.layers[: -1]:
            h_dict = layer(g, h_dict, **kwds)
            self.hidden_states.append(h_dict)
        # custom for the last layer (facilitate residual layer)
        h_dict = self.layers[-1](g, h_dict, norm=norm, bias=bias, activate=activate, **kwds)
        self.hidden_states.append(h_dict)

        return h_dict


class HiddenRRGCN(nn.Module):
    """
    Stacked hidden layers sharing parameters with each other, so that the
    dimensions for each layer should be the same.

    Parameters
    ----------

    canonical_etypes: dgl.DGLGraph or a list of 3-length-tuples
        if provide a list of tuples, each of the tuples should be like
        ``(node_type_source, edge_type, node_type_destination)``.
    h_dim: int
        number of dimensions of the hidden states
    num_hidden_layers: int
        number of hidden layers
    out_dim: Optional[int or Tuple[int]]
        if provided, an extra hidden layer will be add before the classifier
    norm: str
        normalization method for message aggregation, should be one of
        {'none', 'both', 'right', 'left'} (Default: 'right')
    use_weight: bool
        True if a linear layer is applied after message passing. Default: True
    dropout: float
        dropout-rate for the hidden layer
    negative_slope: float
        negative slope for ``LeakyReLU``
    batchnorm_ntypes: List[str]
        specify the node types to apply BatchNorm (Default: None)
    layernorm_ntypes: List[str]
        specify the node types to apply ``LayerNorm``
    share_layernorm: bool
        whether to share the LayerNorm across hidden layers

    See Also
    --------
    GeneralRGCLayer
    """

    def __init__(self,
                 canonical_etypes,
                 h_dim: int = 32,
                 num_hidden_layers: int = 2,
                 out_dim: Union[Sequence, int, None] = None,  # integer or (2, 2, ...)
                 norm: Union[str, None] = 'right',
                 use_weight=True,
                 dropout: float = 0.,
                 negative_slope: float = 0.05,
                 batchnorm_ntypes: Optional[Sequence[str]] = None,  # g.ntypes
                 layernorm_ntypes: Optional[Sequence[str]] = None,  # g.ntypes
                 share_layernorm: bool = True,
                 ):
        super(HiddenRRGCN, self).__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.canonical_etypes = canonical_etypes  # list(g.canonical_etypes)
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.layers = nn.ModuleList()
        # h2h
        self.shared_layer = GeneralRGCLayer(
            self.h_dim, self.h_dim, self.canonical_etypes,
            norm=norm,
            use_weight=use_weight,
            activation=self.leaky_relu,
            bias=True,
            dropout=self.dropout,
            batchnorm_ntypes=batchnorm_ntypes,
            layernorm_ntypes=layernorm_ntypes if share_layernorm else None,
            aggregate='sum',
        )

        self.share_layernorm = share_layernorm and (layernorm_ntypes is not None)
        if not self.share_layernorm and (layernorm_ntypes is not None):
            self.layernorms = nn.ModuleList()
        else:
            self.layernorms = None

        for i in range(self.num_hidden_layers):
            self.layers.append(self.shared_layer)

            if self.layernorms:
                dim_dict = dict.fromkeys(layernorm_ntypes, h_dim)
                self.layernorms.append(HeteroLayerNorm(dim_dict))

        if out_dim is not None:
            #            out_dim0 = out_dim if isinstance(self.out_dim, int) else out_dim[0]
            if isinstance(self.out_dim, Sequence):
                _out_dims = (self.h_dim,) + tuple(self.out_dim)
            elif isinstance(self.out_dim, int):
                _out_dims = (self.h_dim, self.out_dim)
            else:
                raise ValueError(
                    f'`out_dim` should be either Sequence or int if provided! '
                    f'Got {out_dim}')
            for i, dim in enumerate(_out_dims[: -1]):
                d_in, d_out = dim, _out_dims[i + 1]
                self.layers.append(GeneralRGCLayer(
                    d_in, d_out, self.canonical_etypes,
                    norm=norm,
                    use_weight=use_weight,
                    activation=self.leaky_relu,
                    bias=True,
                    dropout=self.dropout,
                    batchnorm_ntypes=batchnorm_ntypes,
                    layernorm_ntypes=layernorm_ntypes
                ))

    def forward(
            self,
            g_or_blocks,
            h_dict,  # residual=False,
            norm=True, bias=True, activate=True,
            **kwds,
    ):
        """
        No copies made for h_dict, so make sure the forward functions do not
        make any changes directly on the h_dict !
        """
        self.hidden_states = []
        if isinstance(g_or_blocks, DGLGraph):
            graphs = [g_or_blocks] * len(self.layers)
        else:
            graphs = g_or_blocks

        for i, layer in enumerate(self.layers[: -1]):
            h_dict = layer(graphs[i], h_dict, **kwds)

            if self.layernorms and i < self.num_hidden_layers:
                h_dict = self.layernorms[i](h_dict)
            self.hidden_states.append(h_dict)
        # for residual connection, may not be normalized
        h_dict = self.layers[-1](
            graphs[-1], h_dict, norm=norm, bias=bias, activate=activate, **kwds)
        self.hidden_states.append(h_dict)

        return h_dict


