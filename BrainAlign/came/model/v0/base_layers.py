# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 00:30:14 2020

@author: Xingyan Liu
"""
import torch as th
from torch import nn

import dgl.function as fn
import dgl.nn as dglnn

from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
import numpy as np

from .heteroframe import HeteroGraphConv

DEFAULT_MOD_PARAMS = {
    "GraphAttentionLayer": dict(
        h_dim=8,
        n_heads=8,
        feat_drop=0.1,
        attn_drop=0.1,
        negative_slope=0.2,
        residual=False,
        activation=None,
        attn_type='add',  # 'mul' or 'add' as the original paper
        heads_fuse='flat',  # 'flat' or 'mean'
    ),
    "GraphConvLayer": dict(
        norm='right',
        use_weight=True,
        bias=True,
        activation=None,
    ),

}


def _unzip_canonical_etypes(canonical_etypes):
    scr_ntypes, etypes, dst_ntypes = list(zip(*canonical_etypes))
    return list(set(scr_ntypes)), etypes, list(set(dst_ntypes))


# In[]

class BaseMixConvLayer(nn.Module):
    """
    Basic one single layer for (cell-gene) graph convolution
    
    Parameters
    ----------
    in_dim_dict: dict[str: int]
        Input dimensions for each type of nodes.
    out_dim_dict: dict[str: int]
        Output dimensions for each type of nodes.

    canonical_etypes: A list of 3-length-tuples: (ntype_scr, etype, ntype_dst)
        can be accessed from a `dgl.heterograph.DGLHeteroGraph` object by 
        `G.canonical_etypes`


    mod_kwdicts: a list of tuples like (etype, kind, kwdict)
        etype: str, edge-type name.
        kind: str, either of 'gat' or 'gcn'.
        kwdict: dict.
            parameter dict for GNN model for that edge type. If an empty dict
            {} is given, model will be built using the defaut parameters.

    bias: bool, optional
        True if bias is added. Default: True
    
    activation: callable, optional
        Activation function. Default: None
    
    layernorm_ntypes: Union[Sequence[str], None]
    
    dropout: float, optional
        Dropout rate. Default: 0.0
    
    aggregate: Union[str, Callable] 
        Aggregation function for reducing messages from different types of relations.
        Default: 'sum'
        
    
    """

    def __init__(self,
                 in_dim_dict,
                 out_dim_dict,
                 #                 canonical_etypes,
                 mod_kwdicts,
                 #                 use_weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,  # ignored, just for Code compatibility
                 layernorm_ntypes=None,
                 dropout=0.0,
                 aggregate='sum',
                 ):

        super(BaseMixConvLayer, self).__init__()

        canonical_etypes = (kwtuple[0] for kwtuple in mod_kwdicts)
        scr_ntypes, etypes, dst_ntypes = _unzip_canonical_etypes(canonical_etypes)
        if isinstance(in_dim_dict, int):
            in_dim_dict = dict.fromkeys(scr_ntypes, in_dim_dict)
        if isinstance(out_dim_dict, int):
            out_dim_dict = dict.fromkeys(dst_ntypes, out_dim_dict)

        self.in_dim_dict = in_dim_dict
        self.out_dim_dict = out_dim_dict
        self.canonical_etypes = canonical_etypes
        self.build_mix_conv(mod_kwdicts, aggregate)

        self.bias = bias
        if bias:
            self.build_biases()
        self.activation = activation

        self.dropout = nn.Dropout(dropout)
        self.build_layernorm(layernorm_ntypes)

    def forward(self, g, inputs, **kwds):
        """Forward computation

        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        **kwds : ignored, only for compatibility.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        #        if len(self.use_weight_etypes) > 0:
        #            wdict = {e : {'weight' : w}
        #                     for e, w in self.weights.items()}
        #        else:
        #            wdict = {}
        #        hs = self.conv(g, h_dict, mod_kwargs=wdict)
        #        hs = self.conv(g, h_dict)
        #print('inputs', inputs)
        inputs_src = inputs_dst = inputs
        hs = self.conv(g, (inputs_src, inputs_dst), **kwds)

        def _apply(ntype, h):
            if self.use_layernorm:
                h = self.norm_layers[ntype](h)
            if self.bias:
                h = h + self.h_bias[ntype]
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}

    def build_layernorm(self, layernorm_ntypes=None, elementwise_affine=True, ):
        if layernorm_ntypes is not None:
            self.use_layernorm = True
            self.norm_layers = nn.ModuleDict({
                ntype: nn.LayerNorm(self.out_dim_dict[ntype],
                                    elementwise_affine=elementwise_affine)
                for ntype in layernorm_ntypes
            })
        else:
            self.use_layernorm = False

    def build_biases(self, ):
        self.h_bias = nn.ParameterDict()
        for ntype, out_dim in self.out_dim_dict.items():
            self.h_bias[ntype] = nn.Parameter(th.Tensor(out_dim))
            nn.init.zeros_(self.h_bias[ntype])

    def build_mix_conv(self, mod_kwdicts, aggregate):
        """
        mod_kwdicts: list of tuples, each tuple should be formed as:
            (canonical_etype, mod_kind, kwdict)
            
            for example:
            (('s_ntype', 'etype', 'd_ntype'), 'gat',
            dict(h_dim = 16, n_heads=8, feat_drop=0.05,
                 attn_drop=0.6, negative_slope=0.2,)
            )
        """
        conv_dict = {}

        for canonical_etype, mod_kind, kwdict in mod_kwdicts:
            ntype_scr, etype, ntype_dst = canonical_etype
            in_dim, out_dim = self.in_dim_dict[ntype_scr], self.out_dim_dict[ntype_dst]

            if mod_kind.lower() == 'gcn':
                conv_dict[etype] = self._make_gcn_layer(in_dim, out_dim, **kwdict)

            elif mod_kind.lower() == 'gat':
                if 'h_dim' not in kwdict.keys() or 'n_heads' not in kwdict.keys():
                    raise KeyError('the keyword dict for "gat" should contain '
                                   'keys named "h_dim" and "n_heads"')

                kwdict.update(in_dim=in_dim, out_dim=out_dim)
                conv_dict[etype] = self._make_gat_layer(**kwdict)
        self.conv = HeteroGraphConv(conv_dict, aggregate=aggregate)

    def _make_gcn_layer(self, in_dim, out_dim, **kwds):

        params = dict(
            norm='right',
            weight=True,
            bias=False,
            activation=None
        )
        if len(kwds) > 1:
            params.update(kwds)
        return GraphConvLayer(in_dim, out_dim, **params)

    def _make_gat_layer(self, in_dim, h_dim, n_heads, out_dim, **kwds):

        params = dict(
            feat_drop=0.1,
            attn_drop=0.1,
            negative_slope=0.2,
            residual=True,
            activation=None,
            attn_type='mul',  # or 'add' as the original paper
            heads_fuse='flat',  # 'flat' or 'mean'
        )
        if len(kwds) > 1:
            params.update(kwds)

        # make sure that the output dimensions are matched.
        if params['heads_fuse'] == 'flat':
            out_dim_gat = h_dim * n_heads
        else:
            out_dim_gat = h_dim
        if out_dim_gat != out_dim:
            raise ValueError(f'output dimensions are not matched! '
                             f'({out_dim_gat} != {out_dim})')

        return GraphAttentionLayer((in_dim, in_dim), h_dim, **params)


# In[]
class HeteroLayerNorm(nn.Module):
    """
    LayerNorm for different type of nodes
    """

    def __init__(self, in_dim_dict, **kwds):

        super(HeteroLayerNorm, self).__init__()

        self.in_dim_dict = in_dim_dict
        self.norm_layers = nn.ModuleDict()

        for key, in_dim in in_dim_dict.items():
            self.norm_layers[key] = nn.LayerNorm(in_dim, **kwds)

    def forward(self, h_dict):
        for key, h in h_dict.items():
            h_dict[key] = self.norm_layers[key](h)

        return h_dict


# In[]

class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.

    Parameters
    ----------
    in_dim : int
        Input feature size.
    out_dim : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    use_weight : bool or list[str], optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 rel_names,
                 *,
                 norm='right',
                 use_weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 layernorm_ntypes=None,
                 dropout=0.0,
                 aggregate='sum'):
        super(RelGraphConvLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rel_names = rel_names
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = HeteroGraphConv({
            rel: GraphConvLayer(in_dim, out_dim, norm=norm,
                                weight=False, bias=False)
            for rel in rel_names
        }, aggregate=aggregate)

        ## relation weights
        if not use_weight:
            self.use_weight_etypes = []
        else:
            if not isinstance(use_weight, bool):  # a list of edge-type-names
                self.use_weight_etypes = use_weight
            else:
                self.use_weight_etypes = rel_names
            self.weight = nn.Parameter(
                th.Tensor(len(self.use_weight_etypes), in_dim, out_dim))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        ## bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_dim))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_dim, out_dim))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))
        self.dropout = nn.Dropout(dropout)

        if layernorm_ntypes is not None:
            self.use_layernorm = True
            self.norm_layers = nn.ModuleDict({
                ntype: nn.LayerNorm(out_dim) for ntype in layernorm_ntypes
            })
        else:
            self.use_layernorm = False

    def forward(self, g, inputs, etypes=None):
        """Forward computation

        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        etypes: None, list[str]
            
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if len(self.use_weight_etypes) > 0:
            wdict = {self.use_weight_etypes[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(th.split(self.weight, 1, dim=0))}
        else:
            wdict = {}

        inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs, etypes, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.use_layernorm:
                h = self.norm_layers[ntype](h)

            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


# In[]

class GeneralRGCLayer(nn.Module):
    """ A variant of the Relational graph convolution (RGCN) layer, 
        allowing different number dimensions for each node-type.
    
    Parameters
    ----------
    in_dim_dict: Union[int, Mapping[str, int]]
        Input dimensions for each node-type
        
    out_dim_dict: Union[int, Mapping[str, int]]
        Input dimensions for each node-type
        
    canonical_etypes: A list of 3-length-tuples: (ntype_scr, etype, ntype_dst)
        can be accessed from a `dgl.heterograph.DGLHeteroGraph` object by 
        `G.canonical_etypes`
    
    norm: str
        one of 'right', 'left', 'both', 'none'
        
    use_weight: bool or list[str], optional.
        True if a linear layer is applied after message passing. Default: True
    
    bias: bool, optional
        True if bias is added. Default: True
    
    activation: callable, optional
        Activation function. Default: None
    
    layernorm_ntypes: Union[Sequence[str], None]
    
    dropout: float, optional
        Dropout rate. Default: 0.0
    
    aggregate: Union[str, Callable] 
        Aggregation function for reducing messages from different types of relations.
        Default: 'sum'
        
    """

    def __init__(self,
                 in_dim_dict,
                 out_dim_dict,
                 canonical_etypes,  # A list of 3-length-tuples: ntype_scr, etype, ntype_dst
                 norm='right',
                 use_weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,  # ignored, just for Code compatibility
                 batchnorm_ntypes=None,
                 layernorm_ntypes=None,
                 dropout_feat=0.,
                 dropout=0.0,
                 aggregate='sum',
                 ):
        super(GeneralRGCLayer, self).__init__()

        scr_ntypes, etypes, dst_ntypes = self._unzip_canonical_etypes(canonical_etypes)
        if isinstance(in_dim_dict, int):
            in_dim_dict = dict.fromkeys(scr_ntypes, in_dim_dict)
        if isinstance(out_dim_dict, int):
            out_dim_dict = dict.fromkeys(dst_ntypes, out_dim_dict)

        self.in_dim_dict = in_dim_dict
        self.out_dim_dict = out_dim_dict
        self.canonical_etypes = canonical_etypes
        self.bias = bias
        self.activation = activation

        if not use_weight:
            self.use_weight_etypes = []
        else:
            if not isinstance(use_weight, bool):  # a list of edge-type-names
                self.use_weight_etypes = use_weight
            else:
                self.use_weight_etypes = etypes

        self.weights = nn.ParameterDict()
        conv_dict = {}

        ### layers and weights for given etypes (`self.use_weight_etypes`)
        for ntype_scr, etype, ntype_dst in canonical_etypes:
            in_dim, out_dim = in_dim_dict[ntype_scr], out_dim_dict[ntype_dst]
            conv_dict[etype] = GraphConvLayer(
                in_dim, out_dim, norm=norm, weight=False, bias=False
            )
            if etype in self.use_weight_etypes:
                self.weights[etype] = nn.Parameter(th.Tensor(in_dim, out_dim))
                nn.init.xavier_uniform_(self.weights[etype],
                                        gain=nn.init.calculate_gain('relu'))

        self.conv = HeteroGraphConv(conv_dict, aggregate=aggregate)

        ### bias
        if bias:
            self.h_bias = nn.ParameterDict()
            for ntype, out_dim in self.out_dim_dict.items():
                self.h_bias[ntype] = nn.Parameter(th.Tensor(out_dim))
                nn.init.zeros_(self.h_bias[ntype])

        self.dropout_feat = nn.Dropout(dropout_feat)
        self.dropout = nn.Dropout(dropout)

        ### BatchNorm for given ntypes.
        if batchnorm_ntypes is not None:
            self.use_batchnorm = True
            self.batchnorm_layers = nn.ModuleDict({
                ntype: nn.BatchNorm1d(out_dim_dict[ntype], )
                for ntype in batchnorm_ntypes
            })
        else:
            self.use_batchnorm = False

        ### LayerNorm for given ntypes.
        if layernorm_ntypes is not None:
            self.use_layernorm = True
            self.norm_layers = nn.ModuleDict({
                ntype: nn.LayerNorm(out_dim_dict[ntype], elementwise_affine=True)
                for ntype in layernorm_ntypes
            })
        else:
            self.use_layernorm = False

    def forward(self, g, inputs: dict,
                etypes=None,
                norm=True, bias=True, activate=True,
                static_wdict={}):
        """(GeneralRGCLayer, modified `RelGraphConvLayer`)

        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        etypes: None, list[str]
            if provided, it can be used to subset the edge-types of the graph.
        
        static_wdict: dict[str, torch.Tensor], optional
            etype --> `w` of shape (n_edges, )
            Optional external weight tensor, this can be the intrinsic weights
            of the graph, and is NOT trainable. (for `GraphConvLayer.forward`)
            
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if len(self.use_weight_etypes) > 0:
            wdict = {e: {'weight': w,
                         'static_weight': static_wdict.get(e, None)}
                     for e, w in self.weights.items()}
        else:
            wdict = {}

        #        inputs_src = inputs_dst = inputs
        inputs = {ntype: self.dropout_feat(feat) for ntype, feat in inputs.items()}

        hs = self.conv(g, inputs, etypes, mod_kwargs=wdict)

        def _apply(ntype, h):

            if self.use_batchnorm and norm:
                h = self.batchnorm_layers[ntype](h)
            if self.use_layernorm and norm:
                h = self.norm_layers[ntype](h)

            if self.bias and bias:
                h = h + self.h_bias[ntype]
            if self.activation and activate:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}

    #    @staticmethod
    def _unzip_canonical_etypes(self, canonical_etypes):
        scr_ntypes, etypes, dst_ntypes = list(zip(*canonical_etypes))
        return list(set(scr_ntypes)), etypes, list(set(dst_ntypes))


# In[]


class GraphAttentionLayer(nn.Module):
    """ 
    Modified version of `dgl.nn.GATConv`
    * message passing with attentions.
    * directed and asymmetric message passing, allowing different dimensions
        of source and destination node-features.
    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 n_heads=8,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 attn_type='mul',  # or 'add' as the original paper
                 heads_fuse=None,  # 'flat' or 'mean'
                 ):
        super(GraphAttentionLayer, self).__init__()
        self._n_heads = n_heads
        self._in_src_dim, self._in_dst_dim = expand_as_pair(in_dim)
        self._out_dim = out_dim

        ### weights for linear feature transform
        if isinstance(in_dim, tuple):
            ### asymmetric case
            self.fc_src = nn.Linear(
                self._in_src_dim, out_dim * n_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_dim, out_dim * n_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_dim, out_dim * n_heads, bias=False)
        ### weights for attention computation
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, n_heads, out_dim)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, n_heads, out_dim)))
        if residual:
            if self._in_dst_dim != out_dim:
                self.res_fc = nn.Linear(
                    self._in_dst_dim, n_heads * out_dim, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        self.leaky_relu = nn.LeakyReLU(negative_slope)  # for thresholding attentions
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.reset_parameters()

        self.activation = activation  # output
        self.attn_type = attn_type
        self._set_attn_fn()
        self.heads_fuse = heads_fuse
        self._set_fuse_fn()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:  # bipartite graph neural networks
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, g, feat, return_attn=False):
        r"""Compute graph attention network layer.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        g = g.local_var()
        ### feature linear transform
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            feat_src = self.fc_src(h_src).view(-1, self._n_heads, self._out_dim)
            feat_dst = self.fc_dst(h_dst).view(-1, self._n_heads, self._out_dim)
        else:
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._n_heads, self._out_dim)
        # NOTE: GAT paper uses "first concatenation then linear projection"
        # to compute attention scores, while ours is "first projection then
        # addition", the two approaches are mathematically equivalent:
        # We decompose the weight vector a mentioned in the paper into
        # [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        # Our implementation is much efficient because we do not need to
        # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
        # addition could be optimized with DGL's built-in function u_add_v,
        # which further speeds up computation and saves memory footprint.
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        if self.heads_fuse == 'mul':
            er /= np.sqrt(self._out_dim)
        g.srcdata.update({'ft': feat_src, 'el': el})
        g.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        g.apply_edges(self.attn_fn)

        e = self.leaky_relu(g.edata.pop('e'))
        # compute softmax (normalized weights)
        g.edata['a'] = self.attn_drop(edge_softmax(g, e))
        # message passing
        g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
        rst = g.dstdata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_dim)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)

        # handling multi-heads
        rst = self.fuse_heads(rst)
        if return_attn:
            return rst, g.edata['a']
        return rst

    def _set_attn_fn(self, ):
        if self.attn_type == 'mul':
            self.attn_fn = fn.u_mul_v('el', 'er', 'e')
        elif self.attn_type == 'add':
            # use the same attention as the GAT paper
            self.attn_fn = fn.u_add_v('el', 'er', 'e')
        else:
            raise ValueError('`attn_type` shoul be either "add" (paper) or "mul"')

    def _set_fuse_fn(self, ):
        # function handling multi-heads
        if self.heads_fuse is None:
            self.fuse_heads = lambda x: x
        elif self.heads_fuse == 'flat':
            self.fuse_heads = lambda x: x.flatten(1)  # then the dim_out is of H * D_out
        elif self.heads_fuse == 'mean':
            self.fuse_heads = lambda x: x.mean(1)  # then the dim_out is of D_out
        elif self.heads_fuse == 'max':
            self.fuse_heads = lambda x: th.max(x, 1)[0]  # then the dim_out is of D_out


# In[]

class GraphConvLayer(nn.Module):
    """ 
    Notes
    -----
    * similar to `dgl.nn.GraphConv`, while normalization can be 'left', which
        is not allowed in `dgl.nn.GraphConv`.
    * directed and asymmetric message passing, allowing different dimensions
        of source and destination node-features.
    
    Parameters
    ----------
    in_dim : int
        Input feature size.
    out_dim : int
        Output feature size.
    norm : str, optional
        How to apply the normalizer. If is `'right'`, divide the aggregated messages
        by each node's in-degrees, which is equivalent to averaging the received messages.
        If is `'none'`, no normalization is applied. Default is `'both'`,
        where the :math:`c_{ij}` in the paper is applied.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.
    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 norm='left',
                 weight=True,
                 bias=True,
                 activation=None):
        super(GraphConvLayer, self).__init__()

        if norm not in ('none', 'both', 'right', 'left'):
            raise ValueError("""Invalid norm value. Must be either "none",
                             "both", "right", "left". But got "{}".""".format(norm))
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._norm = norm

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_dim, out_dim))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, g, feat, weight=None, static_weight=None):
        """(modified GCN)
        
        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.
        static_weight: torch.Tensor of shape (n_edges, ), optional
            Optional external weight tensor, this can be the intrinsic weights
            of the graph, and is NOT trainable.
        Returns
        -------
        torch.Tensor
            The output feature
        """
        g = g.local_var()
        if isinstance(feat, tuple):
            feat = feat[0]

        ### left normalization (for each source node)
        if self._norm in ('both', 'left'):
            degs = g.out_degrees().to(feat.device).float().clamp(min=1)
            if self._norm == 'both':
                norm = th.pow(degs, -0.5)
            else:  # left
                norm = 1.0 / degs
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp)
            feat = feat * norm

        if weight is not None:
            if self.weight is not None:
                raise ValueError('External weight is provided while at the same time the'
                                 ' module has defined its own weight parameter. Please'
                                 ' create the module with flag weight=False.')
        else:
            weight = self.weight
        if static_weight is None:
            message_func = fn.copy_src(src='h', out='m')
        else:
            g.edata['w_static'] = static_weight
            message_func = fn.u_mul_e('h', 'w_static', 'm')

        ### feature (linear) transform 
        if self._in_dim > self._out_dim:
            # mult W first to reduce the feature size for aggregation.
            if weight is not None:
                feat = th.matmul(feat, weight)
            g.srcdata['h'] = feat
            g.update_all(message_func,  # fn.copy_src(src='h', out='m'),
                         fn.sum(msg='m', out='h'))
            rst = g.dstdata['h']
        else:
            # aggregate first then mult W
            g.srcdata['h'] = feat
            g.update_all(message_func,  # fn.copy_src(src='h', out='m'),
                         fn.sum(msg='m', out='h'))
            rst = g.dstdata['h']
            if weight is not None:
                rst = th.matmul(rst, weight)

        ### normalize the aggregated (summed) message for each target node
        if self._norm in ('both', 'right'):
            degs = g.in_degrees().to(feat.device).float().clamp(min=1)
            if self._norm == 'both':
                norm = th.pow(degs, -0.5)
            elif self._norm == 'right':
                norm = 1.0 / degs
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_dim}, out={_out_dim}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

# In[]
