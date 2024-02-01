# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:14:05 2020

@author: Xingyan Liu

====================================================

Functions for downstream biological analysis

"""

import os
from pathlib import Path
from typing import Sequence, Union, Mapping, Optional, Callable
import logging
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy import sparse
import networkx as nx

import scanpy as sc
from ..datapair import DataPair, AlignedDataPair
from . import preprocess as pp
from ..model import CGGCNet, CGCNet
from .base import (
    make_pairs_from_lists,
    load_pickle,
    load_json_dict,
    # save_json_dict,
)


# In[]
def _unique_cats(labels, cats=None, ):
    uniques = np.unique(labels)  # sorted

    if cats is None:
        if hasattr(labels, 'cat'):
            cat_ord = labels.cat.categories
        elif isinstance(labels, pd.Categorical):
            cat_ord = labels.categories
        else:
            cat_ord = uniques  # sorted
        return [c for c in cat_ord if c in uniques]

    else:
        return cats


def _int2str(lst):
    return list(map(str, lst))


def _int_to_cat_strings(lst):
    cats = _int2str(np.unique(lst))  # ordered by default
    lst = _int2str(lst)
    return pd.Categorical(lst, categories=cats)


def load_dpair_and_model(
        dirname: Union[str, Path],
        subdir_model: str = '_models',
        ckpt: Union[int, str, None] = None,
):
    """ Load the output results of CAME.

    Parameters
    ----------
    dirname
        result directory of CAME
    subdir_model
        subdirectory where the model checkpoints are saved.
    ckpt
        specify which model checkpoint to load.

    Returns
    -------
    dpair: AlignedDataPair or DataPair
        the datapair object that stores the features and graph
    model: CGGCNet or CGCNet
        the graph neural network model

    Examples
    --------
    >>> dpair, model = came.load_dpair_and_model(came_resdir)

    access the feature dict

    >>> feat_dict = dpair.get_feature_dict(scale=True)

    access the heterogrnrous cell-gene graph

    >>> g = dpair.get_whole_net()

    access the reference and query sample-ids

    >>> obs_ids1, obs_ids2 = dpair.obs_ids1, dpair.obs_ids2

    passing forward

    >>> outputs = model(feat_dict, g)
    """
    import torch
    dirname = Path(dirname)
    model_dir = dirname / subdir_model
    element_dict = load_pickle(dirname / 'datapair_init.pickle')
    model_params = load_json_dict(dirname / 'model_params.json')

    if 'vv_adj' not in element_dict.keys():
        dpair = AlignedDataPair(**element_dict)
        model = CGCNet(**model_params)
    else:
        dpair = DataPair(**element_dict)
        model = CGGCNet(**model_params)
    if os.path.exists(dirname / 'obs.csv'):
        dpair.set_common_obs_annos(
                pd.read_csv(dirname / 'obs.csv', index_col=0),
                ignore_index=True
                )
    if ckpt is None:
        ckpt = _infer_ckpt(model_dir)
    if not torch.cuda.is_available():
        map_location = torch.device('cpu')
    else:
        map_location = None
    model.load_state_dict(
        torch.load(model_dir / f'weights_epoch{ckpt}.pt',
                   map_location=map_location)
    )
    return dpair, model


def _infer_ckpt(dirname) -> int:
    """ infer the proper checkpoint """
    # ckpt_file = model_dir / 'checkpoint_dict.json'
    fnames = [_f for _f in os.listdir(dirname) if _f.endswith('_dict.json')]
    if len(fnames) > 0:
        ckpt_file = dirname / fnames[0]
        ckpt = load_json_dict(ckpt_file)['recommended']

    else:
        from .train import get_checkpoint_list
        all_ckpts = get_checkpoint_list(dirname)
        ckpt = sorted(all_ckpts)[-1]
    return ckpt


def module_homo_weights(
        var: Union[pd.DataFrame, Sequence[pd.DataFrame]],
        df_links: pd.DataFrame,
        module,
        key_module='module',
        key_split='dataset',
        key_name='name',
        include_private=False):
    """ extract the weights between homologous genes in the given module

    Parameters
    ----------
    var
        a DataFrame or a pair of DataFrames storing the annotations of variables
        (genes), should be with columns {`key_module`, `key_split`, `key_name`}
    df_links
        a DataFrame recording the weights between each pair of homologies.
    module
        module name, e.g. '0', '1', ...
    key_module
        column name in var, of which the module-identities are stored
    key_split
        a column name in var, to split by (if var is a single DataFrame).
    key_name
        a column name in var, storing gene names
    include_private
        whether to include the weights (filled by zeros) between
        dataset/species-specific genes

    Returns
    -------
    a list of weights between homologous genes in the given module.
    """
    if isinstance(var, pd.DataFrame):
        if isinstance(module, str):
            var[key_module] = var[key_module].astype(str)
        from .base import split_df
        var1, var2 = split_df(var, by=key_split)
    elif isinstance(var, Sequence):
        var1, var2 = var
    else:
        raise ValueError(
            "`var` should be either a pd.DataFrame or a pair of DataFrames")
    genes1 = var1[var1[key_module] == module][key_name]
    genes2 = var2[var2[key_module] == module][key_name]
    if isinstance(df_links.index, pd.MultiIndex):
        df_links = df_links.reset_index()
    df_sub = pp.subset_matches(df_links, genes1, genes2)
    weights = df_sub['weight'].tolist()

    if include_private:
        genes_private1 = [g for g in genes1 if g not in df_sub.iloc[:, 0]]
        genes_private2 = [g for g in genes2 if g not in df_sub.iloc[:, 1]]
        print(len(weights), len(genes_private1), len(genes_private2))
        weights = weights + [0.] * (len(genes_private1) + len(genes_private2))

    return weights


def compute_common_private(
        genes1: Sequence,  # Union[Sequence, Mapping[str, Sequence]],
        genes2: Sequence,  # Union[Sequence, Mapping[str, Sequence]],
        gmap: pd.DataFrame, ):
    """compute common and private genes based on a given gene mapping
     (e.g.,homologous mapping)

    Parameters
    ----------
    genes1, genes2
        the two gene sets to compare
    gmap
        a DataFrame with at least two columns, storing homologous gene mappings.
        the first column corresponds to ``genes1``, and the second corresponds
        to ``genes2``

    Returns
    -------
    record: dict
    """
    # genes1 = dct_deg1[cl]
    # genes2 = dct_deg2[cl]
    # deg1homo = pp.get_homologies(gmap, dct_deg1[cl], )
    # common = sorted(set(deg1homo).intersection(genes2))  # wrong way!
    record = {}
    gmap_1v1 = pp.take_1v1_matches(gmap)
    if len(gmap_1v1) != len(gmap):
        subdf_varmap_1v1 = pp.subset_matches(gmap_1v1, genes1, genes2)
        record.update({
            "common1v1": subdf_varmap_1v1.apply(tuple, axis=1).tolist()
        })
    subdf_varmap = pp.subset_matches(gmap, genes1, genes2)
    deg_common1 = subdf_varmap.iloc[:, 0].tolist()
    deg_common2 = subdf_varmap.iloc[:, 1].tolist()

    private1 = [g for g in genes1 if genes1 not in deg_common1]
    private2 = [g for g in genes2 if genes2 not in deg_common2]
    record.update({
        # 'common1v1': subdf_varmap_1v1.apply(tuple, axis=1).tolist(),
        'common1': deg_common1,
        'common2': deg_common2,
        'private1': private1,
        'private2': private2,
    })
    return record


def compare_modules(
        mod_labels1, mod_labels2,
        df_var_links,
        avg_scaled: Optional[Sequence[pd.DataFrame]] = None,
        zscore_cut: float = 1.,
) -> Mapping[str, Mapping[str, list]]:
    """
    Compute common and private genes (cross-species) in each gene module.
    If `avg_scaled` is provided, the module genes enriched in each cell-type
    will be computed and compared.

    Parameters
    ----------
    mod_labels1, mod_labels2: pd.Series
        module labels
    df_var_links: pd.DataFrame

    avg_scaled:
        If provided, should be a pair of DataFrame storing the average
        expressions for each dataset (species), and the index should be
        the gene names.
    zscore_cut
        Cut-off of expression z-scores. This will be ignored if ``avg_scaled``
        is not provided.

    Returns
    -------
    record: dict of dicts

    """
    # mod_labels1 = gadt1.obs[key_module]
    # mod_labels2 = gadt2.obs[key_module]
    all_modules = sorted(
        set(mod_labels1).union(mod_labels2),
        key=lambda x: int(x))
    record = {}
    for mod in all_modules:
        genes1 = mod_labels1[mod_labels1 == mod].index
        genes2 = mod_labels2[mod_labels2 == mod].index
        df_sub = pp.subset_matches(df_var_links.reset_index(), genes1, genes2)
        logging.info(f'module {mod}: {len(genes1)}, {len(genes2)}')

        genes_common01 = df_sub.iloc[:, 0]
        genes_common02 = df_sub.iloc[:, 1]
        record[mod] = {
            'genes1': genes1.tolist(),
            'genes2': genes2.tolist(),
            # there may be duplicated genes
            'genes_common1': genes_common01.tolist(),
            'genes_common2': genes_common02.tolist(),
            'weights_common': df_sub['weight'].tolist(),
        }
        if avg_scaled is not None:
            avg_scaled1, avg_scaled2 = avg_scaled
            record_each_cl = module_enrichment_for_classes(
                avg_scaled1, avg_scaled2, genes1, genes2,
                genes_common1=genes_common01,
                genes_common2=genes_common02,
                zscore_cut=zscore_cut,
            )
            # record[mod].update(record_each_cl)
            record[mod]['cls'] = record_each_cl
    return record


def module_enrichment_for_classes(
        avg_scaled1: pd.DataFrame, avg_scaled2: pd.DataFrame,
        mod_genes1: Sequence, mod_genes2: Sequence,
        genes_common1: Sequence, genes_common2: Sequence,
        zscore_cut: float = 1.,
        **ignored
):
    """ For each (cell-)type and the given gene set (module) of two species,
    calculate the relatively highly expressed genes, and find those genes that
    are highly expressed in both species, and species-specific gene.

    Note that `genes_common1` and `genes_common2` should be of the same length.

    Returns
    -------
    record: dict of dicts
    """
    # concatenated avg_scaled should be split, for there may be gene-name
    # collisions that will raise error when indexing by names.
    classes = sorted(set(avg_scaled1.columns).union(avg_scaled2.columns))
    avg_scaled1 = avg_scaled1.loc[mod_genes1]
    avg_scaled2 = avg_scaled2.loc[mod_genes2]

    record = {}
    for cl in classes:
        # expr1 = avg_scaled.loc[mod_genes1, cl]
        # expr2 = avg_scaled.loc[mod_genes2, cl]
        expr1, expr2 = avg_scaled1[cl], avg_scaled2[cl]

        cl_genes1 = expr1[expr1 >= zscore_cut].index
        cl_genes2 = expr2[expr2 >= zscore_cut].index

        expr_common1 = avg_scaled1.loc[genes_common1, cl].values
        expr_common2 = avg_scaled2.loc[genes_common2, cl].values

        indicator = (expr_common1 >= 1.) & (expr_common2 >= 1.)  # 同时高表达的基因
        cl_genes_common1 = genes_common1[indicator].tolist()
        cl_genes_common2 = genes_common2[indicator].tolist()

        genes_private1 = sorted(set(cl_genes1).difference(cl_genes_common1))
        genes_private2 = sorted(set(cl_genes2).difference(cl_genes_common2))
        record[cl] = {
            'cl_genes_common1': cl_genes_common1,
            'cl_genes_common2': cl_genes_common2,
            'genes_private1': genes_private1,
            'genes_private2': genes_private2,
        }
    return record


def compare_degs_seurat(
        df_deg1, df_deg2,
        df_map: Optional[pd.DataFrame] = None,
        cut_padj: float = None,
        ntop: Optional[int] = None,
        key_group = 'cluster',
        key_gene = 'gene',
        key_pval = 'p_val_adj',
):
    """ Compare DEGs of common cell-types/clusters across datasets (or species)

    Parameters
    ----------
    df_deg1
        DEG table output from Seurat's function FindAllMarkers(), with columns
        'p_val', 'avg_logFC', 'pct.1', 'pct.2', 'p_val_adj', 'cluster', 'gene'
    df_deg2
        DEG table output from Seurat's function FindAllMarkers(), with the same
        format with ``df_deg1``
    df_map
        homologous gene mappings.
        If not given, genes will mapped by their names (case sensitive).
    cut_padj
        filter genes with adjusted-p-values lower than this value
    ntop: Optional
        if specified, take only top-{ntop} DEGs to compare

    Returns
    -------
    dict of dicts

    Examples
    --------

    # compute intersections and privates
    >>> record = compare_degs_adata(df_deg1, df_deg2, df_map, cut_padj=0.05)
    # inspect the results
    >>> pd.DataFrame(record)
    >>> pd.DataFrame(record).applymap(lambda x: len(set(x)))

    """
    # sorted by p-values
    df_deg1 = df_deg1.groupby(key_group).apply(
        lambda x: x.sort_values(key_pval)).reset_index(drop=True)
    df_deg2 = df_deg2.groupby(key_group).apply(
        lambda x: x.sort_values(key_pval)).reset_index(drop=True)

    # filter DEGs that are not significant enough
    df_deg1 = df_deg1[df_deg1[key_pval] <= cut_padj]
    df_deg2 = df_deg2[df_deg2[key_pval] <= cut_padj]

    if ntop is None:
        dct_deg1 = df_deg1.groupby(key_group)[key_gene].apply(list).to_dict()
        dct_deg2 = df_deg2.groupby(key_group)[key_gene].apply(list).to_dict()
    else:
        dct_deg1 = df_deg1.groupby(key_group)[key_gene].apply(
            lambda x: x.head(ntop).tolist()).to_dict()
        dct_deg2 = df_deg2.groupby(key_group)[key_gene].apply(
            lambda x: x.head(ntop).tolist()).to_dict()
    return compare_deg_dicts(dct_deg1, dct_deg2, df_map)


def compare_degs_adata(
        adata1: sc.AnnData, adata2: sc.AnnData,
        df_map: Optional[pd.DataFrame] = None,
        cut_padj=0.05,
        cut_logfc=0.25,
        ntop: Optional[int] = None,
        key_group: str = 'group',
        key_gene: str = 'names',
):
    """ Compare DEGs of common cell-types/clusters across datasets (or species)

    Parameters
    ----------
    adata1, adata2: ``sc.AnnData`` objects whose DEGs to be compared
    df_map
        homologous gene mappings
    cut_padj
        filter genes with adjusted-p-values lower than this value
    cut_logfc
        filter genes with long-fold-changes higher than this value
    ntop: Optional
        if specified, take only top-{ntop} DEGs to compare
    key_group
    key_gene

    Returns
    -------
    dict of dicts

    Examples
    --------
    # normalize data
    >>> pp.normalize_default(adata1)
    >>> pp.normalize_default(adata2)
    # compute DEGs
    >>> sc.tl.rank_genes_groups(adata1, groupby='cell_ontology_class')
    >>> sc.tl.rank_genes_groups(adata2, groupby='cell_ontology_class')
    # compute intersections and privates
    >>> record = compare_degs_adata(adata1, adata2, df_map, cut_padj=0.05)
    # inspect the results
    >>> pd.DataFrame(record)
    >>> pd.DataFrame(record).applymap(lambda x: len(set(x)))

    """
    df_deg1 = pp.get_marker_info_table(
        adata1, cut_padj=cut_padj, cut_logfc=cut_logfc)
    df_deg2 = pp.get_marker_info_table(
        adata2, cut_padj=cut_padj, cut_logfc=cut_logfc)

    if ntop is None:
        dct_deg1 = df_deg1.groupby(key_group)[key_gene].apply(list).to_dict()
        dct_deg2 = df_deg2.groupby(key_group)[key_gene].apply(list).to_dict()
    else:
        dct_deg1 = df_deg1.groupby(key_group)[key_gene].apply(
            lambda x: x.head(ntop).tolist()).to_dict()
        dct_deg2 = df_deg2.groupby(key_group)[key_gene].apply(
            lambda x: x.head(ntop).tolist()).to_dict()

    return compare_deg_dicts(dct_deg1, dct_deg2, df_map)


def compare_deg_dicts(
        dct_deg1, dct_deg2,
        df_map: Optional[pd.DataFrame] = None):
    """

    Parameters
    ----------
    dct_deg1: Dict[Any, Sequence]
        a dict of DEG lists, where each key corresponds to a cell group.
    dct_deg2: Dict[Any, Sequence]
        a dict of DEG lists, where each key corresponds to a cell group.
    df_map:
        homologous gene mappings
    Returns
    -------
    record: dict
    """
    common_types = sorted(set(dct_deg1.keys()).intersection(dct_deg2.keys()))
    # gmap = df_varmap if tag_1v1 == '' else df_varmap_1v1

    record = {}
    for cl in common_types:
        deg1 = dct_deg1[cl]
        deg2 = dct_deg2[cl]
        if df_map is None:
            common1v1 = set(deg1).intersection(deg2)
            deg_common1 = deg_common2 = common1v1
        else:
            subdf_varmap_1v1 = pp.subset_matches(
                pp.take_1v1_matches(df_map), deg1, deg2)
            common1v1 = subdf_varmap_1v1.apply(tuple, axis=1).tolist()

            subdf_varmap = pp.subset_matches(df_map, deg1, deg2)
            deg_common1 = subdf_varmap.iloc[:, 0].tolist()
            deg_common2 = subdf_varmap.iloc[:, 1].tolist()

        private1 = [g for g in deg1 if deg1 not in deg_common1]
        private2 = [g for g in deg2 if deg2 not in deg_common2]
        record[cl] = {
            'common1v1': common1v1,
            'common1': deg_common1,
            'common2': deg_common2,
            'private1': private1,
            'private2': private2,
        }
    return record


def weight_linked_vars_by_expr(
        dpair: DataPair,
        labels_or_probs: Union[tuple, np.ndarray, pd.DataFrame],
):
    """ Compute the weights between homologies by their average expressions
    across (cell) groups.

    Parameters
    ----------
    dpair
        the ``DataPair`` object
    labels_or_probs
        group (cell-type) labels or soft assignments (probabilities)

    Returns
    -------
    df_var_links_expr: pd.DataFrame
        the weights between each pair of homologous genes
    avg: pd.DataFrame
        concatenated average expressions

    """
    index_names = dpair.dataset_names
    if not isinstance(labels_or_probs, pd.DataFrame):
        # using the original labels
        # labels1 = adt.obs['celltype'][dpair.obs_ids1]
        # labels2 = adt.obs['celltype'][dpair.obs_ids2]
        if isinstance(labels_or_probs, tuple) and len(labels_or_probs) == 2:
            labels1, labels2 = labels_or_probs
        else:
            labels = labels_or_probs
            labels1, labels2 = labels[dpair.obs_ids1], labels[dpair.obs_ids2]
        _avg1 = pp.group_mean(dpair._ov_adjs[0], labels1,
                              features=dpair.vnode_names1)
        _avg2 = pp.group_mean(dpair._ov_adjs[1], labels2,
                              features=dpair.vnode_names2)
        avg = pd.concat([_avg1, _avg2], axis=0).fillna(0.)

        classes_common = sorted(set(_avg1.columns).intersection(_avg2.columns))
        avg = avg[classes_common]
    else:
        logging.info("Compute average expression by soft group-assignment")
        df_probs = labels_or_probs
        probs = df_probs.values
        classes = df_probs.columns
        # probs = came.as_probabilities(out_cell, mode='softmax')
        _avg_soft1 = pd.DataFrame(
            dpair._ov_adjs[0].T.dot(probs[dpair.obs_ids1]) / probs[
                dpair.obs_ids1].sum(axis=0),
            index=dpair.vnode_names1, columns=classes
        )
        _avg_soft2 = pd.DataFrame(
            dpair._ov_adjs[1].T.dot(probs[dpair.obs_ids2]) / probs[
                dpair.obs_ids2].sum(axis=0),
            index=dpair.vnode_names2, columns=classes
        )

        avg = pd.concat([_avg_soft1, _avg_soft2], axis=0)  # .fillna(0)

    df_var_links_expr = weight_linked_vars(
        avg.values,
        dpair._vv_adj, names=dpair.get_vnode_names(),
        matric='correlation', index_names=index_names,
    )
    # there may be species-specific genes with zero-expressions in the other species
    df_var_links_expr['weight'] = df_var_links_expr['weight'].fillna(0.)
    df_var_links_expr['distance'] = df_var_links_expr['distance'].fillna(1.)
    return df_var_links_expr, avg


def weight_linked_vars(
        X: np.ndarray,
        adj: sparse.spmatrix,
        names: Optional[Sequence] = None,
        metric: str = 'cosine',
        func_dist2weight: Optional[Callable] = None,
        sigma: Optional[float] = None,
        sort: bool = True,
        index_names=(0, 1),
        **ignored) -> pd.DataFrame:
    """ Computes the similarity of each linked (homologous) pair of variables.

    Parameters
    ----------
    X: np.ndarray
        feature matrix of shape (N, M), where N is the number of sample and M is
        the feature dimensionality.
    adj: 
        sparse.spmatrix; binary adjacent matrix of shape (N, N). Note that only
        the upper triangle of the matrix will be considered!
    names: 
        a sequence of names for rows of `X`, of shape (N,)
    metric:
        the metric to quantify the similarities of the given vectors (embeddings)
    sort
        whether to sort by the resulting weights
    index_names
        a pair of names for the multi-index of the resulting DataFrame.
        e.g., a pair of dataset or species names (in cross-species scenario)
    
    Returns
    -------
    df: pd.DataFrame
        with columns [``index_names[0]``, ``index_names[1]``, "distance", "weight"]
    """
    adj = sparse.triu(adj).tocoo()

    foo = lambda x1, x2: cdist(x1[None, :], x2[None, :], metric=metric)[0, 0]
    dists = np.array(list(map(foo, X[adj.row, :], X[adj.col, :])))

    # transform dists to weights
    if func_dist2weight is None:
        if metric == 'euclidean':  # for 'euclidean' distances
            sigma = 2 * np.var(dists) if sigma is None else sigma
            weights = np.exp(-np.square(dists) / sigma)
        else:
            weights = 1 - dists
    else:
        weights = func_dist2weight(dists)

    # constructing the resulting dataframe
    if names is None:
        index = pd.MultiIndex.from_tuples(
            list(zip(adj.row, adj.col)), names=index_names)
    else:
        index = pd.MultiIndex.from_tuples(
            list(zip(np.take(names, adj.row), np.take(names, adj.col))),
            names=index_names)
    df = pd.DataFrame({'distance': dists, 'weight': weights, },
                      index=index)
    if sort:
        logging.info('sorting links by weights')
        df.sort_values(by='weight', ascending=False, inplace=True)

    return df


# =================[ module extraction ]=================
def _filter_for_abstract(
        var_labels1: pd.Series,
        var_labels2: pd.Series,
        avg_expr1: pd.DataFrame,
        avg_expr2: pd.DataFrame,
        df_var_links: pd.DataFrame,
        name=None):
    if name is None:
        foo_filter = pd.notna
    else:
        foo_filter = lambda x: x != name
    kept1 = var_labels1.apply(foo_filter)# list(map(foo_filter, var_labels1))
    kept2 = var_labels2.apply(foo_filter)#list(map(foo_filter, var_labels2))

    var_labels1, var_labels2 = var_labels1[kept1], var_labels2[kept2]
    # var_labels1, var_labels2 = np.take(var_labels1, kept1), np.take(var_labels2, kept2)
    avg_expr1, avg_expr2 = avg_expr1[kept1], avg_expr2[kept2]
    vars1, vars2 = avg_expr1.index, avg_expr2.index
    kept_index = list(filter(
        lambda x: x[0] in vars1 and x[1] in vars2,
        df_var_links.index.values
    ))
    df_var_links = df_var_links.loc[kept_index, :]
    # print(sum(kept1), sum(kept1))
    return var_labels1, var_labels2, avg_expr1, avg_expr2, df_var_links,


def _net_density(mat: sparse.spmatrix):
    n, m = mat.shape
    if n == m:
        max_nnz = n * (n - 1)
    else:
        max_nnz = n * m
    if sparse.issparse(mat):
        nnz = mat.nnz
    else:
        nnz = (mat != 0).sum()
    return nnz / max_nnz


def _match_groups(mod_lbs1: Union[pd.Series, Mapping],
                  mod_lbs2: Union[pd.Series, Mapping],
                  df_homo: Optional[pd.DataFrame] = None,
                  key_weight=None,
                  keys_link: Union[None, Sequence[str]] = None,
                  fix_left=True,
                  as_cats=True,
                  **kwds):
    """
    match two different sets of groups (modules), and re-arrange the group_labels
    """
    tokens1 = list(mod_lbs1.keys())
    tokens2 = list(mod_lbs2.keys())

    #    print(mod_lbs1.value_counts())
    if df_homo is None:
        unique_tokens = list(set(tokens1).union(tokens2))
        df_homo = pd.DataFrame({
            'src': unique_tokens,
            'tgt': unique_tokens,
            'weight': 1.,
        })
        keys_link = ('src', 'tgt')
    # cross-set module connections
    mod_conn = aggregate_links(
        df_homo, mod_lbs1, mod_lbs2,
        key_weight=key_weight,
        keys_link=keys_link,
        norm_sizes='auto',
        **kwds
    )
    n1, n2 = mod_conn.shape
    # re-arrange the module-numbers
    mod_conn = pp.order_contingency_mat(mod_conn, axis=1)  # re-order the rows
    mod_conn = pp.order_contingency_mat(mod_conn, axis=0)  # re-order the columns
    #    print(mod_conn)

    # change the labels
    new_cats1 = _int2str(np.arange(n1))
    map1 = dict(list(zip(mod_conn.index, new_cats1)))
    mod_conn.index = new_cats1
    new_mod_lbs1 = np.array([map1[_lb] for _lb in mod_lbs1])

    new_cats2 = _int2str(np.arange(n2))
    map2 = dict(list(zip(mod_conn.columns, new_cats2)))
    mod_conn.columns = new_cats2
    new_mod_lbs2 = np.array([map2[_lb] for _lb in mod_lbs2])

    if as_cats:
        new_mod_lbs1 = pd.Categorical(new_mod_lbs1, categories=new_cats1)
        new_mod_lbs2 = pd.Categorical(new_mod_lbs2, categories=new_cats2)

    return new_mod_lbs1, new_mod_lbs2, mod_conn


def _subgroup_edges(
        adj, labels, groups, node_names=None,
        key_row='source',
        key_col='target',
        key_data='weight',
        as_attrs=False):
    kept = pp.take_group_labels(labels, groups, indicate=True)
    ids = np.flatnonzero(kept)
    adj = adj[ids, :][:, ids]
    if node_names is None:
        names = ids
    else:
        names = node_names[ids]
    # transformed into edge-df
    df_edge = adj_to_edges(
        adj,
        rownames=names,
        colnames=names,
        key_row=key_row,
        key_col=key_col,
        key_data=key_data,
        as_attrs=as_attrs)

    return df_edge


def adata_subgroup_edges(
        adata, groups, groupby='module', key=None,
        **kwds):
    """
    take a subset of the KNN-graph from adata, melt to an edge-dataframe, with
    columns ['source', 'target', 'weight'] by default.
    """

    dist, conn = get_adata_neighbors(adata, key=key, )
    lbs = adata.obs[groupby]
    ndnames = adata.obs_names

    df_edge = _subgroup_edges(conn, lbs, groups, ndnames, **kwds)
    return df_edge


def edges_from_adata(
        adata, kind='conn',
        key_neigh=None,
        key_row='source',
        key_col='target',
        as_attrs=False):
    """ transform the connectivities in `adata` into an edge-dataframe, with
    columns ['source', 'target', 'weight'] by default.
    
    kind: 'conn', 'dist'
    """
    dist, conn = get_adata_neighbors(adata, key=key_neigh)
    names = adata.obs_names
    if kind == 'conn':
        df_edge = adj_to_edges(
            sparse.triu(conn),
            rownames=names,
            colnames=names,
            key_row=key_row,
            key_col=key_col,
            key_data='weight',
            as_attrs=as_attrs)
    elif kind == 'dist':
        df_edge = adj_to_edges(
            sparse.triu(dist),
            rownames=names,
            colnames=names,
            key_row=key_row,
            key_col=key_col,
            key_data='distance',
            as_attrs=as_attrs)
    else:
        raise ValueError

    return df_edge


def nx_from_adata(
        adata, key_neigh=None,
        keys_attr=None,
) -> nx.Graph:
    """ nx.Graph from the KNN graph of `adata`
    """
    node_data = adata.obs if keys_attr is None else adata.obs[keys_attr]
    edges = edges_from_adata(
        adata, 'conn', key_neigh=key_neigh, as_attrs=True)
    nodes = make_nx_input_from_df(node_data)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    logging.info(nx.info(g))
    return g


def neighbor_induced_edgedf(
        df, nodes,
        keys_edge=['source', 'target'],
        return_nodes=False,
        fp=None):
    _inds = df[keys_edge].isin(nodes).max(1)
    subnodes = np.unique(df[_inds][['source', 'target']].values.flatten())
    df_sub = pp.subset_matches(df, subnodes, subnodes, cols=keys_edge, union=False)
    if fp is not None:
        df_sub.to_csv(fp, index=False)

    if return_nodes:
        return df_sub, subnodes
    else:
        return df_sub


def neighbor_induced_subgraph(g: nx.Graph, nodes):
    """ Note that the returned graph also contains the edges connecting the 
    neighbor-nodes if there is any.
    """
    nodes = [n for n in nodes if n in list(g)]
    subnodes = nx_neighbors(g, nodes, with_self=True)

    return g.subgraph(subnodes)


def nx_neighbors(g, nodes, with_self=False):
    """ find all the neighbors of given nodes
    """
    if isinstance(nodes, str):
        nodes = [nodes]
    tmp = []
    for nd in nodes:
        tmp += nx.neighbors(g, nd)
    if with_self:
        tmp += list(nodes)

    return np.unique(tmp)


def nx_to_dfs(
        g: nx.Graph,
        nodelist: Union[Sequence, None] = None,
        source="source", target="target",
        **kwds):
    """
    Returns
    -------
        edgedf, nodedf

    Examples
    --------
    >>> edgedf, nodedf = nx_to_dfs(g)
    """
    if isinstance(nodelist, str):
        nodelist = [nodelist]

    edgedf = nx.to_pandas_edgelist(
        g, source=source, target=target, nodelist=nodelist)
    if nodelist is not None:
        nodelist = np.unique(edgedf[[source, target]].values.flatten())
    nodedf = nx_to_nodedf(g, nodelist)
    return edgedf, nodedf


def nx_to_nodedf(g, nodelist=None):
    nodedf = pd.DataFrame(g.nodes.values(), index=g.nodes.keys())
    if nodelist is not None:
        nodedf = nodedf.loc[nodelist, :]
    return nodedf


def export_subgraph_df(
        g, nodenames=None, fdir=Path('.'), tag='', tfset=None):
    """
    export graph to '.csv' files for other visualization
    """
    if nodenames is not None:
        g = nx.subgraph(g, nodenames)
    print(nx.info(g))
    edgedf, nodedf = nx_to_dfs(g, )
    if tfset is not None:
        nodedf['TF'] = nodedf['name'].isin(tfset)
    fpe = fdir / f'nxEdges-{tag}.csv'
    fpn = fdir / f'nxAnnos-{tag}.csv'
    edgedf.to_csv(fpe, index=False)
    nodedf.to_csv(fpn, index=False)
    print(fpe, fpn, sep='\n')
    return g


def export_neighbor_subgraph_df(
        g, nodenames=None, fdir=Path('.'), tag='', tfset=None):
    """
    temp function; 
    """
    if nodenames is not None:
        g = neighbor_induced_subgraph(g, nodenames)
    print(nx.info(g))
    edgedf, nodedf = nx_to_dfs(g, )
    if tfset is not None:
        nodedf['TF'] = nodedf['name'].isin(tfset)
    fpe = fdir / f'gene_edges-{tag}.csv'
    fpn = fdir / f'gene_annos-{tag}.csv'
    edgedf.to_csv(fpe, index=False)
    nodedf.to_csv(fpn, index=False)
    print(fpe, fpn, sep='\n')
    return g


# In[]
"""     functions for cell/gene embeddings
==================================================
"""


def get_adata_neighbors(adata, key: Union[str, None] = None):
    """
    getting distances and connectivities from adata
    """
    key_dist = 'distances'
    key_conn = 'connectivities'
    if key is not None:
        key_dist = f'{key}_' + key_dist
        key_conn = f'{key}_' + key_conn

    return adata.obsp[key_dist], adata.obsp[key_conn]


def make_abstracted_graph(
        obs_labels1: Sequence,
        obs_labels2: Sequence,
        var_labels1: pd.Series,
        var_labels2: pd.Series,
        avg_expr1: pd.DataFrame,
        avg_expr2: pd.DataFrame,
        df_var_links: pd.DataFrame,
        tags_obs=('', ''),
        tags_var=('', ''),
        key_weight: str = 'weight',
        # key_count='size',
        # key_identity: str = 'identity',
        cut_ov: float = 0.,  # 0.55,
        norm_mtd_ov: Optional[str] = 'zs',  # 'max',
        ov_norm_first: bool = True,
        global_adjust_ov: bool = True,
        global_adjust_vv: bool = True,
        vargroup_filtered='filtered',
        **kwds):
    """ Compute and make the abstracted graph from expression matrices and the
    linkage weights between homologous genes

    Parameters
    ----------
    obs_labels1, obs_labels2
        group labels of the reference and query observations (cells), respectively.
    var_labels1, var_labels2
        group labels of the reference and query variables (genes), respectively.
    avg_expr1
        averaged expression matrix of the reference data
    avg_expr2
        averaged expression matrix of the query data

    df_var_links
        the linkage-weights between homologous genes
    tags_obs
        a tuple of two strings, for specifying homologous cel-types from
        different species.
        For example, if set ``tags_obs=('human ', 'mouse ')``, the result
        node names (for 'T cell') will be 'human T cell' and 'mouse T cell',
        respectively.
    tags_var
        a tuple of two strings, for specifying gene modules from different
        species.
        For example, if set ``tags_obs=('human module ', 'mouse module ')``,
        the result node names (for gene module '1') will be 'human module 1'
        and 'mouse module 1', respectively.
    key_weight
        column name in ``df_var_links``, specifying weights between each pair
        of variables.
    cut_ov
        the threshold to cut edges with values lower than it.
    norm_mtd_ov
        one of {None, 'zs', 'maxmin', 'max'}
    global_adjust_ov
        whether to globally adjust the weights between the observations and
        the variables
    global_adjust_vv
        whether to globally adjust the weights between the observations
    """
    tag_obs1, tag_obs2 = tags_obs
    tag_var1, tag_var2 = tags_var
    var_labels1, var_labels2, avg_expr1, avg_expr2, df_var_links = \
        _filter_for_abstract(
            var_labels1, var_labels2, avg_expr1, avg_expr2, df_var_links,
            name=vargroup_filtered)
    #    obs_group_order1 = _unique_cats(obs_labels1, obs_group_order1)
    #    obs_group_order2 = _unique_cats(obs_labels2, obs_group_order2)
    # var_group_order1 = _unique_cats(var_labels1, var_group_order1)
    # var_group_order2 = _unique_cats(var_labels2, var_group_order2)
    # print('--->', var_group_order1)
    # obs-var edge abstraction #
    edges_ov1, avg_vo1 = abstract_ov_edges(
        avg_expr1, var_labels1,
        norm_method=norm_mtd_ov,
        cut=cut_ov,
        tag_var=tag_var1, tag_obs=tag_obs1,
        norm_first=ov_norm_first,
        global_adjust=global_adjust_ov,
        return_full_adj=True)
    edges_ov2, avg_vo2 = abstract_ov_edges(
        avg_expr2, var_labels2,
        norm_method=norm_mtd_ov,
        cut=cut_ov,
        tag_var=tag_var2, tag_obs=tag_obs2,
        norm_first=ov_norm_first,
        global_adjust=global_adjust_ov,
        return_full_adj=True)
    # print('---> avg_vo1\n', avg_vo1)
    # var-weights abstraction #
    edges_vv, adj_vv = abstract_vv_edges(
        df_var_links,  # res.var_link_weights,
        var_labels1,
        var_labels2,
        norm_sizes='auto',
        # norm_sizes=(df_vnodes1[key_count], df_vnodes2[key_count]),
        return_full_adj=True,
        global_adjust=global_adjust_vv,
        key_weight=key_weight,
        tag_var1=tag_var1,
        tag_var2=tag_var2,
        **kwds)
    # print('---> adj_vv\n', adj_vv)

    # deciding orders of the groups #
    avg_vo1 = pp.order_contingency_mat(avg_vo1, 1)  # 1 for the rows (vars)
    avg_vo1 = pp.order_contingency_mat(avg_vo1, 0)
    var_group_order1, obs_group_order1 = avg_vo1.index, avg_vo1.columns

    var_name_order1 = [f'{tag_var1}{x}' for x in var_group_order1]
    # print(var_name_order1)
    adj_vv = pp.order_contingency_mat(adj_vv.loc[var_name_order1, :], 0)
    var_group_order2 = [x.replace(tag_var2, '') for x in adj_vv.columns]
    if set(obs_group_order1) == set(obs_labels2):
        obs_group_order2 = obs_group_order1
    else:
        avg_vo2 = pp.order_contingency_mat(avg_vo2.loc[var_group_order2, :], 0)
        obs_group_order2 = avg_vo2.columns

    # obs-nodes abstraction #    
    node_attrs_obs1 = abstract_nodes(
        obs_labels1, obs_group_order1, tag=tag_obs1)
    node_attrs_obs2 = abstract_nodes(
        obs_labels2, obs_group_order2, tag=tag_obs2)

    # var-nodes abstraction #    
    node_attrs_var1, df_vnodes1 = abstract_nodes(
        var_labels1, var_group_order1, tag=tag_var1,
        return_df=True)
    node_attrs_var2, df_vnodes2 = abstract_nodes(
        var_labels2, var_group_order2, tag=tag_var2,
        return_df=True)

    # df_vnodes1, df_vnodes2 = list(map(lambda x: x.set_index(key_identity),
    #                                   [df_vnodes1, df_vnodes2]))

    # graph construction #
    g = nx_multipartite_graph(
        node_attrs_obs1,
        node_attrs_var1, node_attrs_var2,
        node_attrs_obs2,
        edges=edges_ov1 + edges_vv + edges_ov2)

    return g


def abstract_vv_edges(
        df_links: pd.DataFrame,
        var_labels1: Union[pd.Series, Mapping],
        var_labels2: Union[pd.Series, Mapping],
        norm_sizes: Union[None, Sequence, str] = 'auto',
        return_full_adj=True,
        global_adjust: bool = True,
        global_norm_func=max,
        key_weight: str = 'weight',
        keys_link: Union[None, Sequence[str]] = None,
        tag_var1: str = '',
        tag_var2: str = '',
):
    """
    Parameters
    ----------
    df_links: pd.DataFrame, shape=(n_edges, *)
        If `keys_edge` is provided, it should be a tuple of 2 column-names
        in df_links.columns, indicating the edge columns. 
        Otherwise, the `df_links.index` should be `pd.MultiIndex` to indicate 
        the source and target edges.

    keys_link:
        If `keys_edge` is provided, it should be a tuple of 2 column-names
        in df_links.columns, indicating the edge columns.
    var_labels1, var_labels2:
        grouping labels for the two sets of variables, respectively.
    """
    # _var_labels1 = pd.Series({k: f'{tag_var1}{c}' for k, c in var_labels1.items()})
    # _var_labels2 = pd.Series({k: f'{tag_var1}{c}' for k, c in var_labels2.items()})

    abs_vv = aggregate_links(
        df_links, var_labels1, var_labels2, norm_sizes=None,  # normalize latter
        key_weight=key_weight, keys_link=keys_link)

    abs_vv.index = [f'{tag_var1}{c}' for c in abs_vv.index]
    abs_vv.columns = [f'{tag_var2}{c}' for c in abs_vv.columns]

    if norm_sizes is not None:
        if norm_sizes == 'auto':
            _var_labels1 = [f'{tag_var1}{c}' for c in var_labels1]
            _var_labels2 = [f'{tag_var2}{c}' for c in var_labels2]
            sizes1, sizes2 = [pd.value_counts(_lbs) for _lbs in [_var_labels1, _var_labels2]]
        elif isinstance(norm_sizes, Sequence):
            sizes1, sizes2 = norm_sizes
        else:
            raise ValueError
        abs_vv = weight_normalize_by_size(
            abs_vv, sizes1, sizes2, global_adjust=global_adjust,
            norm_func=global_norm_func)

    edges_vv = adj_to_edges(abs_vv, as_attrs=True)
    if return_full_adj:
        return edges_vv, abs_vv
    else:
        return edges_vv


def aggregate_links(
        df_links: pd.DataFrame,
        labels1: Union[pd.Series, Mapping],
        labels2: Union[pd.Series, Mapping],
        norm_sizes: Union[None, Sequence, str] = None,
        global_adjust: bool = False,
        global_norm_func=max,
        key_weight: str = 'weight',
        keys_link: Union[None, Sequence[str]] = None,
):
    """
    Parameters
    ----------
    df_links: pd.DataFrame, shape=(n_edges, *)
        If `keys_edge` is provided, it should be a tuple of 2 column-names
        in df_links.columns, indicating the edge columns. 
        Otherwise, the `df_links.index` should be `pd.MultiIndex` to indicate 
        the source and target edges.

    labels1, labels2:
        grouping labels for the rows and columns, respectively.
        
    norm_sizes:
        if provided, should be a pair of pd.Series, dict, Mapping, list
        it can be also set as 'auto', which means decide sizes by the group-labels.
    keys_link: str ('weight' by default)
        If `keys_edge` is provided, it should be a tuple of 2 column-names
        in df_links.columns, indicating the edge columns.

    """
    rnames0 = labels1.keys()
    cnames0 = labels2.keys()

    if keys_link is not None:
        df_links = df_links.set_index(keys_link)

    # print(key_weight, df_links, sep='\n')
    _data = df_links[key_weight] if key_weight in df_links.columns else None
    # print(_data)
    adj_var, rnames, cnames = pp.pivot_to_sparse(
        rows=df_links.index.get_level_values(0),
        cols=df_links.index.get_level_values(1),
        data=_data,
        rownames=rnames0, colnames=cnames0)
    # make sure the labels are correctly ordered
    lbs1 = np.array([labels1[r] for r in rnames])  # var_labels1[rnames]
    lbs2 = np.array([labels2[c] for c in cnames])
    # print(pd.value_counts(lbs1, dropna=False))
    aggregated = pp.agg_group_edges(
        adj_var, lbs1, lbs2, groups1=None, groups2=None, )

    if norm_sizes is None:
        return aggregated

    if norm_sizes == 'auto':
        sizes1, sizes2 = [pd.value_counts(_lbs) for _lbs in [lbs1, lbs2]]
    elif isinstance(norm_sizes, Sequence):
        sizes1, sizes2 = norm_sizes
    else:
        raise ValueError
    aggregated = weight_normalize_by_size(
        aggregated, sizes1, sizes2,
        global_adjust=global_adjust, norm_func=global_norm_func)

    return aggregated


def weight_normalize_by_size(adj, sizes1, sizes2,
                             norm_func=max,  # denominator
                             global_adjust=False):
    """
    Parameters
    ----------
    adj: pd.DataFrame
        adjacent matrix of shape (n1, n2)
    sizes1, sizes2: pd.Series, dict, Mapping, list-like
        shape = (n1,) and (n2,)
    global_adjust: bool
        whether to perform a global adjustment after
        size-normalization.
    """
    if not isinstance(adj, pd.DataFrame): adj = pd.DataFrame(adj)
    if isinstance(sizes1, list):
        sizes1 = dict(list(zip(adj.index, sizes1)))
    if isinstance(sizes2, list):
        sizes2 = dict(list(zip(adj.columns, sizes2)))

    _adj = adj.copy()
    for r, c in make_pairs_from_lists(_adj.index, _adj.columns, skip_equal=False):
        _adj.loc[r, c] /= norm_func([sizes1[r], sizes2[c]])

    if global_adjust:
        _adj /= _adj.max().max()
        # abs_vv = pp.wrapper_normalize(abs_vv, 'max', axis=1)

    return _adj


def abstract_ov_edges(
        avg_expr,
        var_labels,
        norm_method=None,
        norm_axis=1,
        tag_var='',
        tag_obs='',
        cut=0.,
        norm_first: bool = True,
        global_adjust: bool = False,
        return_full_adj=False,
):
    """
    Parameters
    ----------
    avg_expr: pd.DataFrame
        each column represent the average expressions
        for each observation group, and each row as a variable.
    norm_method:
        one of {None, 'zs', 'maxmin', 'max'}
    """
    df = avg_expr.copy()
    if norm_method is not None and norm_first:
        df = pp.wrapper_normalize(df, method=norm_method, axis=norm_axis)

    # averaged by varible-groups
    groupby_var = '__temp_labels__'
    df[groupby_var] = var_labels
    df_avg = df.groupby(groupby_var).mean()
    df_avg.dropna(inplace=True)
    if norm_method and not norm_first:
        df_avg = pp.wrapper_normalize(df_avg, method=norm_method, axis=norm_axis)
    df_avg0 = df_avg.copy()

    df_avg.index = [f'{tag_var}{i}' for i in df_avg.index]
    df_avg.columns = [f'{tag_obs}{i}' for i in df_avg.columns]

    if cut is not None:
        if isinstance(cut, Sequence):
            df_avg[df_avg <= cut[0]] = 0
            df_avg[df_avg > cut[1]] = cut[1]
            logging.warning(
                f'Edges with weights out of range {cut} were cut out '
                f'or clipped')
        else:
            df_avg[df_avg <= cut] = 0
            logging.warning(
                f'Edges with weights lower than {cut} were cut out')

    if global_adjust:
        df_avg /= df_avg.max().max()

    edge_attrs = adj_to_edges(df_avg, cut=0, as_attrs=True)
    if return_full_adj:
        return edge_attrs, df_avg0
    else:
        return edge_attrs


def adj_to_edges(
        adj,
        rownames=None,
        colnames=None,
        key_row='source',
        key_col='target',
        key_data='weight',
        cut=None,
        as_attrs=False):
    """
    Melt (Un-pivot) a pd.DataFrame of adjacent matrix into a list of edges, works
    similar to `pd.melt()` but ignoring ALL of the zeros instead.
    """
    if isinstance(adj, pd.DataFrame):
        rownames = adj.index
        colnames = adj.columns
        data = adj.values
    else:
        rownames = np.arange(adj.shape[0]) if rownames is None else rownames
        colnames = np.arange(adj.shape[1]) if colnames is None else colnames
        data = adj

    coo_data = sparse.coo_matrix(data)
    df_edge = pd.DataFrame({
        key_row: np.take(rownames, coo_data.row),
        key_col: np.take(colnames, coo_data.col),
        key_data: coo_data.data,
    })
    if cut is not None:
        if isinstance(cut, Sequence):
            df_edge = df_edge[df_edge[key_data] > cut[0]]
            df_edge[df_edge[key_data] > cut[1]] = cut[1]
            logging.warning(
                f'Edges with weights out of range {cut} were cut out '
                f'or clipped')
        else:
            df_edge = df_edge[df_edge[key_data] > cut]
            logging.warning(
                f'Edges with weights lower than {cut} were cut out.')
    if as_attrs:
        edge_attrs = make_nx_input_from_df(df_edge, key_id=[key_row, key_col])
        return edge_attrs
    else:
        return df_edge


def abstract_nodes(labels,  # df, groupby,
                   group_ord=None,
                   key_count='size',
                   key_identity='identity',
                   key_orig='original name',
                   key_tag='tag',
                   tag='',
                   return_df=False,
                   **attrs):
    """
    Taking each of the groups in `labels` as an abstracted node, and making a 
    list of node attributes for further input to `networkx.Graph.add_nodes_from()`
    
    Parameters
    ----------
    key_identity: key name for the unique identifier for each abstracted node.
    key_orig: key name for the original (group) name for each abstracted node.
    **kwds: ignored currently
    
    Examples
    --------
    >>> df_nodes = abstract_nodes(df, groupby='module')
    >>> g = nx.Graph()
    >>> g.add_nodes_from(node_attrs, 'node name')

    """
    #    labels = df[groupby]
    labels = pd.Series(labels)
    if group_ord is None:
        df_nodes = labels.value_counts().to_frame(key_count)
    else:
        df_nodes = labels.value_counts()[group_ord].to_frame(key_count)
    df_nodes[key_identity] = [f'{tag}{i}' for i in df_nodes.index]
    df_nodes[key_orig] = list(df_nodes.index)
    #    if tag != '':
    df_nodes[key_tag] = tag.strip()

    node_attrs = make_nx_input_from_df(df_nodes, key_id=key_identity, **attrs)
    if return_df:
        return node_attrs, df_nodes
    else:
        return node_attrs


def make_nx_input_from_df(
        df: pd.DataFrame,
        key_id: Union[None, str, list] = None,
        **attrs):
    """
    Parameters
    ----------
    df: pd.DataFrame with each column containing node attributes
        
    key_id: 
        which column(s) should be used for unique identifier for each nodes.
        str or list of strings from df.columns, or None (using `df.index` in this case)
        If specified, the values should be unique.
    **attrs: 
        other common attributes for this batch of nodes
    
    
    Returns
    -------
    A list of tuples, with each tuple formed like `(node, attrdict)` when 
    `len(key_id) == 1` or `(u, v, attrdict)` when `len(key_id) == 2` 
        
        
    Examples
    --------
    >>> node_attrs = make_nx_input_from_df(df, **attrs)
    >>> g = nx.Graph()
    >>> g.add_nodes_from(node_attrs)
    
    """
    if key_id is not None:
        df = df.set_index(key_id, drop=True)

    else:
        df = df.copy()
    if len(attrs) >= 1:
        for k, v in attrs.items():
            df[k] = v

    dct = df.apply(lambda x: x.to_dict(), axis=1).to_dict()
    if isinstance(df.index, pd.MultiIndex):
        return [k + (vd,) for k, vd in dct.items()]
    else:
        return [(k, vd) for k, vd in dct.items()]


def nx_multipartite_graph(*node_layers,
                          edges=None,
                          subset_key='subset', **attrs):
    """

    Examples
    --------
    >>> g = nx_multipartite_graph([0, 1], [2, 3, 4, 5], [6, 7, 8], )
    >>> pos = nx.multipartite_layout(g, subset_key=subset_key, )
    >>> nx.draw(g, pos, with_labels=True, )
    """
    import networkx as nx

    #    print(nodes)
    g = nx.Graph()
    for i, nodes in enumerate(node_layers):
        g.add_nodes_from(nodes, **{subset_key: i})
    if edges is None:
        print('`edges` are not provides, adding full connected edges '
              'between adjacent layers by default')
        from itertools import product
        for layer1, layer2 in nx.utils.pairwise(node_layers):
            g.add_edges_from(product(layer1, layer2))
    else:
        g.add_edges_from(edges)
    return g


# In[]
def arrange_contingency_mat(
        mat: pd.DataFrame,
        novel_names=['unknown', 'uncertain', 'multi-type'],
):
    """
    alignment of column and row names
    """
    set1, set2 = set(mat.index), set(mat.columns)
    common = list(set1.intersection(set2))
    index = common + list(set1.difference(common))
    columns = common + list(set2.difference(common))
    for nm in novel_names:
        if nm in columns:
            columns.remove(nm)
            columns += [nm]

    mat = mat.reindex(index)[columns]
    return mat


def wrapper_confus_mat(y_true, y_pred, classes_on=None,
                       normalize='true', as_df=True):
    """ 
    normalize: 'true', 'pred', 'all', None
        by default, normalized by row (true classes)
    """
    from sklearn import metrics
    if classes_on is None:
        classes_on = np.unique(list(y_true) + list(y_pred))
    try:
        mat = metrics.confusion_matrix(y_true, y_pred, labels=classes_on,
                                       normalize=normalize)
    except:
        logging.warning(
            'The argument `normalize` may not be accepted by '
            'the previous version of scikit-learn')
        mat = metrics.confusion_matrix(y_true, y_pred, labels=classes_on, )
    if as_df:
        mat = pd.DataFrame(mat, index=classes_on, columns=classes_on)
    return mat


def wrapper_contingency_mat(y_true, y_pred,
                            order_rows=True,
                            order_cols=False,
                            normalize_axis=None,
                            as_df=True,
                            eps=None,
                            assparse=False
                            ):
    """ 
    Modified and wrapped function from `sklearn`:
    >>> mat = sklearn.metrics.cluster.contingency_matrix(
    ...        y_true, y_pred, eps=eps, sparse=assparse)
    """
    if eps is not None and sparse:
        raise Warning("Cannot set 'eps' when sparse=True")
    # to avoid mix-type error when taking unique, transform all labels to str
    y_true = [str(x) for x in y_true]
    y_pred = [str(x) for x in y_pred]
    classes, class_idx = np.unique(y_true, return_inverse=True)
    clusters, cluster_idx = np.unique(y_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    mat = sparse.coo_matrix(
        (np.ones(class_idx.shape[0]), (class_idx, cluster_idx)),
        shape=(n_classes, n_clusters), dtype=np.int
    )
    if assparse:
        mat = mat.tocsr()
        mat.sum_duplicates()
        if normalize_axis is not None:  # 0 for columns
            mat = pp.normalize_norms(mat, axis=normalize_axis)
    else:
        mat = mat.toarray()
        if eps is not None:
            # don't use += as mat is integer
            mat = mat + eps
        if normalize_axis is not None:  # 0 for columns
            mat = pp.normalize_norms(mat, axis=normalize_axis)

        if as_df:
            mat = pd.DataFrame(mat, index=classes, columns=clusters)
        # reorder to make clusters and classes matching each other as possible
        if order_cols:
            mat = pp.order_contingency_mat(mat, 0)
        if order_rows:
            mat = pp.order_contingency_mat(mat, 1)
    return mat


def set_precomputed_neighbors(
        adata,
        distances,
        connectivities=None,
        n_neighbors=15,
        metric='cosine',  # pretended parameter
        method='umap',  # pretended parameter
        metric_kwds=None,  # pretended parameter
        use_rep=None,  # pretended parameter
        n_pcs=None,  # pretended parameter
        key_added=None,  #
):
    if key_added is None:
        key_added = 'neighbors'
        conns_key = 'connectivities'
        dists_key = 'distances'
    else:
        conns_key = key_added + '_connectivities'
        dists_key = key_added + '_distances'

    if connectivities is None:
        connectivities = distances.copy().tocsr()
        connectivities[connectivities > 0] = 1

    adata.obsp[dists_key] = distances
    adata.obsp[conns_key] = connectivities

    adata.uns[key_added] = {}
    neighbors_dict = adata.uns[key_added]
    neighbors_dict['connectivities_key'] = conns_key
    neighbors_dict['distances_key'] = dists_key

    neighbors_dict['params'] = {'n_neighbors': n_neighbors, 'method': method}
    neighbors_dict['params']['metric'] = metric
    if metric_kwds is not None:
        neighbors_dict['params']['metric_kwds'] = metric_kwds
    if use_rep is not None:
        neighbors_dict['params']['use_rep'] = use_rep
    if n_pcs is not None:
        neighbors_dict['params']['n_pcs'] = n_pcs

    return adata
