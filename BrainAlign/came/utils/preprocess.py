# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 00:21:15 2020
@author: Xingyan Liu

Functions for handling AnnData:

    * I/O functions
    * preprocessing
        - normalization / z-scoring / ...
        - groups / ...
    * statistics computation



"""
import logging
import os
from pathlib import Path
from typing import Sequence, Union, Mapping, List, Optional, Dict, Callable
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, label_binarize
import scanpy as sc

from scipy import sparse, io
from .base import check_dirs, dec_timewrapper


def unpack_dict_of_lists(d):
    lst = []
    for k, l in d.items():
        lst += l
    return lst


def dict_has_keys(d: Mapping, *keys):
    """
    return True if all the keys are in the dict, else False.
    """
    dk = d.keys()
    for k in keys:
        if k not in dk:
            logging.info(f'key {k} is not included in the dict')
            return False
    return True


# In[]
# I/O functions (handling AnnData object(s))


def save_named_mtx(adata, dirname, field=None, raw=True, backup_npz=True,
                   **kwds):
    check_dirs(dirname)
    """ better set field='integer' if possible """
    if adata.raw is not None:
        adata = adata.raw

    mtx_file = '%s/matrix.mtx' % (dirname)
    bcd_file = '%s/barcodes.tsv' % (dirname)
    gene_file = '%s/genes.tsv' % (dirname)

    genes = adata.var_names.to_frame(index=False, name='genes')
    barcodes = adata.obs_names.to_frame(index=False, name='barcodes')
    logging.info(adata.X.data[:5])
    genes.to_csv(gene_file, index=False, header=False)
    barcodes.to_csv(bcd_file, index=False, header=False)
    if backup_npz:
        sparse.save_npz(f'{dirname}/matrix.npz', adata.X.T)
    io.mmwrite(mtx_file, adata.X.T, field=field, **kwds)

    print('Matrix saved to directory `%s`' % dirname)


def save_mtx2df(adata, fname, index=True, header=True, **kwds):
    logging.warning("NOTE: saving the dense matrix might take some time. \
          if needed, you can consider the function: `funx.saveNamedMtx`\
          to handle a sparse matrix in a efficient way.")
    df = mtx2df(adata)
    df.to_csv(fname, index=index, header=header)


def mtx2df(adata):
    df = pd.DataFrame(data=adata.X.toarray(),
                      index=adata.obs_names,
                      columns=adata.var_names)
    return df


def load_namelist(fpath, header=None, tolist=True, icol=0, **kw):
    """
    icol: int
    only return the first column by default
    """
    names = pd.read_csv(fpath, header=header, **kw)
    names = names.iloc[:, icol]
    return names.tolist() if tolist else names


def save_namelist(lst, fname, header=False, index=False, **kw):
    pd.Series(lst).to_csv(fname, header=header, index=index, **kw)
    print('name list seved into:', fname)


def load_dense(fpath, ):
    print(f'loading dense matrix from {fpath}')
    mat = pd.read_csv(fpath, sep='\t', index_col=0)
    return mat


def load_sparse(fpath, backup_npz=True):
    """
    `fpath` should be ended with '.mtx' or 'npz'
    """
    fpath = Path(fpath)
    if fpath.suffix == '.mtx':
        mat = io.mmread(str(fpath))
        if backup_npz:
            print('backup the `.npz` file for speedup loading next time...')
            mat = sparse.csc_matrix(mat)
            sparse.save_npz(fpath.parent / 'matrix.npz', mat)
    elif fpath.suffix == '.npz':
        mat = sparse.load_npz(str(fpath))
    else:
        raise ValueError("the file path should be ended with '.mtx' or 'npz'")
    return mat


def adata_from_raw(dirname, backup_npz=True, name_mtx='matrix',
                   dtype='float32', **kw):
    """
    read matrix and annotated names from directory `dirname`

    * dirname
        - matrix.npz
        - genes.tsv
        - barcodes.tsv (or meta_cell.tsv)

    Alternative (for load mtx file):
    >>> adata = sc.read_10x_mtx(dirname)
        
    """
    fn_mat_npz = f'{name_mtx}.npz'
    files = os.listdir(dirname)
    if fn_mat_npz in files:
        mat = load_sparse(dirname / fn_mat_npz)
    elif f'{name_mtx}.mtx' in files:
        mat = load_sparse(dirname / f'{name_mtx}.mtx', backup_npz=backup_npz)
    else:
        raise FileNotFoundError(
            f"None of file named `{name_mtx}.npz` or `{name_mtx}.mtx` exists!")
    mat = sparse.csr_matrix(mat.T)

    params = dict(header=None, sep='\t', index_col=0)
    if 'meta_cell.tsv' in files:
        barcodes = pd.read_csv(dirname / 'meta_cell.tsv', sep='\t', index_col=0)
    else:
        barcodes = pd.read_csv(dirname / 'barcodes.tsv', **params)
    genes = pd.read_csv(dirname / 'genes.tsv', **params)

    adata = sc.AnnData(mat.astype(dtype), dtype=dtype)
    adata.var_names = genes.index.values
    adata.obs_names = barcodes.index.values
    for col in barcodes.columns:
        adata.obs[col] = barcodes[col]
    # print(adata)
    return adata


def add_columns(
        df0, df=None,
        only_diff=True,
        ignore_index=True,
        copy=True, **kwannos):
    def _set_values(_df, k, v, ignore_index=ignore_index):
        if k in _df.columns:
            logging.warning(
                f'NOTE that column "{k}" will be covered by new values')
        _df[k] = list(v) if ignore_index else v
        # print(k, end=', ')

    if copy: df0 = df0.copy()
    if df is not None:
        if only_diff:
            cols = [c for c in df.columns if c not in df0.columns]
        else:
            cols = df.columns
        for col in cols:
            _set_values(df0, col, df[col])

    if len(kwannos) >= 1:
        for k, v in kwannos.items():
            # be careful of the index mismatching problems!!!
            _set_values(df0, k, v)
    # print('done!')
    return df0 if copy else None


# In[]
# functions for prepare data [0]


def add_obs_annos(adata: sc.AnnData,
                  df: Union[Mapping, pd.DataFrame],
                  ignore_index=True,
                  only_diff=False,
                  copy=False, **kwds):
    adata = adata.copy() if copy else adata
    print(f'adding columns to `adata.obs` (ignore_index={ignore_index}):')

    add_columns(adata.obs, df, only_diff=only_diff,
                ignore_index=ignore_index,
                copy=copy, **kwds)

    return adata if copy else None


def add_var_annos(adata: sc.AnnData,
                  df: Union[Mapping, pd.DataFrame],
                  ignore_index=True,
                  only_diff=False,
                  copy=False,
                  **kwds):
    adata = adata.copy() if copy else adata
    print('adding columns to `adata.var`:')
    add_columns(adata.var, df, only_diff=only_diff,
                ignore_index=ignore_index,
                copy=copy, **kwds)

    return adata if copy else None


def make_adata(mat: Union[np.ndarray, sparse.spmatrix],
               obs: Union[Mapping, pd.DataFrame, None] = None,
               var: Union[Mapping, pd.DataFrame, None] = None,
               fn: Union[Path, str] = None,
               ignore_index: bool = True,
               assparse: bool = True,
               ):
    """ An alternative way to construct AnnData (BUG fixed).
    Something might go wrong when saving the AnnData object constructed 
    by the build-in function `sc.AnnData`.
    """
    if assparse and not sparse.issparse(mat):
        mat = sparse.csr_matrix(mat)
    adata = sc.AnnData(mat)
    if isinstance(obs, pd.DataFrame):
        adata.obs_names = obs.index.values.astype(str)
    if isinstance(var, pd.DataFrame):
        adata.var_names = var.index.values.astype(str)
    if obs is not None:
        add_obs_annos(adata, obs, copy=False, ignore_index=ignore_index)
    if var is not None:
        add_var_annos(adata, var, copy=False, ignore_index=ignore_index)

    if fn is not None:
        adata.write(fn)
    return adata


def merge_metas(adatas, obs_keys=None):
    if obs_keys:
        obs_keys = [obs_keys] if isinstance(obs_keys, str) else obs_keys
        obss = [adt.obs[obs_keys] for adt in adatas]
    else:
        obss = [adt.obs for adt in adatas]
    return pd.concat(obss, axis=0)


def merge_adatas(
        adatas: Union[Mapping[str, sc.AnnData], Sequence[sc.AnnData]],
        union: bool = False,
        obs_keys: Optional[Sequence] = None,
        dsnames: Optional[Sequence] = None,
        key_dsname: str = None
) -> sc.AnnData:
    """ a new adata will be created, use the raw matrix

    Parameters
    ----------
    adatas:
        a list or a dict of AnnData objects.
        use data from adata.raw if not None.
    union: bool
        whether to take union of the features (e.g. genes)
    obs_keys:
        column-names from sample annotations to be merged
    dsnames:
        names for these datasets, if None, use numbers ranging from
        0 to (n_datasets-1)
    key_dsname:
        if provided as a str, a column will be added to the merged `obs`.

    Returns
    -------
    adata_merged: ``AnnData``

    """
    if isinstance(adatas, Mapping):
        _dsnames, _adatas = list(zip(*adatas.items()))
        dsnames = _dsnames if dsnames is None else dsnames
    else:
        _adatas = adatas
        dsnames = list(range(len(_adatas))) if dsnames is None else dsnames
    #    if dsnames is None:
    #        dsnames = list(range(len(_adatas))) if dsnames is None else dsnames
    print("merging raw matrix...")
    mats = []
    genes = []
    bcds = []
    lbs_datasets = []
    for i, adt in enumerate(_adatas):
        data = adt.raw if adt.raw is not None else adt
        mats.append(data.X)
        genes.append(data.var_names)
        bcds.append(data.obs_names)
        lbs_datasets += [dsnames[i]] * data.shape[0]

    mats, genes = merge_named_matrices(mats, genes, union=union)
    merged_mat = sparse.vstack(mats, )  # dtype = mats[0].dtype)
    genes = pd.DataFrame(index=genes)
    logging.debug(genes.head())

    # if obs_keys is None:
    #     obs = pd.DataFrame(index=np.concatenate(bcds, axis=0))
    # else:
    logging.info("merging metadata...")
    obs = merge_metas(_adatas, obs_keys)
    if key_dsname is not None:
        obs[key_dsname] = pd.Categorical(lbs_datasets, categories=dsnames)
    logging.debug(obs.head())
    return make_adata(merged_mat, obs=obs, var=genes)


def merge_named_matrices(
        mat_list: Sequence[Union[np.ndarray, sparse.csr_matrix]],
        gene_lists: Sequence[Sequence],
        dsnames: Sequence = None,
        union: bool = False,
        verbose: bool = True,
):
    """ merge matrices and align their columns
    This function code is copied from `scanorama.merge_datasets`.

    Parameters
    ----------
    mat_list:
        a list of cell-by-gene matrices (np.ndarray or sparse.csr_matrix) 
    gene_lists:
        a list of gene-list corresponding to the columns of each matrix in 
        `mat_list`
    dsnames:
        names for these datasets, if None, use numbers ranging from
        0 to (n_datasets-1)
    union: bool
        whether to take union of the features from all the matrices

    Returns
    -------
    mat_list:
        a list of cell-by-gene matrices (sparse.csr_matrix) with aligned gene 
        columns.
    ret_genes:
        a gene list shared by all the mat_list

    """
    import sys
    if union:
        sys.stderr.write(
            'WARNING: Integrating based on the union of genes is '
            'highly discouraged, consider taking the intersection '
            'or requantifying gene expression.\n'
        )

    # Find genes in common.
    keep_genes = set()
    for idx, gene_list in enumerate(gene_lists):
        if len(keep_genes) == 0:
            keep_genes = set(gene_list)
        elif union:
            keep_genes |= set(gene_list)
        else:
            keep_genes &= set(gene_list)
        if not union and (dsnames is not None) and verbose:
            print('After {}: {} genes'.format(dsnames[idx], len(keep_genes)))
        if len(keep_genes) == 0:
            raise ValueError(
                'Error: No genes found in all datasets, exiting...')
    if verbose:
        print('Found {} genes among all datasets'
              .format(len(keep_genes)))

    #    print(gene_list)
    if union:
        union_genes = sorted(keep_genes)
        for i in range(len(mat_list)):
            if verbose:
                print('Processing data set {}'.format(i))
            # TODO!!!: Dense matirx!!! using sparse matrix instead!!!!!
            X_new = np.zeros((mat_list[i].shape[0], len(union_genes)))
            X_old = sparse.csc_matrix(mat_list[i])
            gene_to_idx = {gene: idx for idx, gene in enumerate(gene_lists[i])}
            for j, gene in enumerate(union_genes):
                if gene in gene_to_idx:
                    X_new[:, j] = X_old[:,
                                  gene_to_idx[gene]].toarray().flatten()
            mat_list[i] = sparse.csr_matrix(X_new)
        # print('TEST:', mat_list[i].data[:5])
        ret_genes = np.array(union_genes)
    else:
        # Only keep genes in common.
        ret_genes = np.array(sorted(keep_genes))
        for i in range(len(mat_list)):
            # Remove duplicate genes.
            uniq_genes, uniq_idx = np.unique(gene_lists[i], return_index=True)
            mat_list[i] = mat_list[i][:, uniq_idx]

            # Do gene filtering.
            gene_sort_idx = np.argsort(uniq_genes)
            gene_idx = [idx for idx in gene_sort_idx
                        if uniq_genes[idx] in keep_genes]
            mat_list[i] = mat_list[i][:, gene_idx]
            assert (np.array_equal(uniq_genes[gene_idx], ret_genes))

    return mat_list, ret_genes


def all_vars_of_adata(adata):
    if hasattr(adata, 'raw') and adata.raw is not None:
        var_names = list(adata.raw.var_names)
    else:
        var_names = list(adata.var_names)
    return var_names


def align_adata_vars(adata1: sc.AnnData,  # better be raw data
                     adata2: sc.AnnData,
                     df_varmap_1v1: Optional[pd.DataFrame] = None,
                     unify_names: bool = False,
                     ) -> [sc.AnnData, sc.AnnData]:
    """ Align the vaiables of two ``sc.AnnData`` objects

    Parameters
    ----------
    adata1: AnnData
        reference data
    adata2: AnnData
        query data
    df_varmap_1v1:
        a 2-column DataFrame containing one-to-one mapping between
        variables in ``adata1`` and ``adata2``
    unify_names:
        whether to change the variable-names in ``adata2`` to match the
        ``adata1.var_names``
    """
    vars_all1, vars_all2 = list(map(all_vars_of_adata, [adata1, adata2]))
    if df_varmap_1v1 is None:
        vars1 = vars2 = list(
            set(adata1.var_names).intersection(adata2.var_names))
    else:
        submaps_1v1 = subset_matches(df_varmap_1v1, vars_all1, vars_all2,
                                     union=False)
        vars1, vars2 = submaps_1v1.iloc[:, 0], submaps_1v1.iloc[:, 1]
    adata1 = adata1[:, vars1].copy()
    adata2 = adata2[:, vars2].copy()
    if unify_names:
        adata2.var_names = list(vars1)
    return adata1, adata2


def make_dict_from_str(s: str, n2id=True, ) -> dict:
    s = s.split()
    if not n2id:
        s = s[::-1]
    dct = {}
    for k, v in zip(s[1::2], s[::2]):
        if k in dct.keys():
            dct[k] += f',{v}'
        else:
            dct[k] = v
    return dct


def regu_gname(gname, pattern='^X[0-9]{2,3}'):
    # [\d] is equal to [0-9]
    # import re
    if re.match(pattern, gname) is not None:
        gname = gname[1:]
    return gname


def change_names(names: Sequence,
                 foo_change: Callable,
                 **kwargs):
    """
    Parameters
    ----------
    names
        a list of names to be modified
    foo_change: function to map a name-string to a new one
    **kwargs: other kwargs for foo_change
    """
    return list(map(foo_change, names, **kwargs))


def get_homologies(df_match: pd.DataFrame,
                   vals: Sequence,
                   cols: Union[None, Sequence[str]] = None,
                   # [col_from, col_to]
                   reverse: bool = False,
                   uniquelist: bool = True,
                   with_null=False):
    """ Get the homologous gene of input ones based on the homology-mappings

    Parameters
    ----------
    df_match
        homologous gene mappings
    vals
        the gene names.
    cols
        Two column names as ``[col_from, col_to]``, specifying the homology
        mapping direction.
    reverse
        whether to reverse the mappings
    uniquelist
        whether to remove the duplicates from the results.
    with_null
        whether to return those genes without any homologies (copy names)

    Returns
    -------
    If ``uniquelist=True`` and ``with_null=False``, a single list will be returned;
    otherwise, returns a tuple of two lists (homo, null)

    """
    if cols is None:
        cols = df_match.columns[: 2]
    col_from, col_to = cols
    if reverse:
        col_from, col_to = col_to, col_from
    _df_match = df_match.set_index([col_from, col_to], drop=False)
    # print(_df_match.head())
    homos = []
    null = []
    for v in vals:
        try:
            homos.append(_df_match.loc[v, :])
        # print(homos[-1])
        except KeyError:
            null.append(v)
            continue
    homos = pd.concat(homos, axis=0, ignore_index=True)
    # homos = _df_match.loc[vals, col_to]
    if len(null) >= 1:
        logging.warning(f'the following have no homologies:\n{null}')
    if uniquelist:
        homos = pd.unique(homos[col_to].values)
        return list(homos) + null if with_null else homos
    else:
        return (homos, null) if with_null else homos


get_homologues = get_homologies


def subset_matches(df_match: pd.DataFrame,
                   left: Sequence,
                   right: Sequence,
                   union: bool = False,
                   cols: Union[None, Sequence[str]] = None,
                   indicators=False):
    """ Take a subset of token matches (e.g., gene homologies)

    Parameters
    ----------
    df_match: pd.DataFrame
        a dataframe with at least 2 columns
    left:
        list-like, for extracting elements in the first column.
    right:
        list-like, for extracting elements in the second column.
    union:
        whether to take union of both sides of genes
    cols:
        if specified, should be 2 column names for left and right genes,
        respectively.
    indicators:
        if True, only return the indicators.
    """
    if cols is None:
        cols = df_match.columns[: 2]

    c1, c2 = cols
    tmp = df_match[c1].isin(left).to_frame(c1)
    tmp[c2] = df_match[c2].isin(right)

    keep = tmp.max(1) if union else tmp.min(1)

    return keep if indicators else df_match[keep]


def take_freq1(lst):
    """ take elements that occur only once in `lst`
    """
    vcnts = pd.value_counts(lst)
    return vcnts[vcnts == 1].index.to_list()


def take_freq1more(lst):
    """ take elements that occur more than once in `lst`
    """
    vcnts = pd.value_counts(lst)
    return vcnts[vcnts > 1].index.to_list()


def take_1v1_matches(df: pd.DataFrame,
                     cols: Optional[List] = None):
    """ Take the one-to-one matches of the given two columns of a ``pd.DataFrame``

    Parameters
    ----------
    df
        a dataframe with at least two columns
    cols
        column names of ``df``, specifying the columns to match.

    Returns
    -------
    A pd.DataFrame contains only one-to-one mappings and the corresponding rows
    """
    _df = df[cols] if cols is not None else df
    left, right = take_freq1(_df.iloc[:, 0]), take_freq1(_df.iloc[:, 1])
    return subset_matches(df, left, right, union=False)


def is1v1pairs(df: pd.DataFrame):
    left, right = take_freq1(df.iloc[:, 0]), take_freq1(df.iloc[:, 1])
    return subset_matches(df, left, right, union=False, indicators=True)


def reverse_dict_of_list(d: Mapping, ) -> Mapping:
    """
    the values of the dict must be list-like type
    """
    d_rev = {}
    for k in d.keys():
        vals = d[k]
        _d = dict.fromkeys(vals, k)
        d_rev.update(_d)
    return d_rev


def reverse_dict(d: Mapping, ) -> Mapping:
    """
    the values of the dict must be un-changable type
    """
    d_rev = {d[k]: k for k in d.keys()}
    return d_rev


def _has_duplicates(lst, error=True):
    if len(lst) != len(set(lst)):
        if error:
            raise ValueError('The input sequence should not have duplicates!')
        else:
            return True
    return False


def make_id_name_maps(names1, names2):
    """
    concatenate 2 list of names, and return 2 kinds of maps:
        1. integer id --> name
        2. name --> integer id (a tuple of 2 dicts)

    Note:
        using 2 dict to map a name to its id for that
        -- the 2 name-lists may share the same name!
        (while the indices are unique)
    """
    _has_duplicates(names1)
    _has_duplicates(names2)
    # id --> name
    name_srs = pd.Series(list(names1) + list(names2))
    # name --> id
    # using 2 dicts to map a name to its id for that
    # -- the 2 name-lists may share the same name!
    n2i_dict1 = reverse_dict(name_srs[: len(names1)])
    n2i_dict2 = reverse_dict(name_srs[len(names1):])
    return name_srs, (n2i_dict1, n2i_dict2)


def make_bipartite_adj(df_map: pd.DataFrame,
                       nodes1=None, nodes2=None,
                       key_data: Optional[str] = None,
                       with_singleton: bool = True,
                       symmetric: bool = True):
    """ Make a bipartite adjacent (sparse) matrix from a ``pd.DataFrame`` with
    two mapping columns.

    Parameters
    ----------
    df_map: pd.DataFrame with 2 columns.
        each row represent an edge between 2 nodes from the left and right 
        group of nodes of the bipartite-graph.
    nodes1, nodes2: list-like
        node name-list representing left and right nodes, respectively.
    key_data: str or None
        if provided, should be a name in df.columns, where the values will be
        taken as the weights of the adjacent matrix.
    with_singleton:
        whether allow nodes without any neighbors. (default: True)
        if nodes1 and nodes2 are not provided, this parameter makes no difference.
    symmetric
        whether make it symmetric, i.e. X += X.T

    Examples
    --------
    >>> bi_adj, nodes1, nodes2 = make_bipartite_adj(df_map)

    """
    nodes1 = list(set(df_map.iloc[:, 0])) if nodes1 is None else nodes1
    nodes2 = list(set(df_map.iloc[:, 1])) if nodes2 is None else nodes2

    if with_singleton:
        nodes1 = list(set(df_map.iloc[:, 0]).union(nodes1))
        nodes2 = list(set(df_map.iloc[:, 1]).union(nodes2))
    i2name_srs, name2i_dicts = make_id_name_maps(nodes1, nodes2)
    if key_data is None:
        data = np.ones(df_map.shape[0], dtype=int)
    else:
        data = df_map[key_data].values
    # sparse adjacent matrix construction:
    # ids representing nodes from left-nodes (ii) and right-nodes (jj)
    ii = change_names(df_map.iloc[:, 0], lambda x: name2i_dicts[0][x])
    jj = change_names(df_map.iloc[:, 1], lambda x: name2i_dicts[1][x])
    bi_adj = sparse.coo_matrix(
        (data, (ii, jj)),
        shape=(len(i2name_srs),) * 2)
    if symmetric:
        bi_adj += bi_adj.T

    return bi_adj, nodes1, nodes2


def pivot_df_to_sparse(df: pd.DataFrame, row=0, col=1, key_data=None, **kwds):
    """
    row, col:
        str or int, int for column index, and str for column name
    """

    def _get_df_vals(key):
        if isinstance(key, str):
            return df[key].values
        else:
            return df.iloc[:, key].values

    rows, cols = list(map(_get_df_vals, [row, col]))

    if key_data is None:
        data = np.ones(df.shape[0], dtype=int)
    else:
        data = _get_df_vals(key_data)
    return pivot_to_sparse(rows, cols, data, **kwds)


def pivot_to_sparse(rows: Sequence, cols: Sequence,
                    data: Optional[Sequence] = None,
                    rownames: Sequence = None,
                    colnames: Sequence = None):
    """
    Parameters
    ----------
    rownames:
        If provided, the resulting matrix rows are restricted to `rownames`,
        Names will be removed if they are not in `rownames`, and names that
        not occur in `rows` but in `rownames` will take ALL-zeros in that row
        of the resulting matrix.
        if not provided, will be set as `rows.unique()`
    colnames: 
        if not provided, will be set as `cols.unique()`

    Notes
    -----
        * `rows` and `cols` should be of the same length!

    """

    def _make_ids_from_name(args):  # vals, names=None
        vals, names = args
        if names is None:
            names = np.unique(vals)
        name2id = pd.Series(np.arange(len(names)), index=names)
        ii = change_names(vals, lambda x: name2id[x])
        return names, ii

    if data is None:
        data = np.ones_like(rows, dtype=float)
    # make sure that all of the row or column names are in the provided names
    if rownames is not None or colnames is not None:
        # r_kept, c_kept = np.ones_like(rows).astype(bool), np.ones_like(rows).astype(bool)
        if rownames is not None:
            tmp = set(rownames)
            r_kept = list(map(lambda x: x in tmp, rows))
            logging.debug(sum(r_kept))
        else:
            r_kept = np.ones_like(rows).astype(bool)
        if colnames is not None:
            tmp = set(colnames)
            c_kept = list(map(lambda x: x in tmp, cols))
        else:
            c_kept = np.ones_like(rows).astype(bool)
        kept = np.minimum(r_kept, c_kept)
        logging.debug(len(kept), sum(kept))
        rows, cols, data = rows[kept], cols[kept], data[kept]

    (rownames, ii), (colnames, jj) = list(map(
        _make_ids_from_name, [(rows, rownames), (cols, colnames)]))

    sparse_mat = sparse.coo_matrix(
        (data, (ii, jj)), shape=(len(rownames), len(colnames)))
    return sparse_mat, rownames, colnames


class AdjacentTrans(object):
    def __init__(self, adj, vars1, vars2, trans_to=0):
        assert adj.shape[0] == len(vars1)
        assert adj.shape[1] == len(vars2)
        self.adj = adj
        self.vars1 = vars1  # for rows
        self.vars2 = vars2  # for columns
        self.trans_to = int(trans_to)  # 0 for `vars1` and 1 for `vars2`

    @property
    def shape(self):
        return self.adj.shape

    @staticmethod
    def from_edge_df(edge_df, col_weight=None):
        # edge_df: the first column for rows and the second for columns
        adj, vars1, vars2 = pivot_df_to_sparse(edge_df, key_data=col_weight)
        return AdjacentTrans(adj, vars1, vars2)

    def reduce_to_align(self, sep='__'):
        trans_to = self.trans_to
        logging.info(f"[AdjacentTrans] reduce to align, trans_to={trans_to}")
        if trans_to == 0:
            return self.vars1, reduce_to_align(self.adj, self.vars2, sep=sep)
        else:
            return reduce_to_align(self.adj.T, self.vars1, sep=sep), self.vars2

    def reduce_to_align_features(self, feats):
        """feats: a sample-by-vars matrix or (sparse) array"""
        trans_to = self.trans_to
        adj = self.adj
        logging.info(
            f"[AdjacentTrans] reduce features to align, trans_to={trans_to}")
        if trans_to == 1:
            adj = adj.T
        assert feats.shape[1] == adj.shape[1]
        feats = adj.dot(feats.T) / adj.sum(1)  # divide row-sums as averages
        # if self.trans_to == 0:
        #     assert feats.shape[1] == adj.shape[1]
        #     feats = adj.dot(feats.T) / adj.sum(1)
        # else:
        #     adj = adj.T
        #     assert feats.shape[1] == adj.shape[1]
        #     feats = adj.T.dot(feats.T)
        return feats.T

    def T(self):
        """Transpose"""
        return AdjacentTrans(self.adj.T, self.vars2, self.vars1)


def reduce_to_align(adj: sparse.spmatrix, names: Sequence, sep='__'):
    """reduce column-names to align rows"""
    assert adj.shape[1] == len(names)
    adj = sparse.csr_matrix(adj)
    # adj.indices.shape == adj.data.shape
    return [sep.join(np.take(names, adj[i, :].indices).astype(str))
            for i in range(adj.shape[0])]


def label_binarize_each(labels, classes, sparse_out=True):
    lb1hot = label_binarize(labels, classes=classes, sparse_output=sparse_out)
    if len(classes) == 2:
        lb1hot = lb1hot.toarray()
        lb1hot = np.c_[1 - lb1hot, lb1hot]
        if sparse_out:
            lb1hot = sparse.csc_matrix(lb1hot)
    return lb1hot


def agg_group_edges(adj, labels1, labels2=None,
                    groups1=None, groups2=None,
                    asdf=True, verbose=False):
    """
    Parameters
    ----------
    adj: 
        adjacent matrix of shape (N0, N1), if `labels2` is None, then set N0=N1.
    labels1:
        a list or a np.array of shape (N0,)
    labels2:
        a list or a np.array of shape (N1,)    
    
    Returns
    -------
    group_conn
        summation of connected edges between given groups
    """
    #    if sparse.issparse(adj):
    adj = sparse.csc_matrix(adj)

    groups1 = pd.unique(labels1) if groups1 is None else groups1
    lb1hot1 = label_binarize_each(labels1, classes=groups1, sparse_out=True)
    if labels2 is None:
        lb1hot2 = lb1hot1
        groups2 = groups1
    else:
        groups2 = pd.unique(labels2) if groups2 is None else groups2
        lb1hot2 = label_binarize_each(labels2, classes=groups2, sparse_out=True)
    if verbose:
        print('---> aggregating edges...')
        print('unique labels of rows:', groups1)
        print('unique labels of columns:', groups2)
        print('grouping elements (edges)')
        print('shape of the one-hot-labels:', lb1hot1.shape, lb1hot2.shape)
    group_conn = lb1hot1.T.dot(adj).dot(lb1hot2)
    if asdf:
        group_conn = pd.DataFrame(group_conn.toarray(),
                                  index=groups1, columns=groups2)

    return group_conn


def scipy_edge_dict_for_dgl(edge_dict, foo=None):
    """
    foo: a callable object. 
        For example, `torch.LongTensor`
    """

    def _spsparse2edge(spmat, ):
        spmat = sparse.coo_matrix(spmat)
        #        print(type(spmat.row), spmat.col)
        return (spmat.row.astype(int), spmat.col.astype(int))

    for etuple, adj in edge_dict.items():
        if foo is None:
            edge_dict[etuple] = _spsparse2edge(adj)
        else:
            edge_dict[etuple] = tuple(map(foo, _spsparse2edge(adj)))

    return edge_dict


def homograph_from_scipy(adj, as_dgl=True, self_loop=True):
    """ making a homologous DGL graph from a scipy.sparse matrix
    """
    from networkx import from_scipy_sparse_matrix
    g = from_scipy_sparse_matrix(adj)
    if as_dgl:
        from dgl import DGLGraph
        g = DGLGraph(g)
    if self_loop:
        logging.info('adding self-loops to the homologous graph')
        g.add_edges(g.nodes(), g.nodes())
    return g


def add_new_edgetype_to_dglgraph(
        g,
        src_edge_tgt: tuple,
        adj: sparse.spmatrix):
    edge_dict = {
        e: g.adj(scipy_fmt='csr', etype=e) for e in g.canonical_etypes
    }
    edge_dict[src_edge_tgt] = adj
    edge_dict = scipy_edge_dict_for_dgl(edge_dict)
    from dgl import heterograph
    return heterograph(edge_dict, )


# In[]

def _order_contingency_array(mat, axis=1):
    """
    axis = 0: re-order the columns
    axis = 1: re-order the rows
    """
    order = np.argsort(np.argmax(mat, axis=axis))
    if axis == 1:
        logging.debug('Re-order the rows')
        return mat[order, :]
    else:
        logging.debug('Re-order the columns')
        return mat[:, order]


def _order_contingency_df(df: pd.DataFrame, axis=1):
    """
    axis = 0: re-order the columns
    axis = 1: re-order the rows
    """
    order = np.argsort(np.argmax(df.values, axis=axis))
    if axis == 1:
        logging.debug('Re-order the rows')
        return df.iloc[order, :]
    else:
        logging.debug('Re-order the columns')
        return df.iloc[:, order]


def order_contingency_mat(mat, axis=1):
    if isinstance(mat, pd.DataFrame):
        return _order_contingency_df(mat, axis=axis)
    else:
        return _order_contingency_array(mat, axis=axis)


def describe_mat_nnz(mat, axis=1):
    print('shape:', mat.shape)
    tag = 'row' if axis == 1 else 'column'
    nnz = (mat > 0).sum(axis=axis)
    if sparse.issparse(mat):
        nnz = nnz.A1
    print(f'non-zeros for each {tag}:')
    print(pd.Series(nnz).describe())


def describe_mat(mat, axis=1):
    print('shape:', mat.shape)
    tag = 'row' if axis == 1 else 'column'
    nnz = mat.sum(axis=axis).flatten()
    if sparse.issparse(mat):
        nnz = nnz.A1
    print(f'non-zeros for each {tag}:')
    print(pd.Series(nnz).describe())


def is_symetric(mat):
    if (mat != mat.T).sum() == 0:
        return True
    else:
        return False


def _binarize_mat(mat, inplace=False):
    if not inplace:
        mat = mat.copy()
    mat[mat > 0] = 1
    return mat


# In[]
""" merge / split / take ... groups;
    group averages
    
"""


def compute_unkn_rate(gcnt, ):
    """ unknown rate computing """
    n1, n2 = gcnt.sum(0)
    dsn1, dsn2 = gcnt.columns
    unkn_rate = gcnt.loc[gcnt[dsn1].isna(), dsn2].sum() / n2

    return unkn_rate


def compute_unkn_rate_for2(gcnt1, gcnt2, names):
    """ unknown rate computing """
    gcnt = pd.concat([gcnt1, gcnt2], axis=1, keys=names)
    n1, n2 = gcnt.sum(0)
    dsn1, dsn2 = gcnt.columns
    unkn_rate = gcnt.loc[gcnt[dsn1].isna(), dsn2].sum() / n2

    return unkn_rate


def take_group_labels(labels: Sequence, group_names: Sequence,
                      indicate=False, remove=False):
    """
    Parameters
    ----------
    labels: list-like
    group_names:
        names of groups that you want to take out
    indicate: bool
        if True, return a Series of bool indicators of the groups
        else, return the labels.
    remove: bool
        False by default, set as True if you want to keep groups that
        NOT in the `given group_names`
    """
    if isinstance(group_names, (str, int)):
        group_names = [group_names]
    if remove:
        indicators = change_names(labels, lambda x: x not in group_names)
    else:
        indicators = change_names(labels, lambda x: x in group_names)
    if indicate:
        return np.array(indicators)
    else:
        return np.array(labels)[indicators]


def remove_small_groups(labels, min_samples=10,
                        indicate=False, ):
    """ return labels with small groups removed
    """
    vcnts = pd.value_counts(labels)
    #    print(vcnts)
    groups_rmv = list(vcnts[vcnts <= min_samples].index)
    logging.info('groups to be removed:\n\t%s', groups_rmv)
    return take_group_labels(labels, group_names=groups_rmv,
                             indicate=indicate, remove=True)


def remove_adata_small_groups(adata: sc.AnnData,
                              key: str,
                              min_samples=10):
    """ return adata with small groups removed, grouped by `key`
    """
    indicators = remove_small_groups(adata.obs[key],
                                     min_samples=min_samples,
                                     indicate=True, )
    return adata[indicators, :].copy()


def remove_adata_groups(adata: sc.AnnData, key: str,
                        group_names: Sequence, copy=True):
    """ Remove given groups from an AnnData object """
    indicators = take_group_labels(adata.obs[key], group_names,
                                   indicate=True, remove=True)
    return adata[indicators, :].copy() if copy else adata[indicators, :]


def take_adata_groups(adata: sc.AnnData,
                      key: str,
                      group_names: Sequence,
                      onlyx: bool = False,
                      copy: bool = False):
    """ Take given groups from an AnnData object """
    indicators = take_group_labels(adata.obs[key], group_names,
                                   indicate=True)
    if copy:
        return adata[indicators, :].X.copy() if onlyx else adata[indicators,
                                                           :].copy()
    else:
        return adata[indicators, :].X if onlyx else adata[indicators, :]


def merge_group_labels(labels: Sequence,
                       group_lists: Sequence):
    """
    Parameters
    ----------
    labels:
        list-like
    group_lists:
        a list of lists of group names to be merged
    
    Examples
    --------
    >>> merge_group_labels(adata.obs['batch'], [['a', 'b'], ['c', 'd']]).unique()
    """
    if not isinstance(group_lists[0], list):
        # merge only one set of groups
        groups = group_lists  # list(map(str, group_lists))
        new_name = '_'.join(groups)
        labels = change_names(labels, lambda x: new_name if x in groups else x)
        logging.info('groups are merged into a new single group: %s', new_name)
        return labels
    else:
        # merge multiple sets of groups
        for groups in group_lists:
            labels = merge_group_labels(labels, groups)

        return labels


def merge_adata_groups(adata: sc.AnnData,
                       key: str,
                       group_lists: Sequence,
                       new_key=None,
                       rename=False,
                       copy=False):
    """ Merge the given groups into one single group
    which is named as ``'_'.join(groups[i])`` by default.

    Parameters
    ----------
    adata: AnnData
    key:
        a column name in `adata.obs`
    group_lists
        a list of lists of group names
    new_key:
        a column name to be added to `adata.obs`
    rename: bool
        wheter to re-label the groups after merge
    copy: bool
    
    Examples
    --------
    >>> adata_new = merge_adata_groups(
    ...     adata, 'batch', group_lists=[['A', 'B'], ['E', 'F']], copy=True)
    >>> adata

    """
    adata = adata.copy() if copy else adata
    labels = adata.obs[key].copy()
    # group_lists = [group_lists] if not isinstance(group_lists[0], list) else group_lists
    labels = merge_group_labels(labels, group_lists)

    new_key = key + '_new' if new_key is None else new_key
    adata.obs[new_key] = pd.Categorical(labels)
    print(f'A new key `{new_key}` added.')
    if rename:
        n_groups = len(adata.obs[new_key].cat.categories)
        new_cat = list(map(str, range(n_groups)))
        adata.obs[new_key].cat.categories = new_cat
        print(f'categories are renamed as:\n{new_cat}')
    if copy: return adata


def split_adata(adata: sc.AnnData,
                key: str,
                ) -> Dict[str, sc.AnnData]:
    """ Split an ``AnnData`` object into a dict of multiple objects

    Parameters
    ----------
    adata:
        the AnnData object to be split

    key:
        a column-name of `adata.obs`

    Returns
    -------
    a dict of AnnData
    """
    groups = adata.obs[key].unique()
    adts = {}
    for gname in groups:
        adts[gname] = take_adata_groups(adata, key, [gname],
                                        onlyx=False, copy=True)
    return adts


def bisplit_adata(adata: sc.AnnData,
                  key: str,
                  left_groups: Union[Sequence, None] = None,
                  reset_index_by: Union[None, str, Sequence] = None,
                  ) -> List[sc.AnnData]:
    """ Split an ``AnnData`` object into a pair of AnnData objects

        Parameters
        ----------
        adata: sc.AnnData
        key:
            a column-name of `adata.obs`
        left_groups
            a list of names specifying which groups to be returned as the left one.
        reset_index_by
            the column(s) that store the observation names.
            if provided, the observation names will be reset.
        Returns
        -------
        a list of two ``AnnData``
        """
    if left_groups is None:
        # take the first group-label by default
        lbs = adata.obs[key]
        if hasattr(lbs, 'cat'):
            left_groups = lbs.cat.categories[0]
        else:
            left_groups = lbs[0]

    indicators = take_group_labels(adata.obs[key], left_groups,
                                   indicate=True)
    left = adata[indicators, :].copy()
    right = adata[~ indicators, :].copy()
    if reset_index_by:
        if isinstance(reset_index_by, str):
            reset_index_by = [reset_index_by] * 2
        left.obs_names = left.obs[reset_index_by[0]].tolist()
        right.obs_names = right.obs[reset_index_by[1]].tolist()
    return [left, right]


# In[]
#  filtering out mito-genes


def filter_mitogenes(adata):
    vars_kept = _filter_mito(adata.var_names)
    return adata[:, vars_kept].copy()


def _filter_mito(lst):
    return [x for x in lst if not x.lower().startswith('mt-')]


# In[]
#    normalize / z-score grouped by ...


def normalize_default(adata: sc.AnnData,
                      target_sum=None,
                      copy: bool = False,
                      log_only: bool = False,
                      force_return: bool = False, ):
    """ Normalizing datasets with default settings (total-counts normalization
    followed by log(x+1) transform).

    Parameters
    ----------
    adata
        ``AnnData`` object
    target_sum
        scale factor of total-count normalization
    copy
        whether to copy the dataset
    log_only
        whether to skip the "total-counts normalization" and only perform
        log(x+1) transform
    force_return
        whether to return the data, even if changes are made for the
        original object

    Returns
    -------
    ``AnnData`` or None

    """
    if copy:
        adata = adata.copy()
        logging.info('A copy of AnnData made!')
    else:
        logging.info('No copy was made, the input AnnData will be changed!')
    logging.info('normalizing datasets with default settings.')
    if not log_only:
        logging.info(
            f'performing total-sum normalization, target_sum={target_sum}...')
        sc.pp.normalize_total(adata, target_sum=target_sum)
    else:
        logging.info('skipping total-sum normalization')
    sc.pp.log1p(adata)
    return adata if copy or force_return else None


def normalize_log_then_total(
        adata, target_sum=5e2,
        copy=False, force_return=False):
    """ For SplitSeq data, performing log(x+1) BEFORE total-sum normalization
        will results a better UMAP visualization (e.g. clusters would be less
        confounded by different total-counts ).
    """
    if copy:
        adata = adata.copy()
    logging.info(
        'normalizing datasets with default settings (log1p first).'
        f'target_sum = {target_sum}')
    sc.pp.log1p(adata)  # x <- log(x + 1)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    return adata if copy or force_return else None


def zscore(X, with_mean=True, scale=True, ):
    """ For each column of X, do centering (z-scoring)
    """
    # code borrowed from `scanpy.pp._simple`
    scaler = StandardScaler(with_mean=with_mean, copy=True).partial_fit(X)
    if scale:
        # user R convention (unbiased estimator)
        e_adjust = np.sqrt(X.shape[0] / (X.shape[0] - 1))
        scaler.scale_ *= e_adjust
    else:
        scaler.scale_ = np.array([1] * X.shape[1])
    X_new = scaler.transform(X)
    if isinstance(X, pd.DataFrame):
        X_new = pd.DataFrame(X_new, index=X.index, columns=X.columns)
    return X_new


def group_zscore(X: Union[np.ndarray, pd.DataFrame],
                 labels: Union[Sequence, np.ndarray],
                 with_mean: bool = True,
                 scale: bool = True,
                 max_value: float = None):
    """
    For each column of X, do within-group centering (z-scoring)

    Parameters
    ----------
    X: np.ndarray or pd.DataFrame
        A matrix of shape (n_samples, n_features), each row of X is an
        observation, wile each column is a feature
    labels: np.ndarray
        the group labels, of shape (n_samples,)
    with_mean: boolean, True by default
        If True, center the data before scaling, and X shoud be a dense matrix.
    scale: bool
        whether to scale with standard deviation
    max_value: float
        if given, the absolute values of the result matrix will be
        clipped at this value.

    Returns
    -------
    the scaled data matrix
    """
    isdf = False
    if isinstance(X, pd.DataFrame):
        isdf = True
        index, columns, X = X.index, X.columns, X.values
    X = X.astype(np.float).copy()
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    for lb in unique_labels:
        ind = labels == lb
        if sum(ind) == 1:
            logging.warning(f'ignoring class {lb} with only one sample.')
            continue
        X[ind, :] = zscore(X[ind, :], with_mean=with_mean, scale=scale)

    if max_value is not None:
        X[X > max_value] = max_value
        logging.info(f'... clipping at max_value {max_value}')

    if isdf:
        X = pd.DataFrame(X, index=index, columns=columns)
    return X


def group_zscore_adata(adt: sc.AnnData,
                       groupby: str = 'batch',
                       key: str = 'counts',
                       key_new: str = None,
                       max_value: float = None,
                       with_mean: bool = True,
                       cover: bool = True,
                       **kwds):
    """Calculate z-scores for each group of observations in an ``AnnData`` object

    Parameters
    ----------
    adt: AnnData
    groupby: str
        A key from adt.obs, from which the labels are take
    key: str, {'X_pca', 'count'}
        can be a key from adt.obsm, e.g. `key='X_pca'`
        If key == 'counts', then do scaling on `adt.X`
        and cover the old count matrix, ignoring the `cover` parameter
    key_new: str
        used when ``key != 'counts'``
    with_mean: boolean, True by default
        If True, center the data before scaling, and X shoud be a dense matrix.
    max_value: float
        if given, the absolute values of the result matrix will be
        clipped at this value.
    cover: bool
        whether to cover the old X with the scored X
    """
    labels = adt.obs[groupby]
    if key == 'counts':
        logging.info(
            'Z-score scaling on count matrix, transformed into a dense array')
        if sparse.issparse(adt.X):
            X = adt.X.toarray()
        else:
            X = adt.X
        if not cover: adt = adt.copy()
        adt.X = group_zscore(
            X, labels, with_mean=with_mean,
            max_value=max_value, **kwds)
    else:
        if cover:
            key_new = key
        else:
            key_new = key + '_new' if key_new is None else key_new
        adt.obsm[key_new] = group_zscore(
            adt.obsm[key], labels,
            with_mean=with_mean, max_value=None, **kwds)

    return adt if not cover else None


def wrapper_scale(adata, zero_center=True, max_value=None,
                  groupby=None, copy=False, **kwds):
    """
    Wrapper function for centering and scaling data matrix `X` in sc.AnnData,
    extended for within-batch processing.
    
    Examples
    --------
    >>> wrapper_scale(adata, groupby='batch')
    """
    if groupby is not None:
        logging.info(f'doing within-group scaling, group by [ {groupby} ]')
        return group_zscore_adata(adata,
                                  max_value=max_value,
                                  groupby=groupby,
                                  key='counts',
                                  with_mean=zero_center,
                                  cover=not copy,
                                  **kwds)
    else:
        logging.info('using the build-in function `sc.pp.scale(..)`')
        return sc.pp.scale(adata, zero_center=zero_center,
                           max_value=max_value, copy=copy)


def normalize_col(X, scale_factor=1., by='sum'):
    """
    make the column elements of X to unit sum

    Parameters
    ----------
    X:
        a (sparse) matrix
    scale_factor: float or None
        if None, use the median of sum level as the scaling factor.
    by: str, {'sum', 'max'}

    """
    if by == 'sum':
        norms = X.sum(axis=0)
    elif by == 'max':
        norms = X.max(axis=0)
    else:
        raise ValueError(f'`by` should be either "sum" or "max", got {by}')
    if hasattr(norms, 'A'):
        norms = norms.A.flatten()
    if scale_factor is None:
        is_zero = norms == 0
        scale_factor = np.median(norms[~ is_zero])
    norms /= scale_factor
    # for those rows or columns that summed to 0, just do nothing
    norms[norms == 0] = 1

    norm_ = 1 / norms

    if sparse.isspmatrix(X):
        logging.info('sparse normalization')
        X_new = X.dot(sparse.diags(norm_))
    else:
        logging.info('dense normalization')
        X_new = X.dot(np.diag(norm_))

    if isinstance(X, pd.DataFrame):
        X_new.columns = X.columns
    return X_new


def normalize_row(X, scale_factor=1, by='sum'):
    """
    make the row elements of X to unit sum

    Parameters
    ----------
    X:
        a (sparse) matrix
    scale_factor: float or None
        if None, use the median of sum level as the scaling factor.
    by: str, {'sum', 'max'}

    """
    if by == 'sum':
        norms = X.sum(axis=1)
    elif by == 'max':
        norms = X.max(axis=1)
    else:
        raise ValueError(f'`by` should be either "sum" or "max", got {by}')
    if hasattr(norms, 'A'):
        norms = norms.A.flatten()
    if scale_factor is None:
        is_zero = norms == 0
        scale_factor = np.median(norms[~ is_zero])
    norms /= scale_factor
    # for those rows or columns that summed to 0, just do nothing
    norms[norms == 0] = 1
    norm_ = 1 / norms

    if sparse.isspmatrix(X):
        logging.info('sparse normalization')
        X_new = sparse.diags(norm_).dot(X)
    else:
        logging.info('dense normalization')
        X_new = np.diag(norm_).dot(X)

    if isinstance(X, pd.DataFrame):
        X_new = pd.DataFrame(X_new, columns=X.columns)
    return X_new


def normalize_norms(X, scale_factor=1, axis=0, by='sum'):
    """ wrapper of `normalize_colsum` and `normalize_rowsum`

    Parameters
    ----------
    X:
        a (sparse) matrix
    scale_factor: numeric, None
        if None, use the median of sum level as the scaling factor.
    axis: int, {0, 1}
        if axis = 0, apply to each column;
        if axis = 1, apply to each row.
    by: str, {'sum', 'max'}
        normalization method

    """
    foo = normalize_col if axis == 0 else normalize_row
    return foo(X, scale_factor=scale_factor, by=by)


def normalize_max(df, axis=0, **kwds):
    vmax, vmin = df.max().max(), df.min().min()
    if vmin < 0:
        raise ValueError('Data with ALL non-negative values are required '
                         'for "max-normalization"')
    elif vmax == 0:
        logging.warning('Full-zero values.')
        return df

    if axis is None:
        func_norm = lambda x: x / vmax
        return df.applymap(func_norm, )
    else:
        func_norm = lambda x: x / x.max() if x.max() > 0 else x
        return df.apply(func_norm, axis=axis)


def normalize_maxmin(df, axis=0, eps=1e-8, **kwds):
    vmax, vmin = df.max().max(), df.min().min()
    if vmax == vmin:
        logging.warning('DataFrame with  constant values, zeros are returned')
        return df - vmin

    if axis is None:
        func_norm = lambda x: (x - vmin) / (vmax - vmin + eps)
        return df.applymap(func_norm, )
    else:
        func_norm = lambda x: (x - x.min()) / (x.max() - x.min() + eps)
        return df.apply(func_norm, axis=axis)


def wrapper_normalize(df, method='maxmin', axis=0, **kwds):
    #    df = df.copy()
    # normalize values
    if method.lower() in {'minmax', 'maxmin'}:
        return normalize_maxmin(df, axis=axis, **kwds)
    if method.lower() == 'max':
        return normalize_max(df, axis=axis, **kwds)

    elif method == 'zs':
        if axis == 0:
            df = zscore(df)  # column normalization
        elif axis == 1:
            df = zscore(df.T).T
    return df


def mean_of_nozeros(mat, axis=0):
    """
    Parameters
    ----------
    mat: np.arrary or sparse matrix
    axis: int
    """
    mat = mat.copy()
    logging.info('making a copy')
    sums = mat.sum(axis=axis)
    # mat.eliminate_zeros()
    mat[mat > 0] = 1
    nnz = mat.sum(axis=axis)
    m = sums / nnz
    if hasattr(m, 'A'):
        m = m.A
    return m.flatten()


# In[]
#     group averages


def group_value_counts(df, count_on, group_by, split=True, **kwds):
    """ extended function of pd.value_counts

    Parameters
    ----------
    count_on: str
        which column to apply function `pd.value_counts`
    group_by: str
        which column to group-by
    split: bool
        split the Series and concatenate them as columns if True.
    """
    vcnt = df.groupby(group_by)[count_on].apply(pd.value_counts, **kwds)
    if split:  # split the Series and concatenate them as columns
        lvls = df[group_by].unique()
        #        lvls = vcnt.index.get_level_values(group_by).unique()
        vcnt = pd.concat([vcnt[nm] for nm in lvls], axis=1, keys=lvls)
    return vcnt


def group_mean(
        X: Union[np.ndarray, sparse.spmatrix],
        labels: Sequence,
        binary: bool = False,
        classes=None, features=None,
        **kwds
):
    """ compute the group averaged features

    Parameters
    ----------
    X: np.ndarray or sparse.spmatrix
        shape (n_samples, n_features)
    labels:
        shape (n_samples, )
    binary
        if True, the results will be the non-zero proportions
    classes:
        optional, names of groups
    features:
        optional, names of features

    Returns
    -------
    average_mat: pd.DataFrame
    """
    if sparse.issparse(X):
        return group_mean_sparse(
            X, labels, binary=binary, classes=classes, features=features, **kwds
        )
    else:
        return group_mean_dense(
            X, labels, binary=binary, classes=classes, features=features, **kwds
        )


def group_mean_sparse(
        X: sparse.spmatrix,
        labels: Sequence,
        binary=False, classes=None, features=None,
        print_groups=False):
    """
    This function may work with more efficiency than `df.groupby().mean()` 
    when handling sparse matrix.

    Parameters
    ----------
    X: np.ndarray or sparse.spmatrix
        shape (n_samples, n_features)
    labels:
        shape (n_samples, )
    classes:
        optional, names of groups
    features:
        optional, names of features
        
    """
    classes = np.unique(labels, ) if classes is None else classes
    if binary:
        X = (X > 0)  # .astype('float')
        logging.info('Binarized; the results will be non-zero proportions')

    if len(classes) == 1:
        grp_mean = X.mean(axis=0).T
    else:
        lb1hot = label_binarize_each(labels, classes=classes, sparse_out=True)
        logging.info(f'Calculating feature averages for {len(classes)} groups')
        if print_groups:
            print(classes)
        grp_mean = X.T.dot(lb1hot) / lb1hot.sum(axis=0)
    grp_mean = pd.DataFrame(grp_mean, columns=classes, index=features)
    return grp_mean


def group_mean_dense(
        X, labels, binary=False,
        index_name='group',
        classes=None,
        features=None,
):
    classes = np.unique(labels, ) if classes is None else classes
    if binary:
        X = (X > 0)  # .astype('float')
        logging.info('Binarized...the results will be the non-zero '
                     'proportions.')
    tmp = pd.DataFrame(X, columns=features)
    tmp[index_name] = list(labels)
    avgs = tmp.groupby(index_name).mean().T
    del tmp[index_name]
    # print(avgs.shape)
    for c in classes:
        if c not in avgs.columns:
            avgs[c] = 0.
    return avgs[classes]  # each column as a group


def group_median_dense(
        X, labels,
        index_name='group',
        classes=None,
):
    classes = np.unique(labels, ) if classes is None else classes
    tmp = pd.DataFrame(X)
    tmp[index_name] = list(labels)
    avgs = tmp.groupby(index_name).median()
    # print(avgs.shape)
    return avgs.T[classes]  # each column as a group


def group_mean_adata(adata: sc.AnnData,
                     groupby: str,
                     features: Sequence = None,
                     binary: bool = False,
                     use_raw: bool = False):
    """Compute averaged feature-values for each group

    Parameters
    ----------
    adata: AnnData
    groupby: str
        a column name in adata.obs
    features:
        a subset of names in adata.var_names (or adata.raw.var_names)
    binary: bool
        if True, the results will turn to be the non-zeor proportions for
        all (or the given) features
    use_raw: bool
        whether to access adata.raw to compute the averages.

    Returns
    -------
    a pd.DataFrame with features as index and groups as columns
    """
    labels = adata.obs[groupby]
    #print(f'Computing averages grouped by {groupby}')
    if use_raw and adata.raw is not None:
        if features is not None:
            features = [f for f in features if f in adata.raw.var_names]
            X = adata.raw[:, features].X
        else:
            features = adata.raw.var_names
            X = adata.raw.X
    else:
        if features is not None:
            features = [f for f in features if f in adata.var_names]
            X = adata[:, features].X
        else:
            features = adata.var_names
            X = adata.X
    if sparse.issparse(X):
        return group_mean(X, labels, binary=binary, features=features)
    else:
        return group_mean_dense(X, labels, binary=binary, )


def group_mean_multiadata(adatas: Sequence[sc.AnnData],
                          keys: Union[str, Sequence],
                          use_genes=None, binary=False,
                          use_raw=True, tags=None, ):
    """
    tags: tag of each adata to avoid column-name collision
    """
    n = len(adatas)
    if isinstance(keys, str):
        keys = [keys] * n
    tags = list(map(str, np.arange(n))) if tags is None else tags
    # within-group averages 
    gmeans = [
        group_mean_adata(adt, ky, use_genes, binary=binary, use_raw=use_raw) \
        for adt, ky in zip(adatas, keys)]
    gmeans = pd.concat(gmeans, axis=1, keys=tags, sort=False)
    gmeans.fillna(0, inplace=True)
    # Make names
    clnames = ['_'.join(c) for c in gmeans.columns.to_list()]
    gmeans.set_axis(clnames, axis=1, inplace=True)
    return gmeans


# In[]

def quick_preprocess(
        adata: sc.AnnData,
        hvgs: Optional[Sequence] = None,
        normalize_data: bool = True,
        target_sum: Optional[float] = 1e4,
        batch_key=None,
        n_top_genes: int = 5000, #2000
        n_pcs: int = 30,  # if None, stop before PCA step
        nneigh: int = 10,  # 20 was used for clustering
        metric='cosine',
        copy: bool = True,
        **hvg_kwds):
    """
    Quick preprocess of the raw data.

    Notes
    -----
    if `normalize_data` is True, an adata with RAW counts is required!
    """
    if copy:
        _adata = adata.copy()
        logging.info('A copy of AnnData made!')
    else:
        _adata = adata
        logging.info('No copy was made, the input AnnData will be changed!')
    # 1: normalization
    if normalize_data:
        normalize_default(_adata, target_sum=target_sum)
    _adata.raw = _adata
    # 2: HVG selection (skipped if `hvgs` is not None)
    # Mistake???
    if hvgs is None:
        sc.pp.highly_variable_genes(
            _adata, batch_key=batch_key, n_top_genes=n_top_genes, **hvg_kwds)
        _adata = _adata[:, _adata.var['highly_variable']].copy()
    else:
        _adata = _adata[:, hvgs].copy()  # detach form view-data
    # 3: z-score 
    wrapper_scale(_adata, groupby=batch_key)
    #    sc.pp.scale(_adata)
    # 4: PCA
    if n_pcs is None:
        do_pca = False
    else:
        sc.tl.pca(_adata, n_comps=n_pcs)
        do_pca = True
    # 5: k-nearest-neighbor graph
    if do_pca and nneigh is not None:
        sc.pp.neighbors(_adata, n_pcs=n_pcs, n_neighbors=nneigh, metric=metric)
    # 6: leiden-clustering...(separated function)

    return _adata  # always return data


def quick_pre_vis(adata, hvgs=None,
                  normalize_data=True,
                  target_sum=None,
                  batch_key=None,
                  n_pcs=30, nneigh=10,
                  vis='umap',
                  color=None,
                  metric='cosine',
                  copy=True, **hvg_kwds):
    """ Go through the default pipeline and have a overall visualization of
    the data.
    """

    _adata = quick_preprocess(
        adata, hvgs=hvgs,
        batch_key=batch_key,
        normalize_data=normalize_data,
        target_sum=target_sum,
        n_pcs=n_pcs, nneigh=nneigh,  # 20
        metric=metric,
        copy=copy,
        **hvg_kwds)

    if vis.lower() == 'umap':
        sc.tl.umap(_adata)
        sc.pl.umap(_adata, color=color,
                   legend_fontsize=12, ncols=1)
    elif vis.lower() == 'tsne':
        sc.tl.tsne(_adata)
        sc.pl.tsne(_adata, color=color,
                   legend_fontsize=12, ncols=1)
    else:
        raise ValueError(
            f'the `vis` argument should either be "umap" or "tsne", got {vis}')
    return _adata


def quick_pre_clust(
        adata, hvgs=None,
        normalize_data=True,
        target_sum=1e4,
        batch_key=None,
        n_pcs=30, nneigh=20,
        reso=0.4,
        copy=True, **hvg_kwds):
    _adata = quick_preprocess(
        adata, hvgs=hvgs,
        batch_key=batch_key,
        normalize_data=normalize_data,
        target_sum=target_sum,
        n_pcs=n_pcs, nneigh=nneigh,  # 20
        copy=copy,
        **hvg_kwds)

    sc.tl.leiden(_adata, resolution=reso)
    return _adata


def get_scnet(adata: sc.AnnData):
    """ Extract the pre-computed single-cell KNN network

    If the adata has not been preprocessed, please run 
    `adata_processed = quick_preprocess(adata, **kwds)` first.

    Parameters
    ----------
    adata
        the data object
    """
    key0 = 'neighbors'
    key = 'connectivities'
    if key in adata.obsp.keys():
        adj = adata.obsp[key]
    else:
        adj = adata.uns[key0][key]
    return adj


def get_hvgs(adata, force_redo=False, batch_key=None,
             n_top_genes=2000,
             key_hvg: str = 'highly_variable',
             **hvg_kwds):
    """ packed function for computing and get HVGs from ``sc.AnnData`` object.
    if `force_redo` is True, data should be normalized and log-transformed.
    
    Parameters
    ----------
    adata:
        The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    force_redo: bool
        whether to recompute HVGs even when it has already been done
    batch_key:
        A column name in `adata.obs`
    n_top_genes:
        Number of highly-variable genes to keep. 
        If specified, other parameters will be ignored.
    key_hvg:
        A column that should be in ``adata.var.columns``
    hvg_kwds:
        Other parameters, e.g. min_mean, max_mean, min_disp, max_disp
    """
    # key_hvg = 'highly_variable'
    if force_redo or key_hvg not in adata.var.columns:
        logging.info(
            f'performing HVG-selection...\n '
            '(note that the input adata should have already been normalized '
            'and log-transformed)')
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_top_genes,
            batch_key=batch_key, **hvg_kwds)
    all_vars = adata.var_names
    is_hvg = adata.var[key_hvg]
    return list(all_vars[is_hvg])


def get_marker_info_table(
        adata, groups=None, key='rank_genes_groups',
        cut_padj: Optional[float] = 0.05,
        cut_logfc: Optional[float] = 0.25,
        cut_pts: Optional[float] = None,
):
    result = adata.uns[key]
    if groups is None:
        groups = result['names'].dtype.names

    dfs = []
    cols = ['names', 'logfoldchanges', 'pvals', 'pvals_adj', 'scores', ]
    # cols = [c for c in cols if c in result.keys()]
    flag_pts = 'pts' in result.keys()

    for group in groups:
        _df = pd.DataFrame({
            key: result[key][group] for key in cols
        })
        _df['group'] = group

        if flag_pts:
            # expression proportions, avoid row mismatching
            _df['pts'] = _df['names'].map(result['pts'][group])
            _df['pts_rest'] = _df['names'].map(result['pts_rest'][group])
            if cut_pts is not None:
                _df = _df[_df['pts'] >= cut_pts]
        if cut_padj is not None:
            _df = _df[_df['pvals_adj'] <= cut_padj]
        if cut_logfc is not None:
            _df = _df[_df['logfoldchanges'] >= cut_logfc]
        dfs.append(_df.copy())  # [['group'] + cols])
    df = pd.concat(dfs, axis=0, keys=groups)
    if flag_pts:
        cols += ['pts', 'pts_rest']
    return df[['group'] + cols]


def get_marker_name_table(adata, key='rank_genes_groups'):
    return pd.DataFrame(adata.uns[key]['names'])


def top_markers_from_df(marker_df, n=5, groups=None, unique=True, ):
    """
    Parameters
    ----------
    marker_df:
        a data-frame with cluster names as columns, and genes as values
    n: int
    groups:
        a list of cluster names (column names)
    unique: bool
        whether to flatten the results into a unique gene-list, default is True.

    Returns
    -------
    A dataframe (``union=False``) or a flattened marker list (``union=True``)
    """
    groups = marker_df.columns if groups is None else groups
    top = marker_df[groups].iloc[: n]
    #    top = marker_df[groups].iloc[: n].values.T.flatten()
    if unique:
        top = top.values.T.flatten()
        print(f'{len(top)} genes before taking unique')
        top = pd.unique(top)
        print(f'taking total of {len(top)} unique differential expressed genes')
    return top


def top_markers_from_info(
        df_info, n=5, groups=None, unique=True,
        col_group='group',
        col_name='names') -> list:
    """
    df_info: DEG-info table that can be take from `top_markers_from_adata`
    """
    # if groups is not None:
    #     df_info = df_info[df_info[col_group].isin(groups)]
    subdf = df_info.groupby(col_group).apply(lambda x: x.head(n))
    if groups is not None:
        subdf = subdf.loc[groups]  # filter, or keep the group orders
    names = subdf[col_name]
    return names.unique().tolist() if unique else names.tolist()


def top_markers_from_adata(adata: sc.AnnData,
                           n: Optional[int] = 5,
                           groups: Optional[Sequence] = None,
                           unique=True,
                           cut_padj=0.05,
                           cut_logfc=0.25,
                           cut_pts=0.1,
                           key='rank_genes_groups'):
    df_info = get_marker_info_table(
        adata, groups, key=key,
        cut_padj=cut_padj, cut_logfc=cut_logfc, cut_pts=cut_pts)

    if n is None:
        if unique:
            return df_info['names'].unique().tolist()
        return df_info['names'].tolist()
    else:
        # df = get_marker_name_table(adata, key=key)
        # return top_markers_from_df(df, n=n, groups=groups, unique=unique)
        if groups is None:
            _groupby = adata.uns[key]['params']['groupby']
            try:
                # to keep the group order
                groups = adata.obs[_groupby].cat.categories
            except:
                pass
        return top_markers_from_info(df_info, n=n, groups=groups, unique=unique)


def compute_and_get_DEGs(adata: sc.AnnData,
                         groupby: str,
                         n=20,
                         groups=None,
                         unique=True,
                         force_redo=False,
                         key_added='rank_genes_groups',
                         inplace=True,
                         do_normalize=False,
                         method='t-test',
                         return_info=False,
                         cuts={},
                         **kwds):
    """ Compute and get DEGs from ``sc.AnnData`` object
    
    By default, assume that the counts in adata has been normalized.
    If `force_redo`: re-compute DEGs and the original DEGs in adata will be ignored

    cuts: dict with keys 'cut_padj', 'cut_pts', and 'cut_logfc'

    """
    if not inplace:
        logging.info('making a copy')
        adata = adata.copy()
    adata.obs[groupby] = adata.obs[groupby].astype('category')
    if force_redo or key_added not in adata.uns.keys():
        logging.info(f'computing differentially expressed genes using {method}')
        if do_normalize:
            # todo: ``target_sum`` used to be 1e4
            normalize_default(adata, target_sum=None, )
        else:
            logging.info(
                'computing differential expression genes using default settings'
                '(assume that the expression values are already normalized)')
        if True:  # TODO: singletons will raise error
            adata = remove_adata_small_groups(adata, key=groupby, min_samples=1)
        sc.tl.rank_genes_groups(adata, groupby=groupby,
                                key_added=key_added,
                                pts=True,
                                method=method,
                                **kwds)
    if return_info:
        return get_marker_info_table(adata, key=key_added, **cuts)
    return top_markers_from_adata(adata, n=n,
                                  groups=groups,
                                  unique=unique,
                                  key=key_added, **cuts)


@dec_timewrapper('leiden')
def get_leiden_labels(adata, hvgs=None,
                      force_redo=False,
                      nneigh=20, reso=2, n_pcs=30,
                      neighbors_key=None,
                      key_added='leiden',
                      copy=False,
                      ):
    """ assume that the `X_pca` is already computed
    """
    adata = adata.copy() if copy else adata
    if 'X_pca' not in adata.obsm.keys():
        sc.tl.pca(adata, n_comps=n_pcs)
    _key_adj = 'connectivities'
    if neighbors_key is not None:
        _key_adj = f'{neighbors_key}_' + _key_adj
    if force_redo or _key_adj not in adata.obsp.keys():
        sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=nneigh, n_pcs=n_pcs,
                        key_added=neighbors_key)

    sc.tl.leiden(adata, resolution=reso,
                 key_added=key_added,
                 neighbors_key=neighbors_key)
    lbs = adata.obs[key_added]
    logging.info("Leiden results:\n%s", lbs.value_counts())
    return lbs


def _augment_full_repeat(
        x: np.ndarray,
        y: np.ndarray = None,
        n_add: int = 10,
        seed: int = 1234,
):
    # This function do NOT consider the groups labeled in y
    n0 = x.shape[0]
    np.random.seed(seed)
    ids_repeat = np.random.choice(n0, size=n_add, replace=True)
    if sparse.issparse(x):
        x_pseudo = x[ids_repeat,]
    else:
        x_pseudo = np.take(x, ids_repeat, axis=0)
    y_pseudo = np.take(y, ids_repeat, axis=0) if y is not None else None
    return x_pseudo, y_pseudo


def _augment_balance_group_repeat(
        x: np.ndarray,
        y: np.ndarray,
        n_tot_each: int = 1000,
        groups: Optional[Sequence] = None,
        concat: bool = True,
        seed: int = 1234,
):
    # TODO: in case that y has multi-dimensions
    # groups = np.unique(y, axis=0)
    groups = set(groups) if groups else set(y)
    groups = sorted(groups)
    x_is_sparse = sparse.issparse(x)
    if not x_is_sparse:
        x = np.asarray(x)
    y = np.asarray(y)
    x_pseudo_all, y_pseudo_all = [], []
    for lb in groups:
        ids = np.flatnonzero(y == lb)
        _n_add = n_tot_each - len(ids)
        if _n_add <= 0:
            continue
        else:
            logging.debug(lb)
            _x = x[ids,]
            _y = np.take(y, ids, axis=0)
            _x_pseudo, _y_pseudo = _augment_full_repeat(_x, _y, n_add=_n_add,
                                                        seed=seed)
        x_pseudo_all.append(_x_pseudo)
        y_pseudo_all.append(_y_pseudo)
    if len(x_pseudo_all) == 1:
        logging.warning(f"Only one group {groups[0]} was found!")
        x_pseudo_all = x_pseudo_all[0]
        y_pseudo_all = y_pseudo_all[0]
    else:
        if x_is_sparse:
            x_pseudo_all = sparse.vstack(x_pseudo_all)
        else:
            x_pseudo_all = np.concatenate(x_pseudo_all, axis=0)
        y_pseudo_all = np.concatenate(y_pseudo_all, axis=0)
    if concat:
        x_aug, y_aug, is_pseudo = _concat_x_and_y(
            [x, y], [x_pseudo_all, y_pseudo_all])
        is_pseudo = is_pseudo.astype(bool)
        return x_aug, y_aug, is_pseudo
    else:
        return x_pseudo_all, y_pseudo_all


def _concat_x_and_y(xy1, xy2):
    x1, y1 = xy1
    x2, y2 = xy2
    n1, n2 = x1.shape[0], x2.shape[0]
    if sparse.issparse(x1):
        x = sparse.vstack([x1, x2])
    else:
        x = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([y1, y2], axis=0)
    is_pseudo = np.array([False] * n1 + [True] * n2)
    return x, y, is_pseudo


def augment_repeat_adata(
        adata: sc.AnnData,
        key_y: str or int,  # column name in adata.obs
        n_tot_each: int = 1000,
        groups: Optional[Sequence] = None,
        concat: bool = True,
        seed: int = 1234,
        id_prefix: str = 'aug',
):
    x = adata.X
    y = adata.obs[key_y].values  # group labels
    key_pseudo = 'is_pseudo'
    x_pseudo, y_pseudo = _augment_balance_group_repeat(x, y, n_tot_each,
                                                       groups=groups,
                                                       concat=False, seed=seed)
    ids_pseudo = [f'{id_prefix}.{i}' for i in range(x_pseudo.shape[0])]
    obs_pseudo = pd.DataFrame({
        key_y: y_pseudo, key_pseudo: True,
    }, index=ids_pseudo)

    if concat:
        adata.obs[key_pseudo] = False
        return sc.AnnData(
            sparse.vstack([x, x_pseudo]),
            obs=pd.concat([adata.obs, obs_pseudo], axis=0),
            var=adata.var, )
    else:
        return sc.AnnData(x_pseudo, obs=obs_pseudo, var=adata.var, )


def __test__():
    fp = '/Users/yanyan/PycharmProjects/CAME/came/sample_data/raw-Baron_mouse.h5ad'
    adt = sc.read_h5ad(fp)
    key = 'cell_ontology_class'
    print(adt.obs[key].value_counts())
    adt = augment_repeat_adata(
        adt, key_y=key, n_tot_each=100,
        # groups=None,
        groups=['T cell'],
    )
    print(adt.obs[key].value_counts())
    print(adt.obs['is_pseudo'].value_counts())
