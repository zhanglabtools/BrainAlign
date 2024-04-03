# -*- coding: utf-8 -*-
"""
@author: Xingyan Liu
@file: vis_hidden_layers.py
@time: Dec 23 22:06:09 2020
"""
from typing import Union, Optional, Tuple, Collection, Sequence, Iterable
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
import logging
import numba
import numpy as np
from scanpy import AnnData


def downsample_total_counts(adata, frac=0.5, copy=False, **kwds):
    total0 = adata.X.data.sum()
    total1 = int(total0 * frac)
    logging.info('before:', total0, '\tafter:', total1)
    return downsample_counts(
                adata, total_counts=total1, 
                copy=copy, **kwds)


def downsample_counts_per_cell(adata, frac=0.5, copy=False, **kwds):
    cnts_per_cell0 = adata.X.sum(1)
    if hasattr(cnts_per_cell0, 'A1'):
        cnts_per_cell0 = cnts_per_cell0.A1
    cnts_per_cell = (cnts_per_cell0 * frac).flatten().astype(int)
    return downsample_counts(
                adata, counts_per_cell=cnts_per_cell, 
                copy=copy, **kwds)
    
    
def downsample_counts(
    adata: AnnData,
    counts_per_cell: Optional[Union[int, Collection[int]]] = None,
    total_counts: Optional[int] = None,
    *,
    random_state = 0,
    replace: bool = False,
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    Downsample counts from count matrix.

    If `counts_per_cell` is specified, each cell will downsampled.
    If `total_counts` is specified, expression matrix will be downsampled to
    contain at most `total_counts`.

    Parameters
    ----------
    adata
        Annotated data matrix.
    counts_per_cell
        Target total counts per cell. If a cell has more than 'counts_per_cell',
        it will be downsampled to this number. Resulting counts can be specified
        on a per cell basis by passing an array.Should be an integer or integer
        ndarray with same length as number of obs.
    total_counts
        Target total counts. If the count matrix has more than `total_counts`
        it will be downsampled to have this number.
    random_state
        Random seed for subsampling.
    replace
        Whether to sample the counts with replacement.
    copy
        Determines whether a copy of `adata` is returned.

    Returns
    -------
    Depending on `copy` returns or updates an `adata` with downsampled `.X`.
    """
    # This logic is all dispatch
    total_counts_call = total_counts is not None
    counts_per_cell_call = counts_per_cell is not None
    if total_counts_call is counts_per_cell_call:
        raise ValueError(
            "Must specify exactly one of `total_counts` or `counts_per_cell`."
        )
    if copy:
        adata = adata.copy()
    if total_counts_call:
        adata.X = _downsample_total_counts(
            adata.X, total_counts, random_state, replace
        )
        print(adata.X.data.sum())
    elif counts_per_cell_call:
        adata.X = _downsample_per_cell(
            adata.X, counts_per_cell, random_state, replace
        )
    if copy:
        return adata


def _downsample_per_cell(X, counts_per_cell, random_state, replace):
    n_obs = X.shape[0]
    if isinstance(counts_per_cell, int):
        counts_per_cell = np.full(n_obs, counts_per_cell)
    else:
        counts_per_cell = np.asarray(counts_per_cell)
    # np.random.choice needs int arguments in numba code:
    counts_per_cell = counts_per_cell.astype(np.int_, copy=False)
    if (
        not isinstance(counts_per_cell, np.ndarray)
        or len(counts_per_cell) != n_obs
    ):
        raise ValueError(
            "If provided, 'counts_per_cell' must be either an integer, or "
            "coercible to an `np.ndarray` of length as number of observations"
            " by `np.asarray(counts_per_cell)`."
        )
    if issparse(X):
        original_type = type(X)
        if not isspmatrix_csr(X):
            X = csr_matrix(X)
        totals = np.ravel(X.sum(axis=1))  # Faster for csr matrix
        under_target = np.nonzero(totals > counts_per_cell)[0]
        rows = np.split(X.data, X.indptr[1:-1])
        for rowidx in under_target:
            row = rows[rowidx]
            _downsample_array(
                row,
                row, #.astype(int),
                counts_per_cell[rowidx],
                random_state=random_state,
                replace=replace,
                inplace=True,
            )
        X.eliminate_zeros()
        if original_type is not csr_matrix:  # Put it back
            X = original_type(X)
    else:
        totals = np.ravel(X.sum(axis=1))
        under_target = np.nonzero(totals > counts_per_cell)[0]
        for rowidx in under_target:
            row = X[rowidx, :]
            _downsample_array(
                row,
                row, #.astype(int),
                counts_per_cell[rowidx],
                random_state=random_state,
                replace=replace,
                inplace=True,
            )
    return X


def _downsample_total_counts(X, total_counts, random_state, replace):
    total_counts = int(total_counts)
    total = X.sum()
    if total < total_counts:
        return X
    if issparse(X):
        original_type = type(X)
        if not isspmatrix_csr(X):
            X = csr_matrix(X)
        _downsample_array(
            X.data,
            X.data.astype(int),
            total_counts,
            random_state=random_state,
            replace=replace,
            inplace=True,
        )
        X.eliminate_zeros()
        if original_type is not csr_matrix:
            X = original_type(X)
    else:
        v = X.reshape(np.multiply(*X.shape))
        _downsample_array(
            v, v.astype(int), total_counts, random_state, replace=replace, inplace=True
        )
    return X


@numba.njit(cache=True)
def _downsample_array(
    col: np.ndarray,
    col_int: np.ndarray, 
    target: int,
    random_state = 0,
    replace: bool = True,
    inplace: bool = False,
):
    """\
    Evenly reduce counts in cell to target amount.

    This is an internal function and has some restrictions:

    * total counts in cell must be less than target
    """
    np.random.seed(random_state)
    cumcounts = col_int.cumsum()
    if inplace:
        col[:] = 0
    else:
        col = np.zeros_like(col)
    total = np.int_(cumcounts[-1])
#    print('TEST `_downsample_array`:', total, target)
    sample = np.random.choice(total, target, replace=replace)
    sample.sort()
    geneptr = 0
    for count in sample:
        while count >= cumcounts[geneptr]:
            geneptr += 1
        col[geneptr] += 1
    return col



