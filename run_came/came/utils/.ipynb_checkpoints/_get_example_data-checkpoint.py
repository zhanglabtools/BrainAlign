# -*- coding: UTF-8 -*-
"""
@author: Xingyan Liu
@file: _get_example_data.py
@time: 2021-06-12
"""

import os
from pathlib import Path
from typing import Sequence, Union, Dict, List, Optional  # , Callable
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import logging

CAME_ROOT = Path(__file__).parents[1]


def _extract_zip(
        fp_zip=CAME_ROOT / 'sample_data.zip',
        fp_unzip=CAME_ROOT / 'sample_data',
):
    import zipfile
    with zipfile.ZipFile(fp_zip) as zipf:
        zipf.extractall(fp_unzip)


def load_example_data() -> Dict:
    """ Load example data, for a quick start with CAME.

    This pair of cross-species datasets contains the pancreatic scRNA-seq data
    of human ("Baron_human") and mouse ("Baron_human"),
    initially published with paper [1].

    NOTE that "Baron_human" is a 20%-subsample from the original data.
    The resulting cell-typing accuracy may not be as good as one
    using full dataset as the reference.

    [1] Baron, M. et al. (2016) A Single-Cell Transcriptomic Map of the Human
    and Mouse Pancreas Reveals Inter- and Intra-cell Population Structure.
    Cell Syst 3 (4), 346-360.e4.

    Returns
    -------
    dict:
        a dict with keys ['adatas', 'varmap', 'varmap_1v1', 'dataset_names', 'key_class']

    Examples
    --------
    >>> example_data_dict = load_example_data()
    >>> print(example_data_dict.keys())
    # Out[]: dict_keys(['adatas', 'varmap', 'varmap_1v1', 'dataset_names', 'key_class'])

    >>> adatas = example_data_dict['adatas']
    >>> dsnames = example_data_dict['dataset_names']  # ('Baron_human', 'Baron_mouse')
    >>> df_varmap = example_data_dict['varmap']
    >>> df_varmap_1v1 = example_data_dict['varmap_1v1']
    >>> key_class1 = key_class2 = example_data_dict['key_class']

    """
    datadir = CAME_ROOT / 'sample_data'

    sp1, sp2 = ('human', 'mouse')
    dsnames = ('Baron_human', 'Baron_mouse')
    dsn1, dsn2 = dsnames
    fp1, fp2 = datadir / f'raw-{dsn1}.h5ad', datadir / f'raw-{dsn2}.h5ad'
    fp_varmap_1v1 = datadir / f'gene_matches_1v1_{sp1}2{sp2}.csv'
    fp_varmap = datadir / f'gene_matches_{sp1}2{sp2}.csv'

    if not (datadir.exists() and fp1.exists() and fp2.exists() and
            fp_varmap.exists() and fp_varmap_1v1.exists()):
        _extract_zip()

    df_varmap_1v1 = pd.read_csv(fp_varmap_1v1, )
    df_varmap = pd.read_csv(fp_varmap, )

    adata_raw1, adata_raw2 = sc.read_h5ad(fp1), sc.read_h5ad(fp2)

    key_class = 'cell_ontology_class'
    example_dict = {
        'adatas': [adata_raw1, adata_raw2],
        'varmap': df_varmap,
        'varmap_1v1': df_varmap_1v1,
        'dataset_names': dsnames,
        'key_class': key_class,
    }
    logging.info(example_dict.keys())
    logging.debug(example_dict)
    return example_dict


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s-%(lineno)d-%(funcName)s(): '
               '%(levelname)s\n %(message)s')
    d = load_example_data()
    print(d.keys())
