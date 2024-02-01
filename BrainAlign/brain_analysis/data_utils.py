import numpy as np
import pickle
import numpy as np
import scanpy as sc, anndata as ad
import scipy.sparse as sp
import pandas as pd
import sys
import matplotlib.pyplot as plt
import re, seaborn as sns
from collections import OrderedDict
import os
import time
import sklearn

import scipy

from collections.abc import Mapping
from typing import Union, Sequence, Dict, Optional, Tuple

from scipy.spatial.distance import pdist, squareform

def get_correlation(X1:Union[np.ndarray, pd.DataFrame], X2:Union[np.ndarray, pd.DataFrame], metric_name:str='correlation') -> Sequence:
    """

    Use pdist to compute correlation list for two matrix with the same column length. The correlation is acquired via 1 - distance

    :param X1: matrix 1
    :param X2: matrix 2
    :param metric_name: The distance name used by scipy.spatial.distance
    :return: correlation list
    """
    # df1 = pd.DataFrame(X1).T
    # df2 = pd.DataFrame(X2).T
    # sample_df = pd.concat([df1.T, df2.T], axis=0)
    # distances = pdist(sample_df.values, metric=metric_name)
    # dist_matrix = squareform(distances)
    # X1_num = X1.shape[0]
    # X2_num = X2.shape[0]
    # dist_matrix_all = dist_matrix[0:X1_num, X1_num:X1_num + X2_num]
    # corr_matrix_all = 1 - dist_matrix_all
    # corr_all = list(corr_matrix_all.flatten())
    if type(X1) == pd.DataFrame:
        X1 = X1.values
    if type(X2) == pd.DataFrame:
        X2 = X2.values

    mouse_df = pd.DataFrame(X1).T
    human_df = pd.DataFrame(X2).T
    #if type(X1) == np.ndarray:
    mouse_df.columns = ['X1_' + str(x) for x in mouse_df.columns]
    human_df.columns = ['X2_' + str(x) for x in human_df.columns]
    ## Compute correlation of homologous regions compared to the other pairs
    corr_result = pd.concat([mouse_df, human_df], axis=1).corr()

    Var_Corr = corr_result[human_df.columns].loc[mouse_df.columns]
    #print(Var_Corr)
    corr_all = list(Var_Corr.values.flatten())
    return corr_all


def get_distance(X, Y, method='euclidean'):
    D_mat = np.zeros((X.shape[0], Y.shape[0]))
    if method == 'euclidean':
        D_mat = scipy.spatial.distance.euclidean(X, Y)
    return np.flatten(D_mat)

def get_submin(X:np.ndarray):
    """
    :param X: input matrix for computing minimum values
    :return:
    """
    row_n = X.shape[0]
    X_sub = 1000 * np.ones((X.shape[0], X.shape[1]-1))
    for i in range(row_n):
        if i == 0:
            X_sub[i, i:] = X[i, i+1:]
        elif i == row_n-1:
            X_sub[i, 0:i-1] = X[i, i:i-1]
        else:
            X_sub[i, 0:i-1] = X[i, 0:i-1]
            X_sub[i, i-1:] = X[i, i:]

    return np.min(X_sub, axis=1)


class Dict2Mapping(Mapping):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, key):
        return key in self.data

def threshold_array(X):
    '''
    input: array row: sample, column: gene vector
    output: a binary matrix
    For each value in (i, j), the binary value = if(M_ij > avg(column_j))
    '''
    return X > np.mean(X, axis=0)

def threshold_quantile(X, quantile=0.8):
    '''
    input: array row: sample, column: gene vector
    output: a binary matrix
    For each value in (i, j), the binary value = if(M_ij > avg(column_j))
    '''
    return X > np.quantile(X, quantile, axis=0)


def threshold_top(X, percent=1):
    '''
    input: array row: sample, column: gene vector
    output: a binary matrix
    For each value in (i, j), the binary value = if(M_ij > avg(column_j))
    '''
    #topk = int(round(X.shape[0] * percent))
    topk = percent
    #print(topk)
    #topk_pos = X.shape[0] - topk
    X_sort = np.sort(X, axis=0)
    return X >= X_sort[-topk, :]

def threshold_std(X, std_times=1):
    '''
    input: array row: sample, column: gene vector
    output: a binary matrix
    '''
    std_array = np.std(X, axis=0)
    mean_array = np.mean(X, axis=0)
    threshold_array = mean_array + std_times * std_array
    return X >= threshold_array

def num_zero_rows(X):
    Y = np.sum(X, axis=1)
    return np.max(Y.shape[0]) - np.count_nonzero(Y)


def threshold_array_nonzero(X):
    '''
    input: array row: sample, column: gene vector
    output: a binary matrix
    For each value in (i, j), the binary value = if(M_ij > avg(column_j))
    '''
    return X > 0


from sklearn.model_selection import train_test_split

def train_test_val_split(data_x, data_y, train_ratio = 0.8,validation_ratio = 0.1,test_ratio = 0.1, random_state=1):
    # random_state for reproduction
    # shuffle must be 'True'
    [x_train, x_test, y_train, y_test] = train_test_split(data_x, data_y, test_size=validation_ratio+test_ratio, random_state=random_state, shuffle=True)
    [x_val, x_test, y_val, y_test] = train_test_split(
    x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=random_state)
    return x_train,y_train, x_val, y_val, x_test, y_test

from scipy.stats import pearsonr
def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

def generate_correlation_map(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                   mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])

# plot marker genes using umap
def plot_marker_selection_umap(data, markers, names, umap_neighbor=30):
    print('Computing UMAP embedding')
    t = time.time()

    adata_X = ad.AnnData(data)
    adata_X.obs['labels'] = names
    sc.pp.neighbors(adata_X, n_neighbors=umap_neighbor, metric='cosine', use_rep='X')
    sc.tl.umap(adata_X)
    X_original = adata_X.obsm['X_umap']

    adata_X_marker = ad.AnnData(data[:, markers])
    adata_X_marker.obs['labels'] = names
    sc.pp.neighbors(adata_X_marker, n_neighbors=umap_neighbor, metric='cosine', use_rep='X')
    sc.tl.umap(adata_X_marker)
    X_embedded = adata_X_marker.obsm['X_umap']
    '''
    X_original = sklearn.manifold.TSNE(
        n_components=2, perplexity=perplexity).fit_transform(data)
    X_embedded = sklearn.manifold.TSNE(n_components=2, perplexity=perplexity).fit_transform(
        data[:, markers])
    '''
    print('Elapsed time: {} seconds'.format(time.time() - t))
    cmap = plt.cm.jet
    unique_names = list(set(names))
    num_labels = len(unique_names)
    colors = [cmap(int(i * 256 / num_labels)) for i in range(num_labels)]
    aux = [colors[unique_names.index(name)] for name in names]

    fig = plt.figure()
    ax = fig.add_subplot(121)
    for g in unique_names:
        i = [s for s in range(len(names)) if names[s] == g]
        ax.scatter(X_original[i, 0], X_original[i, 1],
                   c=[aux[i[0]]], s=1, label=names[i[0]])
    ax.set_title('Original data')
    ax2 = fig.add_subplot(122)
    for g in np.unique(names):
        i = [s for s in range(len(names)) if names[s] == g]
        ax2.scatter(X_embedded[i, 0], X_embedded[i, 1],
                    c=[aux[i[0]]], s=1, label=names[i[0]])
    ax2.set_title('{} markers'.format(len(markers)))
    plt.legend(bbox_to_anchor=(1, 1), frameon=False, ncol=5)
    plt.subplots_adjust(right=0.7)
    return fig


def plot_marker_selection_umap_embedding(data, embedding, markers, names, umap_neighbor=30):
    print('Computing UMAP embedding')
    t = time.time()

    adata_X = ad.AnnData(embedding)
    adata_X.obs['labels'] = names
    sc.pp.neighbors(adata_X, n_neighbors=umap_neighbor, metric='cosine', use_rep='X')
    sc.tl.umap(adata_X)
    X_original = adata_X.obsm['X_umap']

    adata_X_marker = ad.AnnData(data[:, markers])
    adata_X_marker.obs['labels'] = names
    sc.pp.neighbors(adata_X_marker, n_neighbors=umap_neighbor, metric='cosine', use_rep='X')
    sc.tl.umap(adata_X_marker)
    X_embedded = adata_X_marker.obsm['X_umap']
    '''
    X_original = sklearn.manifold.TSNE(
        n_components=2, perplexity=perplexity).fit_transform(data)
    X_embedded = sklearn.manifold.TSNE(n_components=2, perplexity=perplexity).fit_transform(
        data[:, markers])
    '''
    print('Elapsed time: {} seconds'.format(time.time() - t))
    cmap = plt.cm.jet
    unique_names = list(set(names))
    num_labels = len(unique_names)
    colors = [cmap(int(i * 256 / num_labels)) for i in range(num_labels)]
    aux = [colors[unique_names.index(name)] for name in names]

    fig = plt.figure()
    ax = fig.add_subplot(121)
    for g in unique_names:
        i = [s for s in range(len(names)) if names[s] == g]
        ax.scatter(X_original[i, 0], X_original[i, 1],
                   c=[aux[i[0]]], s=1, label=names[i[0]])
    ax.set_title('Original data')
    ax2 = fig.add_subplot(122)
    for g in np.unique(names):
        i = [s for s in range(len(names)) if names[s] == g]
        ax2.scatter(X_embedded[i, 0], X_embedded[i, 1],
                    c=[aux[i[0]]], s=1, label=names[i[0]])
    ax2.set_title('{} markers'.format(len(markers)))
    plt.legend(bbox_to_anchor=(1, 1), frameon=False, ncol=5)
    plt.subplots_adjust(right=0.7)
    return fig


def plot_marker_selection_TSNE_embedding(data, embedding, markers, names, perplexity=40):
    print('Computing UMAP embedding')
    t = time.time()


    X_original = sklearn.manifold.TSNE(
        n_components=2, perplexity=perplexity).fit_transform(embedding)
    X_embedded = sklearn.manifold.TSNE(n_components=2, perplexity=perplexity).fit_transform(
        data[:, markers])

    print('Elapsed time: {} seconds'.format(time.time() - t))
    cmap = plt.cm.jet
    unique_names = list(set(names))
    num_labels = len(unique_names)
    colors = [cmap(int(i * 256 / num_labels)) for i in range(num_labels)]
    aux = [colors[unique_names.index(name)] for name in names]

    fig = plt.figure()
    ax = fig.add_subplot(121)
    for g in unique_names:
        i = [s for s in range(len(names)) if names[s] == g]
        ax.scatter(X_original[i, 0], X_original[i, 1],
                   c=[aux[i[0]]], s=1, label=names[i[0]])
    ax.set_title('Original data')
    ax2 = fig.add_subplot(122)
    for g in np.unique(names):
        i = [s for s in range(len(names)) if names[s] == g]
        ax2.scatter(X_embedded[i, 0], X_embedded[i, 1],
                    c=[aux[i[0]]], s=1, label=names[i[0]])
    ax2.set_title('{} markers'.format(len(markers)))
    plt.legend(bbox_to_anchor=(1, 1), frameon=False, ncol=5)
    plt.subplots_adjust(right=0.7)
    return fig

def load_dimensions_binary(datapair_path):
    path_datapiar_file = open(datapair_path, 'rb')
    datapair = pickle.load(path_datapiar_file)
    #print(datapair)
    binary_S = datapair['ov_adjs'][0].shape[0]
    binary_M = datapair['ov_adjs'][0].shape[1]
    binary_H = datapair['ov_adjs'][1].shape[1]
    binary_V = datapair['ov_adjs'][1].shape[0]
    #print('binary_S:', binary_S, 'binary_M:', binary_M, 'binary_H:', binary_H, 'binary_V:', binary_V)
    S = binary_S + binary_V
    M = binary_M + binary_H
    #print('S:', S, 'M:', M)
    return S, M, binary_S, binary_M, binary_H, binary_V


class load_gb_2020:
    def __init__(self, expression_csv_folder: str ='./bulk/'):
        self.species_list = ['Bonobo', 'Chimpanzee', 'Human', 'Macaque']
        self.exp_file_list = ['Bonobo', 'Chimpanzee', 'Human', 'Macaques']
        self.expression_csv_folder = expression_csv_folder
        self.expression_csv_list = [self.expression_csv_folder + 'GSE127898_' + x.lower() + '.csv' for x in self.exp_file_list]

        self.gene_id2name_file_list =  [self.expression_csv_folder + 'gene_id2name/' + x.lower() + '.tsv' for x in self.exp_file_list]

        self.adata_save_path = self.expression_csv_folder + '/adatas/'
        if not os.path.exists(self.adata_save_path):
            os.makedirs(self.adata_save_path)

        sampleID_region_df = pd.read_csv(expression_csv_folder + 'sampleID_region.tsv', sep='\t')
        sample_id_region = sampleID_region_df['sample_id_region'].values
        sample_name = []
        sample_region_id = []
        sample_region_name = []
        sample_region_id_name = []

        for s in sample_id_region:
            sample_n = s.split(': ')[0]
            sample_name.append(sample_n)
            s_1 = s.split(': ')[1]
            sample_region_id_name.append(s_1)
            s_list = s_1.split(' ')
            sample_region_id.append(s_list[0])
            sample_r_n = s_1.strip(s_list[0]).strip(' ')
            sample_region_name.append(sample_r_n)

        region_df = pd.DataFrame.from_dict({'sample_name': sample_name,
                                            'sample_region_id:': sample_region_id,
                                            'sample_region_name:': sample_region_name,
                                            'sample_region_id_name:': sample_region_id_name})
        region_df.to_csv(expression_csv_folder + 'sampleID_region_info.tsv', sep='\t')

        self.sample_region_id_dict = {k: v for k, v in zip(sample_name, sample_region_id)}
        self.sample_region_name_dict = {k: v for k, v in zip(sample_name, sample_region_name)}
        self.sample_region_id_name_dict = {k: v for k, v in zip(sample_name, sample_region_id_name)}

    def csv2adata_single(self, expression_csv_file, gene_id2name_file):
        df = pd.read_csv(expression_csv_file, index_col=0).T

        adata = sc.AnnData(df, dtype=np.float32)

        gene_id2name_df = pd.read_csv(gene_id2name_file, sep='\t')

        id2name_dict = {k: v for k, v in
                        zip(gene_id2name_df['Gene stable ID'].values, gene_id2name_df['Gene name'].values)}

        adata.var['gene_stable_id'] = adata.var_names
        adata.var_names = [g_id if g_id not in set(id2name_dict.keys()) or str(id2name_dict[g_id]) == 'nan' else id2name_dict[g_id] for g_id in adata.var_names]

        adata.obs['region_id'] = [self.sample_region_id_dict[s] for s in adata.obs_names]
        adata.obs['region_name'] = [self.sample_region_name_dict[s] for s in adata.obs_names]
        adata.obs['region_id_name'] = [self.sample_region_id_name_dict[s] for s in adata.obs_names]

        adata.var_names_make_unique()

        return adata

    def csv2adata_all(self):
        for specie_name, expression_csv_file, gene_id2name_file in zip(self.species_list, self.expression_csv_list, self.gene_id2name_file_list):
            print('Process '+ specie_name + ':')
            print(specie_name, expression_csv_file, gene_id2name_file)
            adata = self.csv2adata_single(expression_csv_file, gene_id2name_file)
            print(adata)
            adata.write_h5ad(self.adata_save_path + specie_name+'.h5ad')
            print('Finished.')



    def forward(self):

        self.csv2adata_all()

        return None


def check_dirs(path):
    if os.path.exists(path):
        print('already exists:\n\t%s' % path)
    else:
        os.makedirs(path)
        print('a new directory made:\n\t%s' % path)



if __name__ == '__main__':
    X1 = np.array([[1.1, 1], [0.1, 1], [1, 0.5]]) # 3 * 2
    X2 = np.array([[1, 1.2], [1, 1.1]]) # 2 * 2
    li = scipy.spatial.distance_matrix(X1, X2, p=2)
    print(li)