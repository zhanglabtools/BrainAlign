# -- coding: utf-8 --
# @Time : 2022/10/17 14:28
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : analysis.py
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
sys.path.append('../code/')
from BrainAlign.code.utils import set_params
from BrainAlign.brain_analysis.configs import heco_config
from BrainAlign.brain_analysis.data_utils import plot_marker_selection_umap, plot_marker_selection_umap_embedding
from BrainAlign.brain_analysis.process import normalize_before_pruning
from scipy.spatial.distance import squareform, pdist
sys.path.append('../')
#import ./came
import BrainAlign.came as came
from BrainAlign.came import pipeline, pp, pl, make_abstracted_graph, weight_linked_vars

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from matplotlib import rcParams


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from scipy.spatial.distance import pdist, squareform

from scipy import stats

# call marker gene
#from scGeneFit.functions import *

# color map
import colorcet as cc

try:
    import matplotlib as mpl
    mpl.use('agg')
except Exception as e:
    print(f"An error occurred when setting matplotlib backend ({e})")



def transform_embedding(Mat, M_embedding):
    '''
    Add one in the  denominator to avoid infinite values
    '''
    divide_mat = np.sum(Mat, axis=1)
    for x in range(divide_mat.shape[0]):
        if divide_mat[x] == 0:
            divide_mat[x] = 1
    res_mat = (Mat.dot(M_embedding).T / divide_mat ).T
    N_sample = res_mat.shape[0]
    for i in range(N_sample):
        if np.sum(np.abs(res_mat[i, :])) == 0:
            res_mat[i, 0] = 1e-8
    return res_mat


def load_came_embeddings(cfg):
    # expression_human_path = '../brain_human_mouse/human_brain_region_88_sparse_with3d.h5ad'
    adata_human = sc.read_h5ad(cfg.CAME.path_rawdata2)
    # expression_mouse_path = '../../Brain_ST_human_mouse/data/mouse_brain_region_67_sparse_no_threshold.h5ad'
    adata_mouse = sc.read_h5ad(cfg.CAME.path_rawdata1)


    adata_human_mouse = sc.read_h5ad(cfg.CAME.ROOT + 'adt_hidden_cell.h5ad')
    adata_human_mouse.obs_names = adata_human_mouse.obs['original_name']
    #print(adata_human_mouse.obs_names)
    #print(adata_human_mouse)
    # Step 1
    #embedding_len = 128
    mouse_anndata = adata_human_mouse[adata_mouse.obs_names]
    #print('1:', mouse_anndata)
    mouse_anndata.obs['region_name'] = adata_mouse.obs['region_name']
    for sample_id in mouse_anndata.obs_names:
        mouse_anndata[sample_id].obs['region_name'] = adata_mouse[sample_id].obs['region_name']
    #print('2:', mouse_anndata)
    if not os.path.exists(cfg.BrainAlign.embeddings_file_path):
        os.makedirs(cfg.BrainAlign.embeddings_file_path)

    mouse_anndata.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/s_embeddings.h5ad')

    adata_human_mouse.obs_names = adata_human_mouse.obs['original_name']
    human_anndata = adata_human_mouse[adata_human.obs_names]
    human_anndata.obs['region_name'] = adata_human.obs['region_name']
    for sample_id in human_anndata.obs_names:
        human_anndata[sample_id].obs['region_name'] = adata_human[sample_id].obs['region_name']
    human_anndata.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/v_embeddings.h5ad')

    # save came gene embeddings
    m_embeddings = sc.read_h5ad(cfg.CAME.ROOT + 'adt_hidden_gene1.h5ad')
    m_embeddings.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/m_embeddings.h5ad')
    h_embeddings = sc.read_h5ad(cfg.CAME.ROOT + 'adt_hidden_gene2.h5ad')
    h_embeddings.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/h_embeddings.h5ad')


def load_srrsc_embeddings(cfg):
    ## Load heco learned embeddings
    f = open(cfg.BrainAlign.embeddings_file_path + 'node.pkl', "rb")
    all_embeddings = pickle.load(f)
    print(all_embeddings.shape)
    s_embeddings = all_embeddings[0:cfg.BrainAlign.binary_S, :]
    v_embeddings = all_embeddings[cfg.BrainAlign.binary_S:cfg.BrainAlign.binary_S+cfg.BrainAlign.binary_V, :]
    m_embeddings = all_embeddings[(cfg.BrainAlign.binary_S+cfg.BrainAlign.binary_V):(cfg.BrainAlign.binary_S+cfg.BrainAlign.binary_V+cfg.BrainAlign.binary_M), :]
    h_embeddings = all_embeddings[cfg.BrainAlign.binary_S+cfg.BrainAlign.binary_V+cfg.BrainAlign.binary_M:, :]

    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)

    mouse_sample_list = list(datapair['obs_dfs'][0].index)
    # print(mouse_sample_list)
    print(len(mouse_sample_list))

    human_sample_list = list(datapair['obs_dfs'][1].index)
    # print(human_sample_list)
    print(len(human_sample_list))

    print(datapair)

    print(len(datapair['varnames_node'][0]))
    np.save(cfg.BrainAlign.embeddings_file_path + 'mouse_gene_names.npy', datapair['varnames_node'][0])
    print(len(datapair['varnames_node'][1]))
    np.save(cfg.BrainAlign.embeddings_file_path + 'human_gene_names.npy', datapair['varnames_node'][1])

    mouse_gene_list = datapair['varnames_node'][0]
    human_gene_list = datapair['varnames_node'][1]


    # save heco mouse embeddings
    mouse_anndata = ad.AnnData(s_embeddings)
    mouse_anndata.obs_names = mouse_sample_list
    mouse_anndata.obs['region_name'] = datapair['obs_dfs'][0]['region_name']
    print('Region name', datapair['obs_dfs'][0]['region_name'])
    mouse_anndata.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/s_embeddings.h5ad')

    # save heco human embeddings
    human_anndata = ad.AnnData(v_embeddings)
    human_anndata.obs_names = human_sample_list
    human_anndata.obs['region_name'] = datapair['obs_dfs'][1]['region_name']
    print('Region name', datapair['obs_dfs'][1]['region_name'])
    human_anndata.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/v_embeddings.h5ad')

    # save mouse genes embeddings
    # if cfg.BrainAlign.NODE_TYPE_NUM == 2:
    mouse_gene_anndata = ad.AnnData(m_embeddings)
    mouse_gene_anndata.obs_names = mouse_gene_list
    mouse_gene_anndata.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/m_embeddings.h5ad')

    # save human genes embeddings
    human_gene_anndata = ad.AnnData(h_embeddings)
    human_gene_anndata.obs_names = human_gene_list
    human_gene_anndata.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/h_embeddings.h5ad')


def load_heco_embeddings(cfg):

    ## Load heco learned embeddings
    if cfg.BrainAlign.NODE_TYPE_NUM == 4:
        f = open(cfg.BrainAlign.embeddings_file_path+'node_S.pkl', "rb")
        s_embeddings = pickle.load(f)
        print(s_embeddings)
        print(s_embeddings.shape)

        '''
        f = open(cfg.BrainAlign.embeddings_file_path + 'node_1.pkl', "rb")
        m_embeddings = pickle.load(f)
    
        f = open(cfg.BrainAlign.embeddings_file_path + 'node_2.pkl', "rb")
        h_embeddings = pickle.load(f)
        '''
        f = open(cfg.BrainAlign.embeddings_file_path + 'node_V.pkl', "rb")
        v_embeddings = pickle.load(f)


        f = open(cfg.BrainAlign.embeddings_file_path + 'node_M.pkl', "rb")
        m_embeddings = pickle.load(f)

        f = open(cfg.BrainAlign.embeddings_file_path + 'node_H.pkl', "rb")
        h_embeddings = pickle.load(f)




    elif cfg.BrainAlign.NODE_TYPE_NUM == 2:
        f = open(cfg.BrainAlign.embeddings_file_path + 'node_S.pkl', "rb")
        sv_embeddings = pickle.load(f)
        print(sv_embeddings.shape)
        s_embeddings = sv_embeddings[0:cfg.BrainAlign.binary_S, :]
        v_embeddings = sv_embeddings[cfg.BrainAlign.binary_S:, :]

        f = open(cfg.BrainAlign.embeddings_file_path + 'node_M.pkl', "rb")
        mh_embeddings = pickle.load(f)
        print(mh_embeddings.shape)
        m_embeddings = mh_embeddings[0:cfg.BrainAlign.binary_M, :]
        h_embeddings = mh_embeddings[cfg.BrainAlign.binary_M:, :]


    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)

    mouse_sample_list = list(datapair['obs_dfs'][0].index)
    # print(mouse_sample_list)
    print(len(mouse_sample_list))

    human_sample_list = list(datapair['obs_dfs'][1].index)
    # print(human_sample_list)
    print(len(human_sample_list))

    print(datapair)

    print(len(datapair['varnames_node'][0]))
    np.save(cfg.BrainAlign.embeddings_file_path + 'mouse_gene_names.npy', datapair['varnames_node'][0])
    print(len(datapair['varnames_node'][1]))
    np.save(cfg.BrainAlign.embeddings_file_path + 'human_gene_names.npy', datapair['varnames_node'][1])

    mouse_gene_list = datapair['varnames_node'][0]
    human_gene_list = datapair['varnames_node'][1]


    '''
    sm_ = datapair['ov_adjs'][0].toarray()
    if cfg.BrainAlign.if_threshold:
        sm_ = threshold_top(sm_, percent=cfg.BrainAlign.gene_top) + threshold_top(sm_.T, percent=cfg.BrainAlign.sample_top).T
    print(sm_)
    print('sm_', sm_.shape)

    vh_ = datapair['ov_adjs'][1].toarray()
    if cfg.BrainAlign.if_threshold:
        vh_ = threshold_top(vh_, percent=cfg.BrainAlign.gene_top) + threshold_top(vh_.T, percent=cfg.BrainAlign.sample_top).T
    print('vh_', vh_.shape)
    mh_ = datapair['vv_adj'].toarray()[0:cfg.BrainAlign.M, cfg.BrainAlign.M:] > 0
    print('mh_', mh_.shape)
    #if cfg.BrainAlign.if_threshold:
    sh_ = np.matmul(sm_, mh_) > 0
    sv_ = np.matmul(sh_, vh_.T) > 0

    m_embeddings = transform_embedding(sm_.T, s_embeddings)
    h_embeddings = transform_embedding(sh_.T, s_embeddings)
    v_embeddings = transform_embedding(sv_.T, s_embeddings)
    '''
    # save heco mouse embeddings
    mouse_anndata = ad.AnnData(s_embeddings)
    mouse_anndata.obs_names = mouse_sample_list
    mouse_anndata.obs['region_name'] = datapair['obs_dfs'][0]['region_name']
    print('Region name', datapair['obs_dfs'][0]['region_name'])
    mouse_anndata.write_h5ad(cfg.BrainAlign.embeddings_file_path +'/s_embeddings.h5ad')

    # save heco human embeddings
    human_anndata = ad.AnnData(v_embeddings)
    human_anndata.obs_names = human_sample_list
    human_anndata.obs['region_name'] = datapair['obs_dfs'][1]['region_name']
    print('Region name', datapair['obs_dfs'][1]['region_name'])
    human_anndata.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/v_embeddings.h5ad')

    # save mouse genes embeddings
    #if cfg.BrainAlign.NODE_TYPE_NUM == 2:
    mouse_gene_anndata = ad.AnnData(m_embeddings)
    mouse_gene_anndata.obs_names = mouse_gene_list
    mouse_gene_anndata.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/m_embeddings.h5ad')

    # save human genes embeddings
    human_gene_anndata = ad.AnnData(h_embeddings)
    human_gene_anndata.obs_names = human_gene_list
    human_gene_anndata.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/h_embeddings.h5ad')


##########################################################################
'''
load genes multiple to multiple homologous relations:
1. get relation matrix from CAME init data;
2. assign accurate genes names;
3. save to h5ad file.
'''
#############################################################################

def load_homo_genes(cfg):
    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)
    if cfg.BrainAlign.NODE_TYPE_NUM == 2:
        mh_ = datapair['vv_adj'].toarray()[0:cfg.BrainAlign.binary_M, cfg.BrainAlign.binary_M:]
    else:
        mh_ = datapair['vv_adj'].toarray()[0:cfg.BrainAlign.binary_M, cfg.BrainAlign.binary_M:]

    mouse_gene_list = datapair['varnames_node'][0]
    human_gene_list = datapair['varnames_node'][1]

    mouse_human_homo_anndata = ad.AnnData(mh_)
    mouse_human_homo_anndata.obs_names = mouse_gene_list
    mouse_human_homo_anndata.var_names = human_gene_list

    mouse_human_homo_anndata.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/mouse_human_gene_homos.h5ad')




##########################################################################
'''
The following four functions are used to calcalute the correlations between homologous brain 
regions and non-homologous regions. 
For well-learned general embeddings, we expect to see homologous regions of two species, such
as Mouse and Human, correlate significantly stronger than those between non-homologous regions.
'''
#############################################################################
def homo_corr(cfg):
    # No label order version
    '''
    Step 1: Compute average embedding of every region in two species, use two dict to store;
    Step 2: Compute similarity matrix, use np array to store;
    Step 3: Heatmap.
    '''
    expression_human_path = cfg.CAME.path_rawdata2
    adata_human = sc.read_h5ad(expression_human_path)
    print(adata_human)

    expression_mouse_path = cfg.CAME.path_rawdata1
    adata_mouse = sc.read_h5ad(expression_mouse_path)
    print(adata_mouse)
    adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
    human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)
    print(human_mouse_homo_region)
    home_region_dict = OrderedDict()
    for x, y in zip(human_mouse_homo_region['Human'], human_mouse_homo_region['Mouse'].values):
        home_region_dict[x] = y
    k = 0
    human_correlation_dict = {'human_region_list': [], 'mean': [], 'std': []}
    mouse_correlation_dict = {'mouse_region_list': [], 'mean': [], 'std': []}
    human_mouse_correlation_dict = {'human_region_list': [], 'mouse_region_list': [], 'mean': [], 'std': []}
    distance_type = 'correlation'

    save_path_root = cfg.BrainAlign.embeddings_file_path + '/figs/human_88/Hiercluster_came_embedding_{}/'.format(distance_type)

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    with open(save_path_root + 'homo_region_dict.pkl', 'wb') as f:
        pickle.dump(home_region_dict, f)

    for human_region, mouse_region in home_region_dict.items():

        save_path = cfg.BrainAlign.embeddings_file_path + '/figs/human_88/Hiercluster_came_embedding_{}/human_{}_mouse_{}/'.format(distance_type,
                                                                                              human_region,
                                                                                              mouse_region)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print('human_region', human_region)
        adata_human_embedding_region = adata_human_embedding[
            adata_human_embedding.obs['region_name'].isin([human_region])]
        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        # plt.figure(figsize=(16, 16))
        print('type of adata_human_embedding_region.X', type(adata_human_embedding_region.X))
        sns.clustermap(adata_human_embedding_region.X, metric=distance_type).fig.suptitle(
            'Human - {}'.format(human_region))
        plt.savefig(save_path + 'human_{}.png'.format(human_region))
        # plt.show()

        print('mouse_region', mouse_region)
        adata_mouse_embedding_region = adata_mouse_embedding[adata_mouse_embedding.obs['region_name'] == mouse_region]
        # plt.figure(figsize=(16, 16))
        if adata_mouse_embedding_region.X.shape[0] <= 0:
            continue
        sns.clustermap(adata_mouse_embedding_region.X, metric=distance_type).fig.suptitle(
            'Mouse - {}'.format(mouse_region))
        plt.savefig(save_path + 'mouse_{}.png'.format(mouse_region))
        # plt.show()

        # ---------human corr---------------------
        human_df = pd.DataFrame(adata_human_embedding_region.X).T
        human_corr = human_df.corr()
        print('human corr shape:', human_corr.shape)
        mean, std = human_corr.mean().mean(), human_corr.stack().std()
        print('mean', mean, 'std', std)
        human_correlation_dict['human_region_list'].append(human_region)
        human_correlation_dict['mean'].append(mean)
        human_correlation_dict['std'].append(std)
        # ---------mouse corr---------------------
        mouse_df = pd.DataFrame(adata_mouse_embedding_region.X).T
        mouse_corr = mouse_df.corr()
        print('mouse corr shape:', mouse_corr.shape)
        mean, std = mouse_corr.mean().mean(), mouse_corr.stack().std()
        print('mean', mean, 'std', std)
        mouse_correlation_dict['mouse_region_list'].append(mouse_region)
        mouse_correlation_dict['mean'].append(mean)
        mouse_correlation_dict['std'].append(std)
        # ---------------------------------------------------------------------
        ## Cross clustering of human and mouse
        result = pd.concat([human_df, mouse_df], axis=1).corr()

        # print(pd.DataFrame(adata_human_embedding_region.X).shape)
        # print(mouse_embedding_df.columns)
        # print(human_embedding_df.columns)
        Var_Corr = result[mouse_df.columns].loc[human_df.columns]
        mean, std = Var_Corr.mean().mean(), Var_Corr.stack().std()
        print('mean', mean, 'std', std)
        human_mouse_correlation_dict['human_region_list'].append(human_region)
        human_mouse_correlation_dict['mouse_region_list'].append(mouse_region)
        human_mouse_correlation_dict['mean'].append(mean)
        human_mouse_correlation_dict['std'].append(std)

        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        # plt.figure(figsize=(16, 16))
        sns.clustermap(Var_Corr, metric=distance_type).fig.suptitle(
            'Human-{} and Mouse-{}'.format(human_region, mouse_region))
        # plt.title('Human-{} and Mouse-{}'.format(human_region, mouse_region), loc='right')
        plt.savefig(save_path + 'human_{}_mouse_{}.png'.format(human_region, mouse_region))

        k += 1
        print('{}-th region finished!'.format(k))
        # plt.show()

    with open(save_path_root + 'human_mouse_correlation_dict.pkl', 'wb') as f:
        pickle.dump(human_mouse_correlation_dict, f)
    with open(save_path_root + 'human_correlation_dict.pkl', 'wb') as f:
        pickle.dump(human_correlation_dict, f)
    with open(save_path_root + 'mouse_correlation_dict.pkl', 'wb') as f:
        pickle.dump(mouse_correlation_dict, f)


def homo_corr_umap(cfg):
    # No label order version
    '''
    Step 1: Compute average embedding of every region in two species, use two dict to store;
    Step 2: Compute similarity matrix, use np array to store;
    Step 3: Heatmap.
    '''
    expression_human_path = cfg.CAME.path_rawdata2
    adata_human = sc.read_h5ad(expression_human_path)
    print(adata_human)

    expression_mouse_path = cfg.CAME.path_rawdata1
    adata_mouse = sc.read_h5ad(expression_mouse_path)
    print(adata_mouse)

    #path_human = cfg.CAME.ROOT + '/adt_hidden_cell.h5ad'

    adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    #
    adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')

    adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
    print(adata_embedding)

    sc.pp.neighbors(adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
    #sc.tl.umap(adata_embedding, n_components=cfg.ANALYSIS.umap_homo_random)
    sc.tl.pca(adata_embedding, svd_solver='arpack', n_comps=cfg.ANALYSIS.umap_homo_random)

    human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)

    print(human_mouse_homo_region)
    home_region_dict = OrderedDict()
    for x, y in zip(human_mouse_homo_region['Human'], human_mouse_homo_region['Mouse'].values):
        home_region_dict[x] = y

    k = 0

    human_correlation_dict = {'human_region_list': [], 'mean': [], 'std': []}
    mouse_correlation_dict = {'mouse_region_list': [], 'mean': [], 'std': []}
    human_mouse_correlation_dict = {'human_region_list': [], 'mouse_region_list': [], 'mean': [], 'std': []}

    distance_type = 'correlation'

    save_path_root = cfg.BrainAlign.embeddings_file_path + '/figs/human_88/umap_Hiercluster_came_embedding_{}/'.format(distance_type)

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    with open(save_path_root + 'homo_region_dict.pkl', 'wb') as f:
        pickle.dump(home_region_dict, f)

    for human_region, mouse_region in home_region_dict.items():

        save_path = cfg.BrainAlign.embeddings_file_path + '/figs/human_88/umap_Hiercluster_came_embedding_{}/human_{}_mouse_{}/'.format(distance_type,
                                                                                              human_region,
                                                                                              mouse_region)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print('human_region', human_region)
        adata_human_embedding_region = adata_embedding[
            adata_embedding.obs['region_name'].isin([human_region])]

        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        # plt.figure(figsize=(16, 16))
        '''
        print('type of adata_human_embedding_region.X', type(adata_human_embedding_region.X))
        sns.clustermap(adata_human_embedding_region.X, metric=distance_type).fig.suptitle(
            'Human - {}'.format(human_region))
        plt.savefig(save_path + 'human_{}.png'.format(human_region))
        '''
        # plt.show()

        print('mouse_region', mouse_region)
        adata_mouse_embedding_region = adata_embedding[adata_embedding.obs['region_name'] == mouse_region]
        # plt.figure(figsize=(16, 16))
        if adata_mouse_embedding_region.X.shape[0] <= 0:
            continue
        '''
        sns.clustermap(adata_mouse_embedding_region.X, metric=distance_type).fig.suptitle(
            'Mouse - {}'.format(mouse_region))
        plt.savefig(save_path + 'mouse_{}.png'.format(mouse_region))
        # plt.show()
        '''
        # ---------human corr---------------------
        human_df = pd.DataFrame(adata_human_embedding_region.obsm['X_pca']).T
        human_corr = human_df.corr()
        print('human corr shape:', human_corr.shape)
        mean, std = human_corr.mean().mean(), human_corr.stack().std()
        print('mean', mean, 'std', std)
        human_correlation_dict['human_region_list'].append(human_region)
        human_correlation_dict['mean'].append(mean)
        human_correlation_dict['std'].append(std)
        # ---------mouse corr---------------------
        mouse_df = pd.DataFrame(adata_mouse_embedding_region.obsm['X_pca']).T
        mouse_corr = mouse_df.corr()
        print('mouse corr shape:', mouse_corr.shape)
        mean, std = mouse_corr.mean().mean(), mouse_corr.stack().std()
        print('mean', mean, 'std', std)
        mouse_correlation_dict['mouse_region_list'].append(mouse_region)
        mouse_correlation_dict['mean'].append(mean)
        mouse_correlation_dict['std'].append(std)
        # ---------------------------------------------------------------------
        ## Cross clustering of human and mouse
        result = pd.concat([human_df, mouse_df], axis=1).corr()

        # print(pd.DataFrame(adata_human_embedding_region.X).shape)
        # print(mouse_embedding_df.columns)
        # print(human_embedding_df.columns)
        Var_Corr = result[mouse_df.columns].loc[human_df.columns]
        mean, std = Var_Corr.mean().mean(), Var_Corr.stack().std()
        print('mean', mean, 'std', std)
        human_mouse_correlation_dict['human_region_list'].append(human_region)
        human_mouse_correlation_dict['mouse_region_list'].append(mouse_region)
        human_mouse_correlation_dict['mean'].append(mean)
        human_mouse_correlation_dict['std'].append(std)

        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        # plt.figure(figsize=(16, 16))
        sns.clustermap(Var_Corr, metric=distance_type).fig.suptitle(
            'Human-{} and Mouse-{}'.format(human_region, mouse_region))
        # plt.title('Human-{} and Mouse-{}'.format(human_region, mouse_region), loc='right')
        plt.savefig(save_path + 'human_{}_mouse_{}.png'.format(human_region, mouse_region))

        k += 1
        print('{}-th region finished!'.format(k))
        # plt.show()

    with open(save_path_root + 'human_mouse_correlation_dict.pkl', 'wb') as f:
        pickle.dump(human_mouse_correlation_dict, f)
    with open(save_path_root + 'human_correlation_dict.pkl', 'wb') as f:
        pickle.dump(human_correlation_dict, f)
    with open(save_path_root + 'mouse_correlation_dict.pkl', 'wb') as f:
        pickle.dump(mouse_correlation_dict, f)


def random_corr(cfg):
    # No label order version
    '''
    Step 1: Compute average embedding of every region in two species, use two dict to store;
    Step 2: Compute similarity matrix, use np array to store;
    Step 3: Heatmap.
    '''

    # Read ordered labels
    human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
    mouse_67_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)

    adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
    #
    human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)

    print(human_mouse_homo_region)
    home_region_dict = OrderedDict()
    mouse_region_list = human_mouse_homo_region['Mouse'].values
    # random.shuffle(mouse_region_list)
    for x, y in zip(human_mouse_homo_region['Human'].values, mouse_region_list):
        home_region_dict[x] = y

    # home_region_dict = OrderedDict()
    human_88_labels_list = list(human_88_labels['region_name'])
    mouse_67_labels_list = list(mouse_67_labels['region_name'])


    k = 0

    human_correlation_dict = {'human_region_list': [], 'mean': [], 'std': []}
    mouse_correlation_dict = {'mouse_region_list': [], 'mean': [], 'std': []}
    human_mouse_correlation_dict = {'human_region_list': [], 'mouse_region_list': [], 'mean': [], 'std': []}

    distance_type = 'correlation'

    if cfg.HOMO_RANDOM.random_field == 'all':
        save_path_root = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/random_Hiercluster_came_embedding_{}/'.format(distance_type)
    else:
        save_path_root = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/35_random_Hiercluster_came_embedding_{}/'.format(distance_type)

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)
    with open(save_path_root + 'homo_region_dict.pkl', 'wb') as f:
        pickle.dump(home_region_dict, f)

    for human_region in human_88_labels_list:
        for mouse_region in mouse_67_labels_list:
            if human_region in home_region_dict.keys() and home_region_dict[human_region] == mouse_region:
                continue
            else:
                # home_region_dict[human_region] = mouse_region
                print('human_region', human_region)
                adata_human_embedding_region = adata_human_embedding[
                    adata_human_embedding.obs['region_name'] == human_region]
                # print(adata_human_embedding_region)
                if min(adata_human_embedding_region.X.shape) <= 1:
                    continue

                # color_map = sns.color_palette("coolwarm", as_cmap=True)
                # plt.figure(figsize=(16, 16))
                if cfg.HOMO_RANDOM.random_plot == True:
                    save_path = save_path_root + '/human_{}_mouse_{}/'.format(distance_type, human_region, mouse_region)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    sns.clustermap(adata_human_embedding_region.X, metric=distance_type).fig.suptitle('Human - {}'.format(human_region))
                    plt.savefig(save_path + 'human_{}.png'.format(human_region))

                print('mouse_region', mouse_region)
                adata_mouse_embedding_region = adata_mouse_embedding[
                    adata_mouse_embedding.obs['region_name'] == mouse_region]
                if min(adata_mouse_embedding_region.X.shape) <= 1:
                    continue
                # if max(adata_mouse_embedding_region.X.shape) >= 4500:
                #    continue
                # print(adata_mouse_embedding_region)
                # plt.figure(figsize=(16, 16))
                if cfg.HOMO_RANDOM.random_plot == True:
                    sns.clustermap(adata_mouse_embedding_region.X, metric=distance_type).fig.suptitle('Mouse - {}'.format(mouse_region))
                    plt.savefig(save_path + 'mouse_{}.png'.format(mouse_region))

                # ---------human corr---------------------
                human_df = pd.DataFrame(adata_human_embedding_region.X).T
                human_corr = human_df.corr()
                print('human corr shape:', human_corr.shape)
                mean, std = human_corr.mean().mean(), human_corr.stack().std()
                print('mean', mean, 'std', std)
                human_correlation_dict['human_region_list'].append(human_region)
                human_correlation_dict['mean'].append(mean)
                human_correlation_dict['std'].append(std)
                # ---------mouse corr---------------------
                mouse_df = pd.DataFrame(adata_mouse_embedding_region.X).T
                mouse_corr = mouse_df.corr()
                print('mouse corr shape:', mouse_corr.shape)
                mean, std = mouse_corr.mean().mean(), mouse_corr.stack().std()
                print('mean', mean, 'std', std)
                mouse_correlation_dict['mouse_region_list'].append(mouse_region)
                mouse_correlation_dict['mean'].append(mean)
                mouse_correlation_dict['std'].append(std)
                # ---------------------------------------------------------------------
                ## Cross clustering of human and mouse
                result = pd.concat([human_df, mouse_df], axis=1).corr()

                # print(pd.DataFrame(adata_human_embedding_region.X).shape)
                # print(mouse_embedding_df.columns)
                # print(human_embedding_df.columns)
                Var_Corr = result[mouse_df.columns].loc[human_df.columns]
                mean, std = Var_Corr.mean().mean(), Var_Corr.stack().std()
                print('mean', mean, 'std', std)
                human_mouse_correlation_dict['human_region_list'].append(human_region)
                human_mouse_correlation_dict['mouse_region_list'].append(mouse_region)
                human_mouse_correlation_dict['mean'].append(mean)
                human_mouse_correlation_dict['std'].append(std)

                # color_map = sns.color_palette("coolwarm", as_cmap=True)
                # plt.figure(figsize=(16, 16))
                if cfg.HOMO_RANDOM.random_plot == True:
                    sns.clustermap(Var_Corr, metric=distance_type).fig.suptitle('Human-{} and Mouse-{}'.format(human_region, mouse_region))
                    #plt.title('Human-{} and Mouse-{}'.format(human_region, mouse_region), loc='right')
                    plt.savefig(save_path + 'human_{}_mouse_{}.png'.format(human_region, mouse_region))

                k += 1
                print('{}-th region finished!'.format(k))
                # plt.show()

    with open(save_path_root + 'human_mouse_correlation_dict.pkl', 'wb') as f:
        pickle.dump(human_mouse_correlation_dict, f)
    with open(save_path_root + 'human_correlation_dict.pkl', 'wb') as f:
        pickle.dump(human_correlation_dict, f)
    with open(save_path_root + 'mouse_correlation_dict.pkl', 'wb') as f:
        pickle.dump(mouse_correlation_dict, f)


def random_corr_umap(cfg):
    # No label order version
    '''
    Step 1: Compute average embedding of every region in two species, use two dict to store;
    Step 2: Compute similarity matrix, use np array to store;
    Step 3: Heatmap.
    '''

    # Read ordered labels
    human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
    mouse_67_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)

    adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')

    # umap of all the embeddings
    adata_mouse_embedding.obs['dataset'] = 'mouse'
    adata_human_embedding.obs['dataset'] = 'human'

    adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
    print(adata_embedding)

    sc.pp.neighbors(adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
    #sc.tl.umap(adata_embedding, n_components=cfg.ANALYSIS.umap_homo_random)
    sc.tl.pca(adata_embedding, svd_solver='arpack', n_comps=cfg.ANALYSIS.umap_homo_random)
    #
    human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)

    print(human_mouse_homo_region)
    home_region_dict = OrderedDict()
    mouse_region_list = human_mouse_homo_region['Mouse'].values
    # random.shuffle(mouse_region_list)
    for x, y in zip(human_mouse_homo_region['Human'].values, mouse_region_list):
        home_region_dict[x] = y

    # home_region_dict = OrderedDict()
    human_88_labels_list = list(human_88_labels['region_name'])
    mouse_67_labels_list = list(mouse_67_labels['region_name'])

    k = 0

    human_correlation_dict = {'human_region_list': [], 'mean': [], 'std': []}
    mouse_correlation_dict = {'mouse_region_list': [], 'mean': [], 'std': []}
    human_mouse_correlation_dict = {'human_region_list': [], 'mouse_region_list': [], 'mean': [], 'std': []}

    distance_type = 'correlation'

    if cfg.HOMO_RANDOM.random_field == 'all':
        save_path_root = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/umap_random_Hiercluster_came_embedding_{}/'.format(distance_type)
    else:
        save_path_root = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/umap_35_random_Hiercluster_came_embedding_{}/'.format(distance_type)

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)
    with open(save_path_root + 'homo_region_dict.pkl', 'wb') as f:
        pickle.dump(home_region_dict, f)

    for human_region in human_88_labels_list:
        for mouse_region in mouse_67_labels_list:
            if human_region in home_region_dict.keys() and home_region_dict[human_region] == mouse_region:
                continue
            else:
                # home_region_dict[human_region] = mouse_region
                print('human_region', human_region)
                adata_human_embedding_region = adata_embedding[
                    adata_embedding.obs['region_name'] == human_region]
                # print(adata_human_embedding_region)
                if min(adata_human_embedding_region.X.shape) <= 1:
                    continue

                # color_map = sns.color_palette("coolwarm", as_cmap=True)
                # plt.figure(figsize=(16, 16))
                save_path = save_path_root + '/human_{}_mouse_{}/'.format(distance_type, human_region, mouse_region)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                '''
                if cfg.HOMO_RANDOM.random_plot == True:
                    save_path = save_path_root + '/human_{}_mouse_{}/'.format(distance_type, human_region, mouse_region)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    sns.clustermap(adata_human_embedding_region.X, metric=distance_type).fig.suptitle('Human - {}'.format(human_region))
                    plt.savefig(save_path + 'human_{}.png'.format(human_region))
                '''
                print('mouse_region', mouse_region)
                adata_mouse_embedding_region = adata_embedding[adata_embedding.obs['region_name'] == mouse_region]
                if min(adata_mouse_embedding_region.X.shape) <= 1:
                    continue
                # if max(adata_mouse_embedding_region.X.shape) >= 4500:
                #    continue
                # print(adata_mouse_embedding_region)
                # plt.figure(figsize=(16, 16))
                #if cfg.HOMO_RANDOM.random_plot == True:
                #    sns.clustermap(adata_mouse_embedding_region.X, metric=distance_type).fig.suptitle('Mouse - {}'.format(mouse_region))
                #    plt.savefig(save_path + 'mouse_{}.png'.format(mouse_region))

                # ---------human corr---------------------
                human_df = pd.DataFrame(adata_human_embedding_region.obsm['X_pca']).T #X_umap
                human_corr = human_df.corr()
                print('human corr shape:', human_corr.shape)
                mean, std = human_corr.mean().mean(), human_corr.stack().std()
                print('mean', mean, 'std', std)
                human_correlation_dict['human_region_list'].append(human_region)
                human_correlation_dict['mean'].append(mean)
                human_correlation_dict['std'].append(std)
                # ---------mouse corr---------------------
                mouse_df = pd.DataFrame(adata_mouse_embedding_region.obsm['X_pca']).T
                mouse_corr = mouse_df.corr()
                print('mouse corr shape:', mouse_corr.shape)
                mean, std = mouse_corr.mean().mean(), mouse_corr.stack().std()
                print('mean', mean, 'std', std)
                mouse_correlation_dict['mouse_region_list'].append(mouse_region)
                mouse_correlation_dict['mean'].append(mean)
                mouse_correlation_dict['std'].append(std)
                # ---------------------------------------------------------------------
                ## Cross clustering of human and mouse
                result = pd.concat([human_df, mouse_df], axis=1).corr()

                # print(pd.DataFrame(adata_human_embedding_region.X).shape)
                # print(mouse_embedding_df.columns)
                # print(human_embedding_df.columns)
                Var_Corr = result[mouse_df.columns].loc[human_df.columns]
                mean, std = Var_Corr.mean().mean(), Var_Corr.stack().std()
                print('mean', mean, 'std', std)
                human_mouse_correlation_dict['human_region_list'].append(human_region)
                human_mouse_correlation_dict['mouse_region_list'].append(mouse_region)
                human_mouse_correlation_dict['mean'].append(mean)
                human_mouse_correlation_dict['std'].append(std)

                # color_map = sns.color_palette("coolwarm", as_cmap=True)
                # plt.figure(figsize=(16, 16))
                if cfg.HOMO_RANDOM.random_plot == True:
                    sns.clustermap(Var_Corr, metric=distance_type).fig.suptitle('Human-{} and Mouse-{}'.format(human_region, mouse_region))
                    #plt.title('Human-{} and Mouse-{}'.format(human_region, mouse_region), loc='right')
                    plt.savefig(save_path + 'human_{}_mouse_{}.png'.format(human_region, mouse_region))

                k += 1
                print('{}-th region finished!'.format(k))
                # plt.show()

    with open(save_path_root + 'human_mouse_correlation_dict.pkl', 'wb') as f:
        pickle.dump(human_mouse_correlation_dict, f)
    with open(save_path_root + 'human_correlation_dict.pkl', 'wb') as f:
        pickle.dump(human_correlation_dict, f)
    with open(save_path_root + 'mouse_correlation_dict.pkl', 'wb') as f:
        pickle.dump(mouse_correlation_dict, f)


'''
from matplotlib import rcParams

TINY_SIZE = 16
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=TINY_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams["legend.frameon"] = False
'''

'''
def plot_homo_random_umap(cfg):
    # -----------------------------------------------------------------
    # Use UMAP coordinates to measure homologous region relations
    #Step 1:load human and mouse umap coordinates of embeddings of homologous regions, and random regions
    #Step 2: plot bar, mean and std

    homo_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/umap_Hiercluster_came_embedding_correlation/'
    with open(homo_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
        human_mouse_correlation_dict = pickle.load(f)

    home_len = len(human_mouse_correlation_dict['mean'])
    home_random_type = ['homologous'] * home_len
    human_mouse_correlation_dict['type'] = home_random_type
    # data_df = pd.DataFrame.from_dict(human_mouse_correlation_dict)

    if cfg.HOMO_RANDOM.random_field == 'all':
        random_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/umap_random_Hiercluster_came_embedding_correlation/'
    else:
        random_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/umap_35_random_Hiercluster_came_embedding_correlation/'
    with open(random_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
        random_human_mouse_correlation_dict = pickle.load(f)

    random_len = len(random_human_mouse_correlation_dict['mean'])
    home_random_type = ['random'] * random_len
    random_human_mouse_correlation_dict['type'] = home_random_type
    concat_dict = {}
    for k, v in random_human_mouse_correlation_dict.items():
        concat_dict[k] = human_mouse_correlation_dict[k] + random_human_mouse_correlation_dict[k]
    data_df = pd.DataFrame.from_dict(concat_dict)

    if cfg.HOMO_RANDOM.random_field == 'all':
        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/umap_homo_random/'
    else:
        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/umap_35_homo_random/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # my_pal = {"homo region": (0/255,149/255,182/255), "random":(178/255, 0/255, 32/255)}
    my_pal = {"homologous": (0 / 255, 149 / 255, 182 / 255), "random": (178 / 255, 0 / 255, 32 / 255)}
    # sns.set_theme(style="whitegrid")
    # tips = sns.load_dataset("tips")

    plt.figure(figsize=(8, 8))
    ax = sns.boxplot(x="type", y="mean", data=data_df, order=["homologous", "random"], palette=my_pal)
    plt.savefig(save_path + 'mean.svg')
    plt.show()

    plt.figure(figsize=(8, 8))
    ax = sns.boxplot(x="type", y="std", data=data_df, order=["homologous", "random"], palette=my_pal)
    plt.savefig(save_path + 'std.svg')
    plt.show()

    homo_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/umap_Hiercluster_came_embedding_correlation/'
    with open(homo_region_data_path + 'human_correlation_dict.pkl', 'rb') as f:
        human_correlation_dict = pickle.load(f)

    with open(homo_region_data_path + 'mouse_correlation_dict.pkl', 'rb') as f:
        mouse_correlation_dict = pickle.load(f)

    human_mouse_dict_mean = {'Human': [], 'Mouse': []}
    human_mouse_dict_std = {'Human': [], 'Mouse': []}

    human_mouse_dict_mean['Human'] = human_correlation_dict['mean']
    human_mouse_dict_mean['Mouse'] = mouse_correlation_dict['mean']

    human_mouse_dict_std['Human'] = human_correlation_dict['std']
    human_mouse_dict_std['Mouse'] = mouse_correlation_dict['std']

    sns.jointplot(data=pd.DataFrame.from_dict(human_mouse_dict_mean), x="Human", y="Mouse", kind="reg")
    plt.savefig(save_path + 'mean_human_mouse.svg')
    plt.show()

    sns.jointplot(data=pd.DataFrame.from_dict(human_mouse_dict_std), x="Human", y="Mouse", kind="reg")
    plt.savefig(save_path + 'std_human_mouse.svg')
    plt.show()
'''

def plot_homo_random(cfg):

    # No label order version
    '''
    Step 1:load human and mouse cross expression data of homologous regions, and random regions
    Step 2: plot bar, mean and std
    '''

    homo_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/Hiercluster_came_embedding_correlation/'
    with open(homo_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
        human_mouse_correlation_dict = pickle.load(f)


    home_len = len(human_mouse_correlation_dict['mean'])
    home_random_type = ['homologous'] * home_len
    human_mouse_correlation_dict['type'] = home_random_type
    #data_df = pd.DataFrame.from_dict(human_mouse_correlation_dict)

    if cfg.HOMO_RANDOM.random_field == 'all':
        random_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/random_Hiercluster_came_embedding_correlation/'
    else:
        random_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/35_random_Hiercluster_came_embedding_correlation/'
    with open(random_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
        random_human_mouse_correlation_dict = pickle.load(f)

    random_len = len(random_human_mouse_correlation_dict['mean'])
    home_random_type = ['random'] * random_len
    random_human_mouse_correlation_dict['type'] = home_random_type
    concat_dict = {}
    for k,v in random_human_mouse_correlation_dict.items():
        concat_dict[k] = human_mouse_correlation_dict[k] + random_human_mouse_correlation_dict[k]
    data_df = pd.DataFrame.from_dict(concat_dict)


    if cfg.HOMO_RANDOM.random_field == 'all':
        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/homo_random/'
    else:
        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/35_homo_random/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    #my_pal = {"homo region": (0/255,149/255,182/255), "random":(178/255, 0/255, 32/255)}
    my_pal = {"homologous": (0 / 255, 149 / 255, 182 / 255), "random": (178 / 255, 0 / 255, 32 / 255)}
    #sns.set_theme(style="whitegrid")
    #tips = sns.load_dataset("tips")

    plt.figure(figsize=(8,8))
    ax = sns.boxplot(x="type", y="mean", data=data_df, order=["homologous", "random"], palette = my_pal)
    plt.savefig(save_path + 'mean.svg')
    plt.show()

    plt.figure(figsize=(8, 8))
    ax = sns.boxplot(x="type", y="std", data=data_df, order=["homologous", "random"], palette=my_pal)
    plt.savefig(save_path + 'std.svg')
    plt.show()


    homo_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/Hiercluster_came_embedding_correlation/'
    with open(homo_region_data_path + 'human_correlation_dict.pkl', 'rb') as f:
        human_correlation_dict = pickle.load(f)

    with open(homo_region_data_path + 'mouse_correlation_dict.pkl', 'rb') as f:
        mouse_correlation_dict = pickle.load(f)

    human_mouse_dict_mean = {'Human':[], 'Mouse':[]}
    human_mouse_dict_std = {'Human':[], 'Mouse':[]}

    human_mouse_dict_mean['Human'] = human_correlation_dict['mean']
    human_mouse_dict_mean['Mouse'] = mouse_correlation_dict['mean']

    human_mouse_dict_std['Human'] = human_correlation_dict['std']
    human_mouse_dict_std['Mouse'] = mouse_correlation_dict['std']

    sns.jointplot(data=pd.DataFrame.from_dict(human_mouse_dict_mean), x="Human", y="Mouse", kind="reg")
    plt.savefig(save_path + 'mean_human_mouse.svg')
    plt.show()

    sns.jointplot(data=pd.DataFrame.from_dict(human_mouse_dict_std), x="Human", y="Mouse", kind="reg")
    plt.savefig(save_path + 'std_human_mouse.svg')
    plt.show()


def ttest_homo_random(cfg):

    # No label order version
    '''
    Step 1:load human and mouse cross expression data of homologous regions, and random regions
    Step 2: plot bar, mean and std
    '''
    # Read ordered labels
    human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
    mouse_67_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)

    homo_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/Hiercluster_came_embedding_correlation/'
    with open(homo_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
        human_mouse_correlation_dict = pickle.load(f)

    home_len = len(human_mouse_correlation_dict['mean'])
    home_random_type = ['homologous'] * home_len
    human_mouse_correlation_dict['type'] = home_random_type
    # data_df = pd.DataFrame.from_dict(human_mouse_correlation_dict)

    if cfg.HOMO_RANDOM.random_field == 'all':
        random_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/random_Hiercluster_came_embedding_correlation/'
    else:
        random_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/35_random_Hiercluster_came_embedding_correlation/'
    with open(random_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
        random_human_mouse_correlation_dict = pickle.load(f)

    random_len = len(random_human_mouse_correlation_dict['mean'])
    home_random_type = ['random'] * random_len
    random_human_mouse_correlation_dict['type'] = home_random_type
    concat_dict = {}
    for k, v in random_human_mouse_correlation_dict.items():
        concat_dict[k] = human_mouse_correlation_dict[k] + random_human_mouse_correlation_dict[k]
    data_df = pd.DataFrame.from_dict(concat_dict)

    if cfg.HOMO_RANDOM.random_field == 'all':
        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/homo_random/'
    else:
        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/35_homo_random/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # my_pal = {"homo region": (0/255,149/255,182/255), "random":(178/255, 0/255, 32/255)}
    my_pal = {"homologous": (0 / 255, 149 / 255, 182 / 255), "random": (178 / 255, 0 / 255, 32 / 255)}
    # sns.set_theme(style="whitegrid")
    # tips = sns.load_dataset("tips")

    print(data_df)

    random_df = data_df[data_df['type'] == 'random']
    print(random_df)
    mean_random_list = random_df['mean'].values
    mean_r = np.mean(mean_random_list)
    std_r = np.std(mean_random_list)

    homologous_df = data_df[data_df['type'] == 'homologous']
    print(homologous_df)
    mean_homo_list = homologous_df['mean'].values
    mean_h = np.mean(mean_homo_list)
    std_h = np.std(mean_homo_list)

    from scipy import stats

    print(stats.ttest_ind(
        mean_homo_list,
        mean_random_list,
        equal_var=False
    ))

    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(save_path+'t_test_result.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(stats.ttest_ind(
            mean_homo_list,
            mean_random_list,
            equal_var=False
        ))
        sys.stdout = original_stdout  # Reset the standard output to its original value



##################################################################################
'''
The following function does a binary hierarchical clustering of embeddings of two species, to evaluate 
whether the data scale difference between two species is well eliminated. 
'''
##################################################################################

def cross_species_binary_clustering(cfg):

    fig_format = cfg.BrainAlign.fig_format
    import sys
    sys.setrecursionlimit(100000)
    # No label order version
    '''
    Step 1: Compute average embedding of every region in two species, use two dict to store;
    Step 2: Compute similarity matrix, use np array to store;
    Step 3: Heatmap.
    '''
    adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')

    custom_palette = sns.color_palette("Paired", 2)
    print(custom_palette)
    # lut = dict(zip(['Human', 'Mouse'], custom_palette))
    lut = {"Human": (26 / 255, 85 / 255, 153 / 255), "Mouse": (128 / 255, 0 / 255, 32 / 255)}
    print(lut)
    row_colors = []
    for i in range(adata_human_embedding.X.shape[0]):
        row_colors.append(lut['Human'])
    ## ---------------------------------------------------------------------mouse------------------------
    for i in range(adata_human_embedding.X.shape[0]):
        row_colors.append(lut['Mouse'])

    rng = np.random.default_rng(12345)
    rints = rng.integers(low=0, high=adata_mouse_embedding.X.shape[0], size=adata_human_embedding.X.shape[0])
    # rints
    print('adata_human_embedding.X.shape', adata_human_embedding.X.shape)
    print('adata_mouse_embedding.X.shape', adata_mouse_embedding.X.shape)

    sns.clustermap(np.concatenate((adata_human_embedding.X, adata_mouse_embedding.X[rints, :]), axis=0),
                   row_colors=row_colors)

    # color_map = sns.color_palette("coolwarm", as_cmap=True)
    # sns.clustermap(Var_Corr, cmap=color_map, center=0.6)
    save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + 'Hiercluster_heco_embedding_concate.'+fig_format, format=fig_format)
    plt.show()

    adata_mouse_embedding.obs['dataset'] = 'mouse'
    adata_human_embedding.obs['dataset'] = 'human'

    adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
    print(adata_embedding)

    # Umap of the whole dataset
    #sc.pp.neighbors(adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
    #sc.tl.umap(adata_embedding)
    sc.pp.neighbors(adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
    sc.tl.umap(adata_embedding)



    save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    with plt.rc_context({"figure.figsize": (8, 8)}):
        sc.pl.umap(adata_embedding, color=['dataset'], return_fig=True, legend_loc='on data').savefig(
            save_path + 'umap_dataset.' + fig_format, format=fig_format)
        #plt.subplots_adjust(right=0.3)

    #sc.set_figure_params(dpi_save=200)

    rcParams["figure.subplot.left"] = 0.05
    rcParams["figure.subplot.right"] = 0.45

    #if cfg.BrainAlign.homo_region_num >= 10:
    human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)
    homo_region_list = list(human_mouse_homo_region['Mouse'].values) + list(human_mouse_homo_region['Human'].values)
    print(homo_region_list)
    adata_embedding_homo = adata_embedding[adata_embedding.obs['region_name'].isin(homo_region_list)]
    with plt.rc_context({"figure.figsize": (12, 6)}):
        sc.pl.umap(adata_embedding_homo, color=['region_name'], return_fig=True, legend_loc='right margin').savefig(
            save_path + 'umap_types.' + fig_format, format=fig_format)


    rcParams["figure.subplot.left"] = 0.125
    rcParams["figure.subplot.right"] = 0.9
    with plt.rc_context({"figure.figsize": (8, 8)}):
        sc.pl.umap(adata_embedding_homo, color=['region_name'], return_fig=True, legend_loc='on data', legend_fontsize='x-small').savefig(
            save_path + 'umap_types_ondata.' + fig_format, format=fig_format)

    with plt.rc_context({"figure.figsize": (8, 8)}):
        sc.pl.umap(adata_embedding_homo, color=['dataset'], return_fig=True, legend_loc='on data').savefig(
            save_path + 'umap_dataset_homo_regions_ondata.' + fig_format, format=fig_format)

    rcParams["figure.subplot.right"] = 0.7
    with plt.rc_context({"figure.figsize": (8, 8)}):
        sc.pl.umap(adata_embedding_homo, color=['dataset'], return_fig=True, legend_loc='right margin').savefig(
            save_path + 'umap_dataset_homo_regions.' + fig_format, format=fig_format)
    #plt.subplots_adjust(right=0.3)

    #rcParams["figure.subplot.left"] = 0.125
    #rcParams["figure.subplot.right"] = 0.9
    rcParams["figure.subplot.left"] = 0.05
    rcParams["figure.subplot.right"] = 0.45

    # PCA and kmeans clustering of the whole dataset
    sc.tl.pca(adata_embedding, svd_solver='arpack', n_comps=30)
    sc.pl.pca(adata_embedding, color=['region_name'], components=['1,2', '3,4', '5,6', '7,8'], ncols=2,
              return_fig=True).savefig \
        (save_path + 'pca.' + fig_format, format=fig_format)
    # extract pca coordinates
    X_pca = adata_embedding.obsm['X_pca']
    kmeans = KMeans(n_clusters=11, random_state=0).fit(X_pca)
    adata_embedding.obs['kmeans11'] = kmeans.labels_.astype(str)
    kmeans = KMeans(n_clusters=22, random_state=0).fit(X_pca)
    adata_embedding.obs['kmeans22'] = kmeans.labels_.astype(str)
    kmeans = KMeans(n_clusters=64, random_state=0).fit(X_pca)
    adata_embedding.obs['kmeans64'] = kmeans.labels_.astype(str)
    kmeans = KMeans(n_clusters=100, random_state=0).fit(X_pca)
    adata_embedding.obs['kmeans100'] = kmeans.labels_.astype(str)

    kmeans_list = ['kmeans11', 'kmeans22', 'kmeans64', 'kmeans100']
    for k_num in kmeans_list:
        with plt.rc_context({"figure.figsize": (12, 6)}):
            sc.pl.umap(adata_embedding, color=[k_num], return_fig=True).savefig(
                save_path + 'umap_pca_{}.'.format(k_num) + fig_format, format=fig_format)
    human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)
    homo_region_list = list(human_mouse_homo_region['Mouse'].values) + list(human_mouse_homo_region['Human'].values)
    print(homo_region_list)
    adata_embedding_homo = adata_embedding[adata_embedding.obs['region_name'].isin(homo_region_list)]
    for k_num in kmeans_list:
            with plt.rc_context({"figure.figsize": (12, 6)}):
                sc.pl.umap(adata_embedding_homo, color=[k_num], return_fig=True).savefig(
                    save_path + 'umap_pca_homoregions_{}.'.format(k_num) + fig_format, format=fig_format)
    rcParams["figure.subplot.right"] = 0.9


# Evaluation of mouse and human embedding alignment
# 1. clustering of mouse and human embeddngs;
# 2. compute the coverage between two labels, keep samples those are consistent;
# 3. compute the similarity of between the clusters of the two species and identify those highly correlated pairs.
# 4. Check the overlap of homolous clusters pair (brain regions) and clusters pairs.
def cross_evaluation_aligment(cfg):
    fig_format = cfg.BrainAlign.fig_format

    save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/cross_evaluation_alignment/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
    adata_mouse_embedding.obs['dataset'] = 'mouse'
    adata_human_embedding.obs['dataset'] = 'human'

    mouse_64_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)
    mouse_64_labels_list = list(mouse_64_labels['region_name'])
    human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
    human_88_labels_list = list(human_88_labels['region_name'])

    # plot human mouse homologous samples
    human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)
    homo_region_list_mouse = list(human_mouse_homo_region['Mouse'].values)
    homo_region_list_human = list(human_mouse_homo_region['Human'].values)

    homo_region_mouse_human_dict = {k:v for k,v in zip(homo_region_list_mouse, homo_region_list_human)}

    sc.pp.neighbors(adata_mouse_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
    sc.tl.umap(adata_mouse_embedding)

    adata_mouse_embedding_homo = adata_mouse_embedding[adata_mouse_embedding.obs['region_name'].isin(homo_region_list_mouse)]

    rcParams["figure.subplot.left"] = 0.05
    rcParams["figure.subplot.right"] = 0.45

    with plt.rc_context({"figure.figsize": (15, 6)}):
        sc.pl.umap(adata_mouse_embedding_homo, color=['region_name'], return_fig=True, legend_loc='right margin').savefig(
            save_path + 'umap_mouse_homologous_region.' + fig_format, format=fig_format)

    sc.pp.neighbors(adata_human_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
    sc.tl.umap(adata_human_embedding)

    adata_human_embedding_homo = adata_human_embedding[
        adata_human_embedding.obs['region_name'].isin(homo_region_list_human)]
    with plt.rc_context({"figure.figsize": (15, 6)}):
        sc.pl.umap(adata_human_embedding_homo, color=['region_name'], return_fig=True, legend_loc='right margin').savefig(
            save_path + 'umap_human_homologous_region.' + fig_format, format=fig_format)

    rcParams["figure.subplot.left"] = 0.05
    rcParams["figure.subplot.right"] = 0.9

    with plt.rc_context({"figure.figsize": (8, 8)}):
        sc.pl.umap(adata_mouse_embedding_homo, color=['region_name'], return_fig=True, legend_loc='on data').savefig(
            save_path + 'umap_mouse_homologous_region_ondata.' + fig_format, format=fig_format)

    with plt.rc_context({"figure.figsize": (8, 8)}):
        sc.pl.umap(adata_human_embedding_homo, color=['region_name'], return_fig=True, legend_loc='on data').savefig(
            save_path + 'umap_human_homologous_region_ondata.' + fig_format, format=fig_format)

    ## Method 1: Do PCA and clustering separately
    # Mouse
    sc.tl.pca(adata_mouse_embedding, svd_solver='arpack', n_comps=30)
    X_pca = adata_mouse_embedding.obsm['X_pca']
    kmeans = KMeans(n_clusters=64, random_state=29).fit(X_pca)
    adata_mouse_embedding.obs['kmeans64'] = kmeans.labels_.astype(str)
    region_labels = adata_mouse_embedding.obs['region_name'].values
    cluster_labels = adata_mouse_embedding.obs['kmeans64'].values
    #print(adata_mouse_embedding)
    cluster_labels_unique = [str(x) for x in list(range(0, 64))]

    overlap_dict = {}
    for re in mouse_64_labels_list:
        overlap_dict[re] = {}
        for cl in cluster_labels_unique:
            overlap_dict[re][cl] = 0

    for re,cl in zip(region_labels, cluster_labels):
            overlap_dict[re][cl] = overlap_dict[re][cl] + 1

    overlap_df = pd.DataFrame.from_dict(overlap_dict)

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(overlap_df, annot=False, ax=ax)
    plt.savefig(save_path + 'unnormalized_mouse_overlap_region_cluster.' + fig_format, format=fig_format)

    for re in mouse_64_labels_list:
        norm_sum = np.sum(list(overlap_dict[re].values()))
        print('norm_sum:', norm_sum)
        for cl in cluster_labels_unique:
            overlap_dict[re][cl] = overlap_dict[re][cl] / norm_sum

    overlap_df = pd.DataFrame.from_dict(overlap_dict)

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(overlap_df, annot=False, ax=ax)
    plt.savefig(save_path + 'normalized_mouse_overlap_region_cluster.' + fig_format, format=fig_format)

    mouse_idxmax = overlap_df.idxmax()
    print('Mouse:', mouse_idxmax)

    mouse_overlap_df = overlap_df

    # Human
    sc.tl.pca(adata_human_embedding, svd_solver='arpack', n_comps=30)
    X_pca = adata_human_embedding.obsm['X_pca']
    kmeans = KMeans(n_clusters=88, random_state=29).fit(X_pca)
    adata_human_embedding.obs['kmeans88'] = kmeans.labels_.astype(str)
    region_labels = adata_human_embedding.obs['region_name'].values
    cluster_labels = adata_human_embedding.obs['kmeans88'].values
    # print(adata_mouse_embedding)
    cluster_labels_unique = [str(x) for x in list(range(0, 88))]

    overlap_dict = {}
    for re in human_88_labels_list:
        overlap_dict[re] = {}
        for cl in cluster_labels_unique:
            overlap_dict[re][cl] = 0

    for re, cl in zip(region_labels, cluster_labels):
        overlap_dict[re][cl] = overlap_dict[re][cl] + 1

    overlap_df = pd.DataFrame.from_dict(overlap_dict)

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(overlap_df, annot=False, ax=ax)
    plt.savefig(save_path + 'unnormalized_human_overlap_region_cluster.' + fig_format, format=fig_format)

    for re in human_88_labels_list:
        norm_sum = np.sum(list(overlap_dict[re].values()))
        for cl in cluster_labels_unique:
            overlap_dict[re][cl] = overlap_dict[re][cl] / norm_sum
    overlap_df = pd.DataFrame.from_dict(overlap_dict)

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(overlap_df, annot=False, ax=ax)
    plt.savefig(save_path + 'normalized_human_overlap_region_cluster.' + fig_format, format=fig_format)


    #overlap_df['max_proportion'] = overlap_df.apply(lambda x: ', '.join(x[x.eq(x.max())].index), axis=1)
    #print(overlap_df['max_proportion'])
    #print(overlap_df)
    human_idxmax = overlap_df.idxmax()
    print('Human:', human_idxmax)

    human_overlap_df = overlap_df

    homologous_pair_num = 0
    equal_subtype_dict = {'mouse':[], 'human':[]}
    print('homologous pairs:')
    for re_mouse in mouse_overlap_df.columns:
        for re_human in human_overlap_df.columns:
            if mouse_idxmax[re_mouse] == human_idxmax[re_human]:
                equal_subtype_dict['mouse'].append(re_mouse)
                equal_subtype_dict['human'].append(re_human)
                print(re_mouse, re_human)


    ################################################################################
    ## Method 2: Do PCA and clustering together and align to brain regions
    # init input

    save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/cross_evaluation_alignment/wholecluster/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
    adata_mouse_embedding.obs['dataset'] = 'mouse'
    adata_human_embedding.obs['dataset'] = 'human'

    adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
    print(adata_embedding)

    sc.pp.neighbors(adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
    sc.tl.umap(adata_embedding)

    # PCA and kmeans clustering of the whole dataset
    sc.tl.pca(adata_embedding, svd_solver='arpack', n_comps=30)

    X_pca = adata_embedding.obsm['X_pca']
    kmeans = KMeans(n_clusters=100, random_state=29).fit(X_pca)
    adata_embedding.obs['kmeans100'] = kmeans.labels_.astype(str)

    region_labels = adata_embedding.obs['region_name'].values
    cluster_labels = adata_embedding.obs['kmeans100'].values
    # print(adata_mouse_embedding)
    cluster_labels_unique = [str(x) for x in list(range(0, 100))]
    # mouse
    overlap_dict = {}
    for re in mouse_64_labels_list:
        overlap_dict[re] = {}
        for cl in cluster_labels_unique:
            overlap_dict[re][cl] = 0

    for re, cl in zip(region_labels, cluster_labels):
        if re in mouse_64_labels_list:
            overlap_dict[re][cl] = overlap_dict[re][cl] + 1

    overlap_df = pd.DataFrame.from_dict(overlap_dict)

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(overlap_df, annot=False, ax=ax)
    plt.savefig(save_path + 'unnormalized_mouse_overlap_region_cluster.' + fig_format, format=fig_format)

    for re in mouse_64_labels_list:
        norm_sum = np.sum(list(overlap_dict[re].values()))
        print('norm_sum:', norm_sum)
        for cl in cluster_labels_unique:
            overlap_dict[re][cl] = overlap_dict[re][cl] / norm_sum

    overlap_df = pd.DataFrame.from_dict(overlap_dict)

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(overlap_df, annot=False, ax=ax)
    plt.savefig(save_path + 'normalized_mouse_overlap_region_cluster.' + fig_format, format=fig_format)

    mouse_idxmax = overlap_df.idxmax()
    print('Mouse:', mouse_idxmax)

    mouse_overlap_df = overlap_df

    # human
    overlap_dict = {}
    for re in human_88_labels_list:
        overlap_dict[re] = {}
        for cl in cluster_labels_unique:
            overlap_dict[re][cl] = 0

    for re, cl in zip(region_labels, cluster_labels):
        if re in human_88_labels_list:
            overlap_dict[re][cl] = overlap_dict[re][cl] + 1

    overlap_df = pd.DataFrame.from_dict(overlap_dict)

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(overlap_df, annot=False, ax=ax)
    plt.savefig(save_path + 'unnormalized_human_overlap_region_cluster.' + fig_format, format=fig_format)

    for re in human_88_labels_list:
        norm_sum = np.sum(list(overlap_dict[re].values()))
        for cl in cluster_labels_unique:
            overlap_dict[re][cl] = overlap_dict[re][cl] / norm_sum
    overlap_df = pd.DataFrame.from_dict(overlap_dict)

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(overlap_df, annot=False, ax=ax)
    plt.savefig(save_path + 'normalized_human_overlap_region_cluster.' + fig_format, format=fig_format)


    # overlap_df['max_proportion'] = overlap_df.apply(lambda x: ', '.join(x[x.eq(x.max())].index), axis=1)
    # print(overlap_df['max_proportion'])
    # print(overlap_df)
    human_idxmax = overlap_df.idxmax()
    print('Human:', human_idxmax)

    human_overlap_df = overlap_df

    homologous_pair_num = 0
    equal_subtype_dict = {'mouse': [], 'human': []}
    print('homologous pairs:')
    for re_mouse in mouse_overlap_df.columns:
        for re_human in human_overlap_df.columns:
            if mouse_idxmax[re_mouse] == human_idxmax[re_human]:
                equal_subtype_dict['mouse'].append(re_mouse)
                equal_subtype_dict['human'].append(re_human)
                print(re_mouse, re_human)


    #################################################################################
    # Method 3:
    print('#########################################################################')
    print('Method 3')

    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(save_path + 'subset_size_homologous_regions_rate.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        max_subset_num = 1
        mouse_idxmax_subset_dict = {}
        human_idxmax_subset_dict = {}
        for re_mouse in mouse_64_labels_list:
            subset_list = mouse_overlap_df[re_mouse].values
            mouse_idxmax_subset_dict[re_mouse] = np.argpartition(subset_list, -max_subset_num)[-max_subset_num:]

        for re_human in human_88_labels_list:
            subset_list = human_overlap_df[re_human].values
            human_idxmax_subset_dict[re_human] = np.argpartition(subset_list, -max_subset_num)[-max_subset_num:]

        homologous_pair_num = 0
        equal_subtype_dict = {'mouse': [], 'human': []}
        print('homologous pairs:')
        for re_mouse in mouse_overlap_df.columns:
            for re_human in human_overlap_df.columns:
                if len(set(mouse_idxmax_subset_dict[re_mouse]).intersection(
                        set(human_idxmax_subset_dict[re_human]))) != 0:
                    equal_subtype_dict['mouse'].append(re_mouse)
                    equal_subtype_dict['human'].append(re_human)
                    print(re_mouse, re_human)

        print(f'Subset size = {max_subset_num}')
        homo_region_mouse_human_dict_identified = {}
        for mouse_re, human_re in zip(equal_subtype_dict['mouse'], equal_subtype_dict['human']):
            if mouse_re in set(homo_region_mouse_human_dict.keys()) and homo_region_mouse_human_dict[
                mouse_re] == human_re:
                homo_region_mouse_human_dict_identified[mouse_re] = human_re

        print('homo_region_mouse_human_dict_identified', homo_region_mouse_human_dict_identified)
        print(f'Identified {len(homo_region_mouse_human_dict_identified)} homologous regions.')
        print('Rate of homologous regions identified = ',
              len(homo_region_mouse_human_dict_identified) / len(homo_region_mouse_human_dict))

        max_subset_num = 2
        mouse_idxmax_subset_dict = {}
        human_idxmax_subset_dict = {}
        for re_mouse in mouse_64_labels_list:
            subset_list = mouse_overlap_df[re_mouse].values
            mouse_idxmax_subset_dict[re_mouse] = np.argpartition(subset_list, -max_subset_num)[-max_subset_num:]

        for re_human in human_88_labels_list:
            subset_list = human_overlap_df[re_human].values
            human_idxmax_subset_dict[re_human] = np.argpartition(subset_list, -max_subset_num)[-max_subset_num:]

        homologous_pair_num = 0
        equal_subtype_dict = {'mouse': [], 'human': []}
        print('homologous pairs:')
        for re_mouse in mouse_overlap_df.columns:
            for re_human in human_overlap_df.columns:
                if len(set(mouse_idxmax_subset_dict[re_mouse]).intersection(
                        set(human_idxmax_subset_dict[re_human]))) != 0:
                    equal_subtype_dict['mouse'].append(re_mouse)
                    equal_subtype_dict['human'].append(re_human)
                    print(re_mouse, re_human)

        print(f'Subset size = {max_subset_num}')
        homo_region_mouse_human_dict_identified = {}
        for mouse_re, human_re in zip(equal_subtype_dict['mouse'], equal_subtype_dict['human']):
            if mouse_re in set(homo_region_mouse_human_dict.keys()) and homo_region_mouse_human_dict[
                mouse_re] == human_re:
                homo_region_mouse_human_dict_identified[mouse_re] = human_re

        print('homo_region_mouse_human_dict_identified', homo_region_mouse_human_dict_identified)
        print(f'Identified {len(homo_region_mouse_human_dict_identified)} homologous regions.')
        print('Rate of homologous regions identified = ',
              len(homo_region_mouse_human_dict_identified) / len(homo_region_mouse_human_dict))

        sys.stdout = original_stdout  # Reset the standard output to its original value



    plot_dict = {'subset size':[], 'Rate of homologous regions': [], 'Rate of homologous pairs': [], 'Normalized rate': []}
    plot_dict['subset size'] = list(range(1, 11))
    for max_subset_num in plot_dict['subset size']:
        rate, rate_pair = get_homologous_rate(max_subset_num,
                        mouse_64_labels_list,
                        human_88_labels_list,
                        mouse_overlap_df,
                        human_overlap_df,
                        homo_region_mouse_human_dict)
        plot_dict['Rate of homologous regions'].append(rate)
        plot_dict['Rate of homologous pairs'].append(rate_pair)
        plot_dict['Normalized rate'].append(rate/rate_pair)

    sns.set(style='white')
    TINY_SIZE = 32  # 39
    SMALL_SIZE = 34  # 42
    MEDIUM_SIZE = 42  # 46
    BIGGER_SIZE = 42  # 46

    plt.rc('font', size=30)  # 35 controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']

    fig, ax = plt.subplots(figsize=(12, 10))
    plot_rate_df = pd.DataFrame.from_dict(plot_dict)
    sns.lineplot(data=plot_rate_df, x="subset size", y="Rate of homologous regions", marker='o', markersize=14, linewidth=2.5, color='black')
    plt.subplots_adjust(bottom=0.3, left=0.2)
    plt.savefig(save_path + 'subset_size_homologous_regions_rate.' + fig_format, format=fig_format)

    fig, ax = plt.subplots(figsize=(12, 10))
    plot_rate_df = pd.DataFrame.from_dict(plot_dict)
    sns.lineplot(data=plot_rate_df, x="subset size", y="Rate of homologous pairs", marker='s', markersize=14,
                 linewidth=2.5, color='black')
    plt.subplots_adjust(bottom=0.3, left=0.2)
    plt.savefig(save_path + 'subset_size_homologous_pairs_rate.' + fig_format, format=fig_format)

    fig, ax = plt.subplots(figsize=(12, 10))
    plot_rate_df = pd.DataFrame.from_dict(plot_dict)
    sns.lineplot(data=plot_rate_df, x="subset size", y="Normalized rate", marker='*', markersize=14,
                 linewidth=2.5, color='black')
    plt.subplots_adjust(bottom=0.3, left=0.2)
    plt.savefig(save_path + 'subset_size_homologous_normalized_rate.' + fig_format, format=fig_format)
    return None


# Evaluation of mouse and human embedding alignment
# 1. Clustering of mouse and human embeddings, identify marker genes for each cluster,
#    and name each cluster by cluster order_mouse marker gene_human marker gene.
# 2. Compute the similarity or distance between the clusters of the two species
#    and identify those highly correlated pairs.
# 3. For each cluster, compute its brain region composition, check each pair and align it to the brain region pair.
# 4. Plot the 3D alignment figure, for each pair of highly correlated clusters, plot the mean of each cluster (black);
#    for each pair of homologous brain regions, plot the mean of each region (red).
# 4. Check the overlap of homologous clusters pair (brain regions) and clusters pairs (precision, recall, etc).
def cross_evaluation_aligment_cluster(cfg):

    fig_format = cfg.BrainAlign.fig_format

    save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/cross_evaluation_alignment_cluster/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
    adata_mouse_embedding.obs['dataset'] = 'mouse'
    adata_human_embedding.obs['dataset'] = 'human'

    mouse_64_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)
    mouse_64_labels_list = list(mouse_64_labels['region_name'])
    human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
    human_88_labels_list = list(human_88_labels['region_name'])

    # plot human mouse homologous samples
    human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)
    homo_region_list_mouse = list(human_mouse_homo_region['Mouse'].values)
    homo_region_list_human = list(human_mouse_homo_region['Human'].values)

    homo_region_mouse_human_dict = {k:v for k,v in zip(homo_region_list_mouse, homo_region_list_human)}

    adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
    print(adata_embedding)

    sc.pp.neighbors(adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
    sc.tl.umap(adata_embedding)

    # PCA and kmeans clustering of the whole dataset
    sc.tl.pca(adata_embedding, svd_solver='arpack', n_comps=30)

    X_pca = adata_embedding.obsm['X_pca']

    # Do clustering
    clustering_num = 50
    clustering_name = f'kmeans{clustering_num}'
    kmeans = KMeans(n_clusters=clustering_num, random_state=29).fit(X_pca)
    adata_embedding.obs[clustering_name] = kmeans.labels_.astype(str)

    # Now we know each cluster is actually a pair
    region_labels = adata_embedding.obs['region_name'].values
    sample_names = adata_embedding.obs_names.values
    cluster_labels = adata_embedding.obs[clustering_name].values
    print('cluster_labels.shape', cluster_labels.shape)
    sample_cluter_dict = {k:v for k,v in zip(sample_names, cluster_labels)}
    # All cluster labels
    cluster_labels_unique = [str(x) for x in list(range(0, clustering_num))]


    # load gene expression data selected by CAME
    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)
    # mouse expression data
    adata_sm = ad.AnnData(datapair['ov_adjs'][0])
    adata_sm.obs_names = list(datapair['obs_dfs'][0].index)
    adata_sm.var_names = datapair['varnames_node'][0]
    adata_sm.obs['region_name'] = adata_mouse_embedding.obs['region_name']
    adata_sm = normalize_before_pruning(adata_sm, method=cfg.BrainAlign.normalize_before_pruning_method_1,
                                        target_sum=cfg.BrainAlign.pruning_target_sum_1, force_return=True)
    # mouse expression data: assign cluster labels
    adata_sm_cluster_labels = []
    for obs_name in adata_sm.obs_names:
        adata_sm_cluster_labels.append(sample_cluter_dict[obs_name])
    adata_sm.obs[clustering_name] = adata_sm_cluster_labels

    # human expression data
    adata_vh = ad.AnnData(datapair['ov_adjs'][1])
    adata_vh.obs_names = list(datapair['obs_dfs'][1].index)
    adata_vh.var_names = datapair['varnames_node'][1]
    adata_vh.obs['region_name'] = adata_human_embedding.obs['region_name']
    adata_vh = normalize_before_pruning(adata_vh, method=cfg.BrainAlign.normalize_before_pruning_method_2,
                                        target_sum=cfg.BrainAlign.pruning_target_sum_2, force_return=True)
    # human expression data: assign cluster labels
    adata_vh_cluster_labels = []
    for obs_name in adata_vh.obs_names:
        adata_vh_cluster_labels.append(sample_cluter_dict[obs_name])
    adata_vh.obs[clustering_name] = adata_vh_cluster_labels
    ###################################################################
    # mouse expression data: call maker genes
    num_markers = 100
    method = 'centers'
    redundancy = 0.25

    # region name marker genes
    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(save_path + f'region_name_marker{num_markers}.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.

        markers = get_markers(adata_sm.X.toarray(), adata_sm.obs['region_name'], num_markers, method=method,
                              redundancy=redundancy)
        print('mouse markers:', markers)
        mouse_plot_marker = plot_marker_selection(adata_sm.X.toarray(), markers, adata_sm.obs['region_name'])
        mouse_plot_marker.set_size_inches(18, 6)
        mouse_plot_marker.savefig(save_path + f'region_mouse_plot_{num_markers}_marker.' + fig_format, format=fig_format)

        mouse_plot_marker = plot_marker_selection_umap(adata_sm.X.toarray(), markers, adata_sm.obs['region_name'])
        mouse_plot_marker.set_size_inches(18, 6)
        mouse_plot_marker.savefig(save_path + f'region_umap_mouse_plot_{num_markers}_marker.' + fig_format, format=fig_format)

        mouse_plot_marker = plot_marker_selection_umap_embedding(adata_sm.X.toarray(), adata_mouse_embedding.X, markers,
                                                                 adata_sm.obs['region_name'])
        mouse_plot_marker.set_size_inches(18, 6)
        mouse_plot_marker.savefig(save_path + f'region_embedding_umap_mouse_plot_{num_markers}_marker.' + fig_format,
                                  format=fig_format)

        markers = get_markers(adata_vh.X.toarray(), adata_vh.obs['region_name'], num_markers, method=method,
                              redundancy=redundancy)
        print('human markers:', markers)
        human_plot_marker = plot_marker_selection(adata_vh.X.toarray(), markers, adata_vh.obs['region_name'])
        human_plot_marker.set_size_inches(18, 6)
        human_plot_marker.savefig(save_path + f'region_human_plot_{num_markers}_marker.' + fig_format, format=fig_format)

        human_plot_marker = plot_marker_selection_umap(adata_vh.X.toarray(), markers, adata_vh.obs['region_name'])
        human_plot_marker.set_size_inches(18, 6)
        human_plot_marker.savefig(save_path + f'region_umap_human_plot_{num_markers}_marker.' + fig_format, format=fig_format)

        human_plot_marker = plot_marker_selection_umap_embedding(adata_vh.X.toarray(), adata_human_embedding.X, markers,
                                                                 adata_vh.obs['region_name'])
        human_plot_marker.set_size_inches(18, 6)
        human_plot_marker.savefig(save_path + f'region_embedding_umap_human_plot_{num_markers}_marker.' + fig_format,
                                  format=fig_format)

        sys.stdout = original_stdout  # Reset the standard output to its original value


    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(save_path + f'cluster{clustering_num}_marker{num_markers}.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.

        markers = get_markers(adata_sm.X.toarray(), adata_sm.obs[clustering_name], num_markers, method=method, redundancy=redundancy)
        print('mouse markers:', markers)
        mouse_plot_marker = plot_marker_selection(adata_sm.X.toarray(), markers, adata_sm.obs[clustering_name])
        mouse_plot_marker.set_size_inches(18, 6)
        mouse_plot_marker.savefig(save_path + f'mouse_plot_{num_markers}_marker.' + fig_format, format=fig_format)

        mouse_plot_marker = plot_marker_selection_umap(adata_sm.X.toarray(), markers, adata_sm.obs[clustering_name])
        mouse_plot_marker.set_size_inches(18, 6)
        mouse_plot_marker.savefig(save_path + f'umap_mouse_plot_{num_markers}_marker.' + fig_format, format=fig_format)

        mouse_plot_marker = plot_marker_selection_umap_embedding(adata_sm.X.toarray(), adata_mouse_embedding.X, markers, adata_sm.obs[clustering_name])
        mouse_plot_marker.set_size_inches(18, 6)
        mouse_plot_marker.savefig(save_path + f'embedding_umap_mouse_plot_{num_markers}_marker.' + fig_format, format=fig_format)


        markers = get_markers(adata_vh.X.toarray(), adata_vh.obs[clustering_name], num_markers, method=method,
                              redundancy=redundancy)
        print('human markers:', markers)
        human_plot_marker = plot_marker_selection(adata_vh.X.toarray(), markers, adata_vh.obs[clustering_name])
        human_plot_marker.set_size_inches(18, 6)
        human_plot_marker.savefig(save_path + f'human_plot_{num_markers}_marker.' + fig_format, format=fig_format)

        human_plot_marker = plot_marker_selection_umap(adata_vh.X.toarray(), markers, adata_vh.obs[clustering_name])
        human_plot_marker.set_size_inches(18, 6)
        human_plot_marker.savefig(save_path + f'umap_human_plot_{num_markers}_marker.' + fig_format, format=fig_format)

        human_plot_marker = plot_marker_selection_umap_embedding(adata_vh.X.toarray(), adata_human_embedding.X, markers, adata_vh.obs[clustering_name])
        human_plot_marker.set_size_inches(18, 6)
        human_plot_marker.savefig(save_path + f'embedding_umap_human_plot_{num_markers}_marker.' + fig_format, format=fig_format)

        sys.stdout = original_stdout  # Reset the standard output to its original value
    ###################################
    # Compute the similarity or distance between the clusters of the two species
    # and identify those highly correlated pairs.
    # here, we only need to list the cluster pairs
    for c_label in cluster_labels_unique:
        mouse_adata = adata_embedding[adata_embedding.obs[clustering_name].isin([c_label])]
        mouse_num = mouse_adata[mouse_adata.obs['dataset'].isin(['mouse'])].n_obs
        human_adata = adata_embedding[adata_embedding.obs[clustering_name].isin([c_label])]
        human_num = human_adata[human_adata.obs['dataset'].isin(['human'])].n_obs
        print(c_label, f'mouse number = {mouse_num}', f'human number = {human_num}')

    adata_mouse_umap = adata_embedding[adata_embedding.obs['dataset'].isin(['mouse'])]
    adata_human_umap = adata_embedding[adata_embedding.obs['dataset'].isin(['human'])]

    mouse_count_all = Counter(adata_mouse_umap.obs['region_name'])
    human_count_all = Counter(adata_human_umap.obs['region_name'])

    palette = sns.color_palette(cc.glasbey, n_colors=clustering_num)

    umap1_x, umap1_y = adata_mouse_umap.obsm['X_umap'].toarray()[:, 0], adata_mouse_umap.obsm[
                                                                                 'X_umap'].toarray()[:, 1]
    umap1_z_value = 1 / 36 * (
                ((np.max(umap1_x) - np.min(umap1_x)) ** 2 + (np.max(umap1_y) - np.min(umap1_y)) ** 2) ** (1 / 2))

    umap2_x, umap2_y = adata_human_umap.obsm['X_umap'].toarray()[:, 0], adata_human_umap.obsm[
                                                                                 'X_umap'].toarray()[:, 1]
    umap2_z = np.zeros(umap2_x.shape)

    most_common_num = 1
    rank_region_method = 'count&count_normalized' #count, count_normalized, count&count_normalized

    # Init a dict to save the homologous regions in each cluster
    cluster_mouse_human_homologs_dict = {}

    identified_homo_pairs = {'mouse':[], 'human':[]}
    plt.figure(figsize=(20, 8))
    axes = plt.axes(projection='3d')
    #plt.subplots_adjust(bottom=0.2, left=0.6)

    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(save_path + f'cluster{clustering_num}_most_common_num{most_common_num}.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.

        for c_index, c_label in enumerate(cluster_labels_unique):
            cluster_mouse_human_homologs_dict[c_label] = {}

            scatter_color = palette[c_index]
            mouse_adata = adata_mouse_umap[adata_mouse_umap.obs[clustering_name].isin([c_label])]
            umap1_x, umap1_y = mouse_adata.obsm['X_umap'].toarray()[:, 0], mouse_adata.obsm[
                                                                                    'X_umap'].toarray()[:, 1]
            umap1_z = umap1_z_value * np.ones(umap1_x.shape)
            # mouse: get the most frequent region in each cluster
            if rank_region_method == 'count':
                mouse_count = Counter(mouse_adata.obs['region_name'])
            elif rank_region_method == 'count_normalized':
                dict_c = Counter(mouse_adata.obs['region_name'])
                mouse_count = Counter({k:(v1/v2) for k, v1, v2 in zip(dict_c.keys(), dict_c.values(), mouse_count_all.values())})
            elif rank_region_method == 'count&count_normalized':
                mouse_count_1 = Counter(mouse_adata.obs['region_name'])
                dict_c = Counter(mouse_adata.obs['region_name'])
                mouse_count_2 = Counter(
                    {k: (v1 / v2) for k, v1, v2 in zip(dict_c.keys(), dict_c.values(), mouse_count_all.values())})
                mouse_count = Counter({k:(v1+1)*(v2+1) for k, v1, v2 in zip(mouse_count_1.keys(), mouse_count_1.values(), mouse_count_2.values())})
            #print('mouse_count:', mouse_count)
            mouse_region_set = mouse_count.most_common(most_common_num)
            cluster_mouse_human_homologs_dict[c_label]['mouse'] = {'region name': [],
                                                                   'proportion': []}
            for mouse_region, mouse_region_count in mouse_region_set:
                #mouse_count = dict(mouse_count)
                mouse_proportion = mouse_count[mouse_region] / mouse_adata.n_obs
                cluster_mouse_human_homologs_dict[c_label]['mouse']['region name'].append(mouse_region)
                cluster_mouse_human_homologs_dict[c_label]['mouse']['proportion'].append(mouse_proportion)
                mouse_adata_region = mouse_adata[mouse_adata.obs['region_name'].isin([mouse_region])]

            human_adata = adata_human_umap[adata_human_umap.obs[clustering_name].isin([c_label])]
            umap2_x, umap2_y = human_adata.obsm['X_umap'].toarray()[:, 0], human_adata.obsm['X_umap'].toarray()[:, 1]
            umap2_z = np.zeros(umap2_x.shape)
            # human: get the most frequent region in each cluster
            if rank_region_method == 'count':
                human_count = Counter(human_adata.obs['region_name'])
            elif rank_region_method == 'count_normalized':
                dict_c = Counter(human_adata.obs['region_name'])
                human_count = Counter(
                    {k: (v1 / v2) for k, v1, v2 in zip(dict_c.keys(), dict_c.values(), human_count_all.values())})
            elif rank_region_method == 'count&count_normalized':
                human_count_1 = Counter(human_adata.obs['region_name'])
                dict_c = Counter(human_adata.obs['region_name'])
                human_count_2 = Counter(
                    {k: (v1 / v2) for k, v1, v2 in zip(dict_c.keys(), dict_c.values(), human_count_all.values())})
                human_count = Counter({k:(v1+1)*(v2+1) for k, v1, v2 in zip(human_count_1.keys(), human_count_1.values(), human_count_2.values())})

            human_region_set = human_count.most_common(most_common_num)
            cluster_mouse_human_homologs_dict[c_label]['human'] = {'region name': [],
                                                                   'proportion': []}
            for human_region, human_region_count in human_region_set:
                # human_count = dict(human_count)
                human_proportion = human_count[human_region] / human_adata.n_obs
                cluster_mouse_human_homologs_dict[c_label]['human']['region name'].append(human_region)
                cluster_mouse_human_homologs_dict[c_label]['human']['proportion'].append(human_proportion)
                human_adata_region = human_adata[human_adata.obs['region_name'].isin([human_region])]

            for mouse_region in cluster_mouse_human_homologs_dict[c_label]['mouse']['region name']:
                for human_region in cluster_mouse_human_homologs_dict[c_label]['human']['region name']:
                    print(f'Identified homologous regions for cluster {c_label}: mouse-{mouse_region}, human-{human_region}')
                    for k,v in homo_region_mouse_human_dict.items():
                        if k == mouse_region and v == human_region:
                            identified_homo_pairs['mouse'].append(mouse_region)
                            identified_homo_pairs['human'].append(human_region)

                            adata_mouse_homo = mouse_adata[
                                mouse_adata.obs['region_name'].isin([mouse_region])]
                            umap1_x, umap1_y = adata_mouse_homo.obsm['X_umap'].toarray()[:, 0], \
                                               adata_mouse_homo.obsm['X_umap'].toarray()[:, 1]
                            umap1_z = umap1_z_value * np.ones(umap1_x.shape)

                            adata_human_homo = human_adata[
                                human_adata.obs['region_name'].isin([human_region])]
                            umap2_x, umap2_y = adata_human_homo.obsm['X_umap'].toarray()[:, 0], adata_human_homo.obsm[
                                                                                                    'X_umap'].toarray()[
                                                                                                :, 1]
                            umap2_z = np.zeros(umap2_x.shape)

                            # Compute the center point and plot them
                            umap1_x_mean = np.mean(umap1_x)
                            umap1_y_mean = np.mean(umap1_y)
                            umap1_z_mean = umap1_z_value

                            umap2_x_mean = np.mean(umap2_x)
                            umap2_y_mean = np.mean(umap2_y)
                            umap2_z_mean = 0

                            axes.plot3D([umap1_x_mean, umap2_x_mean], [umap1_y_mean, umap2_y_mean],
                                        [umap1_z_mean, umap2_z_mean],
                                        color='red', linewidth=0.5, alpha=1)

            axes.scatter3D(umap1_x, umap1_y, umap1_z, color=scatter_color, s=1, label=c_label)
            axes.scatter3D(umap2_x, umap2_y, umap2_z, color=scatter_color, s=1)

            # Compute the center point and plot them
            umap1_x_mean = np.mean(umap1_x)
            umap1_y_mean = np.mean(umap1_y)
            umap1_z_mean = umap1_z_value

            umap2_x_mean = np.mean(umap2_x)
            umap2_y_mean = np.mean(umap2_y)
            umap2_z_mean = 0

            axes.plot3D([umap1_x_mean, umap2_x_mean], [umap1_y_mean, umap2_y_mean], [umap1_z_mean, umap2_z_mean], color='gray', linewidth=0.5, alpha=0.5)

        print('identified_homo_pairs', identified_homo_pairs)
        identified_homo_pairs_num = len(set(identified_homo_pairs['mouse']))
        print(f'Identified {identified_homo_pairs_num} homologous pairs.')
        with open(save_path + f'identified_homo_pairs_dict_n{most_common_num}_cluster{clustering_num}.pkl', 'wb') as f:
            pickle.dump(identified_homo_pairs, f)

        # plot homologous regions pairs in blue lines
        for mouse_region, human_region in homo_region_mouse_human_dict.items():
            adata_mouse_homo = adata_mouse_umap[adata_mouse_umap.obs['region_name'].isin([mouse_region])]
            umap1_x, umap1_y = adata_mouse_homo.obsm['X_umap'].toarray()[:, 0], adata_mouse_homo.obsm['X_umap'].toarray()[:, 1]
            umap1_z = umap1_z_value * np.ones(umap1_x.shape)

            adata_human_homo = adata_human_umap[adata_human_umap.obs['region_name'].isin([human_region])]
            umap2_x, umap2_y = adata_human_homo.obsm['X_umap'].toarray()[:, 0], adata_human_homo.obsm['X_umap'].toarray()[:, 1]
            umap2_z = np.zeros(umap2_x.shape)

            # Compute the center point and plot them
            umap1_x_mean = np.mean(umap1_x)
            umap1_y_mean = np.mean(umap1_y)
            umap1_z_mean = umap1_z_value

            umap2_x_mean = np.mean(umap2_x)
            umap2_y_mean = np.mean(umap2_y)
            umap2_z_mean = 0

            axes.plot3D([umap1_x_mean, umap2_x_mean], [umap1_y_mean, umap2_y_mean], [umap1_z_mean, umap2_z_mean],
                        color='black', linewidth=0.5, alpha=1)


        #axes.set_zlim(0, umap1_z_value*1.5)
        # Hide grid lines
        axes.grid(False)
        axes.set_xlabel('umap x')
        axes.set_ylabel('umap y')
        axes.set_zlabel('z')
        plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), ncol=5, frameon=False)
        #plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        plt.savefig(save_path + f'Umap3d_human_mouse_n{most_common_num}_clusters{clustering_num}.' + fig_format)

        sys.stdout = original_stdout  # Reset the standard output to its original value

    # For each cluster, compute its brain region composition, check each pair and align it to the brain region pair.

###################################################################
# Evaluation of mouse and human embedding alignment
# 1. Clustering of mouse and human embeddings seperately, identify marker genes for each cluster,
#    and name each cluster by cluster order_mouse marker gene_human marker gene.
# 2. Compute the similarity or distance between the clusters of the two species
#    and identify those highly correlated pairs.
# 3. For each cluster, compute its brain region composition, check each pair and align it to the brain region pair.
# 4. Plot the 3D alignment figure, for each pair of highly correlated clusters, plot the mean of each cluster (black);
#    for each pair of homologous brain regions, plot the mean of each region (red).
# 4. Check the overlap of homologous clusters pair (brain regions) and clusters pairs (precision, recall, etc).
def cross_evaluation_aligment_seperate(cfg):

    fig_format = cfg.BrainAlign.fig_format

    save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/cross_evaluation_alignment_seperate/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
    adata_mouse_embedding.obs['dataset'] = 'mouse'
    adata_human_embedding.obs['dataset'] = 'human'

    mouse_64_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)
    mouse_64_labels_list = list(mouse_64_labels['region_name'])
    human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
    human_88_labels_list = list(human_88_labels['region_name'])

    # plot human mouse homologous samples
    human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)
    homo_region_list_mouse = list(human_mouse_homo_region['Mouse'].values)
    homo_region_list_human = list(human_mouse_homo_region['Human'].values)

    homo_region_mouse_human_dict = {k: v for k, v in zip(homo_region_list_mouse, homo_region_list_human)}

    adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
    print(adata_embedding)

    sc.pp.neighbors(adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
    sc.tl.umap(adata_embedding)

    # PCA and kmeans clustering of the whole dataset
    sc.tl.pca(adata_embedding, svd_solver='arpack', n_comps=30)

    adata_embedding_pca_mouse = adata_embedding[adata_embedding.obs['dataset'].isin(['mouse'])]
    adata_embedding_pca_human = adata_embedding[adata_embedding.obs['dataset'].isin(['human'])]


    # Do clustering
    mouse_clustering_num = 40
    human_clustering_num = 60
    clustering_name = f'kmeans_mouse{mouse_clustering_num}_human{human_clustering_num}'

    X_pca_mouse = adata_embedding_pca_mouse.obsm['X_pca']
    kmeans = KMeans(n_clusters=mouse_clustering_num, random_state=29).fit(X_pca_mouse)
    adata_embedding_pca_mouse.obs[clustering_name] = ['mouse_'+x for x in kmeans.labels_.astype(str)]

    X_pca_human = adata_embedding_pca_human.obsm['X_pca']
    kmeans = KMeans(n_clusters=human_clustering_num, random_state=29).fit(X_pca_human)
    adata_embedding_pca_human.obs[clustering_name] = ['human_' + x for x in kmeans.labels_.astype(str)]


    adata_embedding = ad.concat([adata_embedding_pca_mouse, adata_embedding_pca_human])

    # Now we know each cluster is actually a pair
    region_labels = adata_embedding.obs['region_name'].values
    sample_names = adata_embedding.obs_names.values
    cluster_labels = adata_embedding.obs[clustering_name].values
    print('cluster_labels.shape', cluster_labels.shape)
    sample_cluter_dict = {k: v for k, v in zip(sample_names, cluster_labels)}
    # All cluster labels
    # cluster_labels_unique = [str(x) for x in list(range(0, clustering_num))]

    strong_corr_region_pairs = {'mouse':[], 'human':[]}
    # compute correlation of each pair of clusters, sort them by correlation value, and keep N as strong correlations




    # load gene expression data selected by CAME
    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)
    # mouse expression data
    adata_sm = ad.AnnData(datapair['ov_adjs'][0])
    adata_sm.obs_names = list(datapair['obs_dfs'][0].index)
    adata_sm.var_names = datapair['varnames_node'][0]
    adata_sm = normalize_before_pruning(adata_sm, method=cfg.BrainAlign.normalize_before_pruning_method_1,
                                        target_sum=cfg.BrainAlign.pruning_target_sum_1, force_return=True)
    # mouse expression data: assign cluster labels
    adata_sm_cluster_labels = []
    for obs_name in adata_sm.obs_names:
        adata_sm_cluster_labels.append(sample_cluter_dict[obs_name])
    adata_sm.obs[clustering_name] = adata_sm_cluster_labels

    ###################################################################
    # mouse expression data: call maker genes
    num_markers = 30
    method = 'centers'
    redundancy = 0.25

    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(save_path + f'cluster_{mouse_clustering_num}_{human_clustering_num}_marker{num_markers}.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.

        markers = get_markers(adata_sm.X.toarray(), adata_sm.obs[clustering_name], num_markers, method=method,
                              redundancy=redundancy)
        print('mouse markers:', markers)
        mouse_plot_marker = plot_marker_selection(adata_sm.X.toarray(), markers, adata_sm.obs[clustering_name])
        mouse_plot_marker.set_size_inches(18, 6)
        mouse_plot_marker.savefig(save_path + f'mouse_plot_{num_markers}_marker.' + fig_format, format=fig_format)

        mouse_plot_marker = plot_marker_selection_umap(adata_sm.X.toarray(), markers, adata_sm.obs[clustering_name])
        mouse_plot_marker.set_size_inches(18, 6)
        mouse_plot_marker.savefig(save_path + f'umap_mouse_plot_{num_markers}_marker.' + fig_format, format=fig_format)

        # human expression data
        adata_vh = ad.AnnData(datapair['ov_adjs'][1])
        adata_vh.obs_names = list(datapair['obs_dfs'][1].index)
        adata_vh.var_names = datapair['varnames_node'][1]
        adata_vh = normalize_before_pruning(adata_vh, method=cfg.BrainAlign.normalize_before_pruning_method_2,
                                            target_sum=cfg.BrainAlign.pruning_target_sum_2, force_return=True)
        # human expression data: assign cluster labels
        adata_vh_cluster_labels = []
        for obs_name in adata_vh.obs_names:
            adata_vh_cluster_labels.append(sample_cluter_dict[obs_name])
        adata_vh.obs[clustering_name] = adata_vh_cluster_labels

        markers = get_markers(adata_vh.X.toarray(), adata_vh.obs[clustering_name], num_markers, method=method,
                              redundancy=redundancy)
        print('human markers:', markers)
        human_plot_marker = plot_marker_selection(adata_vh.X.toarray(), markers, adata_vh.obs[clustering_name])
        human_plot_marker.set_size_inches(18, 6)
        human_plot_marker.savefig(save_path + f'human_plot_{num_markers}_marker.' + fig_format, format=fig_format)

        human_plot_marker = plot_marker_selection_umap(adata_vh.X.toarray(), markers, adata_vh.obs[clustering_name])
        human_plot_marker.set_size_inches(18, 6)
        human_plot_marker.savefig(save_path + f'umap_human_plot_{num_markers}_marker.' + fig_format, format=fig_format)

        sys.stdout = original_stdout  # Reset the standard output to its original value


def get_center(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return sum(x) / len(points), sum(y) / len(points)

def get_homologous_rate(max_subset_num,
                        mouse_64_labels_list,
                        human_88_labels_list,
                        mouse_overlap_df,
                        human_overlap_df,
                        homo_region_mouse_human_dict):
    mouse_idxmax_subset_dict = {}
    human_idxmax_subset_dict = {}
    for re_mouse in mouse_64_labels_list:
        subset_list = mouse_overlap_df[re_mouse].values
        mouse_idxmax_subset_dict[re_mouse] = np.argpartition(subset_list, -max_subset_num)[-max_subset_num:]

    for re_human in human_88_labels_list:
        subset_list = human_overlap_df[re_human].values
        human_idxmax_subset_dict[re_human] = np.argpartition(subset_list, -max_subset_num)[-max_subset_num:]

    homologous_pair_num = 0
    equal_subtype_dict = {'mouse': [], 'human': []}
    print('homologous pairs:')
    for re_mouse in mouse_overlap_df.columns:
        for re_human in human_overlap_df.columns:
            if len(set(mouse_idxmax_subset_dict[re_mouse]).intersection(
                    set(human_idxmax_subset_dict[re_human]))) != 0:
                equal_subtype_dict['mouse'].append(re_mouse)
                equal_subtype_dict['human'].append(re_human)
                print(re_mouse, re_human)

    print(f'Subset size = {max_subset_num}')
    homo_region_mouse_human_dict_identified = {}
    for mouse_re, human_re in zip(equal_subtype_dict['mouse'], equal_subtype_dict['human']):
        if mouse_re in set(homo_region_mouse_human_dict.keys()) and homo_region_mouse_human_dict[
            mouse_re] == human_re:
            homo_region_mouse_human_dict_identified[mouse_re] = human_re

    print('homo_region_mouse_human_dict_identified', homo_region_mouse_human_dict_identified)
    print(f'Identified {len(homo_region_mouse_human_dict_identified)} homologous regions.')
    print('Rate of homologous regions identified = ',
          len(homo_region_mouse_human_dict_identified) / len(homo_region_mouse_human_dict))
    return len(homo_region_mouse_human_dict_identified) / len(homo_region_mouse_human_dict), len(homo_region_mouse_human_dict_identified)/len(equal_subtype_dict['mouse'])

##################################################################################
'''
Clustering and Umap of samples from each specie to check if embeddings are well generated.
'''
##################################################################################
def umap_seperate(cfg):

    fig_format = cfg.BrainAlign.fig_format
    adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')

    sc.pp.neighbors(adata_mouse_embedding, n_neighbors=cfg.ANALYSIS.mouse_umap_neighbor, metric='cosine', use_rep='X')
    sc.tl.umap(adata_mouse_embedding)

    save_path = cfg.BrainAlign.embeddings_file_path + 'figs/mouse_67/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    from matplotlib import rcParams
    rcParams["figure.subplot.left"] = 0.05
    rcParams["figure.subplot.right"] = 0.55

    #plt.tight_layout()
    #sc.set_figure_params(dpi_save=200)
    sc.pl.umap(adata_mouse_embedding, color=['region_name'], return_fig=True, legend_loc='right margin').savefig(
    save_path + 'umap.'+fig_format, format=fig_format)
        #plt.subplots_adjust(left = 0.1, right=5)
    with plt.rc_context({"figure.figsize": (12, 6)}):
        sc.pp.neighbors(adata_human_embedding, n_neighbors=cfg.ANALYSIS.human_umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(adata_human_embedding)

    save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with plt.rc_context({"figure.figsize": (12, 6)}):

    #sc.set_figure_params(dpi_save=200)
        sc.pl.umap(adata_human_embedding, color=['region_name'], return_fig=True, size=30, legend_loc='right margin').savefig(
        save_path + 'umap.'+fig_format, format=fig_format)
    #plt.subplots_adjust(left=0.1, right=5)
        #plt.tight_layout()
    rcParams["figure.subplot.left"] = 0.125
    rcParams["figure.subplot.right"] = 0.9



def cross_species_genes_clustering(cfg):

    fig_format = cfg.BrainAlign.fig_format
    import sys
    sys.setrecursionlimit(100000)
    # No label order version
    '''
    Step 1: Compute average embedding of every region in two species, use two dict to store;
    Step 2: Compute similarity matrix, use np array to store;
    Step 3: Heatmap.
    '''
    adata_mouse_gene_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'm_embeddings.h5ad')
    adata_human_gene_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'h_embeddings.h5ad')

    custom_palette = sns.color_palette("Paired", 2)
    print(custom_palette)
    # lut = dict(zip(['Human', 'Mouse'], custom_palette))
    lut = {"Human": (26 / 255, 85 / 255, 153 / 255), "Mouse": (128 / 255, 0 / 255, 32 / 255)}
    print(lut)
    row_colors = []
    for i in range(adata_human_gene_embedding.X.shape[0]):
        row_colors.append(lut['Human'])
    ## ---------------------------------------------------------------------mouse------------------------
    for i in range(adata_human_gene_embedding.X.shape[0]):
        row_colors.append(lut['Mouse'])

    rng = np.random.default_rng(12345)
    rints = rng.integers(low=0, high=adata_mouse_gene_embedding.X.shape[0], size=adata_human_gene_embedding.X.shape[0])
    # rints
    print('adata_human_embedding.X.shape', adata_human_gene_embedding.X.shape)
    print('adata_mouse_embedding.X.shape', adata_mouse_gene_embedding.X.shape)

    sns.clustermap(np.concatenate((adata_human_gene_embedding.X, adata_mouse_gene_embedding.X[rints, :]), axis=0),
                   row_colors=row_colors)

    # color_map = sns.color_palette("coolwarm", as_cmap=True)
    # sns.clustermap(Var_Corr, cmap=color_map, center=0.6)
    save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + 'Genes_Hiercluster_heco_embedding_concate.'+fig_format, format=fig_format)
    plt.show()

    adata_mouse_gene_embedding.obs['dataset'] = 'Mouse'
    adata_human_gene_embedding.obs['dataset'] = 'Human'

    adata_gene_embedding = ad.concat([adata_mouse_gene_embedding, adata_human_gene_embedding])
    print(adata_gene_embedding)

    sc.pp.neighbors(adata_gene_embedding, n_neighbors=cfg.ANALYSIS.genes_umap_neighbor, metric='cosine', use_rep='X')
    sc.tl.umap(adata_gene_embedding, min_dist=0.85)

    #save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/genes/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with plt.rc_context({"figure.figsize": (8, 8)}): #, "figure.dpi": (300)
        sc.pl.umap(adata_gene_embedding, color=['dataset'], return_fig=True, legend_loc='on data').savefig(
            save_path + 'Genes_umap_dataset.' + fig_format, format=fig_format)
        #plt.subplots_adjust(right=0.3)

    # ----------------------------------------------------------------------------------------------------
    ## 3d plot of genes, map between two species
    '''
    1. load homologous genes relations;
    2. load umap values, increase mouse umap coordinates to a non-zero value;
    3. plot umap points of two species;
    4. plot homologous genes rellations
    '''
    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)
    mh_mat = datapair['vv_adj'].toarray()[0:cfg.BrainAlign.binary_M, cfg.BrainAlign.binary_M:]

    #adata_gene_embedding =
    adata_gene_umap_mouse = adata_gene_embedding[adata_gene_embedding.obs['dataset'] == 'Mouse']
    adata_gene_umap_human = adata_gene_embedding[adata_gene_embedding.obs['dataset'] == 'Human']
    umap1_x, umap1_y = adata_gene_umap_mouse.obsm['X_umap'].toarray()[:, 0], adata_gene_umap_mouse.obsm['X_umap'].toarray()[:, 1]
    umap1_z = 1/4 * np.ones(umap1_x.shape) * (((np.max(umap1_x)-np.min(umap1_x))**2 + (np.max(umap1_y)-np.min(umap1_y))**2) ** (1/2))

    umap2_x, umap2_y = adata_gene_umap_human.obsm['X_umap'].toarray()[:, 0], adata_gene_umap_human.obsm['X_umap'].toarray()[:, 1]
    umap2_z = np.zeros(umap2_x.shape)

    point1_list = []
    for u1_x, u1_y, u1_z in zip(umap1_x, umap1_y, umap1_z):
        point1_list.append([u1_x, u1_y, u1_z])

    point2_list = []
    for u2_x, u2_y, u2_z in zip(umap2_x, umap2_y, umap2_z):
        point2_list.append([u2_x, u2_y, u2_z])

    plt.figure(figsize=(8, 8))
    axes = plt.axes(projection='3d')
    #print(type(axes))
    axes.scatter3D(umap1_x, umap1_y, umap1_z, color=(255/255, 133/255, 25/255), s=1)
    axes.scatter3D(umap2_x, umap2_y, umap2_z, color=(32 / 255, 119 / 255, 180 / 255), s=1)
    for i in range(len(point1_list)):
        for j in range(len(point2_list)):
            point1 = point1_list[i]
            point2 = point2_list[j]
            if mh_mat[i, j] != 0:
                axes.plot3D([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color='gray', linewidth=0.25, alpha=0.4)
    axes.set_xlabel('umap x')
    axes.set_ylabel('umap y')
    axes.set_zlabel('z')
    #axes.view_init(45, 215)
    plt.savefig(save_path + 'Genes_umap_plot_map.' + fig_format)
    #----------------------------------------------------------------------------



    # plot gene modules
    with plt.rc_context({"figure.figsize": (8, 8)}):
        sc.tl.leiden(adata_gene_embedding, resolution=.8, key_added='module')
        sc.pl.umap(adata_gene_embedding, color='module', ncols=1, palette='tab20b', return_fig=True).savefig(
            save_path + 'Genes_module_concat.' + fig_format, format=fig_format)

    #adata_gene_embedding.obs_names = adata_gene_embedding.obs_names.astype(str)
    gadt1, gadt2 = pp.bisplit_adata(adata_gene_embedding, 'dataset', cfg.BrainAlign.dsnames[0]) #weight_linked_vars

    color_by = 'module'
    palette = 'tab20b'
    sc.pl.umap(gadt1, color=color_by, s=10, #edges=True, edges_width=0.05,
               palette=palette,
               save=f'_{color_by}-{cfg.BrainAlign.dsnames[0]}', return_fig=True).savefig(save_path + '{}_genes_module_concat.'.format(cfg.BrainAlign.dsnames[0]) + fig_format, format=fig_format)
    sc.pl.umap(gadt2, color=color_by, s=10, #edges=True, edges_width=0.05,
               palette=palette,
               save=f'_{color_by}-{cfg.BrainAlign.dsnames[1]}', return_fig=True).savefig(save_path + '{}_genes_module_concat.'.format(cfg.BrainAlign.dsnames[1]) + fig_format, format=fig_format)

    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)
    vnode_names = datapair['varnames_node'][0] + datapair['varnames_node'][1]
    print(len(vnode_names))
    df_var_links = weight_linked_vars(
        adata_gene_embedding.X, datapair['vv_adj'], names=vnode_names,
        matric='euclidean', index_names=cfg.BrainAlign.dsnames,
    )

    gadt1.write(cfg.BrainAlign.embeddings_file_path + 'adt_hidden_gene1.h5ad')
    gadt2.write(cfg.BrainAlign.embeddings_file_path + 'adt_hidden_gene2.h5ad')

    # Compute average expressions for each brain region.
    key_class1 = 'region_name'  # set by user
    key_class2 = 'region_name'  # set by user
    # averaged expressions

    #adata_mouse_sample_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    #adata_human_sample_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
    #adatas = [adata_mouse_sample_embedding, adata_human_sample_embedding]
    adata_mouse_sample_embedding = sc.read_h5ad(cfg.CAME.path_rawdata1)
    adata_human_sample_embedding = sc.read_h5ad(cfg.CAME.path_rawdata2)
    adatas = [adata_mouse_sample_embedding, adata_human_sample_embedding]
    # compute average embedding of
    '''
    avg_expr1 = pp.group_mean_adata(
        adatas[0], groupby=key_class1,
        features=datapair.vnode_names1, use_raw=True)
    avg_expr2 = pp.group_mean_adata(
        adatas[1], groupby=key_class2 if key_class2 else 'predicted',
        features=datapair.vnode_names2, use_raw=True)
    '''
    avg_expr1 = pp.group_mean_adata(
        adatas[0], groupby=key_class1, features=datapair['varnames_node'][0],
        use_raw=True)

    avg_expr2 = pp.group_mean_adata(
        adatas[1], groupby=key_class2 if key_class2 else 'predicted', features=datapair['varnames_node'][1],
        use_raw=True)

    plkwds = dict(cmap='RdYlBu_r', vmax=2.5, vmin=-1.5, do_zscore=True,
                  axscale=3, ncols=5, with_cbar=True)

    print('gadt1', gadt1)
    print('gadt2', gadt2)
    fig1, axs1 = pl.adata_embed_with_values(
        gadt1, avg_expr1, embed_key='UMAP',
        fp=save_path + f'umap_exprAvgs-{cfg.BrainAlign.dsnames[0]}-all.'+fig_format,
        **plkwds)
    fig2, axs2 = pl.adata_embed_with_values(
        gadt2, avg_expr2, embed_key='UMAP',
        fp=save_path + f'umap_exprAvgs-{cfg.BrainAlign.dsnames[1]}-all.'+fig_format,
        **plkwds)

    ## Abstracted graph #####################################
    norm_ov = ['max', 'zs', None][1]
    cut_ov = cfg.ANALYSIS.cut_ov

    groupby_var = 'module'
    obs_labels1, obs_labels2 = adata_mouse_sample_embedding.obs['region_name'], adata_human_sample_embedding.obs['region_name']#adt.obs['celltype'][dpair.obs_ids1], adt.obs['celltype'][dpair.obs_ids2]
    var_labels1, var_labels2 = gadt1.obs[groupby_var], gadt2.obs[groupby_var]

    sp1, sp2 = 'mouse', 'human'
    g = came.make_abstracted_graph(
        obs_labels1, obs_labels2,
        var_labels1, var_labels2,
        avg_expr1, avg_expr2,
        df_var_links,
        tags_obs=(f'{sp1} ', f'{sp2} '),
        tags_var=(f'{sp1} module ', f'{sp2} module '),
        cut_ov=cut_ov,
        norm_mtd_ov=norm_ov,
    )

    ''' visualization '''
    fp_abs = save_path + f'abstracted_graph-{groupby_var}-cut{cut_ov}-{norm_ov}.' + fig_format
    ax = pl.plot_multipartite_graph(
        g, edge_scale=10,
        figsize=(8, 20), alpha=0.5, fp=fp_abs)  # nodelist=nodelist,

    #ax.figure
    came.save_pickle(g, cfg.BrainAlign.embeddings_file_path + 'abs_graph.pickle')


def genes_homo_random_distance(cfg, metric_name = 'euclidean'):

    # Step 1: load homologous gene relations, compute homologous genes and non-homologous genes correlations;
    # Step 2: plot box plot according to the two correlation sequences.
    fig_format = cfg.BrainAlign.fig_format
    import sys
    sys.setrecursionlimit(100000)

    # load gene embeddings
    adata_mouse_gene_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'm_embeddings.h5ad')
    adata_human_gene_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'h_embeddings.h5ad')

    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)
    mh_mat = datapair['vv_adj'].toarray()[0:cfg.BrainAlign.binary_M, cfg.BrainAlign.binary_M:]

    homo_random_gene_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/genes_homo_random/'
    if not os.path.exists(homo_random_gene_path):
        os.makedirs(homo_random_gene_path)

    mouse_df = pd.DataFrame(adata_mouse_gene_embedding.X).T
    print(mouse_df.shape)
    human_df = pd.DataFrame(adata_human_gene_embedding.X).T
    print(human_df.shape)

    ## Compute distance of homologous regions compared to the other pairs
    sample_df = pd.concat([mouse_df.T, human_df.T], axis=0)
    y_index_name = metric_name + ' distance'
    distances = pdist(sample_df.values, metric=metric_name)
    dist_matrix = squareform(distances)
    print(dist_matrix.shape)
    mouse_genes_num = len(adata_mouse_gene_embedding.obs_names)
    human_genes_num = len(adata_human_gene_embedding.obs_names)
    dist_matrix_all = dist_matrix[0:mouse_genes_num, mouse_genes_num:mouse_genes_num+human_genes_num]
    Var_Corr = pd.DataFrame(dist_matrix_all, columns=adata_human_gene_embedding.obs_names, index=adata_mouse_gene_embedding.obs_names)
    mean, std = Var_Corr.mean().mean(), Var_Corr.stack().std()
    print('mean:', mean, 'std:', std)
    print('Var_Corr:', Var_Corr.shape)
    Var_mat = Var_Corr.values
    gene_human_mouse_homo_corr_dict = {}
    gene_human_mouse_random_corr_dict = {}
    gene_human_mouse_homo_corr_dict[y_index_name] = []
    gene_human_mouse_random_corr_dict[y_index_name] = []
    for i in range(cfg.BrainAlign.binary_M):
        for j in range(cfg.BrainAlign.binary_H):
            if mh_mat[i, j] > 0:
                gene_human_mouse_homo_corr_dict[y_index_name].append(Var_mat[i, j])
            else:
                gene_human_mouse_random_corr_dict[y_index_name].append(Var_mat[i, j])
    home_len = len(gene_human_mouse_homo_corr_dict[y_index_name])
    home_random_type = ['homologous'] * home_len
    gene_human_mouse_homo_corr_dict['type'] = home_random_type
    # data_df = pd.DataFrame.from_dict(human_mouse_correlation_dict)
    random_len = len(gene_human_mouse_random_corr_dict[y_index_name])
    home_random_type = ['random'] * random_len
    gene_human_mouse_random_corr_dict['type'] = home_random_type

    concat_dict = {}
    for k, v in gene_human_mouse_random_corr_dict.items():
        concat_dict[k] = gene_human_mouse_homo_corr_dict[k] + gene_human_mouse_random_corr_dict[k]
    data_df = pd.DataFrame.from_dict(concat_dict)

    save_path = homo_random_gene_path

    # my_pal = {"homo region": (0/255,149/255,182/255), "random":(178/255, 0/255, 32/255)}
    my_pal = {"homologous": (105 / 255, 105 / 255, 105 / 255), "random": (211 / 255, 211 / 255, 211 / 255)}
    # sns.set_theme(style="whitegrid")
    # tips = sns.load_dataset("tips")
    plt.figure(figsize=(8, 8))
    ax = sns.boxplot(x="type", y=y_index_name, data=data_df, order=["homologous", "random"], palette=my_pal)
    plt.savefig(save_path + f'{metric_name}_distance.' + fig_format)
    plt.show()
    ## t-test of the difference
    print(stats.ttest_ind(
        gene_human_mouse_homo_corr_dict[y_index_name],
        gene_human_mouse_random_corr_dict[y_index_name],
        equal_var=False
    ))
    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(save_path + f't_test_result_{metric_name}.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(stats.ttest_ind(
            gene_human_mouse_homo_corr_dict[y_index_name],
            gene_human_mouse_random_corr_dict[y_index_name],
            equal_var=False
        ))
        sys.stdout = original_stdout  # Reset the standard output to its original value


def genes_homo_random_corr(cfg):

    # Step 1: load homologous gene relations, compute homologous genes and non-homologous genes correlations;
    # Step 2: plot box plot according to the two correlation sequences.
    fig_format = cfg.BrainAlign.fig_format
    import sys
    sys.setrecursionlimit(100000)

    # load gene embeddings
    adata_mouse_gene_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'm_embeddings.h5ad')
    adata_human_gene_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'h_embeddings.h5ad')

    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)
    mh_mat = datapair['vv_adj'].toarray()[0:cfg.BrainAlign.binary_M, cfg.BrainAlign.binary_M:]


    homo_random_gene_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/genes_homo_random/'
    if not os.path.exists(homo_random_gene_path):
        os.makedirs(homo_random_gene_path)

    mouse_df = pd.DataFrame(adata_mouse_gene_embedding.X).T
    mouse_df.columns = ['mouse_' + x for x in adata_mouse_gene_embedding.obs_names]
    print(mouse_df.shape)
    human_df = pd.DataFrame(adata_human_gene_embedding.X).T
    human_df.columns = ['human_'+x for x in adata_human_gene_embedding.obs_names]
    print(human_df.shape)

    ## Compute correlation of homologous regions compared to the other pairs
    corr_result = pd.concat([mouse_df, human_df], axis=1).corr()

    Var_Corr = corr_result[human_df.columns].loc[mouse_df.columns]
    mean, std = Var_Corr.mean().mean(), Var_Corr.stack().std()
    print('mean:', mean, 'std:', std)
    print('Var_Corr:', Var_Corr.shape)

    gene_human_mouse_homo_corr_dict = {}
    gene_human_mouse_random_corr_dict = {}
    gene_human_mouse_homo_corr_dict['Correlation'] = []
    gene_human_mouse_random_corr_dict['Correlation'] = []


    for i in range(cfg.BrainAlign.binary_M):
        for j in range(cfg.BrainAlign.binary_H):
            if mh_mat[i, j] > 0:
                gene_human_mouse_homo_corr_dict['Correlation'].append(Var_Corr.values[i, j])
            else:
                gene_human_mouse_random_corr_dict['Correlation'].append(Var_Corr.values[i, j])

    home_len = len(gene_human_mouse_homo_corr_dict['Correlation'])
    home_random_type = ['homologous'] * home_len
    gene_human_mouse_homo_corr_dict['type'] = home_random_type
    # data_df = pd.DataFrame.from_dict(human_mouse_correlation_dict)

    random_len = len(gene_human_mouse_random_corr_dict['Correlation'])
    home_random_type = ['random'] * random_len
    gene_human_mouse_random_corr_dict['type'] = home_random_type

    concat_dict = {}
    for k, v in gene_human_mouse_random_corr_dict.items():
        concat_dict[k] = gene_human_mouse_homo_corr_dict[k] + gene_human_mouse_random_corr_dict[k]
    data_df = pd.DataFrame.from_dict(concat_dict)

    save_path = homo_random_gene_path

    #my_pal = {"homo region": (0/255,149/255,182/255), "random":(178/255, 0/255, 32/255)}
    my_pal = {"homologous": (105 / 255, 105 / 255, 105 / 255), "random": (211 / 255, 211 / 255, 211 / 255)}
    #sns.set_theme(style="whitegrid")
    #tips = sns.load_dataset("tips")

    plt.figure(figsize=(8,8))
    ax = sns.boxplot(x="type", y="Correlation", data=data_df, order=["homologous", "random"], palette = my_pal)
    plt.savefig(save_path + 'Correlation.' + fig_format)
    plt.show()

    ## t-test of the difference
    from scipy import stats

    print(stats.ttest_ind(
        gene_human_mouse_homo_corr_dict['Correlation'],
        gene_human_mouse_random_corr_dict['Correlation'],
        equal_var=False
    ))

    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(save_path + 't_test_result_correlation.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(stats.ttest_ind(
            gene_human_mouse_homo_corr_dict['Correlation'],
            gene_human_mouse_random_corr_dict['Correlation'],
            equal_var=False
        ))
        sys.stdout = original_stdout  # Reset the standard output to its original value

    ## Compute distance of homologous regions compared to the other pairs
    genes_homo_random_distance(cfg, metric_name='euclidean')
    genes_homo_random_distance(cfg, metric_name='cosine')
    genes_homo_random_distance(cfg, metric_name='chebyshev')
    genes_homo_random_distance(cfg, metric_name='correlation')
    genes_homo_random_distance(cfg, metric_name='braycurtis')
    genes_homo_random_distance(cfg, metric_name='jensenshannon')
    genes_homo_random_distance(cfg, metric_name='canberra')
    genes_homo_random_distance(cfg, metric_name='minkowski')



##################################################################################
'''
Clustering and Umap of genes from each specie to check if embeddings are well generated.
'''
##################################################################################
def umap_genes_seperate(cfg):
    fig_format = cfg.BrainAlign.fig_format
    adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'm_embeddings.h5ad')
    adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'h_embeddings.h5ad')

    sc.pp.neighbors(adata_mouse_embedding, n_neighbors=cfg.ANALYSIS.mouse_umap_neighbor, metric='cosine', use_rep='X')
    sc.tl.umap(adata_mouse_embedding)

    save_path = cfg.BrainAlign.embeddings_file_path + 'figs/mouse_67/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with plt.rc_context({"figure.figsize": (8, 6)}):
        sc.pl.umap(adata_mouse_embedding, color=['region_name'], return_fig=True).savefig(
            save_path + 'umap_genes.' + fig_format, format=fig_format)
        #plt.subplots_adjust(right=0.3)

    sc.pp.neighbors(adata_human_embedding, n_neighbors=cfg.ANALYSIS.human_umap_neighbor, metric='cosine', use_rep='X')
    sc.tl.umap(adata_human_embedding)

    save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with plt.rc_context({"figure.figsize": (8, 6)}):

        sc.pl.umap(adata_human_embedding, color=['region_name'], return_fig=True).savefig(
            save_path + 'umap_genes.' + fig_format, format=fig_format)
        plt.subplots_adjust(right=0.3)



##########################################################################
'''
Alignment of genes:
1. Sankey plots of genes to see gene modules of each specie.
'''
#############################################################################
#def alignment_genes(cfg):




##################################################################################
'''
The following function does visualizations of learned embeddings and evaluate them via
UMAP, alignment with brain region labels, and alignment with spatial distribution of samples.   
'''
##################################################################################
def spatial_alignment(cfg):
    '''
       Step 1: Compute average embedding and position of every region in two species, use two dict to store;
       Step 2: Compute distance between regions, select one-nearest neighbor of each region, compute correlations;
       Step 3: plot Boxplot.

    '''
    fig_format = cfg.BrainAlign.fig_format

    human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
    mouse_67_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)

    adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')

    ## Human experiment
    # Read ordered labels
    expression_human_path = cfg.CAME.path_rawdata2
    adata_human = sc.read_h5ad(expression_human_path)
    #print(adata_human)

    human_embedding_dict = OrderedDict()
    # print(list(human_88_labels['region_name']))
    human_88_labels_list = list(human_88_labels['region_name'])
    for r_n in human_88_labels_list:
        human_embedding_dict[r_n] = None


    for region_name in human_88_labels_list:
        mean_embedding = np.mean(adata_human_embedding[adata_human_embedding.obs['region_name'] == region_name].X, axis=0)
        human_embedding_dict[region_name] = mean_embedding

    human_embedding_df = pd.DataFrame.from_dict(human_embedding_dict)

    # compute mean mri coordinates
    human_mri_xyz_dict = OrderedDict()
    for r_n in human_88_labels_list:
        human_mri_xyz_dict[r_n] = {'mri_voxel_x': 0, 'mri_voxel_y': 0, 'mri_voxel_z': 0}
    #print(np.mean(adata_human.obs['mri_voxel_x'].values))
    for region_name in human_88_labels_list:
        mean_mri_voxel_x = np.mean(adata_human[adata_human.obs['region_name'] == region_name].obs['mri_voxel_x'].values)
        human_mri_xyz_dict[region_name]['mri_voxel_x'] = mean_mri_voxel_x
        mean_mri_voxel_y = np.mean(adata_human[adata_human.obs['region_name'] == region_name].obs['mri_voxel_y'].values)
        human_mri_xyz_dict[region_name]['mri_voxel_y'] = mean_mri_voxel_y
        mean_mri_voxel_z = np.mean(adata_human[adata_human.obs['region_name'] == region_name].obs['mri_voxel_z'].values)
        human_mri_xyz_dict[region_name]['mri_voxel_z'] = mean_mri_voxel_z

    human_mri_xyz_df = pd.DataFrame.from_dict(human_mri_xyz_dict).T
    #print(human_mri_xyz_df)

    ## Compute distance matrix
    human_dist_df = pd.DataFrame(squareform(pdist(human_mri_xyz_df.values)), columns=human_mri_xyz_df.index, index=human_mri_xyz_df.index)

    #print(human_dist_df)
    #values_array = human_dist_df.values
    for i in range(human_dist_df.shape[0]):
        human_dist_df.iloc[i, i] = 1e5

    #print(human_dist_df)

    human_mri_xyz_nearest = OrderedDict()
    for r_n in human_88_labels_list:
        human_mri_xyz_nearest[r_n] = {'region': None, 'distance': None}

    for region_name in human_88_labels_list:
        values_list = human_dist_df.loc[region_name].values
        index_v = np.argmin(values_list)
        human_mri_xyz_nearest[region_name]['region'] = human_dist_df.index[index_v]
        value_min = np.min(values_list)
        human_mri_xyz_nearest[region_name]['distance'] = value_min
    #print(human_mri_xyz_nearest)

    # pearson correlation matrix of embedding
    pearson_corr_df = human_embedding_df.corr()
    #print(pearson_corr_df)

    # Get nearst neighbor correlation and not neighbor correlation lists
    neighbor_pearson_list = []
    not_neighbor_pearson_list = []
    for region_n, r_dict in human_mri_xyz_nearest.items():
        neighbor_pearson_list.append(pearson_corr_df.loc[region_n, r_dict['region']])
    #print(neighbor_pearson_list)

    for region_x in human_88_labels_list:
        for region_y in human_88_labels_list:
            if region_x != region_y and human_mri_xyz_nearest[region_x]['region'] != region_y and human_mri_xyz_nearest[region_y]['region'] != region_x:
                not_neighbor_pearson_list.append(pearson_corr_df.loc[region_x, region_y])

    #print(not_neighbor_pearson_list)

    neighbor_pearson_list_type = ["1-nearest neighbor" for i in range(len(neighbor_pearson_list))]
    not_neighbor_pearson_list_type = ["Not near" for i in range(len(not_neighbor_pearson_list))]

    data_dict = {'Pearson Correlation': neighbor_pearson_list+not_neighbor_pearson_list, 'Spatial Relation':neighbor_pearson_list_type+not_neighbor_pearson_list_type}
    data_df = pd.DataFrame.from_dict(data_dict)

    save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/mri_neighbor_random/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    # t-test
    mean_1_near = np.mean(neighbor_pearson_list)
    std_1_near = np.std(neighbor_pearson_list)

    mean_not_near = np.mean(not_neighbor_pearson_list)
    std_not_near = np.std(not_neighbor_pearson_list)

    #print('len of neighbor_pearson_list', len(neighbor_pearson_list))
    #print('len of not_neighbor_pearson_list', len(not_neighbor_pearson_list))

    from scipy import stats

    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(save_path + 't_test_result.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(stats.ttest_ind(
            neighbor_pearson_list,
            not_neighbor_pearson_list,
            equal_var=False,
        ))
        sys.stdout = original_stdout  # Reset the standard output to its original value

    # my_pal = {"homo region": (0/255,149/255,182/255), "random":(178/255, 0/255, 32/255)}
    my_pal = {"1-nearest neighbor": (64 /255, 125/255, 82/255), "Not near": (142/255, 41/255, 97/255)}
    # sns.set_theme(style="whitegrid")
    # tips = sns.load_dataset("tips")

    plt.figure(figsize=(8,6))
    ax = sns.boxplot(x="Spatial Relation", y="Pearson Correlation", data=data_df, order=["1-nearest neighbor", "Not near"], palette = my_pal)
    plt.savefig(save_path + 'mri_xyz_pearson.' + fig_format, format=fig_format)
    plt.show()


    ## plot joint
    dist_pearson_dict = {'Euclidean Distance':[], 'Pearson Correlation':[]}
    human_dist_list = []
    for i in range(len(human_dist_df.index)):
        for j in range(i+1, len(human_dist_df.index)):
            human_dist_list.append(human_dist_df.iloc[i, j])

    pearson_corr_list = []
    for i in range(len(pearson_corr_df.index)):
        for j in range(i + 1, len(pearson_corr_df.index)):
            pearson_corr_list.append(pearson_corr_df.iloc[i, j])

    dist_pearson_dict['Euclidean Distance'] = human_dist_list
    dist_pearson_dict['Pearson Correlation'] = pearson_corr_list

    plt.figure(figsize=(8, 8))
    sns.jointplot(data=pd.DataFrame.from_dict(dist_pearson_dict), x="Euclidean Distance", y="Pearson Correlation", kind="reg", scatter_kws={"s": 3})
    plt.savefig(save_path + 'human_dist_corr_joint.'+fig_format, format=fig_format)
    plt.show()



def spatial_alignment_mouse(cfg):
    '''
       Step 1: Compute average embedding and position of every region in two species, use two dict to store;
       Step 2: Compute distance between regions, select one-nearest neighbor of each region, compute correlations;
       Step 3: plot Boxplot.

    '''
    fig_format = cfg.BrainAlign.fig_format

    #human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
    mouse_67_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)

    adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    #adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')

    ## Human experiment
    # Read ordered labels
    expression_mouse_path = cfg.CAME.path_rawdata1
    adata_mouse = sc.read_h5ad(expression_mouse_path)
    #print(adata_human)

    mouse_embedding_dict = OrderedDict()
    # print(list(human_88_labels['region_name']))
    mouse_67_labels_list = list(mouse_67_labels['region_name'])
    for r_n in mouse_67_labels_list:
        mouse_embedding_dict[r_n] = None


    for region_name in mouse_67_labels_list:
        mean_embedding = np.mean(adata_mouse_embedding[adata_mouse_embedding.obs['region_name'] == region_name].X, axis=0)
        mouse_embedding_dict[region_name] = mean_embedding

    mouse_embedding_df = pd.DataFrame.from_dict(mouse_embedding_dict)

    # compute mean mri coordinates
    mouse_xyz_dict = OrderedDict()
    for r_n in mouse_67_labels_list:
        mouse_xyz_dict[r_n] = {'x_grid': 0, 'y_grid': 0, 'z_grid': 0}
    #print(np.mean(adata_human.obs['mri_voxel_x'].values))
    for region_name in mouse_67_labels_list:
        mean_voxel_x = np.mean(adata_mouse[adata_mouse.obs['region_name'] == region_name].obs['x_grid'].values)
        mouse_xyz_dict[region_name]['x_grid'] = mean_voxel_x
        mean_voxel_y = np.mean(adata_mouse[adata_mouse.obs['region_name'] == region_name].obs['y_grid'].values)
        mouse_xyz_dict[region_name]['y_grid'] = mean_voxel_y
        mean_voxel_z = np.mean(adata_mouse[adata_mouse.obs['region_name'] == region_name].obs['z_grid'].values)
        mouse_xyz_dict[region_name]['z_grid'] = mean_voxel_z

    xyz_df = pd.DataFrame.from_dict(mouse_xyz_dict).T
    #print(human_mri_xyz_df)

    ## Compute distance matrix
    dist_df = pd.DataFrame(squareform(pdist(xyz_df.values)), columns=xyz_df.index, index=xyz_df.index)

    #print(human_dist_df)
    #values_array = human_dist_df.values
    for i in range(dist_df.shape[0]):
        dist_df.iloc[i, i] = 1e7

    #print(human_dist_df)

    xyz_nearest = OrderedDict()
    for r_n in mouse_67_labels_list:
        xyz_nearest[r_n] = {'region': None, 'distance': None}

    for region_name in mouse_67_labels_list:
        values_list = dist_df.loc[region_name].values
        values_list = np.nan_to_num(values_list, nan=10e10)
        index_v = np.argmin(values_list)
        print('values_list:', values_list)

        xyz_nearest[region_name]['region'] = dist_df.index[index_v]
        value_min = np.min(values_list)
        xyz_nearest[region_name]['distance'] = value_min
        print('value_min:', value_min)
    #print(human_mri_xyz_nearest)

    # pearson correlation matrix of embedding
    pearson_corr_df = mouse_embedding_df.corr()
    #print(pearson_corr_df)

    # Get nearst neighbor correlation and not neighbor correlation lists
    neighbor_pearson_list = []
    not_neighbor_pearson_list = []
    for region_n, r_dict in xyz_nearest.items():
        neighbor_pearson_list.append(pearson_corr_df.loc[region_n, r_dict['region']])
    #print(neighbor_pearson_list)

    for region_x in mouse_67_labels_list:
        for region_y in mouse_67_labels_list:
            if region_x != region_y and xyz_nearest[region_x]['region'] != region_y and xyz_nearest[region_y]['region'] != region_x:
                not_neighbor_pearson_list.append(pearson_corr_df.loc[region_x, region_y])

    #print(not_neighbor_pearson_list)
    print('neighbor_pearson_list', neighbor_pearson_list)

    #print('not_neighbor_pearson_list', not_neighbor_pearson_list)

    neighbor_pearson_list_type = ["1-nearest neighbor" for i in range(len(neighbor_pearson_list))]
    not_neighbor_pearson_list_type = ["Not near" for i in range(len(not_neighbor_pearson_list))]

    data_dict = {'Pearson Correlation': neighbor_pearson_list+not_neighbor_pearson_list, 'Spatial Relation':neighbor_pearson_list_type+not_neighbor_pearson_list_type}
    data_df = pd.DataFrame.from_dict(data_dict)

    save_path = cfg.BrainAlign.embeddings_file_path + 'figs/mouse_67/neighbor_random/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    print('len of neighbor_pearson_list', len(neighbor_pearson_list))
    neighbor_pearson_list = [x for x in neighbor_pearson_list if str(x) != 'nan']#np.nan_to_num(neighbor_pearson_list, nan=0)
    print('len of not_neighbor_pearson_list', len(not_neighbor_pearson_list))
    not_neighbor_pearson_list = [x for x in not_neighbor_pearson_list if str(x) != 'nan']

    from scipy import stats

    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(save_path + 't_test_result.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(stats.ttest_ind(
            neighbor_pearson_list,
            not_neighbor_pearson_list,
            equal_var=False,
        ))
        sys.stdout = original_stdout  # Reset the standard output to its original value

    # my_pal = {"homo region": (0/255,149/255,182/255), "random":(178/255, 0/255, 32/255)}
    my_pal = {"1-nearest neighbor": (64 /255, 125/255, 82/255), "Not near": (142/255, 41/255, 97/255)}
    # sns.set_theme(style="whitegrid")
    # tips = sns.load_dataset("tips")

    plt.figure(figsize=(8,6))
    ax = sns.boxplot(x="Spatial Relation", y="Pearson Correlation", data=data_df, order=["1-nearest neighbor", "Not near"], palette = my_pal)
    plt.savefig(save_path + 'xyz_pearson.' + fig_format, format=fig_format)
    plt.show()


    ## plot joint
    dist_pearson_dict = {'Euclidean Distance':[], 'Pearson Correlation':[]}
    mouse_dist_list = []
    for i in range(len(dist_df.index)):
        for j in range(i+1, len(dist_df.index)):
            mouse_dist_list.append(dist_df.iloc[i, j])

    pearson_corr_list = []
    for i in range(len(pearson_corr_df.index)):
        for j in range(i + 1, len(pearson_corr_df.index)):
            pearson_corr_list.append(pearson_corr_df.iloc[i, j])

    dist_pearson_dict['Euclidean Distance'] = mouse_dist_list
    dist_pearson_dict['Pearson Correlation'] = pearson_corr_list

    plt.figure(figsize=(8, 8))
    sns.jointplot(data=pd.DataFrame.from_dict(dist_pearson_dict), x="Euclidean Distance", y="Pearson Correlation", kind="reg", scatter_kws={"s": 3})
    plt.savefig(save_path + 'mouse_dist_corr_joint.'+fig_format, format=fig_format)
    plt.show()




##################################################################################
'''
The following functions mainly do analysis of gene embeddings:
(1). For each specie, we do UMAP and clutering of genes embeddings and see how genes are gathered, e.g., naturally 
those genes who rely on each other in displaying biological function are supposed to be togather. Those clutering of genes 
are called  gene modules.
(2). For two species, we evaluate how those homologous genes in gene modules are correlated, e.g., homologous genes of 
one gene are supposed to be in the same module.   
'''
##################################################################################

#def


##################################################################################
'''
Analysis of expression data sample x gene distributions
'''
##################################################################################
def analysis_expression(cfg):

    return None


from imblearn.over_sampling import RandomOverSampler

def brain_region_classfier(cfg):
    # srrsc
    print('SRRSC, classification of human samples ')
    adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
    data_X = adata_human_embedding.X
    data_Y = adata_human_embedding.obs['region_name']
    print(type(data_X), data_X.shape)
    print(type(data_Y), data_Y.shape)

    # convert brain region name to labels
    human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
    human_88_labels_list = list(human_88_labels['region_name'])
    region_index_dict = {}
    for i in range(len(human_88_labels_list)):
        region_index_dict[human_88_labels_list[i]] = i
    #print(region_index_dict)
    array_Y = np.zeros((data_Y.shape[0]))
    for i in range(data_Y.shape[0]):
        array_Y[i] = region_index_dict[data_Y[i]]

    # delete rare samples to make classification possible
    from collections import Counter
    print(Counter(array_Y))
    # define oversampling strategy
    oversample = RandomOverSampler(sampling_strategy='minority', random_state=29) #sampling_strategy='minority'
    # fit and apply the transform
    X_over, Y_over = oversample.fit_resample(data_X, array_Y)

    print(Counter(Y_over))

    train_X, test_X, train_Y, test_Y = train_test_split(X_over, Y_over, test_size=0.40, random_state=29)

    clf = XGBClassifier(eval_metric='mlogloss',
                      n_estimators=600,
                      max_depth=4,
                      learning_rate=0.1)
    clf.fit(train_X, train_Y)
    test_Y_pred = clf.predict(test_X)
    test_Y_pred_proba = clf.predict_proba(test_X)

    accuracy = accuracy_score(test_Y, test_Y_pred)
    f1_weighted = f1_score(test_Y, test_Y_pred, average='weighted')
    f1_micro= f1_score(test_Y, test_Y_pred, average='micro')
    f1_macro = f1_score(test_Y, test_Y_pred, average='macro')

    print('Test set accuracy = ', accuracy)
    print('Test set f1_weighted = ', f1_weighted)
    print('Test set f1_micro = ', f1_micro)
    print('Test set f1_macro = ', f1_macro)
    #roc_auc = roc_auc_score(test_Y, test_Y_pred_proba, average='weighted', multi_class='ovr')
    #print('Test set roc_auc = ', roc_auc)

    save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/brain_region_classification/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #return test_accuracy, test_recall, test_F1
    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(save_path + 'classification_result.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print('Method: xgboost')
        print('Test set accuracy = ', accuracy)
        print('Test set f1_weighted = ', f1_weighted)
        print('Test set f1_micro = ', f1_micro)
        print('Test set f1_macro = ', f1_macro)
        #print('Test set roc_auc = ', roc_auc)
        sys.stdout = original_stdout  # Reset the standard output to its original value

from collections import Counter
def align_cross_species(cfg):
    '''
    Step 1: Compute average embedding of every region in two species, use two dict to store;
    Step 2: Compute similarity matrix, use np array to store;
    Step 3: Heatmap.
    '''
    # Read ordered labels
    fig_format = cfg.BrainAlign.fig_format

    human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
    mouse_67_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)

    adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')

    #region_list =
    #print(Counter(adata_mouse_embedding.obs['region_name']).keys)

    #
    human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)

    save_path = cfg.BrainAlign.embeddings_file_path + 'figs/human_88/align_cross_species/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Step 1
    human_embedding_dict = OrderedDict()
    #print(list(human_88_labels['region_name']))
    human_88_labels_list = list(human_88_labels['region_name'])
    for r_n in human_88_labels_list:
        human_embedding_dict[r_n] = None


    mouse_embedding_dict = OrderedDict()
    mouse_67_labels_list = list(mouse_67_labels['region_name'])
    for r_n in mouse_67_labels_list:
        mouse_embedding_dict[r_n] = None


    for region_name in human_88_labels_list:
        mean_embedding = np.mean(adata_human_embedding[adata_human_embedding.obs['region_name'] == region_name].X, axis=0)
        human_embedding_dict[region_name] = mean_embedding

    human_embedding_df = pd.DataFrame.from_dict(human_embedding_dict)


    for region_name in mouse_67_labels_list:
        mean_embedding = np.mean(adata_mouse_embedding[adata_mouse_embedding.obs['region_name'] == region_name].X, axis=0)
        mouse_embedding_dict[region_name] = mean_embedding
        # print(mean_embedding.shape)
    mouse_embedding_df = pd.DataFrame.from_dict(mouse_embedding_dict)

    result = pd.concat([human_embedding_df, mouse_embedding_df], axis=1).corr()
    Var_Corr = result[mouse_embedding_df.columns].loc[human_embedding_df.columns]
    #print(Var_Corr)
    #print(type(Var_Corr))
    # plot the heatmap and annotation on it
    #plt.figure()
    #color_map = sns.color_palette("coolwarm", as_cmap=True)
    color_map = sns.color_palette("vlag", as_cmap=True)
    #color_map = sns.color_palette("rocket_r", as_cmap=

    fig, ax = plt.subplots(figsize=(22, 25))
    sns.heatmap(Var_Corr, xticklabels=mouse_embedding_df.columns, yticklabels=human_embedding_df.columns, annot=False, ax=ax)
    plt.savefig(save_path + 'human_mouse_sim_ordered.' + fig_format, format=fig_format)
    #plt.show()
    #color_map = sns.color_palette("coolwarm", as_cmap=True)
    sns.clustermap(Var_Corr) #cmap=color_map, center=0.6
    plt.savefig(save_path + 'region_Hiercluster_human_mouse.' + fig_format, format=fig_format)
    #plt.show()



from scGeneFit.functions import *
def identify_markergene(cfg):
    np.random.seed(29)


    return 0




if __name__ == '__main__':
    cfg = heco_config._C
    print('Analysis of srrsc embeddings.')

