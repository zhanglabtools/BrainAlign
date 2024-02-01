# -- coding: utf-8 --
# @Time : 2022/10/15 16:05
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : process.py

import pickle
import numpy as np
import scanpy as sc, anndata as ad
import scipy.sparse as sp
import pandas as pd
import sys

import os
sys.path.append('../code/')
#from BrainAlign.code.utils import set_params
from BrainAlign.code.utils.logger import setup_logger
from BrainAlign.brain_analysis.configs import heco_config
from BrainAlign.brain_analysis.data_utils import threshold_top, threshold_quantile, threshold_std, train_test_val_split, corr2_coeff
from BrainAlign.came.utils import preprocess
from functools import reduce
from sklearn.neighbors import kneighbors_graph

def transform_pca_embedding(Mat, M_embedding):
    '''
    input:Mat, sample number * gene number
    output: gene number * embedding dimension
    M_embedding:
    Add one in the  denominator to avoid infinite values
    '''
    divide_mat = np.sum(Mat, axis=0)
    res_mat = (Mat.dot(M_embedding).T / divide_mat ).T
    return res_mat

def transform_pca_embedding_np(Mat, M_embedding):
    '''
    input:Mat, sample number * gene number
    M_embedding:
    Add one in the  denominator to avoid infinite values
    '''
    Mat = sp.csr_matrix(np.nan_to_num(Mat), shape=Mat.shape)
    M_embedding = sp.csc_matrix(np.nan_to_num(M_embedding), shape=M_embedding.shape)
    divide_mat = np.reshape(np.sum(Mat.toarray(), axis=0), (-1, 1))
    for i in range(divide_mat.shape[0]):
        if divide_mat[i,0] == 0:
            divide_mat[i, 0] = 1

    res_mat = Mat.T.dot(M_embedding).toarray() / divide_mat
    return res_mat


def highly_variable_HVGs(_adata, cfg, **hvg_kwds):
    sc.pp.highly_variable_genes(
        _adata, batch_key=cfg.BrainAlign.DEG_batch_key, n_top_genes=cfg.BrainAlign.DEG_n_top_genes, **hvg_kwds)
    _adata = _adata[:, _adata.var['highly_variable']].copy()
    return _adata

def select_deg(cfg):

    print('loading process')
    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)
    #print(datapair)
    #print(datapair['obs_dfs'][0].shape)
    #print(datapair['obs_dfs'][1].shape)

    mouse_sample_list = list(datapair['obs_dfs'][0].index)
    # print(mouse_sample_list)
    print(len(mouse_sample_list))

    human_sample_list = list(datapair['obs_dfs'][1].index)
    # print(human_sample_list)
    print(len(human_sample_list))

    mouse_gene_list = datapair['varnames_node'][0]
    human_gene_list = datapair['varnames_node'][1]

    path_rawdata_mouse = cfg.CAME.path_rawdata1
    adata_mouse = sc.read_h5ad(path_rawdata_mouse)
    adata_mouse = highly_variable_HVGs(adata_mouse, cfg)


    # Check if the order of sample is messed
    diff = False
    for k, v in zip(mouse_sample_list, list(adata_mouse.obs_names)):
        if k != v:
            print("The is difference!")
            diff = True
    if diff == False:
        print('They are identical!')

    path_rawdata_human = cfg.CAME.path_rawdata2
    adata_human = sc.read_h5ad(path_rawdata_human)
    adata_human = highly_variable_HVGs(adata_human, cfg)

    diff = False
    for k, v in zip(human_sample_list, list(adata_human.obs_names)):
        if k != v:
            print("The is difference!")
            diff = True
    if diff == False:
        print('They are identical!')


def load_sample_embedding(mouse_sample_list, human_sample_list, cfg):
    # No label order version
    '''
    Step 1: Compute average embedding of every region in two species, use two dict to store;
    Step 2: Compute similarity matrix, use np array to store;
    Step 3: Heatmap.
    '''
    # Read ordered labels
    expression_human_path = cfg.CAME.path_rawdata2
    adata_human = sc.read_h5ad(expression_human_path)
    #print(adata_human)

    expression_mouse_path = cfg.CAME.path_rawdata1
    adata_mouse = sc.read_h5ad(expression_mouse_path)
    #print(adata_mouse)

    path_human = cfg.CAME.ROOT + 'adt_hidden_cell.h5ad'

    adata_human_mouse = sc.read_h5ad(path_human)
    # ---------------------------human------------------------------------
    # assign embedding to original adata
    adata_human_mouse.obs_names = adata_human_mouse.obs['original_name']

    adata_human_embedding = adata_human_mouse[adata_human.obs_names]
    adata_human_embedding.obs['region_name'] = adata_human.obs['region_name']
    for sample_id in adata_human_embedding.obs_names:
        adata_human_embedding[sample_id].obs['region_name'] = adata_human[sample_id].obs['region_name']

    if cfg.BrainAlign.NODE_TYPE_NUM == 4 or cfg.BrainAlign.NODE_TYPE_NUM == 3:
        sp.save_npz(cfg.BrainAlign.DATA_PATH + 'v_feat.npz', sp.csr_matrix(adata_human_embedding.X))

    diff = False
    for k,v in zip(human_sample_list, list(adata_human_embedding.obs_names)):
        if k != v:
            print("The is difference!")
            diff = True
    if diff == False:
        print('They are identical!')

    ## ---------------------------------------------------------------------mouse------------------------
    adata_mouse_embedding = adata_human_mouse[adata_mouse.obs_names]
    adata_mouse_embedding.obs['region_name'] = adata_mouse.obs['region_name']
    for sample_id in adata_mouse_embedding.obs_names:
        adata_mouse_embedding[sample_id].obs['region_name'] = adata_mouse[sample_id].obs['region_name']

    if cfg.BrainAlign.NODE_TYPE_NUM == 4 or cfg.BrainAlign.NODE_TYPE_NUM == 3:
        sp.save_npz(cfg.BrainAlign.DATA_PATH + 's_feat.npz', sp.csr_matrix(adata_mouse_embedding.X))

    diff = False
    for k, v in zip(mouse_sample_list, list(adata_mouse_embedding.obs_names)):
        if k != v:
            print("The is difference!")
            diff = True
    if diff == False:
        print('They are identical!')

    if cfg.BrainAlign.NODE_TYPE_NUM == 2:
        embedding_s = np.concatenate((adata_mouse_embedding.X, adata_human_embedding.X), axis=0)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + 's_feat.npz', sp.csr_matrix(embedding_s))


def load_gene_embedding(mouse_gene_list, human_gene_list, cfg, logger):
    path_gene_mouse = cfg.CAME.ROOT + 'adt_hidden_gene1.h5ad'
    adata_gene_mouse = sc.read_h5ad(path_gene_mouse)
    logger.debug(adata_gene_mouse)

    if cfg.BrainAlign.NODE_TYPE_NUM == 4:
        sp.save_npz(cfg.BrainAlign.DATA_PATH + 'm_feat.npz', sp.csr_matrix(adata_gene_mouse.X))

    diff = False
    for k, v in zip(mouse_gene_list, list(adata_gene_mouse.obs_names)):
        if k != v:
            logger.debug("The is difference between gene lists!")
            diff = True
    if diff == False:
        logger.debug('Gene lists are identical!')


    path_gene_human = cfg.CAME.ROOT + 'adt_hidden_gene2.h5ad'
    adata_gene_human = sc.read_h5ad(path_gene_human)

    if cfg.BrainAlign.NODE_TYPE_NUM == 4:
        sp.save_npz(cfg.BrainAlign.DATA_PATH + 'h_feat.npz', sp.csr_matrix(adata_gene_human.X))


    diff = False
    for k, v in zip(human_gene_list, list(adata_gene_human.obs_names)):
        if k != v:
            logger.debug("The is difference between gene lists!")
            diff = True
    if diff == False:
        logger.debug('Gene lists are identical!')

    if cfg.BrainAlign.NODE_TYPE_NUM == 2:
        embedding_m = np.concatenate((adata_gene_mouse.X, adata_gene_human.X), axis=0)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + 'm_feat.npz', sp.csr_matrix(embedding_m))
    if cfg.BrainAlign.NODE_TYPE_NUM == 3:
        embedding_m = np.concatenate((adata_gene_mouse.X, adata_gene_human.X), axis=0)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + 'g_feat.npz', sp.csr_matrix(embedding_m))


def load_embedding(cfg, logger):
    '''
        1. Get graph G_mouse, G_human from CAME init data;
        2. Filter original expression data via G_mouse, G_human, acquire M_mouse, M_human;
        3.
    '''
    logger.info('Loading embeddings...')
    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)
    #logger.debug(datapair)
    #print(datapair['obs_dfs'][0].shape)
    #print(datapair['obs_dfs'][1].shape)
    mouse_sample_list = list(datapair['obs_dfs'][0].index)
    # print(mouse_sample_list)
    #print(len(mouse_sample_list))

    human_sample_list = list(datapair['obs_dfs'][1].index)
    # print(human_sample_list)
    #print(len(human_sample_list))
    mouse_gene_list = datapair['varnames_node'][0]
    human_gene_list = datapair['varnames_node'][1]

    if cfg.BrainAlign.embedding_type == 'pca':
        logger.info('Use PCA embedding.')
        logger.info('Processing Mouse PCA embedding...')
        path_rawdata_1 = cfg.CAME.path_rawdata1
        adata_1 = sc.read_h5ad(path_rawdata_1)
        # update adata_1 according to came network
        adata_1 = adata_1[:, mouse_gene_list].copy()

        print('Mouse brain region sample count: ', adata_1.obs['region_name'].value_counts())
        #
        if cfg.BrainAlign.mouse_dataset != '2020sa':
            if cfg.BrainAlign.normalize_before_pca != None:
                adata_1 = normalize_before_pruning(adata_1, method=cfg.BrainAlign.normalize_before_pca,
                                                   target_sum=cfg.BrainAlign.normalize_before_pca_target_sum,
                                                   force_return=True)

        sc.settings.verbosity = 3
        sc.tl.pca(adata_1, svd_solver='arpack', n_comps=cfg.BrainAlign.embedding_pca_dim)
        # extract pca coordinates
        X_pca_1 = adata_1.obsm['X_pca']
        logger.debug('X_pca_1 type: {}'.format(type(X_pca_1)))
        diff = False
        for k, v in zip(mouse_sample_list, list(adata_1.obs_names)):
            if k != v:
                logger.debug("There is difference between sample list!")
                diff = True
        if diff == False:
            logger.debug('Sample lists are identical!')
        diff = False
        for k, v in zip(mouse_gene_list, list(adata_1.var_names)):
            if k != v:
                logger.debug("There is difference between gene list!")
                diff = True
        if diff == False:
            logger.debug('Gene lists are identical!')

        if cfg.BrainAlign.NODE_TYPE_NUM == 4 or cfg.BrainAlign.NODE_TYPE_NUM == 3:
            sp.save_npz(cfg.BrainAlign.DATA_PATH + 's_feat.npz', sp.csr_matrix(X_pca_1))

        adata_1_genes = adata_1.transpose()
        sc.tl.pca(adata_1_genes, svd_solver='arpack', n_comps=cfg.BrainAlign.embedding_pca_dim)
        # extract pca coordinates
        X_pca_1_sample = adata_1_genes.obsm['X_pca']
        if cfg.BrainAlign.NODE_TYPE_NUM == 4:
            sp.save_npz(cfg.BrainAlign.DATA_PATH + 'm_feat.npz', sp.csr_matrix(X_pca_1_sample))

        ################## human pca################
        logger.info('Processing Human PCA embedding')
        path_rawdata_2 = cfg.CAME.path_rawdata2
        adata_2 = sc.read_h5ad(path_rawdata_2)
        # update adata_1 according to came network
        adata_2 = adata_2[:, human_gene_list].copy()

        print('Human brain region sample count: ', adata_2.obs['region_name'].value_counts())

        if cfg.BrainAlign.normalize_before_pca != None:
            adata_2 = normalize_before_pruning(adata_2, method = cfg.BrainAlign.normalize_before_pca, target_sum=cfg.BrainAlign.normalize_before_pca_target_sum, force_return=True)

        sc.settings.verbosity = 3
        sc.tl.pca(adata_2, svd_solver='arpack', n_comps=cfg.BrainAlign.embedding_pca_dim)
        # extract pca coordinates
        X_pca_2 = adata_2.obsm['X_pca']
        logger.debug('X_pca_2 type: {}'.format(type(X_pca_2)))
        diff = False
        for k, v in zip(human_sample_list, list(adata_2.obs_names)):
            if k != v:
                logger.debug("There is difference between sample list!")
                diff = True
        if diff == False:
            logger.debug('Sample lists are identical!')
        diff = False
        for k, v in zip(human_gene_list, list(adata_2.var_names)):
            if k != v:
                logger.debug("There is difference between gene list!")
                diff = True
        if diff == False:
            logger.debug('Gene lists are identical!')
        if cfg.BrainAlign.NODE_TYPE_NUM == 4 or cfg.BrainAlign.NODE_TYPE_NUM == 3:
            sp.save_npz(cfg.BrainAlign.DATA_PATH + 'v_feat.npz', sp.csr_matrix(X_pca_2))

        adata_2_genes = adata_2.transpose()
        sc.tl.pca(adata_2_genes, svd_solver='arpack', n_comps=cfg.BrainAlign.embedding_pca_dim)
        # extract pca coordinates
        X_pca_2_sample = adata_2_genes.obsm['X_pca']
        if cfg.BrainAlign.NODE_TYPE_NUM == 4:
            sp.save_npz(cfg.BrainAlign.DATA_PATH + 'h_feat.npz', sp.csr_matrix(X_pca_2_sample))

        if cfg.BrainAlign.NODE_TYPE_NUM == 2:
            X_pca_s = np.concatenate((X_pca_1, X_pca_2), axis=0)
            logger.debug('Sample init feature dimension = {}'.format(X_pca_s.shape))
            sp.save_npz(cfg.BrainAlign.DATA_PATH + 's_feat.npz', sp.csr_matrix(X_pca_s))

            X_pca_m = np.concatenate((X_pca_1_sample, X_pca_2_sample), axis=0)
            sp.save_npz(cfg.BrainAlign.DATA_PATH + 'm_feat.npz', sp.csr_matrix(X_pca_m))

        if cfg.BrainAlign.NODE_TYPE_NUM == 3:
            X_pca_g = np.concatenate((X_pca_1_sample, X_pca_2_sample), axis=0)
            sp.save_npz(cfg.BrainAlign.DATA_PATH + 'g_feat.npz', sp.csr_matrix(X_pca_g))

    elif cfg.BrainAlign.embedding_type == 'came':
        load_sample_embedding(mouse_sample_list, human_sample_list, cfg)
        load_gene_embedding(mouse_gene_list, human_gene_list, cfg, logger)

    elif cfg.BrainAlign.embedding_type == 'pca_sample_pass_gene':
        logger.info('Use PCA embedding.')
        logger.info('Processing Mouse PCA embedding...')
        path_rawdata_1 = cfg.CAME.path_rawdata1
        adata_1 = sc.read_h5ad(path_rawdata_1)
        # update adata_1 according to came network
        adata_1 = adata_1[:, mouse_gene_list].copy()

        print('Mouse brain region sample count: ', adata_1.obs['region_name'].value_counts())
        #
        if cfg.BrainAlign.mouse_dataset != '2020sa':
            if cfg.BrainAlign.normalize_before_pca != None:
                adata_1 = normalize_before_pruning(adata_1, method=cfg.BrainAlign.normalize_before_pca,
                                                   target_sum=cfg.BrainAlign.normalize_before_pca_target_sum, force_return=True)

        sc.settings.verbosity = 3
        sc.tl.pca(adata_1, svd_solver='arpack', n_comps=cfg.BrainAlign.embedding_pca_dim)
        # extract pca coordinates
        X_pca_1 = adata_1.obsm['X_pca']
        logger.debug('X_pca_1 type: {}'.format(type(X_pca_1)))
        diff = False
        for k, v in zip(mouse_sample_list, list(adata_1.obs_names)):
            if k != v:
                logger.debug("There is difference between sample list!")
                diff = True
        if diff == False:
            logger.debug('Sample lists are identical!')
        diff = False
        for k, v in zip(mouse_gene_list, list(adata_1.var_names)):
            if k != v:
                logger.debug("There is difference between gene list!")
                diff = True
        if diff == False:
            logger.debug('Gene lists are identical!')

        if cfg.BrainAlign.NODE_TYPE_NUM == 4 or cfg.BrainAlign.NODE_TYPE_NUM == 3:
            sp.save_npz(cfg.BrainAlign.DATA_PATH + 's_feat.npz', sp.csr_matrix(X_pca_1))


        #adj_matrix_mouse = datapair['ov_adjs'][0].toarray()
        X_pca_1_gene = transform_pca_embedding_np(adata_1.X.toarray(), X_pca_1)


        #adj_matrix_human = datapair['ov_adjs'][1].toarray()


        if cfg.BrainAlign.NODE_TYPE_NUM == 4:
            sp.save_npz(cfg.BrainAlign.DATA_PATH + 'm_feat.npz', sp.csr_matrix(X_pca_1_gene))

        ################## human pca################
        logger.info('Processing Human PCA embedding')
        path_rawdata_2 = cfg.CAME.path_rawdata2
        adata_2 = sc.read_h5ad(path_rawdata_2)
        # update adata_1 according to came network
        adata_2 = adata_2[:, human_gene_list].copy()

        print('Human brain region sample count: ', adata_2.obs['region_name'].value_counts())

        if cfg.BrainAlign.normalize_before_pca != None:
            adata_2 = normalize_before_pruning(adata_2, method=cfg.BrainAlign.normalize_before_pca,
                                               target_sum=cfg.BrainAlign.normalize_before_pca_target_sum, force_return=True)

        sc.settings.verbosity = 3
        sc.tl.pca(adata_2, svd_solver='arpack', n_comps=cfg.BrainAlign.embedding_pca_dim)
        # extract pca coordinates
        X_pca_2 = adata_2.obsm['X_pca']
        logger.debug('X_pca_2 type: {}'.format(type(X_pca_2)))
        diff = False
        for k, v in zip(human_sample_list, list(adata_2.obs_names)):
            if k != v:
                logger.debug("There is difference between sample list!")
                diff = True
        if diff == False:
            logger.debug('Sample lists are identical!')
        diff = False
        for k, v in zip(human_gene_list, list(adata_2.var_names)):
            if k != v:
                logger.debug("There is difference between gene list!")
                diff = True
        if diff == False:
            logger.debug('Gene lists are identical!')
        if cfg.BrainAlign.NODE_TYPE_NUM == 4 or cfg.BrainAlign.NODE_TYPE_NUM == 3:
            sp.save_npz(cfg.BrainAlign.DATA_PATH + 'v_feat.npz', sp.csr_matrix(X_pca_2))

        X_pca_2_gene = transform_pca_embedding_np(adata_2.X.toarray(), X_pca_2)

        if cfg.BrainAlign.NODE_TYPE_NUM == 4:
            sp.save_npz(cfg.BrainAlign.DATA_PATH + 'h_feat.npz', sp.csr_matrix(X_pca_2_gene))

        if cfg.BrainAlign.NODE_TYPE_NUM == 2:
            X_pca_s = np.concatenate((X_pca_1, X_pca_2), axis=0)
            logger.debug('Sample init feature dimension = {}'.format(X_pca_s.shape))
            sp.save_npz(cfg.BrainAlign.DATA_PATH + 's_feat.npz', sp.csr_matrix(X_pca_s))

            X_pca_m = np.concatenate((X_pca_1_gene, X_pca_2_gene), axis=0)
            sp.save_npz(cfg.BrainAlign.DATA_PATH + 'm_feat.npz', sp.csr_matrix(X_pca_m))
        if cfg.BrainAlign.NODE_TYPE_NUM == 3:
            X_pca_g = np.concatenate((X_pca_1_gene, X_pca_2_gene), axis=0)
            sp.save_npz(cfg.BrainAlign.DATA_PATH + 'g_feat.npz', sp.csr_matrix(X_pca_g))


def get_heco_input(cfg):


    logger = setup_logger("Running BrainAlign and analysis", cfg.BrainAlign.DATA_PATH, if_train=None)
    logger.info("Running with config:\n{}".format(cfg))

    load_embedding(cfg, logger)


    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)
    #print(datapair)



    if cfg.BrainAlign.NODE_TYPE_NUM == 4:
        print(len(datapair['varnames_node'][0]))
        np.save(cfg.BrainAlign.DATA_PATH + 'mouse_gene_names.npy', datapair['varnames_node'][0])
        print(len(datapair['varnames_node'][1]))
        np.save(cfg.BrainAlign.DATA_PATH + 'human_gene_names.npy', datapair['varnames_node'][1])

        if cfg.BrainAlign.normalize_scale == True:
            adata_sm = ad.AnnData(sp.coo_matrix(datapair['ov_adjs'][0].toarray() - datapair['ov_adjs'][0].min().min()).tocsr())
        else:
            adata_sm = ad.AnnData(datapair['ov_adjs'][0])

        #adata_sm = ad.AnnData(datapair['ov_adjs'][0])
        adata_sm.obs_names = list(datapair['obs_dfs'][0].index)
        adata_sm.var_names = datapair['varnames_node'][0]
        adata_sm = normalize_before_pruning(adata_sm, method=cfg.BrainAlign.normalize_before_pruning_method_1, target_sum=cfg.BrainAlign.pruning_target_sum_1, force_return=True)
        sm_ = adata_sm.X.toarray()

        if cfg.BrainAlign.normalize_scale == True:
            vh_X = datapair['ov_adjs'][1].toarray()
            vh_X = vh_X - vh_X.min()
            vh_X = vh_X * (adata_sm.X.max().max() - adata_sm.X.min().min()) / vh_X.max().max()
            vh_X = sp.coo_matrix(vh_X).tocsr()
            adata_vh = ad.AnnData(vh_X)
        else:
            adata_vh = ad.AnnData(datapair['ov_adjs'][1])

        #adata_vh = ad.AnnData(datapair['ov_adjs'][1])
        adata_vh.obs_names = list(datapair['obs_dfs'][1].index)
        adata_vh.var_names = datapair['varnames_node'][1]
        #if cfg.BrainAlign.normalize_before_pruning_method == None:
        #    cfg.BrainAlign.normalize_before_pruning_method = 'default'
        adata_vh = normalize_before_pruning(adata_vh, method=cfg.BrainAlign.normalize_before_pruning_method_2, target_sum=cfg.BrainAlign.pruning_target_sum_2, force_return=True)
        vh_ = adata_vh.X.toarray()

        #sm_ = datapair['ov_adjs'][0].toarray()
        if cfg.BrainAlign.if_threshold:
            if cfg.BrainAlign.pruning_method == 'top':
                sm_ = threshold_top(sm_, percent=cfg.BrainAlign.sm_gene_top) + threshold_top(sm_.T, percent=cfg.BrainAlign.sm_sample_top).T
            elif cfg.BrainAlign.pruning_method == 'std':
                #threshold_std(sm_, std_times=cfg.BrainAlign.pruning_std_times_sm) + \
                sm_ = threshold_top(sm_, percent=cfg.BrainAlign.sm_gene_top) + \
                      threshold_std(sm_, std_times=cfg.BrainAlign.pruning_std_times_sm) + \
                      threshold_top(sm_.T, percent=cfg.BrainAlign.sm_sample_top).T
                #sm_ = threshold_std(sm_, std_times=cfg.BrainAlign.pruning_std_times_sm) + threshold_std(sm_.T,
                #                                                                                  std_times=cfg.BrainAlign.pruning_std_times_sm).T
        else:
            sm_ = sm_ > 0
        print(sm_)
        logger.debug(f'sm_: {sm_.shape}')

        #vh_ = datapair['ov_adjs'][1].toarray()
        if cfg.BrainAlign.if_threshold:
            if cfg.BrainAlign.pruning_method == 'top':
                vh_ = threshold_top(vh_, percent=cfg.BrainAlign.vh_gene_top) + threshold_top(vh_.T,
                                                                                       percent=cfg.BrainAlign.vh_sample_top).T
            elif cfg.BrainAlign.pruning_method == 'std':
                # threshold_std(vh_, std_times=cfg.BrainAlign.pruning_std_times_vh) + \
                vh_ = threshold_top(vh_, percent=cfg.BrainAlign.vh_gene_top) + \
                      threshold_std(vh_, std_times=cfg.BrainAlign.pruning_std_times_vh) + \
                      threshold_top(vh_.T, percent=cfg.BrainAlign.vh_sample_top).T
                #vh_ = threshold_std(vh_, std_times=cfg.BrainAlign.pruning_std_times_vh) + threshold_std(vh_.T,
                #                                                                                  std_times=cfg.BrainAlign.pruning_std_times_vh).T
        else:
            vh_ = vh_ > 0
        logger.debug(f'vh_: {vh_.shape}')
        mh_ = datapair['vv_adj'].toarray()[0:cfg.BrainAlign.M, cfg.BrainAlign.M:] > 0
        logger.debug(f'mh_: {mh_.shape}')

        m_feat = sp.load_npz(cfg.BrainAlign.DATA_PATH + 'm_feat.npz').toarray()
        h_feat = sp.load_npz(cfg.BrainAlign.DATA_PATH + 'h_feat.npz').toarray()
        corr_mh = corr2_coeff(m_feat, h_feat)
        mh_argmax = np.argmax(corr_mh, axis=1)
        for row_i in range(np.shape(corr_mh)[0]):
            if np.sum(mh_[row_i, :]) == 0:
                mh_[row_i, mh_argmax[row_i]] = True
        mh_argmax_col = np.argmax(corr_mh, axis=0)
        for col_j in range(np.shape(corr_mh)[1]):
            if np.sum(mh_[:, col_j]) == 0:
                mh_[mh_argmax_col[col_j], col_j] = True

        ss_ = datapair['oo_adjs'].toarray()[0:cfg.BrainAlign.S, 0:cfg.BrainAlign.S] > 0
        logger.debug(f'ss_: {ss_.shape}')
        # print('ss_ sum', np.sum(ss_)) # == 0
        vv_ = datapair['oo_adjs'].toarray()[cfg.BrainAlign.S:, cfg.BrainAlign.S:] > 0
        logger.debug(f'vv_: {vv_.shape}')


        ## Generate meta-path for target node S
        sms = np.matmul(sm_, sm_.T) > 0
        #print(sms)
        logger.debug('How sparse is sms:%s', np.sum(sms) / (cfg.BrainAlign.S * cfg.BrainAlign.S))
        sms = sp.coo_matrix(sms)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + "sms.npz", sms)
        del sms


        smh = np.matmul(sm_, mh_) > 0
        smhv = np.matmul(smh, vh_.T) > 0
        smhvhms = np.matmul(smhv, smhv.T) > 0
        #print(smhvhms)
        logger.debug('How sparse is smhvhms:%s', np.sum(smhvhms) / (cfg.BrainAlign.S * cfg.BrainAlign.S))
        smhvhms = sp.coo_matrix(smhvhms)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + "smhvhms.npz", smhvhms)
        del smhvhms

        smhvv = np.matmul(smhv, vv_) > 0
        smhvvhms = np.matmul(smhv, smhvv.T) > 0
        #print(smhvvhms)
        logger.debug('How sparse is smhvvhms:%s', np.sum(smhvvhms) / (cfg.BrainAlign.S * cfg.BrainAlign.S))
        smhvvhms = sp.coo_matrix(smhvvhms)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + "smhvvhms.npz", smhvvhms)
        del smhvvhms

        del smh
        del smhv
        del smhvv

        ## Generate meta-path for target node M
        msm = np.matmul(sm_.T, sm_) > 0
        #print(sms)
        logger.info('How sparse is msm:%s', np.sum(msm) / (cfg.BrainAlign.M * cfg.BrainAlign.M))
        msm = sp.coo_matrix(msm)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + "msm.npz", msm)
        del msm

        mss = np.matmul(sm_.T, ss_) > 0
        mssm = np.matmul(mss, sm_) > 0
        logger.debug('How sparse is mssm:%s', np.sum(mssm) / (cfg.BrainAlign.M * cfg.BrainAlign.M))
        mssm = sp.coo_matrix(mssm)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + "mssm.npz", mssm)
        del mssm

        mhv = np.matmul(mh_, vh_.T) > 0
        mhvh = np.matmul(mhv, vh_) > 0
        mhvhm = np.matmul(mhvh, mh_.T) > 0
        logger.debug('How sparse is mhvhm:%f', np.sum(mhvhm) / (cfg.BrainAlign.M * cfg.BrainAlign.M))
        mhvhm = sp.coo_matrix(mhvhm)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + "mhvhm.npz", mhvhm)
        del mhvhm

        mhvv = np.matmul(mhv, vv_) > 0
        mhvvh = np.matmul(mhvv, vh_) > 0
        mhvvhm = np.matmul(mhvvh, mh_.T) > 0
        logger.debug('How sparse is mhvvhm:%f', np.sum(mhvvhm) / (cfg.BrainAlign.M * cfg.BrainAlign.M))
        mhvvhm = sp.coo_matrix(mhvvhm)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + "mhvvhm.npz", mhvvhm)
        del mhvvhm
        del mss, mhv, mhvv, mhvvh

        ## Generate meta-path for target node H
        hvh = np.matmul(vh_.T, vh_) > 0
        logger.debug('How sparse is hvh:%f', np.sum(hvh) / (cfg.BrainAlign.H * cfg.BrainAlign.H))
        hvh = sp.coo_matrix(hvh)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + "hvh.npz", hvh)
        del hvh

        hvv = np.matmul(vh_.T, vv_) > 0
        hvvh = np.matmul(hvv, vh_) > 0
        logger.debug('How sparse is hvvh:%f', np.sum(hvvh) / (cfg.BrainAlign.H * cfg.BrainAlign.H))
        hvvh = sp.coo_matrix(hvvh)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + "hvvh.npz", hvvh)
        del hvvh

        hms = np.matmul(mh_.T, sm_.T) > 0
        hmsm = np.matmul(hms, sm_) > 0
        hmsmh = np.matmul(hmsm, mh_) > 0
        logger.debug('How sparse is hmsmh:%f', np.sum(hmsmh) / (cfg.BrainAlign.H * cfg.BrainAlign.H))
        hmsmh = sp.coo_matrix(hmsmh)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + "hmsmh.npz", hmsmh)
        del hmsmh

        hmss = np.matmul(hms, ss_) > 0
        hmssm = np.matmul(hmss, sm_) > 0
        hmssmh = np.matmul(hmssm, mh_) > 0
        logger.debug('How sparse is hmssmh:%f', np.sum(hmssmh) / (cfg.BrainAlign.H * cfg.BrainAlign.H))
        hmssmh = sp.coo_matrix(hmssmh)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + "hmssmh.npz", hmssmh)
        del hmssmh
        del hvv, hms, hmss, hmssm

        ## Generate meta-path for target node V
        vhv = np.matmul(vh_, vh_.T) > 0
        # print(sms)
        logger.debug('How sparse is vhv:%f', np.sum(vhv) / (cfg.BrainAlign.V * cfg.BrainAlign.V))
        vhv = sp.coo_matrix(vhv)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + "vhv.npz", vhv)
        del vhv

        vhm = np.matmul(vh_, mh_.T) > 0
        vhms = np.matmul(vhm, sm_.T) > 0
        vhmsmhv = np.matmul(vhms, vhms.T) > 0
        logger.debug('How sparse is vhmsmhv:%f', np.sum(vhmsmhv) / (cfg.BrainAlign.V * cfg.BrainAlign.V))
        vhmsmhv = sp.coo_matrix(vhmsmhv)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + "vhmsmhv.npz", vhmsmhv)
        del vhmsmhv

        vhmss = np.matmul(vhms, ss_) > 0
        vhmssmhv = np.matmul(vhms, vhmss.T) > 0
        logger.debug('How sparse is vhmssmhv:%f', np.sum(vhmssmhv) / (cfg.BrainAlign.V * cfg.BrainAlign.V))
        vhmssmhv = sp.coo_matrix(vhmssmhv)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + "vhmssmhv.npz", vhmssmhv)
        del vhmssmhv

        del vhm
        del vhms
        del vhmss


        ## neibor number: S-{M}, M-{S, H}, H-{M, V}, V-{H}
        # Get S neighborhoods
        get_neighbor(sm_, cfg, logger, 's_nei_m', cfg.BrainAlign.S, cfg.BrainAlign.M)

        # Get M neighborhoods
        get_neighbor(sm_.T, cfg, logger, 'm_nei_s', cfg.BrainAlign.M, cfg.BrainAlign.S)
        get_neighbor(mh_, cfg, logger, 'm_nei_h', cfg.BrainAlign.M, cfg.BrainAlign.H)

        # Get M neighborhoods
        get_neighbor(vh_.T, cfg, logger, 'h_nei_v', cfg.BrainAlign.H, cfg.BrainAlign.V)
        get_neighbor(mh_.T, cfg, logger, 'h_nei_m', cfg.BrainAlign.H, cfg.BrainAlign.M)

        # Get V neighborhoods
        get_neighbor(vh_, cfg, logger, 'v_nei_h', cfg.BrainAlign.V, cfg.BrainAlign.H)


        ## Generate positive samples for target node S
        meta_path_list = [cfg.BrainAlign.DATA_PATH + x for x in ['sms.npz', 'smhvhms.npz', 'smhvvhms.npz']]
        get_positive_sample(cfg, meta_path_list, target_node='S')

        ## Generate positive samples for target node M
        meta_path_list = [cfg.BrainAlign.DATA_PATH + x for x in ['msm.npz', 'mssm.npz', 'mhvhm.npz', 'mhvvhm.npz']]
        get_positive_sample(cfg, meta_path_list, target_node='M')

        ## Generate positive samples for target node H
        meta_path_list = [cfg.BrainAlign.DATA_PATH + x for x in ['hvh.npz', 'hvvh.npz', 'hmsmh.npz', 'hmssmh.npz']]
        get_positive_sample(cfg, meta_path_list, target_node='H')

        ## Generate positive samples for target node V
        meta_path_list = [cfg.BrainAlign.DATA_PATH + x for x in ['vhv.npz', 'vhmsmhv.npz', 'vhmssmhv.npz']]
        get_positive_sample(cfg, meta_path_list, target_node='V')

        ratio_list = cfg.BrainAlign_args.ratio
        multiple = 20
        test_val_num = int(cfg.BrainAlign.S * 0.2)
        total_num = cfg.BrainAlign.S
        for ratio in ratio_list:
            train_size = ratio * multiple
            train_test_val_arr = np.random.randint(total_num, size=test_val_num * 2 + train_size)
            train_arr = train_test_val_arr[0:train_size]
            np.save(cfg.BrainAlign.DATA_PATH + 'train_{}.npy'.format(ratio), train_arr)
            test_arr = train_test_val_arr[train_size:train_size + test_val_num]
            np.save(cfg.BrainAlign.DATA_PATH + 'test_{}.npy'.format(ratio), test_arr)
            val_arr = train_test_val_arr[train_size + test_val_num:]
            np.save(cfg.BrainAlign.DATA_PATH + 'val_{}.npy'.format(ratio), val_arr)

        mouse_sample_list = list(datapair['obs_dfs'][0].index)
        mouse_region_list = list(datapair['obs_dfs'][0]['region_name'])
        # print(mouse_region_list)

        label_array = load_sample_label(mouse_region_list, cfg)
        np.save(cfg.BrainAlign.DATA_PATH + 'labels.npy', label_array)

    elif cfg.BrainAlign.NODE_TYPE_NUM == 3:
        print(len(datapair['varnames_node'][0]))
        np.save(cfg.BrainAlign.DATA_PATH + 'mouse_gene_names.npy', datapair['varnames_node'][0])
        print(len(datapair['varnames_node'][1]))
        np.save(cfg.BrainAlign.DATA_PATH + 'human_gene_names.npy', datapair['varnames_node'][1])

        adata_sm = ad.AnnData(datapair['ov_adjs'][0])
        adata_sm.obs_names = list(datapair['obs_dfs'][0].index)
        adata_sm.var_names = datapair['varnames_node'][0]
        adata_sm = normalize_before_pruning(adata_sm, method=cfg.BrainAlign.normalize_before_pruning_method,
                                            target_sum=cfg.BrainAlign.pruning_target_sum, force_return=True)
        sm_ = adata_sm.X.toarray()

        adata_vh = ad.AnnData(datapair['ov_adjs'][1])
        adata_vh.obs_names = list(datapair['obs_dfs'][1].index)
        adata_vh.var_names = datapair['varnames_node'][1]
        adata_vh = normalize_before_pruning(adata_vh, method=cfg.BrainAlign.normalize_before_pruning_method,
                                            target_sum=cfg.BrainAlign.pruning_target_sum, force_return=True)
        vh_ = adata_vh.X.toarray()




    elif cfg.BrainAlign.NODE_TYPE_NUM == 2:


        print(len(datapair['varnames_node'][0] + datapair['varnames_node'][1]))
        np.save(cfg.BrainAlign.DATA_PATH + 'gene_names.npy', datapair['varnames_node'][0]+datapair['varnames_node'][1])

        mouse_sample_list = list(datapair['obs_dfs'][0].index)
        human_sample_list = list(datapair['obs_dfs'][1].index)
        np.save(cfg.BrainAlign.DATA_PATH + 'sample_names.npy',mouse_sample_list + human_sample_list)

        mouse_sample_num = len(mouse_sample_list)
        mouse_gene_num = len(datapair['varnames_node'][0])

        if cfg.BrainAlign.normalize_scale == True:
            adata_sm = ad.AnnData(sp.coo_matrix(datapair['ov_adjs'][0].toarray() - datapair['ov_adjs'][0].min().min()).tocsr())
        else:
            adata_sm = ad.AnnData(datapair['ov_adjs'][0])
        adata_sm.obs_names = list(datapair['obs_dfs'][0].index)
        adata_sm.var_names = datapair['varnames_node'][0]
        adata_sm = normalize_before_pruning(adata_sm, method=cfg.BrainAlign.normalize_before_pruning_method_1, target_sum=cfg.BrainAlign.pruning_target_sum_1, force_return=True)
        sm_1 = adata_sm.X.toarray()

        if cfg.BrainAlign.normalize_scale == True:
            vh_X = datapair['ov_adjs'][1].toarray()
            vh_X = vh_X - vh_X.min()
            vh_X = vh_X * (adata_sm.X.max().max() - adata_sm.X.min().min()) / vh_X.max().max()
            vh_X = sp.coo_matrix(vh_X).tocsr()
            adata_vh = ad.AnnData(vh_X)
        else:
            adata_vh = ad.AnnData(datapair['ov_adjs'][1])
        adata_vh.obs_names = list(datapair['obs_dfs'][1].index)
        adata_vh.var_names = datapair['varnames_node'][1]
        adata_vh = normalize_before_pruning(adata_vh, method=cfg.BrainAlign.normalize_before_pruning_method_2, target_sum=cfg.BrainAlign.pruning_target_sum_2, force_return=True)
        sm_2 = adata_vh.X.toarray()

        #sm_1 = datapair['ov_adjs'][0].tocoo().toarray()
        #sm_2 = datapair['ov_adjs'][1].tocoo().toarray()

        if cfg.BrainAlign.if_threshold:
            if cfg.BrainAlign.pruning_method == 'top':
                sm_1 = threshold_top(sm_1, percent=cfg.BrainAlign.sm_gene_top) + threshold_top(sm_1.T, percent=cfg.BrainAlign.sm_sample_top).T
                sm_2 = threshold_top(sm_2, percent=cfg.BrainAlign.vh_gene_top) + threshold_top(sm_2.T,
                                                                                         percent=cfg.BrainAlign.vh_sample_top).T
            elif cfg.BrainAlign.pruning_method == 'std':
                # threshold_std(sm_1, std_times=cfg.BrainAlign.pruning_std_times_sm) + \
                sm_1 = threshold_top(sm_1, percent=cfg.BrainAlign.sm_gene_top) + \
                       threshold_std(sm_1, std_times=cfg.BrainAlign.pruning_std_times_sm) + \
                       threshold_std(sm_1.T, std_times=cfg.BrainAlign.pruning_std_times_sm).T + \
                       threshold_top(sm_1.T, percent=cfg.BrainAlign.sm_sample_top).T
                #threshold_std(sm_2, std_times=cfg.BrainAlign.pruning_std_times_vh)
                sm_2 = threshold_top(sm_2, percent=cfg.BrainAlign.vh_gene_top) + \
                       threshold_std(sm_2, std_times=cfg.BrainAlign.pruning_std_times_vh) + \
                       threshold_std(sm_2.T, std_times=cfg.BrainAlign.pruning_std_times_vh).T + \
                       threshold_top(sm_2.T, percent=cfg.BrainAlign.vh_sample_top).T
                #sm_1 = threshold_std(sm_1, std_times=cfg.BrainAlign.pruning_std_times_sm) + threshold_std(sm_1.T,
                #                                                                                    std_times=cfg.BrainAlign.pruning_std_times_sm).T
                #sm_2 = threshold_std(sm_2, std_times=cfg.BrainAlign.pruning_std_times_vh) + threshold_std(sm_2.T,
                #                                                                           std_times=cfg.BrainAlign.pruning_std_times_vh).T
                print('sm sparsity = {}'.format(np.sum(sm_1) / (cfg.BrainAlign.binary_S * cfg.BrainAlign.binary_M)))
                print('vh sparsity = {}'.format(np.sum(sm_2) / (cfg.BrainAlign.binary_V * cfg.BrainAlign.binary_H)))
        else:
            sm_1 = sm_1 > 0
            sm_2 = sm_2 > 0

        sm_1 = sp.csr_matrix(sm_1).tocoo()
        sm_2 = sp.csr_matrix(sm_2).tocoo()

        sm_row = np.concatenate((sm_1.row, sm_2.row + mouse_sample_num))
        sm_col = np.concatenate((sm_1.col, sm_2.col + mouse_gene_num))
        sm_data = np.concatenate((sm_1.data, sm_2.data))
        sm_ = sp.coo_matrix((sm_data, (sm_row, sm_col)), shape=(cfg.BrainAlign.S, cfg.BrainAlign.M))

        ms_ = sm_.toarray().T
        s_feat = sp.load_npz(cfg.BrainAlign.DATA_PATH + 's_feat.npz').toarray()
        m_feat = sp.load_npz(cfg.BrainAlign.DATA_PATH + 'm_feat.npz').toarray()
        corr_ms = corr2_coeff(m_feat, s_feat)
        ms_argmax = np.argmax(corr_ms, axis=1)
        for row_i in range(np.shape(corr_ms)[0]):
            if np.sum(ms_[row_i, :]) == 0:
                ms_[row_i, ms_argmax[row_i]] = True
        mh_argmax_col = np.argmax(corr_ms, axis=0)
        for col_j in range(np.shape(corr_ms)[1]):
            if np.sum(ms_[:, col_j]) == 0:
                ms_[mh_argmax_col[col_j], col_j] = True

        sm_ = ms_.T
        #sm_ = sm_.toarray()
        print(sm_)
        logger.debug(f'sm_: {sm_.shape}')



        ss_ = datapair['oo_adjs'].toarray() > 0
        logger.debug(f'ss_: {ss_.shape}')
        # print('ss_ sum', np.sum(ss_)) # == 0
        mm_ = datapair['vv_adj'].toarray() > 0
        logger.debug(f'mm_: {mm_.shape}')

        ## Generate meta-path for target node S
        sms = np.matmul(sm_, sm_.T) > 0
        # print(sms)
        logger.debug('How sparse is sms:%s', np.sum(sms) / (cfg.BrainAlign.S * cfg.BrainAlign.S))
        sms = sp.coo_matrix(sms)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + "sms.npz", sms)
        del sms

        smm = np.matmul(sm_, mm_) > 0
        smms = np.matmul(smm, sm_.T) > 0
        # print(smhvhms)
        logger.debug('How sparse is smms:%s', np.sum(smms) / (cfg.BrainAlign.S * cfg.BrainAlign.S))
        smms = sp.coo_matrix(smms)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + "smms.npz", smms)
        del smm, smms


        ## Generate meta-path for target node M
        msm = np.matmul(sm_.T, sm_) > 0
        # print(sms)
        logger.info('How sparse is msm:%s', np.sum(msm) / (cfg.BrainAlign.M * cfg.BrainAlign.M))
        msm = sp.coo_matrix(msm)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + "msm.npz", msm)
        del msm

        mss = np.matmul(sm_.T, ss_) > 0
        mssm = np.matmul(mss, sm_) > 0
        logger.debug('How sparse is mssm:%s', np.sum(mssm) / (cfg.BrainAlign.M * cfg.BrainAlign.M))
        mssm = sp.coo_matrix(mssm)
        sp.save_npz(cfg.BrainAlign.DATA_PATH + "mssm.npz", mssm)
        del mssm
        

        ## neibor number: S-{M}, M-{S, H}, H-{M, V}, V-{H}
        # Get S neighborhoods
        get_neighbor(sm_, cfg, logger, 's_nei_m', cfg.BrainAlign.S, cfg.BrainAlign.M)

        # Get M neighborhoods
        get_neighbor(sm_.T, cfg, logger, 'm_nei_s', cfg.BrainAlign.M, cfg.BrainAlign.S)

        ## Generate positive samples for target node S
        meta_path_list = [cfg.BrainAlign.DATA_PATH + x for x in ['sms.npz', 'smms.npz']]
        get_positive_sample(cfg, meta_path_list, target_node='S')

        ## Generate positive samples for target node M
        meta_path_list = [cfg.BrainAlign.DATA_PATH + x for x in ['msm.npz', 'mssm.npz']]
        get_positive_sample(cfg, meta_path_list, target_node='M')

        ratio_list = cfg.BrainAlign_args.ratio
        multiple = 20
        test_val_num = int(cfg.BrainAlign.binary_S * 0.2)
        total_num = cfg.BrainAlign.S
        for ratio in ratio_list:
            train_size = ratio * multiple
            train_test_val_arr = np.random.randint(total_num, size=test_val_num * 2 + train_size)
            train_arr = train_test_val_arr[0:train_size]
            np.save(cfg.BrainAlign.DATA_PATH + 'train_{}.npy'.format(ratio), train_arr)
            test_arr = train_test_val_arr[train_size:train_size + test_val_num]
            np.save(cfg.BrainAlign.DATA_PATH + 'test_{}.npy'.format(ratio), test_arr)
            val_arr = train_test_val_arr[train_size + test_val_num:]
            np.save(cfg.BrainAlign.DATA_PATH + 'val_{}.npy'.format(ratio), val_arr)

        mouse_sample_list = list(datapair['obs_dfs'][0].index)
        mouse_region_list = list(datapair['obs_dfs'][0]['region_name'])
        # print(mouse_region_list)

        label_array = load_sample_label(mouse_region_list, cfg)
        np.save(cfg.BrainAlign.DATA_PATH + 'labels.npy', label_array)


    # create the experiments folder


def get_spatial_relation(cfg):
    expression_specie1_path = cfg.CAME.path_rawdata1
    adata_specie1 = sc.read_h5ad(expression_specie1_path)
    print(adata_specie1)
    x = adata_specie1.obs['x_grid']
    y = adata_specie1.obs['y_grid']
    z = adata_specie1.obs['z_grid']
    xyz_arr1 = np.array([x, y, z]).T  # N row, each row is in 3 dimension
    ss_1 = kneighbors_graph(xyz_arr1, 1 + cfg.BrainAlign.spatial_node_neighbor, mode='connectivity', include_self=False) > 0

    expression_specie2_path = cfg.CAME.path_rawdata2
    adata_specie2 = sc.read_h5ad(expression_specie2_path)
    print(adata_specie2)
    x = adata_specie2.obs['mri_voxel_x']
    y = adata_specie2.obs['mri_voxel_y']
    z = adata_specie2.obs['mri_voxel_z']
    xyz_arr2 = np.array([x, y, z]).T  # N row, each row is in 3 dimension
    vv_2 = kneighbors_graph(xyz_arr2, 1 + cfg.BrainAlign.spatial_node_neighbor, mode='connectivity', include_self=False) > 0

    ss_1 = sp.csr_matrix(ss_1).tocoo()
    vv_2 = sp.csr_matrix(vv_2).tocoo()

    ss_row = np.concatenate((ss_1.row, vv_2.row + cfg.BrainAlign.binary_S))
    ss_col = np.concatenate((ss_1.col, vv_2.col + cfg.BrainAlign.binary_S))
    ss_data = np.concatenate((ss_1.data, vv_2.data))
    ss_ = sp.coo_matrix((ss_data, (ss_row, ss_col)), shape=(cfg.BrainAlign.S, cfg.BrainAlign.S)).toarray()
    # print(adata_human)
    #coordinates_3d_specie1 = adata_sm.obs
    return ss_





def get_srrsc_input(cfg):
    logger = setup_logger("Running SR-RSC and analysis", cfg.BrainAlign.DATA_PATH, if_train=None)
    logger.info("Running with config:\n{}".format(cfg))

    load_embedding(cfg, logger)

    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)
    #print(datapair)

    dict_keys = ['t_info', 'rel2id', 'id2rel', 'node2lid', 'node2gid', 'gid2node', 'n_class', 'n_feat', 'relations', 'types',
         'undirected_relations', 'r_info']
    meta_data_dict = {k:None for k in dict_keys}

    if cfg.BrainAlign.NODE_TYPE_NUM == 2:

        logger.info("Generating meta_data...")
        meta_data_dict['t_info'] = {'s':{'ind':range(0, cfg.BrainAlign.S), 'cnt':cfg.BrainAlign.S}, 'm':{'ind':range(cfg.BrainAlign.S, cfg.BrainAlign.S+cfg.BrainAlign.M), 'cnt':cfg.BrainAlign.M}}
        meta_data_dict['rel2id'] = {'s-m': 0, 'm-s':1, 's-s':2, 'm-m':3}
        meta_data_dict['id2real'] = {0:'s-m', 1:'m-s', 2:'s-s', 3:'m-m'}

        #print(len(datapair['varnames_node'][0] + datapair['varnames_node'][1]))
        #np.save(cfg.BrainAlign.DATA_PATH + 'gene_names.npy', datapair['varnames_node'][0] + datapair['varnames_node'][1])
        mouse_gene_name_list = datapair['varnames_node'][0]
        human_gene_name_list = datapair['varnames_node'][1]
        gene_name_list = datapair['varnames_node'][0] + datapair['varnames_node'][1]

        mouse_sample_list = list(datapair['obs_dfs'][0].index)
        human_sample_list = list(datapair['obs_dfs'][1].index)
        sample_name_list =  mouse_sample_list + human_sample_list
        #np.save(cfg.BrainAlign.DATA_PATH + 'sample_names.npy', mouse_sample_list + human_sample_list)
        lid_list = list(range(0, len(sample_name_list))) + list(range(0, len(gene_name_list)))
        node_list = ['s'+str(i) for i in range(0, len(sample_name_list))] + ['m'+str(i) for i in range(0, len(gene_name_list))]
        grid_list = list(range(0, len(sample_name_list+gene_name_list)))
        meta_data_dict['node2lid'] = {k:v for k,v in zip(node_list, lid_list)}
        meta_data_dict['lid2node'] = {k:v for k,v in zip(lid_list, node_list)}

        meta_data_dict['node2gid'] = {k: v for k, v in zip(node_list, grid_list)}
        meta_data_dict['gid2node'] = {k: v for k, v in zip(grid_list, node_list)}

        meta_data_dict['n_class'] = 2
        meta_data_dict['n_feat'] = cfg.BrainAlign.embedding_pca_dim

        meta_data_dict['relations'] = ['s-m', 'm-s', 's-s', 'm-m']
        meta_data_dict['types'] = ['s', 'm']
        #meta_data_dict['undirected_relations'] = {'s-s', 'm-m'}
        meta_data_dict['r_info'] = {}

        f = open(cfg.BrainAlign.DATA_PATH+"meta_data.pkl", "wb")
        pickle.dump(meta_data_dict, f)

        logger.info("Generating labels...")
        all_label_X = np.array(list(range(0, len(sample_name_list+gene_name_list)))).reshape(-1, 1)
        all_label_Y = np.array([0] * len(mouse_sample_list+mouse_gene_name_list) + [1] * len(human_sample_list + human_gene_name_list)).reshape(-1, 1)

        x_train,y_train, x_val, y_val, x_test, y_test = \
            train_test_val_split(all_label_X, all_label_Y, train_ratio=cfg.SRRSC_args.train_ratio, validation_ratio=cfg.SRRSC_args.validation_ratio, test_ratio=cfg.SRRSC_args.test_ratio, random_state=1)
        labels = []
        labels.append(np.concatenate((x_train, y_train), axis=1))
        labels.append(np.concatenate((x_val, y_val), axis=1))
        labels.append(np.concatenate((x_test, y_test), axis=1))
        f = open(cfg.BrainAlign.DATA_PATH + "labels.pkl", "wb")
        pickle.dump(labels, f)

        #################################################
        logger.info("Generating edges...")
        #print(len(datapair['varnames_node'][0] + datapair['varnames_node'][1]))
        np.save(cfg.BrainAlign.DATA_PATH + 'gene_names.npy', datapair['varnames_node'][0] + datapair['varnames_node'][1])

        mouse_sample_list = list(datapair['obs_dfs'][0].index)
        human_sample_list = list(datapair['obs_dfs'][1].index)
        np.save(cfg.BrainAlign.DATA_PATH + 'sample_names.npy', mouse_sample_list + human_sample_list)

        mouse_sample_num = len(mouse_sample_list)
        mouse_gene_num = len(datapair['varnames_node'][0])

        if cfg.BrainAlign.normalize_scale == True:
            adata_sm = ad.AnnData(
                sp.coo_matrix(datapair['ov_adjs'][0].toarray() - datapair['ov_adjs'][0].min().min()).tocsr())
        else:
            adata_sm = ad.AnnData(datapair['ov_adjs'][0])
        adata_sm.obs_names = list(datapair['obs_dfs'][0].index)
        adata_sm.var_names = datapair['varnames_node'][0]
        adata_sm = normalize_before_pruning(adata_sm, method=cfg.BrainAlign.normalize_before_pruning_method_1,
                                            target_sum=cfg.BrainAlign.pruning_target_sum_1, force_return=True)
        sm_1 = adata_sm.X.toarray()

        if cfg.BrainAlign.normalize_scale == True:
            vh_X = datapair['ov_adjs'][1].toarray()
            vh_X = vh_X - vh_X.min()
            vh_X = vh_X * (adata_sm.X.max().max() - adata_sm.X.min().min()) / vh_X.max().max()
            vh_X = sp.coo_matrix(vh_X).tocsr()
            adata_vh = ad.AnnData(vh_X)
        else:
            adata_vh = ad.AnnData(datapair['ov_adjs'][1])
        adata_vh.obs_names = list(datapair['obs_dfs'][1].index)
        adata_vh.var_names = datapair['varnames_node'][1]
        adata_vh = normalize_before_pruning(adata_vh, method=cfg.BrainAlign.normalize_before_pruning_method_2,
                                            target_sum=cfg.BrainAlign.pruning_target_sum_2, force_return=True)
        sm_2 = adata_vh.X.toarray()

        # sm_1 = datapair['ov_adjs'][0].tocoo().toarray()
        # sm_2 = datapair['ov_adjs'][1].tocoo().toarray()

        '''

        if cfg.BrainAlign.if_threshold:
            if cfg.BrainAlign.pruning_method == 'top':
                sm_1 = threshold_top(sm_1, percent=cfg.BrainAlign.sm_gene_top) + threshold_top(sm_1.T,
                                                                                         percent=cfg.BrainAlign.sm_sample_top).T
                sm_2 = threshold_top(sm_2, percent=cfg.BrainAlign.vh_gene_top) + threshold_top(sm_2.T,
                                                                                         percent=cfg.BrainAlign.vh_sample_top).T
            elif cfg.BrainAlign.pruning_method == 'std':
                # threshold_std(sm_1, std_times=cfg.BrainAlign.pruning_std_times_sm) + \
                sm_1 = threshold_top(sm_1, percent=cfg.BrainAlign.sm_gene_top) + \
                       threshold_std(sm_1, std_times=cfg.BrainAlign.pruning_std_times_sm) + \
                       threshold_top(sm_1.T, percent=cfg.BrainAlign.sm_sample_top).T
                # threshold_std(sm_2, std_times=cfg.BrainAlign.pruning_std_times_vh) + \
                sm_2 = threshold_top(sm_2, percent=cfg.BrainAlign.vh_gene_top) + \
                       threshold_std(sm_2, std_times=cfg.BrainAlign.pruning_std_times_vh) + \
                       threshold_top(sm_2.T, percent=cfg.BrainAlign.vh_sample_top).T
                # sm_1 = threshold_std(sm_1, std_times=cfg.BrainAlign.pruning_std_times_sm) + threshold_std(sm_1.T,
                #                                                                                    std_times=cfg.BrainAlign.pruning_std_times_sm).T
                # sm_2 = threshold_std(sm_2, std_times=cfg.BrainAlign.pruning_std_times_vh) + threshold_std(sm_2.T,
                #                                                                           std_times=cfg.BrainAlign.pruning_std_times_vh).T
                print('sm sparsity = {}'.format(np.sum(sm_1) / (cfg.BrainAlign.binary_S * cfg.BrainAlign.binary_M)))
                print('vh sparsity = {}'.format(np.sum(sm_2) / (cfg.BrainAlign.binary_V * cfg.BrainAlign.binary_H)))
        else:
            sm_1 = sm_1 > 0
            sm_2 = sm_2 > 0

        '''

        sm_1 = sp.csr_matrix(sm_1).tocoo()
        sm_2 = sp.csr_matrix(sm_2).tocoo()

        sm_row = np.concatenate((sm_1.row, sm_2.row + mouse_sample_num))
        sm_col = np.concatenate((sm_1.col, sm_2.col + mouse_gene_num))
        sm_data = np.concatenate((sm_1.data, sm_2.data))
        sm_ = sp.coo_matrix((sm_data, (sm_row, sm_col)), shape=(cfg.BrainAlign.S, cfg.BrainAlign.M))

        # mh_ = datapair['vv_adj'].toarray()[0:cfg.BrainAlign.M, cfg.BrainAlign.M:] > 0
        if cfg.BrainAlign.node_relation == 'spatial':
            ss_ = get_spatial_relation(cfg)
        elif cfg.BrainAlign.node_relation == 'knn':
            ss_ = datapair['oo_adjs'].toarray() > 0

        logger.debug(f'ss_: {ss_.shape}')
        # print('ss_ sum', np.sum(ss_)) # == 0
        mm_ = datapair['vv_adj'].toarray() > 0
        logger.debug(f'mm_: {mm_.shape}')

        # Add self loop of nodes
        # This step is optional
        # mm_  = mm_ + np.identity(mm_.shape[0])
        # mm_ = mm_ > 0
        # ss_ = ss_ + np.identity(ss_.shape[0])
        # ss_ = ss_ > 0

        # concatenate to a large matrix
        s_sm_mat = np.concatenate((ss_, sm_.toarray()), axis=1)
        m_sm_mat = np.concatenate((sm_.toarray().T, mm_), axis=1)
        sm_mat = np.concatenate((s_sm_mat, m_sm_mat), axis=0)
        ms_mat = sp.csc_matrix(sm_mat.T)
        sm_mat = sp.csr_matrix(sm_mat)

        #ss_mat = sp.csc_matrix(ss_)
        #mm_mat = sp.csc_matrix(mm_)
        s_sm_mat_mm = np.concatenate((np.zeros(ss_.shape), np.zeros(sm_.toarray().shape)), axis=1)
        m_sm_mat_mm = np.concatenate((np.zeros(sm_.toarray().shape).T, mm_), axis=1)
        mm_mat = np.concatenate((s_sm_mat_mm, m_sm_mat_mm), axis=0)   # Check if this is wrong
        mm_mat = sp.csr_matrix(mm_mat)

        s_sm_mat_ss = np.concatenate((ss_, np.zeros(sm_.toarray().shape)), axis=1)
        m_sm_mat_ss = np.concatenate((np.zeros(sm_.toarray().shape).T, np.zeros(mm_.shape)), axis=1)
        ss_mat = np.concatenate((s_sm_mat_ss, m_sm_mat_ss), axis=0)
        ss_mat = sp.csr_matrix(ss_mat)


        edges_data = {'s-m': sm_mat, 'm-s': ms_mat, 's-s': ss_mat, 'm-m':mm_mat}
        f = open(cfg.BrainAlign.DATA_PATH + "edges.pkl", "wb")
        pickle.dump(edges_data, f)

        feats_s = sp.load_npz(cfg.BrainAlign.DATA_PATH + 's_feat.npz').toarray()
        feats_m = sp.load_npz(cfg.BrainAlign.DATA_PATH + 'm_feat.npz').toarray()

        # No pass of gene features
        #feats_m = np.zeros(feats_m.shape)
        # modify gene features, set homologous genes embeddings the same initial embeddings
        mh_ = datapair['vv_adj'].toarray()[0:cfg.BrainAlign.binary_M, cfg.BrainAlign.binary_M:]

        feats_m_temp = feats_m.copy()
        for i in range(cfg.BrainAlign.binary_M):
            h_index = [x + cfg.BrainAlign.binary_M for x in np.nonzero(mh_[i, :])[0]]
            if len(h_index) > 0:
                h_index.append(i)
                feat_h = feats_m[h_index, :]
                feats_m_temp[i, :] = np.mean(feat_h, axis=0)
        for j in range(cfg.BrainAlign.binary_H):
            m_index = list(np.nonzero(mh_[:, j])[0])
            if len(m_index) > 0:
                m_index.append(int(cfg.BrainAlign.binary_M + j))
                feat_g = feats_m[m_index, :]
                feats_m_temp[cfg.BrainAlign.binary_M + j, :] = np.mean(feat_g, axis=0)
        feats_m = feats_m_temp

        node_feats = sp.coo_matrix(np.concatenate((feats_s, feats_m), axis=0)).tocsr()
        f = open(cfg.BrainAlign.DATA_PATH + "node_features.pkl", "wb")
        pickle.dump(node_feats, f)

    elif cfg.BrainAlign.NODE_TYPE_NUM == 4:

        logger.info("Generating meta_data...")
        meta_data_dict['t_info'] = {'s': {'ind': range(0, cfg.BrainAlign.binary_S), 'cnt': cfg.BrainAlign.binary_S},
                                    'v': {'ind': range(cfg.BrainAlign.binary_S, cfg.BrainAlign.binary_S+cfg.BrainAlign.binary_V), 'cnt': cfg.BrainAlign.binary_V},
                                    'm': {'ind': range(cfg.BrainAlign.binary_S+cfg.BrainAlign.binary_V, cfg.BrainAlign.binary_S+cfg.BrainAlign.binary_V + cfg.BrainAlign.binary_M), 'cnt': cfg.BrainAlign.binary_M},
                                    'h': {'ind': range(cfg.BrainAlign.binary_S+cfg.BrainAlign.binary_V + cfg.BrainAlign.binary_M, cfg.BrainAlign.binary_S+cfg.BrainAlign.binary_V + cfg.BrainAlign.binary_M + cfg.BrainAlign.binary_H), 'cnt': cfg.BrainAlign.binary_H}}
        meta_data_dict['rel2id'] = {'s-m': 0, 'm-s': 1, 'v-h':2, 'h-v':3, 'm-h':4, 'h-m':5}
        meta_data_dict['id2real'] = {0: 's-m', 1: 'm-s', 2:'v-h', 3:'h-v', 4:'m-h', 5:'h-m'}

        #print(len(datapair['varnames_node'][0] + datapair['varnames_node'][1]))
        # np.save(cfg.BrainAlign.DATA_PATH + 'gene_names.npy', datapair['varnames_node'][0] + datapair['varnames_node'][1])
        mouse_gene_name_list = datapair['varnames_node'][0]
        human_gene_name_list = datapair['varnames_node'][1]
        gene_name_list = datapair['varnames_node'][0] + datapair['varnames_node'][1]

        mouse_sample_list = list(datapair['obs_dfs'][0].index)
        human_sample_list = list(datapair['obs_dfs'][1].index)
        sample_name_list = mouse_sample_list + human_sample_list
        # np.save(cfg.BrainAlign.DATA_PATH + 'sample_names.npy', mouse_sample_list + human_sample_list)
        lid_list = list(range(0, len(mouse_sample_list))) + \
                   list(range(0, len(human_sample_list))) + \
                   list(range(0, len(mouse_gene_name_list))) + \
                   list(range(0, len(human_gene_name_list)))
        node_list = ['s' + str(i) for i in range(0, len(mouse_sample_list))] + \
                    ['v' + str(i) for i in range(0, len(human_sample_list))] + \
                    ['m' + str(i) for i in range(0, len(mouse_gene_name_list))] +\
                    ['h' + str(i) for i in range(0, len(human_gene_name_list))]
        grid_list = list(range(0, len(sample_name_list + gene_name_list)))
        meta_data_dict['node2lid'] = {k: v for k, v in zip(node_list, lid_list)}
        meta_data_dict['lid2node'] = {k: v for k, v in zip(lid_list, node_list)}

        meta_data_dict['node2gid'] = {k: v for k, v in zip(node_list, grid_list)}
        meta_data_dict['gid2node'] = {k: v for k, v in zip(grid_list, node_list)}

        meta_data_dict['n_class'] = 2
        meta_data_dict['n_feat'] = cfg.BrainAlign.embedding_pca_dim

        meta_data_dict['relations'] = ['s-m', 'm-s', 'v-h', 'h-v', 'm-h', 'h-m']
        meta_data_dict['types'] = ['s', 'v', 'm', 'h']
        meta_data_dict['undirected_relations'] = {}
        meta_data_dict['r_info'] = {}

        f = open(cfg.BrainAlign.DATA_PATH + "meta_data.pkl", "wb")
        pickle.dump(meta_data_dict, f)

        logger.info("Generating labels...")
        all_label_X = np.array(list(range(0, len(sample_name_list + gene_name_list)))).reshape(-1, 1)
        all_label_Y = np.array([0] * len(mouse_sample_list) + [1] * len(human_sample_list) +
                               [0] * len(mouse_gene_name_list) +  [1] * len(human_gene_name_list)).reshape(-1, 1)

        x_train, y_train, x_val, y_val, x_test, y_test = \
            train_test_val_split(all_label_X, all_label_Y, train_ratio=cfg.SRRSC_args.train_ratio,
                                 validation_ratio=cfg.SRRSC_args.validation_ratio, test_ratio=cfg.SRRSC_args.test_ratio,
                                 random_state=1)
        labels = []
        labels.append(np.concatenate((x_train, y_train), axis=1))
        labels.append(np.concatenate((x_val, y_val), axis=1))
        labels.append(np.concatenate((x_test, y_test), axis=1))
        f = open(cfg.BrainAlign.DATA_PATH + "labels.pkl", "wb")
        pickle.dump(labels, f)


        ########################################################
        # Generate edges

        if cfg.BrainAlign.normalize_scale == True:
            adata_sm = ad.AnnData(
                sp.coo_matrix(datapair['ov_adjs'][0].toarray() - datapair['ov_adjs'][0].min().min()).tocsr())
        else:
            adata_sm = ad.AnnData(datapair['ov_adjs'][0])

        # adata_sm = ad.AnnData(datapair['ov_adjs'][0])
        adata_sm.obs_names = list(datapair['obs_dfs'][0].index)
        adata_sm.var_names = datapair['varnames_node'][0]
        adata_sm = normalize_before_pruning(adata_sm, method=cfg.BrainAlign.normalize_before_pruning_method_1,
                                            target_sum=cfg.BrainAlign.pruning_target_sum_1, force_return=True)
        sm_ = adata_sm.X.toarray()

        if cfg.BrainAlign.normalize_scale == True:
            vh_X = datapair['ov_adjs'][1].toarray()
            vh_X = vh_X - vh_X.min()
            vh_X = vh_X * (adata_sm.X.max().max() - adata_sm.X.min().min()) / vh_X.max().max()
            vh_X = sp.coo_matrix(vh_X).tocsr()
            adata_vh = ad.AnnData(vh_X)
        else:
            adata_vh = ad.AnnData(datapair['ov_adjs'][1])

        adata_vh.obs_names = list(datapair['obs_dfs'][1].index)
        adata_vh.var_names = datapair['varnames_node'][1]
        adata_vh = normalize_before_pruning(adata_vh, method=cfg.BrainAlign.normalize_before_pruning_method_2,
                                            target_sum=cfg.BrainAlign.pruning_target_sum_2, force_return=True)
        vh_ = adata_vh.X.toarray()

        # sm_ = datapair['ov_adjs'][0].toarray()
        '''
        if cfg.BrainAlign.if_threshold:
            if cfg.BrainAlign.pruning_method == 'top':
                sm_ = threshold_top(sm_, percent=cfg.BrainAlign.sm_gene_top) + threshold_top(sm_.T,
                                                                                       percent=cfg.BrainAlign.sm_sample_top).T
            elif cfg.BrainAlign.pruning_method == 'std':
                # threshold_std(sm_, std_times=cfg.BrainAlign.pruning_std_times_sm) + \
                sm_ = threshold_top(sm_, percent=cfg.BrainAlign.sm_gene_top) + \
                      threshold_std(sm_, std_times=cfg.BrainAlign.pruning_std_times_sm) + \
                      threshold_top(sm_.T, percent=cfg.BrainAlign.sm_sample_top).T
                # sm_ = threshold_std(sm_, std_times=cfg.BrainAlign.pruning_std_times_sm) + threshold_std(sm_.T,
                #                                                                                  std_times=cfg.BrainAlign.pruning_std_times_sm).T
        else:
            sm_ = sm_ > 0
        print(sm_)
        logger.debug(f'sm_: {sm_.shape}')

        # vh_ = datapair['ov_adjs'][1].toarray()
        if cfg.BrainAlign.if_threshold:
            if cfg.BrainAlign.pruning_method == 'top':
                vh_ = threshold_top(vh_, percent=cfg.BrainAlign.vh_gene_top) + threshold_top(vh_.T,
                                                                                       percent=cfg.BrainAlign.vh_sample_top).T
            elif cfg.BrainAlign.pruning_method == 'std':
                # threshold_std(vh_, std_times=cfg.BrainAlign.pruning_std_times_vh) + \
                vh_ = threshold_top(vh_, percent=cfg.BrainAlign.vh_gene_top) + \
                      threshold_std(vh_, std_times=cfg.BrainAlign.pruning_std_times_vh) + \
                      threshold_top(vh_.T, percent=cfg.BrainAlign.vh_sample_top).T
                # vh_ = threshold_std(vh_, std_times=cfg.BrainAlign.pruning_std_times_vh) + threshold_std(vh_.T,
                #                                                                                  std_times=cfg.BrainAlign.pruning_std_times_vh).T
        else:
            vh_ = vh_ > 0

        '''
        sm_ = sm_ > 0
        vh_ = vh_ > 0

        logger.debug(f'vh_: {vh_.shape}')
        mh_ = datapair['vv_adj'].toarray()[0:cfg.BrainAlign.binary_M, cfg.BrainAlign.binary_M:] > 0
        logger.debug(f'mh_: {mh_.shape}')

        ss_ = datapair['oo_adjs'].toarray()[0:cfg.BrainAlign.binary_S, 0:cfg.BrainAlign.binary_S] > 0
        logger.debug(f'ss_: {ss_.shape}')
        # print('ss_ sum', np.sum(ss_)) # == 0
        vv_ = datapair['oo_adjs'].toarray()[cfg.BrainAlign.binary_S:, cfg.BrainAlign.binary_S:] > 0
        logger.debug(f'vv_: {vv_.shape}')

        sm_ = sp.csr_matrix(sm_).tocoo()
        vh_ = sp.csr_matrix(vh_).tocoo()
        mh_ = sp.csr_matrix(mh_).tocoo()
        ss_ = sp.csr_matrix(ss_).tocoo()
        vv_ = sp.csr_matrix(vv_).tocoo()


        size_mat = cfg.BrainAlign.binary_S + cfg.BrainAlign.binary_V + cfg.BrainAlign.binary_M + cfg.BrainAlign.binary_H
        sm_row = np.concatenate((sm_.row, sm_.T.row + cfg.BrainAlign.binary_S + cfg.BrainAlign.binary_V))
        sm_col = np.concatenate((sm_.col + cfg.BrainAlign.binary_S + cfg.BrainAlign.binary_V, sm_.T.col))
        sm_data = np.concatenate((sm_.data, sm_.T.data))
        sm_mat = sp.coo_matrix((sm_data, (sm_row, sm_col)), shape=(size_mat, size_mat))
        ms_mat = sm_mat.T

        vh_row = np.concatenate((vh_.row + cfg.BrainAlign.binary_S, vh_.T.row + cfg.BrainAlign.binary_S + cfg.BrainAlign.binary_V + cfg.BrainAlign.binary_M))
        vh_col = np.concatenate((vh_.col + cfg.BrainAlign.binary_S + cfg.BrainAlign.binary_V + cfg.BrainAlign.binary_M, vh_.T.col + cfg.BrainAlign.binary_S))
        vh_data = np.concatenate((vh_.data, vh_.T.data))
        vh_mat = sp.coo_matrix((vh_data, (vh_row, vh_col)), shape=(size_mat, size_mat))
        hv_mat = vh_mat.T

        mh_row = np.concatenate(
            (mh_.row + cfg.BrainAlign.binary_S + cfg.BrainAlign.binary_V, mh_.T.row + cfg.BrainAlign.binary_S + cfg.BrainAlign.binary_V + cfg.BrainAlign.binary_M))
        mh_col = np.concatenate(
            (mh_.col + cfg.BrainAlign.binary_S + cfg.BrainAlign.binary_V + cfg.BrainAlign.binary_M, mh_.T.col + cfg.BrainAlign.binary_S + cfg.BrainAlign.binary_V))
        mh_data = np.concatenate((mh_.data, mh_.T.data))
        mh_mat = sp.coo_matrix((mh_data, (mh_row, mh_col)), shape=(size_mat, size_mat))
        hm_mat = mh_mat.T

        edges_data = {'s-m': sm_mat.tocsr(), 'm-s': ms_mat.tocsc(), 'v-h':vh_mat.tocsr(), 'h-v':hv_mat.tocsc(), 'm-h':mh_mat.tocsr(), 'h-m':hm_mat.tocsc()}
        f = open(cfg.BrainAlign.DATA_PATH + "edges.pkl", "wb")
        pickle.dump(edges_data, f)


        ###### Generate features.....
        feats_s = sp.load_npz(cfg.BrainAlign.DATA_PATH + 's_feat.npz').toarray()
        feats_v = sp.load_npz(cfg.BrainAlign.DATA_PATH + 'v_feat.npz').toarray()
        feats_m = sp.load_npz(cfg.BrainAlign.DATA_PATH + 'm_feat.npz').toarray()
        feats_h = sp.load_npz(cfg.BrainAlign.DATA_PATH + 'h_feat.npz').toarray()
        node_feats = sp.coo_matrix(np.concatenate((feats_s, feats_v, feats_m, feats_h), axis=0)).tocsr()
        f = open(cfg.BrainAlign.DATA_PATH + "node_features.pkl", "wb")
        pickle.dump(node_feats, f)



    return logger

def load_sample_label(mouse_region_list, cfg):

    # No label order version
    '''
    Step 1: Compute average embedding of every region in two species, use two dict to store;
    Step 2: Compute similarity matrix, use np array to store;
    Step 3: Heatmap.

    '''
    mouse_67_labels = pd.read_csv(cfg.CAME.labels_dir+cfg.CAME.labels_mouse_file)
    #print(mouse_67_labels)

    sample_region_map  = {k:v for k,v in zip(mouse_67_labels['region_name'], mouse_67_labels.index)}
    #print(sample_region_map)
    onehot_label_array = np.zeros((len(mouse_region_list), len(mouse_67_labels.index)))
    label_array = [sample_region_map[r] for r in mouse_region_list]
    for i in range(len(mouse_region_list)):
        onehot_label_array[i, label_array[i]] = 1

    #label_array = np.array(label_array).astype(int)
    return onehot_label_array.astype(int)




def get_neighbor(sm_, cfg, logger, filename, target_node_num, neibor_node_num):
    s_m = {}
    for i in range(target_node_num):
        for j in range(neibor_node_num):
            if sm_[i][j]:
                if i not in s_m:
                    s_m[int(i)] = []
                    s_m[int(i)].append(int(j))
                else:
                    s_m[int(i)].append(int(j))
    for i in range(target_node_num):
        if i not in s_m:
            s_m[int(i)] = []
    keys = sorted(s_m.keys())
    s_m = [s_m[i] for i in keys]
    s_m = np.array([np.array(i) for i in s_m])
    np.save(cfg.BrainAlign.DATA_PATH + filename + ".npy", s_m)
    logger.debug(filename+'.shape:'.format(s_m.shape))
    # give some basic statistics about neighbors
    logger.info('Give some basic statistics about neighbors')
    l = [len(i) for i in s_m]
    logger.info('max = {}, min = {}, mean = {}'.format(max(l), min(l), np.mean(l)))


def get_positive_sample(cfg, meta_path_list, target_node):
    pos_num = cfg.BrainAlign.positive_sample_number   # init test 100
    print('Positive number = {}'.format(pos_num))
    if target_node == 'S':
        dim = cfg.BrainAlign.S
    elif target_node == 'M':
        dim = cfg.BrainAlign.M
    elif target_node == 'H':
        dim = cfg.BrainAlign.H
    elif target_node == 'V':
        dim = cfg.BrainAlign.V

    meta_path_list = [sp.load_npz(x) for x in meta_path_list]
    meta_path_list = [x / x.sum(axis=-1).reshape(-1, 1).astype("float16") for x in meta_path_list]
    meta_path_sum = reduce((lambda x, y: x + y), meta_path_list)
    all = meta_path_sum.A.astype("float16")

    all_ = (all > 0).sum(-1)
    # print(all_)
    print(all_.max(), all_.min(), all_.mean())

    pos = np.zeros((dim, dim), dtype=np.float16)
    k = 0
    for i in range(len(all)):
        one = all[i].nonzero()[0]
        if len(one) > pos_num:
            oo = np.argsort(-all[i, one])
            sele = one[oo[:pos_num]]
            pos[i, sele] = 1
            k += 1
        else:
            pos[i, one] = 1
    pos = sp.coo_matrix(pos, dtype=np.int16)
    sp.save_npz(cfg.BrainAlign.DATA_PATH + target_node.lower() + "_pos.npz", pos)





from fitter import Fitter
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings('ignore')


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
        print('A copy of AnnData made!')
    else:
        print('No copy was made, the input AnnData will be changed!')
    print('normalizing datasets with default settings.')
    if not log_only:
        print(
            f'performing total-sum normalization, target_sum={target_sum}...')
        sc.pp.normalize_total(adata, target_sum=target_sum)
    else:
        print('skipping total-sum normalization')
    sc.pp.log1p(adata)
    return adata if copy or force_return else None


#import BrainAlign.came as came

def normalize_before_pruning(adata, method='default', target_sum=None, axis=0, force_return=True):
    '''
    '''
    if method == None:
        return adata
    elif method == 'default':
        adata = preprocess.normalize_default(adata, target_sum=target_sum, log_only=False, force_return=True)
    elif method == 'zscore':
        adata.X = sp.coo_matrix(preprocess.zscore(adata.X.toarray()))
    elif method == 'wrapper_scale':
        adata = preprocess.wrapper_scale(adata, copy=True)
        adata.X = sp.coo_matrix(adata.X).tocsr()
    elif method == 'wrapper_normalize':
        adata.X = sp.coo_matrix(preprocess.wrapper_normalize(pd.DataFrame(adata.X.toarray()))).tocsr()
    elif method == 'normalize_row':
        adata.X = preprocess.normalize_row(adata.X)
    elif method == 'normalize_col':
        adata.X = preprocess.normalize_col(adata.X)
    elif method == 'normalize_norms':
        adata.X = preprocess.normalize_norms(adata.X, axis=axis)
    elif method == 'normalize_max':
        adata.X = sp.coo_matrix(preprocess.normalize_max(pd.DataFrame(adata.X.toarray()), axis=axis)).tocsr()
    elif method == 'normalize_maxmin':
        adata.X = sp.coo_matrix(preprocess.normalize_maxmin(pd.DataFrame(adata.X.toarray()), axis=axis)).tocsr()
    elif method == 'normalize_log_then_total':
        adata = preprocess.normalize_log_then_total(adata, target_sum=target_sum, force_return=True)
    elif method == 'all_zscore':
        X = adata.X.toarray()
        X = np.log(X)
        X = (X - X.mean())/X.std()
        adata.X = sp.coo_matrix(X).tocsr()
    #elif method == 'cross_scale':

    return adata




def test_pruning(cfg):
    #sns.set(style='whitegrid')
    TINY_SIZE = 20  # 39
    SMALL_SIZE = 22  # 42
    MEDIUM_SIZE = 26  # 46
    BIGGER_SIZE = 26  # 46

    plt.rc('font', size=18)  # 35 controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    #rcParams['font.family'] = 'sans-serif'
    #rcParams['font.sans-serif'] = ['Arial']
    # plt.rc('legend',**{'fontsize':16})
    #rcParams["axes.linewidth"] = 1
    #rcParams["legend.frameon"] = True

    img_width = 12
    img_height = 8

    distribution_list = ['poisson', 'gamma', 'lognorm', 'norm', 'powerlaw', 'gamma', 'rayleigh', 'uniform']

    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)
    fig_save_path = cfg.CAME.ROOT+'distribution/'

    print(fig_save_path)

    adata_sm = ad.AnnData(datapair['ov_adjs'][0])
    adata_sm.obs_names = list(datapair['obs_dfs'][0].index)
    adata_sm.var_names = datapair['varnames_node'][0]
    adata_sm = normalize_default(adata_sm, target_sum=1, copy=True)
    sm_ = adata_sm.X.toarray()


    adata_vh = ad.AnnData(datapair['ov_adjs'][1])
    adata_vh.obs_names = list(datapair['obs_dfs'][1].index)
    adata_vh.var_names = datapair['varnames_node'][1]
    adata_vh = normalize_default(adata_vh, target_sum=1, copy=True)
    vh_ = adata_vh.X.toarray()

    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)
    #if cfg.BrainAlign.NODE_TYPE_NUM == 2:
    print(len(datapair['varnames_node'][0]))
    print(len(datapair['varnames_node'][1]))
    #sm_ = datapair['ov_adjs'][0].toarray()
    #vh_ = datapair['ov_adjs'][1].toarray()
    '''
        f = Fitter(sm_x, distributions=['gaussian', 'gamma', 'rayleigh', 'uniform'])
        f.fit()
        # may take some time since by default, all distributions are tried
        # but you call manually provide a smaller set of distributions
        f.summary()
    '''

    print('sm_.shape', sm_.shape)

    sm_sample_list = list(range(0, sm_.shape[0], 2000))
    sm_gene_list = list(range(0, sm_.shape[1], 200))



    for gene_index in sm_gene_list:
        print('Test Mouse sample expression distribution of some gene:')
        sm_x = sm_[:, gene_index]
        plt.figure(figsize=(img_width, img_height))
        ax = sns.distplot(sm_x,
                          bins=50,
                          kde=True,
                          color='red',
                          hist_kws={"linewidth": 15, 'alpha': 1})
        ax.set(xlabel='Expression Value', ylabel='Frequency')
        plt.savefig(fig_save_path + 'sm_gene_{}.png'.format(gene_index))


    for sample_index in sm_sample_list:
        print('Test Mouse gene expression distribution of some sample:')
        sm_y = sm_[sample_index, :]
        plt.figure(figsize=(img_width, img_height))
        ax = sns.distplot(sm_y,
                          bins=50,
                          kde=True,
                          color='red',
                          hist_kws={"linewidth": 15, 'alpha': 1})
        ax.set(xlabel='Expression Value', ylabel='Frequency')
        plt.savefig(fig_save_path + 'sm_sample_{}.png'.format(sample_index))



    # plot all data distributions
    plt.figure(figsize=(img_width, img_height))
    sm_flatten = sm_.flatten()
    n, bins, patches = plt.hist(sm_flatten, bins=200)
    plt.savefig(fig_save_path + 'sm_all.png')
    plt.title('All data distribution of Mouse')


    # test the distribution
    f = Fitter(sm_flatten, distributions=distribution_list, timeout=200)
    f.fit()
    # may take some time since by default, all distributions are tried
    # but you call manually provide a smaller set of distributions
    print(f.summary())


    '''
        f = Fitter(vh_x, distributions=['gaussian', 'gamma', 'rayleigh', 'uniform'])
        f.fit()
        # may take some time since by default, all distributions are tried
        # but you call manually provide a smaller set of distributions
        f.summary()
    '''
    vh_sample_list = list(range(0, vh_.shape[0], 200))
    vh_gene_list = list(range(0, vh_.shape[1], 200))



    for gene_index in vh_gene_list:
        print('Test Human sample expression distribution of some gene:')
        vh_x = vh_[:, gene_index]
        plt.figure(figsize=(img_width, img_height))
        ax = sns.distplot(vh_x,
                          bins=50,
                          kde=True,
                          color='red',
                          hist_kws={"linewidth": 15, 'alpha': 1})
        ax.set(xlabel='Expression Value', ylabel='Frequency')
        plt.savefig(fig_save_path + 'vh_gene_{}.png'.format(gene_index))

    for sample_index in vh_sample_list:
        print('Test Human gene expression distribution of some sample:')
        vh_y = vh_[sample_index, :]
        plt.figure(figsize=(img_width, img_height))
        ax = sns.distplot(vh_y,
                          bins=50,
                          kde=True,
                          color='red',
                          hist_kws={"linewidth": 15, 'alpha': 1})
        ax.set(xlabel='Expression Value', ylabel='Frequency')
        plt.savefig(fig_save_path + 'vh_sample_{}.png'.format(sample_index))



    # plot all data distributions
    plt.figure(figsize=(img_width, img_height))
    vh_flatten = vh_.flatten()
    n, bins, patches = plt.hist(vh_flatten, bins=200)
    plt.savefig(fig_save_path + 'vh_all.png')
    plt.title('All data distribution of Human')


    # test the distribution
    f = Fitter(vh_flatten, distributions=distribution_list, timeout=200)
    f.fit()
    # may take some time since by default, all distributions are tried
    # but you call manually provide a smaller set of distributions
    print(f.summary())






def test_normalize(cfg):
    # sns.set(style='whitegrid')
    TINY_SIZE = 20  # 39
    SMALL_SIZE = 22  # 42
    MEDIUM_SIZE = 26  # 46
    BIGGER_SIZE = 26  # 46

    plt.rc('font', size=18)  # 35 controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # rcParams['font.family'] = 'sans-serif'
    # rcParams['font.sans-serif'] = ['Arial']
    # plt.rc('legend',**{'fontsize':16})
    # rcParams["axes.linewidth"] = 1
    # rcParams["legend.frameon"] = True

    img_width = 12
    img_height = 8

    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)
    fig_save_path = cfg.CAME.ROOT+'test_normalize/{}_{}/'.format(cfg.BrainAlign.normalize_before_pruning_method_1, cfg.BrainAlign.normalize_before_pruning_method_2)

    print(fig_save_path)
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)

    print('mean = ', datapair['ov_adjs'][0].mean().mean(),
          'min = ', datapair['ov_adjs'][0].min().min(),
          'max = ', datapair['ov_adjs'][0].max().max())

    print('mean = ', datapair['ov_adjs'][1].mean().mean(),
          'min = ', datapair['ov_adjs'][1].min().min(),
          'max = ', datapair['ov_adjs'][1].max().max())

    pca_dim = 30

    adata_sm = ad.AnnData(sp.coo_matrix(datapair['ov_adjs'][0].toarray()-datapair['ov_adjs'][0].min().min()).tocsr())
    adata_sm.obs_names = list(datapair['obs_dfs'][0].index)
    adata_sm.var_names = datapair['varnames_node'][0]
    adata_sm = normalize_before_pruning(adata_sm,
                                        method = cfg.BrainAlign.normalize_before_pruning_method_1,
                                        target_sum=cfg.BrainAlign.pruning_target_sum_1,
                                        axis = cfg.BrainAlign.pruning_normalize_axis_1,
                                        force_return=True)
    sc.settings.verbosity = 3
    sc.tl.pca(adata_sm, svd_solver='arpack', n_comps=pca_dim)
    # extract pca coordinates
    adata_sm_pca = ad.AnnData(adata_sm.obsm['X_pca'])
    adata_sm_pca.obs_names = adata_sm.obs_names

    sm_ = adata_sm.X.toarray()
    sm_sample_list = list(range(0, sm_.shape[0], 2000))
    sm_gene_list = list(range(0, sm_.shape[1], 300))

    for gene_index in sm_gene_list:
        print('Test Mouse sample expression distribution of some gene:')
        sm_x = sm_[:, gene_index]
        plt.figure(figsize=(img_width, img_height))
        ax = sns.distplot(sm_x,
                          bins=50,
                          kde=True,
                          color='red',
                          hist_kws={"linewidth": 15, 'alpha': 1})
        ax.set(xlabel='Expression Value', ylabel='Frequency')
        plt.savefig(fig_save_path + 'sm_gene_{}.png'.format(gene_index))

    for sample_index in sm_sample_list:
        print('Test Mouse gene expression distribution of some sample:')
        sm_y = sm_[sample_index, :]
        plt.figure(figsize=(img_width, img_height))
        ax = sns.distplot(sm_y,
                          bins=50,
                          kde=True,
                          color='red',
                          hist_kws={"linewidth": 15, 'alpha': 1})
        ax.set(xlabel='Expression Value', ylabel='Frequency')
        plt.savefig(fig_save_path + 'sm_sample_{}.png'.format(sample_index))
    #sm_ = adata_sm.X.toarray()

    vh_X = datapair['ov_adjs'][1].toarray()
    vh_X = vh_X - vh_X.min()
    vh_X = vh_X * (adata_sm.X.max().max() - adata_sm.X.min().min()) / vh_X.max().max()
    vh_X = sp.coo_matrix(vh_X).tocsr()
    adata_vh = ad.AnnData(vh_X)
    adata_vh.obs_names = list(datapair['obs_dfs'][1].index)
    adata_vh.var_names = datapair['varnames_node'][1]
    adata_vh = normalize_before_pruning(adata_vh,
                                        method = cfg.BrainAlign.normalize_before_pruning_method_2,
                                        target_sum=cfg.BrainAlign.pruning_target_sum_2,
                                        axis=cfg.BrainAlign.pruning_normalize_axis_2,
                                        force_return=True)
    sc.settings.verbosity = 3
    sc.tl.pca(adata_vh, svd_solver='arpack', n_comps=pca_dim)
    # extract pca coordinates
    adata_vh_pca = ad.AnnData(adata_vh.obsm['X_pca'])
    adata_vh_pca.obs_names = adata_vh.obs_names
    #vh_ = adata_vh.X.toarray()

    vh_ = adata_vh.X.toarray()

    vh_sample_list = list(range(0, vh_.shape[0], 500))
    vh_gene_list = list(range(0, vh_.shape[1], 300))

    for gene_index in vh_gene_list:
        print('Test Human sample expression distribution of some gene:')
        vh_x = vh_[:, gene_index]
        plt.figure(figsize=(img_width, img_height))
        ax = sns.distplot(vh_x,
                          bins=50,
                          kde=True,
                          color='red',
                          hist_kws={"linewidth": 15, 'alpha': 1})
        ax.set(xlabel='Expression Value', ylabel='Frequency')
        plt.savefig(fig_save_path + 'vh_gene_{}.png'.format(gene_index))

    for sample_index in vh_sample_list:
        print('Test Human gene expression distribution of some sample:')
        vh_y = vh_[sample_index, :]
        plt.figure(figsize=(img_width, img_height))
        ax = sns.distplot(vh_y,
                          bins=50,
                          kde=True,
                          color='red',
                          hist_kws={"linewidth": 15, 'alpha': 1})
        ax.set(xlabel='Expression Value', ylabel='Frequency')
        plt.savefig(fig_save_path + 'vh_sample_{}.png'.format(sample_index))



    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)
    #if cfg.BrainAlign.NODE_TYPE_NUM == 2:
    print(len(datapair['varnames_node'][0]))
    print(len(datapair['varnames_node'][1]))
    #sm_ = datapair['ov_adjs'][0].toarray()
    #vh_ = datapair['ov_adjs'][1].toarray()

    adata_sm_pca.obs['dataset'] = 'mouse'
    adata_vh_pca.obs['dataset'] = 'human'

    adata_all = ad.concat([adata_sm_pca, adata_vh_pca])
    print(adata_all)

    sc.pp.neighbors(adata_all, n_neighbors=30, metric='cosine', use_rep='X')
    sc.tl.umap(adata_all)

    save_path = fig_save_path#cfg.BrainAlign.embeddings_file_path + 'figs/test_normalization/{}'.format(cfg.BrainAlign.normalize_before_pruning_method)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with plt.rc_context({"figure.figsize": (8, 6)}):
        sc.pl.umap(adata_all, color=['dataset'], return_fig=True, legend_loc='on data').savefig(
            save_path + 'umap_{}_{}.'.format(cfg.BrainAlign.normalize_before_pruning_method_1, cfg.BrainAlign.normalize_before_pruning_method_2) + cfg.BrainAlign.fig_format, format=cfg.BrainAlign.fig_format)
        plt.subplots_adjust(right=0.3)

    plt.figure(figsize=(img_width, img_height))
    sm_flatten = adata_sm.X.toarray().flatten()
    n, bins, patches = plt.hist(sm_flatten, bins=200)
    plt.savefig(save_path + '{}_{}_sm_all.png'.format(cfg.BrainAlign.normalize_before_pruning_method_1, cfg.BrainAlign.normalize_before_pruning_method_2))
    plt.title('All data distribution of Mouse')

    plt.figure(figsize=(img_width, img_height))
    vh_flatten = adata_vh.X.toarray().flatten()
    n, bins, patches = plt.hist(vh_flatten, bins=200)
    plt.savefig(save_path + '{}_{}_vh_all.png'.format(cfg.BrainAlign.normalize_before_pruning_method_1, cfg.BrainAlign.normalize_before_pruning_method_2))
    plt.title('All data distribution of Human')


def test_interation(cfg):
    # sns.set(style='whitegrid')
    TINY_SIZE = 20  # 39
    SMALL_SIZE = 22  # 42
    MEDIUM_SIZE = 26  # 46
    BIGGER_SIZE = 26  # 46

    plt.rc('font', size=18)  # 35 controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # rcParams['font.family'] = 'sans-serif'
    # rcParams['font.sans-serif'] = ['Arial']
    # plt.rc('legend',**{'fontsize':16})
    # rcParams["axes.linewidth"] = 1
    # rcParams["legend.frameon"] = True

    img_width = 12
    img_height = 8

    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)
    fig_save_path = cfg.CAME.ROOT+'test_normalize/{}/'.format(cfg.BrainAlign.normalize_before_pruning_method)

    print(fig_save_path)
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)

    print('mean = ', datapair['ov_adjs'][0].mean().mean(),
          'min = ', datapair['ov_adjs'][0].min().min(),
          'max = ', datapair['ov_adjs'][0].max().max())

    print('mean = ', datapair['ov_adjs'][1].mean().mean(),
          'min = ', datapair['ov_adjs'][1].min().min(),
          'max = ', datapair['ov_adjs'][1].max().max())

    pca_dim = 30

    adata_sm = ad.AnnData(sp.coo_matrix(datapair['ov_adjs'][0].toarray()-datapair['ov_adjs'][0].min().min()).tocsr())
    adata_sm.obs_names = list(datapair['obs_dfs'][0].index)
    adata_sm.var_names = datapair['varnames_node'][0]
    adata_sm = normalize_before_pruning(adata_sm,
                                        method = cfg.BrainAlign.normalize_before_pruning_method,
                                        target_sum=cfg.BrainAlign.pruning_target_sum,
                                        axis = cfg.BrainAlign.pruning_normalize_axis,
                                        force_return=True)
    sc.settings.verbosity = 3
    sc.tl.pca(adata_sm, svd_solver='arpack', n_comps=pca_dim)
    # extract pca coordinates
    adata_sm_pca = ad.AnnData(adata_sm.obsm['X_pca'])
    adata_sm_pca.obs_names = adata_sm.obs_names

    sm_ = adata_sm.X.toarray()
    sm_sample_list = list(range(0, sm_.shape[0], 2000))
    sm_gene_list = list(range(0, sm_.shape[1], 300))

    for gene_index in sm_gene_list:
        print('Test Mouse sample expression distribution of some gene:')
        sm_x = sm_[:, gene_index]
        plt.figure(figsize=(img_width, img_height))
        ax = sns.distplot(sm_x,
                          bins=50,
                          kde=True,
                          color='red',
                          hist_kws={"linewidth": 15, 'alpha': 1})
        ax.set(xlabel='Expression Value', ylabel='Frequency')
        plt.savefig(fig_save_path + 'sm_gene_{}.png'.format(gene_index))

    for sample_index in sm_sample_list:
        print('Test Mouse gene expression distribution of some sample:')
        sm_y = sm_[sample_index, :]
        plt.figure(figsize=(img_width, img_height))
        ax = sns.distplot(sm_y,
                          bins=50,
                          kde=True,
                          color='red',
                          hist_kws={"linewidth": 15, 'alpha': 1})
        ax.set(xlabel='Expression Value', ylabel='Frequency')
        plt.savefig(fig_save_path + 'sm_sample_{}.png'.format(sample_index))
    #sm_ = adata_sm.X.toarray()

    vh_X = datapair['ov_adjs'][1].toarray()
    vh_X = vh_X - vh_X.min()
    vh_X = vh_X * (adata_sm.X.max().max() - adata_sm.X.min().min()) / vh_X.max().max()
    vh_X = sp.coo_matrix(vh_X).tocsr()
    adata_vh = ad.AnnData(vh_X)
    adata_vh.obs_names = list(datapair['obs_dfs'][1].index)
    adata_vh.var_names = datapair['varnames_node'][1]
    adata_vh = normalize_before_pruning(adata_vh,
                                        method = cfg.BrainAlign.normalize_before_pruning_method,
                                        target_sum=cfg.BrainAlign.pruning_target_sum,
                                        axis=cfg.BrainAlign.pruning_normalize_axis,
                                        force_return=True)
    sc.settings.verbosity = 3
    sc.tl.pca(adata_vh, svd_solver='arpack', n_comps=pca_dim)
    # extract pca coordinates
    adata_vh_pca = ad.AnnData(adata_vh.obsm['X_pca'])
    adata_vh_pca.obs_names = adata_vh.obs_names
    #vh_ = adata_vh.X.toarray()

    vh_ = adata_vh.X.toarray()

    vh_sample_list = list(range(0, vh_.shape[0], 500))
    vh_gene_list = list(range(0, vh_.shape[1], 300))

    for gene_index in vh_gene_list:
        print('Test Human sample expression distribution of some gene:')
        vh_x = vh_[:, gene_index]
        plt.figure(figsize=(img_width, img_height))
        ax = sns.distplot(vh_x,
                          bins=50,
                          kde=True,
                          color='red',
                          hist_kws={"linewidth": 15, 'alpha': 1})
        ax.set(xlabel='Expression Value', ylabel='Frequency')
        plt.savefig(fig_save_path + 'vh_gene_{}.png'.format(gene_index))

    for sample_index in vh_sample_list:
        print('Test Human gene expression distribution of some sample:')
        vh_y = vh_[sample_index, :]
        plt.figure(figsize=(img_width, img_height))
        ax = sns.distplot(vh_y,
                          bins=50,
                          kde=True,
                          color='red',
                          hist_kws={"linewidth": 15, 'alpha': 1})
        ax.set(xlabel='Expression Value', ylabel='Frequency')
        plt.savefig(fig_save_path + 'vh_sample_{}.png'.format(sample_index))



    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)
    #if cfg.BrainAlign.NODE_TYPE_NUM == 2:
    print(len(datapair['varnames_node'][0]))
    print(len(datapair['varnames_node'][1]))
    #sm_ = datapair['ov_adjs'][0].toarray()
    #vh_ = datapair['ov_adjs'][1].toarray()

    adata_sm_pca.obs['dataset'] = 'mouse'
    adata_vh_pca.obs['dataset'] = 'human'

    adata_all = ad.concat([adata_sm_pca, adata_vh_pca])
    print(adata_all)

    sc.pp.neighbors(adata_all, n_neighbors=30, metric='cosine', use_rep='X')
    sc.tl.umap(adata_all)

    save_path = fig_save_path#cfg.BrainAlign.embeddings_file_path + 'figs/test_normalization/{}'.format(cfg.BrainAlign.normalize_before_pruning_method)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with plt.rc_context({"figure.figsize": (8, 6)}):
        sc.pl.umap(adata_all, color=['dataset'], return_fig=True, legend_loc='on data').savefig(
            save_path + 'umap_{}_{}_{}.'.format(cfg.BrainAlign.normalize_before_pruning_method, cfg.BrainAlign.pruning_target_sum, cfg.BrainAlign.pruning_normalize_axis) + cfg.BrainAlign.fig_format, format=cfg.BrainAlign.fig_format)
        plt.subplots_adjust(right=0.3)

    plt.figure(figsize=(img_width, img_height))
    sm_flatten = adata_sm.X.toarray().flatten()
    n, bins, patches = plt.hist(sm_flatten, bins=200)
    plt.savefig(save_path + '{}_{}_{}_sm_all.png'.format(cfg.BrainAlign.normalize_before_pruning_method, cfg.BrainAlign.pruning_target_sum, cfg.BrainAlign.pruning_normalize_axis))
    plt.title('All data distribution of Mouse')

    plt.figure(figsize=(img_width, img_height))
    vh_flatten = adata_vh.X.toarray().flatten()
    n, bins, patches = plt.hist(vh_flatten, bins=200)
    plt.savefig(save_path + '{}_{}_{}_vh_all.png'.format(cfg.BrainAlign.normalize_before_pruning_method, cfg.BrainAlign.pruning_target_sum, cfg.BrainAlign.pruning_normalize_axis))
    plt.title('All data distribution of Human')



if __name__ == '__main__':
    print('loading process')
    mh_ = np.array([[1, 0, 0,0], [0, 1, 0, 0], [0, 0, 1, 0]])
    #print(mh_)
    feats_m = np.random.random((7, 2))
    #print(feats_m)
    binary_M = 3
    binary_H = 4
    feats_m_temp = feats_m.copy()
    for i in range(binary_M):
        h_index = [x + binary_M for x in np.nonzero(mh_[i, :])[0]]
        if len(h_index) > 0:
            #print(h_index)
            h_index.append(i)
            #print(h_index)
            feat_h = feats_m[h_index, :]
            #print(np.mean(feat_h, axis=0))
            feats_m_temp[i, :] = np.mean(feat_h, axis=0)

    for j in range(binary_H):
        m_index = list(np.nonzero(mh_[:, j])[0])
        #print(m_index)
        if len(m_index) > 0:
            m_index.append(int(binary_M + j))
            feat_g = feats_m[m_index, :]
            #print(m_index)
            # feat = np.concatenate((feat_m, np.reshape(feats_m[j + binary_M, :], (1, -1))), axis=0)
            #print(np.mean(feat_g, axis=0))
            feats_m_temp[binary_M + j, :] = np.mean(feat_g, axis=0)
    #
    # print(feats_m[[0, 3], :])
    # print(np.mean(feats_m[[0, 3], :], axis=0))
    # print(feats_m[[3, 0], :])
    # print(np.mean(feats_m[[3, 0], :], axis=0))
    feats_m = feats_m_temp
    print(feats_m)
