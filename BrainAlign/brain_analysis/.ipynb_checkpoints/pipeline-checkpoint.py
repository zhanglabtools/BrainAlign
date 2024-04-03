# -- coding: utf-8 --
# @Time : 2022/10/17 15:32
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : pipeline.py
import os.path

import scanpy as sc, anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import scipy.sparse as sp
import re, seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from collections import OrderedDict, Counter
import pickle
import logging
from BrainAlign.brain_analysis.logger import setup_logger
from BrainAlign.brain_analysis.data_utils import check_dirs, train_test_val_split
from typing import Sequence, Union, Mapping, List, Optional, Dict, Callable
from BrainAlign.came.utils.preprocess import *
import BrainAlign.came.utils.preprocess as pp
from BrainAlign.came.datapair.unaligned import make_features
from BrainAlign.brain_analysis.process import transform_pca_embedding_np

from BrainAlign.brain_analysis.pipline_analysis_alignment import alignment_STs_analysis

from sklearn.neighbors import kneighbors_graph

from scipy.sparse import vstack

from BrainAlign.SR_RSC import main_sr_rsc

from matplotlib import rcParams

import sys

class BrainAligner:
    def __init__(self, cfg):
        self.cfg = cfg

        self.species_num = len(cfg.BrainAlign.species_list)

        self.adata_list = [sc.read_h5ad(adata) for adata in cfg.BrainAlign.path_rawdata_list]

        self.adatas = [sc.read_h5ad(adata) for adata in cfg.BrainAlign.path_rawdata_list]

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(filename)s-%(lineno)d-%(funcName)s(): '
                   '%(levelname)s\n %(message)s')
        check_dirs(cfg.BrainAlign.result_save_path)
        self.logger = setup_logger("Running BrainAlign and analysis", cfg.BrainAlign.result_save_path, if_train=None)

        self.df_varmap_species_list = [] # with length = N_species
        for i in range(len(cfg.BrainAlign.species_list)):
            df_varmap = pd.read_csv(cfg.BrainAlign.path_varmap_list[i], sep='\t')
            df_varmap = df_varmap.rename({'Gene stable ID': cfg.BrainAlign.species_list[i] + 'gene stable ID',
                                          'Gene name': cfg.BrainAlign.species_list[i] + ' gene name'}, axis='columns')
            self.df_varmap_species_list.append(df_varmap)

        # generate binary relations for the first species
        self.df_varmap_list_binary = [None] # with length = (N_species)
        df_varmap_multiple = self.df_varmap_species_list[0]
        for i in range(1, len(cfg.BrainAlign.species_list)):
            df_varmap_binary = df_varmap_multiple[[cfg.BrainAlign.species_list[0] + ' gene name',
                                                   cfg.BrainAlign.species_list[i] + ' gene name']]
            self.df_varmap_list_binary.append(df_varmap_binary)
            print(df_varmap_binary)

        # Query and collect all the binary relations among multiple species
        self.df_varmap_list_binary_list = [[None] * self.species_num for s in range(self.species_num)] #[[None, None, None,None], [None, ...]]
        for i in range(self.species_num):
            df_varmap_multiple = self.df_varmap_species_list[i] # multiple species raw homology relationships
            r = list(range(self.species_num)) # [0, 1, 2, 3]
            r.remove(i) # [1, 2, 3]
            for j in r:
                df_varmap_binary = df_varmap_multiple[[cfg.BrainAlign.species_list[i] + ' gene name',
                                                       cfg.BrainAlign.species_list[j] + ' gene name']]
                self.df_varmap_list_binary_list[i][j] = df_varmap_binary.dropna()



    def preprocess_unaligned(self):

        self.logger.info('================ preprocessing ===============')
        params_preproc = dict(
            target_sum=self.cfg.Preprocess.norm_target_sum,
            n_top_genes=self.cfg.Preprocess.n_top_genes,  # 500, # 2000
            n_pcs=self.cfg.Preprocess.n_pcs,
            nneigh=self.cfg.Preprocess.nneigh_scnet,
            copy=True,
        )

        for i in range(len(self.adata_list)):
           self.adata_list[i] = pp.quick_preprocess(self.adata_list[i], normalize_data=self.cfg.BrainAlign.do_normalize[i], **params_preproc)

        # the single-cell network
        if self.cfg.Preprocess.use_scnets:
            scnets = [pp.get_scnet(adata) for adata in self.adata_list]
        else:
            scnets = None

        # get HVGs
        hvgs_list = [pp.get_hvgs(adata) for adata in self.adata_list]

        # cluster labels
        for i in range(len(self.adata_list)):
            clust_lbs = pp.get_leiden_labels(
                self.adata_list[i], force_redo=True,
                nneigh=self.cfg.Preprocess.nneigh_clust,
                neighbors_key='clust',
                key_added=self.cfg.Preprocess.key_clust,
                copy=False
            )
            self.adata_list[i].obs[self.cfg.Preprocess.key_clust] = clust_lbs

        # DEGs
        params_deg = dict(
            n=self.cfg.Preprocess.ntop_deg,
            force_redo=False, inplace=True, do_normalize=False,
            return_info=True,
        )
        # adata1&2 have already been normalized before
        deg_info_list_class = []
        deg_info_list_cluster = []
        for i in range(len(self.adata_list)):
            adata = self.adata_list[i]
            if self.cfg.BrainAlign.key_class_list[i] != None:
                #print(adata.obs_names)
                #print(Counter(adata.var_names))
                deg_info_list_class.append(pp.compute_and_get_DEGs(adata, self.cfg.BrainAlign.key_class_list[i], **params_deg))
            else:
                deg_info_list_class.append(None)
            deg_info_list_cluster.append(pp.compute_and_get_DEGs(adata, self.cfg.Preprocess.key_clust, **params_deg))

        # ======== Node-Features ========
        # select genes as features
        self.logger.info('Select homologous genes as initial cross species features...')
        if self.cfg.Preprocess.ext_feats is None:
            vars_feat_list = [set() for i in range(len(self.cfg.BrainAlign.species_list))]
        else:
            vars_feat_list = [set(self.cfg.Preprocess.ext_feats[i]) for i in range(len(self.cfg.BrainAlign.species_list))]

        for i in range(len(self.cfg.BrainAlign.species_list)):
            if self.cfg.BrainAlign.key_class_list[i] != None:
                vars_feat_list[i].update(pp.top_markers_from_info(deg_info_list_class[i], self.cfg.Preprocess.ntop_deg))
            vars_feat_list[i].update(pp.top_markers_from_info(deg_info_list_cluster[i], self.cfg.Preprocess.ntop_deg))

        vars_feat_list = [sorted(vars_feat) for vars_feat in vars_feat_list]

        # ======== VAR-Nodes ========
        # select genes as nodes
        if self.cfg.Preprocess.ext_nodes is None:
            nodes_list= [set() for i in range(len(self.cfg.BrainAlign.species_list))]
        else:
            nodes_list = [set(self.cfg.Preprocess.ext_nodes[i]) for i in range(len(self.cfg.BrainAlign.species_list))]
        node_source = self.cfg.Preprocess.node_source.lower()
        if 'deg' in node_source:
            for i in range(len(self.cfg.BrainAlign.species_list)):
                if self.cfg.BrainAlign.key_class_list[i] != None:
                    nodes_list[i].update(pp.top_markers_from_info(deg_info_list_class[i], self.cfg.Preprocess.ntop_deg_nodes))
                nodes_list[i].update(pp.top_markers_from_info(deg_info_list_cluster[i], self.cfg.Preprocess.ntop_deg_nodes))
        if 'hvg' in node_source:
            for i in range(len(self.cfg.BrainAlign.species_list)):
                nodes_list[i].update(hvgs_list[i])

        vars_as_nodes = [sorted(nodes) for nodes in nodes_list] # genes as nodes for each species


        self.dct = dict(
            adatas=self.adatas,
            vars_feat=vars_feat_list,
            vars_as_nodes=vars_as_nodes,
            scnets=scnets,
        )

        # normalization of adatas
        self.do_normalize = self.cfg.BrainAlign.do_normalize
        if isinstance(self.cfg.BrainAlign.do_normalize, bool):
            self.do_normalize = [self.cfg.BrainAlign.do_normalize] * self.species_num

        for i in range(len(self.cfg.BrainAlign.species_list)):

            if self.do_normalize[i]:
                self.adatas[i] = pp.normalize_default(
                    self.adatas[i], target_sum=self.cfg.Preprocess.norm_target_sum, force_return=True)

        adata_raw_list = [adata.raw.to_adata() if adata.raw is not None else adata for adata in self.adatas]

        if vars_as_nodes is None:
            vars_as_nodes = vars_feat_list

        # features selected for modeling. (e.g. DEGs, HVGs)
        obs_list = [adata.obs.copy() for adata in self.adatas]

        vars_all_list = [adata_raw.var_names for adata_raw in adata_raw_list]

        #--------------------------------------------------------------------------
        # Compute aligned features for each species
        # 1. Get each pair of relation (homologous gene) for each pair of species (to species 0)
        # 2. Do intersection
        vars_nodes1 = vars_as_nodes[0]
        vars_use1 = vars_feat_list[0]
        vars_all1 = vars_all_list[0]
        adata_raw1 = adata_raw_list[0]
        adata1 = self.adatas[0]

        self.logger.info('Get sample node features...')
        self.feature_list = []
        self.vnames_feat_list = []
        vnames_species0_all = []

        for i in range(1, len(self.cfg.BrainAlign.species_list)):
            vars_nodes2 = vars_as_nodes[i]
            vars_use2 = vars_feat_list[i]
            vars_all2 = vars_all_list[i]
            adata_raw2 = adata_raw_list[i]
            adata2 = self.adatas[i]

            df_varmap = self.df_varmap_list_binary[i]

            submaps = pp.subset_matches(self.df_varmap_list_binary[i], vars_nodes1, vars_nodes2,
                                        union=self.cfg.Preprocess.union_var_nodes)
            submaps = pp.subset_matches(submaps, vars_all1, vars_all2, union=False) # Check if this step is useless

            if self.cfg.Preprocess.with_single_vnodes:
                vv_adj, vnodes1, vnodes2 = pp.make_bipartite_adj(
                    submaps, vars_nodes1, vars_nodes2,
                    with_singleton=True, symmetric=True,
                )
            else:
                vv_adj, vnodes1, vnodes2 = pp.make_bipartite_adj(
                    submaps, with_singleton=False, symmetric=True,
                )
            # --- ov_adjs (unaligned features, for making `ov_adj`)
            ov_adjs1 = adata_raw1[:, vnodes1].X  # for ov_adj
            ov_adjs2 = adata_raw2[:, vnodes2].X
            print('ov_adjs1.shape', ov_adjs1.shape)
            print('ov_adjs2.shape', ov_adjs2.shape)
            var1 = adata1.var.copy().loc[vnodes1, :]
            print('len of var1', len(var1))
            var2 = adata2.var.copy().loc[vnodes2, :]
            print('len of var2', len(var2))

            # --- node features
            # features = list(map(_check_sparse_toarray, [features1, features2]))
            features, trans = make_features(
                (adata1, adata2), vars_use1, vars_use2, df_varmap, col_weight=self.cfg.Preprocess.col_weight,
                union_node_feats=self.cfg.Preprocess.union_node_feats,
                keep_non1v1=self.cfg.Preprocess.keep_non1v1_feats, non1v1_trans_to=0,
            )

            vnames_feat1, vnames_feat2 = trans.reduce_to_align()
            #print("trans.shape=", trans.shape)

            vnames_species0_all.append(vnames_feat1)

            if i==1:
                self.feature_list.append(features[0])
                self.feature_list.append(features[1])
                self.vnames_feat_list.append(vnames_feat1)
                self.vnames_feat_list.append(vnames_feat2)
            else:
                self.feature_list.append(features[1])
                self.vnames_feat_list.append(vnames_feat2)

        for i in range(self.species_num):
            self.logger.info('self.feature_list[i].shape: {}'.format(self.feature_list[i].shape))

        vnames_species0_all_set = [set(x) for x in vnames_species0_all]
        # align for all the species
        if self.species_num >= 3:
            intersetion_feat_species0 = list(set.intersection(*vnames_species0_all_set))

            indices_vnames_species = [list(self.vnames_feat_list[0]).index(x) for x in intersetion_feat_species0]

            self.feature_list[0] = self.feature_list[0][:, indices_vnames_species]

            self.vnames_feat_list[0] = intersetion_feat_species0
            for i in range(1, self.species_num):
                indices_vnames_species, self.vnames_feat_list[i] = \
                    find_match_indices(list(vnames_species0_all[i-1]), intersetion_feat_species0, self.vnames_feat_list[i])
                self.feature_list[i] = self.feature_list[i][:, indices_vnames_species]

            for i in range(self.species_num):
                print(self.vnames_feat_list[i])

        # -------------------------------------------------------------------------------------
        self.vars_all_list = vars_all_list
        self.vars_as_nodes = vars_as_nodes
        self.vars_feat_list = vars_feat_list
        self.obs_list = obs_list
        self.adata_raw_list = adata_raw_list

        # get union gene sets as nodes
        # Genes of each species has relations with the other species, thus for this species, the gene set
        # is the union of all the gene sets.
        self.gvar_as_nodes_list = []
        self.ov_adjs_list = []
        for i in range(self.species_num):
            # This function uses several class-wise variables such
            gvar_as_nodes = self.get_varname_as_nodes(i, self.df_varmap_list_binary_list[i])
            self.gvar_as_nodes_list.append(gvar_as_nodes)
            self.ov_adjs_list.append(adata_raw_list[i][:, gvar_as_nodes].X)
        for gvar_as_nodes in self.gvar_as_nodes_list:
            print('gvar_as_nodes number:', len(gvar_as_nodes))

        # -------------------------------------------------------------------------

        # oo-adj
        self.oo_adj_list = scnets
        print('oo_adj list')
        for oo_adj in self.oo_adj_list:
            print(type(oo_adj), oo_adj.shape)


        self.cfg.BrainAlign.S_list = [adata.n_obs for adata in self.adatas]
        self.cfg.BrainAlign.S = int(np.sum(self.cfg.BrainAlign.S_list))
        self.logger.info('Sample number S_list: {}'.format(self.cfg.BrainAlign.S_list))
        self.logger.info('Sample number S: {}'.format(self.cfg.BrainAlign.S))

        # Concatenate the oo_adjs to the global oo_adj matrix
        obs_nodes_num_list = [adt.n_obs for adt in self.adatas]
        obs_nodes_num_cumu_list = [int(np.sum(np.array(obs_nodes_num_list[0:i + 1]))) for i in range(self.species_num)]
        obs_nodes_num_cumu_list = [int(0)] + obs_nodes_num_cumu_list

        if self.cfg.BrainAlign.node_relation == 'knn':
            oo_row = []
            oo_col = []
            oo_data = []
            for i in range(self.species_num):
                row_col_add = obs_nodes_num_cumu_list[i]
                oo_adj_mat = self.oo_adj_list[i].tocoo()
                oo_row = np.concatenate((oo_row, oo_adj_mat.row + row_col_add))
                oo_col = np.concatenate((oo_col, oo_adj_mat.col + row_col_add))
                oo_data = np.concatenate((oo_data, oo_adj_mat.data))
            self.oo_adjs = sp.coo_matrix((oo_data, (oo_row, oo_col)), shape=(self.cfg.BrainAlign.S, self.cfg.BrainAlign.S))
        elif self.cfg.BrainAlign.node_relation == 'spatial':
            self.oo_adjs = get_spatial_relation_multiple(self.cfg, obs_nodes_num_cumu_list)
        self.logger.info('oo_adjs shape: {}'.format(self.oo_adjs.shape))


        # get var_as_nodes features
        self.logger.info('Get gene node features via ov_adj * o_features...')
        self.feature_list, self.var_node_feature_list = self.init_var_features(self.ov_adjs_list,
                                                            self.feature_list,
                                                            pca_dim = self.cfg.Preprocess.feature_pca_dim,
                                                            if_pca = self.cfg.Preprocess.if_feature_pca)
        for i in range(self.species_num):
            print('feature shape after pca:', self.feature_list[i].shape)
        for i in range(self.species_num):
            print('gene feature shape after pca:', self.var_node_feature_list[i].shape)
        # get the vv_adjs matrix sequentially
        # 1. The global vv_adjs is composed of N*N submatrix, each is constructed from homology relations
        # 2. The submatrix in the diagonal is Identity matrix
        var_nodes_num_list = [len(gvar_as_nodes) for gvar_as_nodes in self.gvar_as_nodes_list]
        var_nodes_num_cumu_list = [int(np.sum(np.array(var_nodes_num_list[0:i+1]))) for i in range(self.species_num)]
        var_nodes_num_cumu_list = [int(0)] + var_nodes_num_cumu_list

        # assign S, M, S_list and M_list

        self.cfg.BrainAlign.M_list = var_nodes_num_list
        self.cfg.BrainAlign.M = int(np.sum(self.cfg.BrainAlign.M_list))
        self.logger.info('Gene number M_list: {}'.format(self.cfg.BrainAlign.M_list))
        self.logger.info('Gene number M: {}'.format(self.cfg.BrainAlign.M))

        vv_row = []
        vv_col = []
        vv_data = []
        for i in range(self.species_num):
                row_add = var_nodes_num_cumu_list[i]
                row_name_list = self.gvar_as_nodes_list[i]
                for j in range(self.species_num):
                    column_add = var_nodes_num_cumu_list[j]
                    column_name_list = self.gvar_as_nodes_list[j]
                    if i == j:
                        vv_mat_row = np.array(range(var_nodes_num_list[i])) + row_add
                        vv_mat_col = np.array(range(var_nodes_num_list[j])) + column_add
                        vv_mat_data = np.ones(var_nodes_num_list[i])
                    else:
                        df_varmap_binary = sub_df_map(self.df_varmap_list_binary_list[i][j], row_name_list, column_name_list)
                        #print('df_varmap_binary', df_varmap_binary)
                        vv_adj_mat = make_bipartite_adj_simple(df_varmap_binary, row_name_list, column_name_list)
                        vv_mat_row = vv_adj_mat.row
                        vv_mat_col = vv_adj_mat.col
                        vv_mat_data = vv_adj_mat.data

                    vv_row = np.concatenate((vv_row, vv_mat_row))
                    vv_col = np.concatenate((vv_col, vv_mat_col))
                    vv_data = np.concatenate((vv_data, vv_mat_data))

        vv_adjs_sparse = sp.coo_matrix((vv_data, (vv_row, vv_col)), shape=(self.cfg.BrainAlign.M, self.cfg.BrainAlign.M))
        self.vv_adjs = vv_adjs_sparse
        self.logger.info('vv_adjs shape: {}'.format(self.vv_adjs.shape))

        # Concatenate the ov_adjs to the global ov_adjs matrix
        ov_row = []
        ov_col = []
        ov_data = []
        for i in range(self.species_num):
            row_add = obs_nodes_num_cumu_list[i]
            col_add = var_nodes_num_cumu_list[i]
            ov_adj_mat = sp.coo_matrix(self.ov_adjs_list[i])
            ov_row = np.concatenate((ov_row, ov_adj_mat.row + row_add))
            ov_col = np.concatenate((ov_col, ov_adj_mat.col + col_add))
            ov_data = np.concatenate((ov_data, ov_adj_mat.data))

        self.ov_adjs = sp.coo_matrix((ov_data, (ov_row, ov_col)), shape=(self.cfg.BrainAlign.S, self.cfg.BrainAlign.M))
        self.logger.info('ov_adjs shape: {}'.format(self.ov_adjs.shape))

        self.S_features = sp.vstack(self.feature_list)
        #print(self.S_features)
        self.logger.info('Sample features shape: {}'.format(self.S_features.shape))
        self.M_features = sp.vstack(self.var_node_feature_list)
        self.logger.info('Gene features shape: {}'.format(self.M_features.shape))

        self.obs_name_list = []
        self.vars_name_list = []
        for i in range(self.species_num):
            self.obs_name_list = self.obs_name_list + list(self.adatas[i].obs_names)
            self.vars_name_list = self.vars_name_list + list(self.gvar_as_nodes_list[i])

        self.logger.info('Saving oo_adjs, ov_adjs, vv_adjs, o_features, and v_features...')
        data_multiple_dict = {'species':self.cfg.BrainAlign.species_list,
                              'oo_adjs':self.oo_adjs,
                              'ov_adjs':self.ov_adjs,
                              'vv_adjs':self.vv_adjs,
                              'o_feat':self.S_features,
                              'o_name':self.obs_name_list,
                              'v_name':self.vars_name_list,
                              'v_feat':self.M_features,
                              'o_list': self.cfg.BrainAlign.S_list,
                              'v_list':self.cfg.BrainAlign.M_list}
        self.logger.info('Initial data:', data_multiple_dict)

        if not os.path.exists(self.cfg.BrainAlign.DATA_PATH):
            os.makedirs(self.cfg.BrainAlign.DATA_PATH)

        with open(self.cfg.BrainAlign.DATA_PATH + 'data_init.pickle', 'wb') as handle:
            pickle.dump(data_multiple_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info('Saved data_init.pickle to {}'.format(self.cfg.BrainAlign.DATA_PATH + 'data_init.pickle'))
        return None


    def get_brainalign_input(self):
        self.logger.info(self.cfg)
        self.logger.info('------------Processing the data to BrainAlign input format...-------------------------')

        with open(self.cfg.BrainAlign.DATA_PATH + 'data_init.pickle', 'rb') as handle:
            data_initial = pickle.load(handle)

        dict_keys = ['t_info', 'rel2id', 'id2rel', 'node2lid', 'node2gid', 'gid2node', 'n_class', 'n_feat', 'relations',
                     'types',
                     'undirected_relations', 'r_info']
        meta_data_dict = {k: None for k in dict_keys}

        self.logger.info("Generating meta_data...")
        meta_data_dict['t_info'] = {'s': {'ind': range(0, self.cfg.BrainAlign.S), 'cnt': self.cfg.BrainAlign.S},
                                    'm': {'ind': range(self.cfg.BrainAlign.S, self.cfg.BrainAlign.S + self.cfg.BrainAlign.M),
                                          'cnt': self.cfg.BrainAlign.M}}
        meta_data_dict['rel2id'] = {'s-m': 0, 'm-s': 1, 's-s': 2, 'm-m': 3}
        meta_data_dict['id2real'] = {0: 's-m', 1: 'm-s', 2: 's-s', 3: 'm-m'}

        sample_name_list = data_initial['o_name']
        gene_name_list = data_initial['v_name']

        lid_list = list(range(0, len(sample_name_list))) + list(range(0, len(gene_name_list)))

        # here, the gene names may be replicated for different species
        node_list = ['s' + str(i) for i in range(0, len(sample_name_list))] + ['m' + str(i) for i in
                                                                               range(0, len(gene_name_list))]
        grid_list = list(range(0, len(sample_name_list + gene_name_list)))
        meta_data_dict['node2lid'] = {k: v for k, v in zip(node_list, lid_list)}
        meta_data_dict['lid2node'] = {k: v for k, v in zip(lid_list, node_list)}

        meta_data_dict['node2gid'] = {k: v for k, v in zip(node_list, grid_list)}
        meta_data_dict['gid2node'] = {k: v for k, v in zip(grid_list, node_list)}

        meta_data_dict['n_class'] = self.species_num
        meta_data_dict['n_feat'] = data_initial['o_feat'].shape[1]

        meta_data_dict['relations'] = ['s-m', 'm-s', 's-s', 'm-m']
        meta_data_dict['types'] = ['s', 'm']
        # meta_data_dict['undirected_relations'] = {'s-s', 'm-m'}
        meta_data_dict['r_info'] = {}

        if not os.path.exists(self.cfg.SRRSC_args.data_path):
            os.makedirs(self.cfg.SRRSC_args.data_path)

        f = open(self.cfg.SRRSC_args.data_path + "meta_data.pkl", "wb")
        pickle.dump(meta_data_dict, f)

        self.logger.info("Generating labels...")
        all_label_X = np.array(list(range(0, len(sample_name_list)+len(gene_name_list)))).reshape(-1, 1)
        print(all_label_X.shape)
        all_label_Y = []
        for i in range(self.species_num):
            all_label_Y = all_label_Y + [int(i)] * data_initial['o_list'][i]
        for i in range(self.species_num):
            all_label_Y = all_label_Y + [int(i)] * data_initial['v_list'][i]
        all_label_Y = np.array(all_label_Y).reshape(-1, 1)
        print(all_label_Y.shape)

        x_train, y_train, x_val, y_val, x_test, y_test = \
            train_test_val_split(all_label_X, all_label_Y, train_ratio=self.cfg.SRRSC_args.train_ratio,
                                 validation_ratio=self.cfg.SRRSC_args.validation_ratio, test_ratio=self.cfg.SRRSC_args.test_ratio,
                                 random_state=1)
        labels = []
        labels.append(np.concatenate((x_train, y_train), axis=1))
        labels.append(np.concatenate((x_val, y_val), axis=1))
        labels.append(np.concatenate((x_test, y_test), axis=1))
        f = open(self.cfg.SRRSC_args.data_path + "labels.pkl", "wb")
        pickle.dump(labels, f)

        #################################################
        self.logger.info("Generating edges...")
        # print(len(datapair['varnames_node'][0] + datapair['varnames_node'][1]))
        np.save(self.cfg.SRRSC_args.data_path + 'gene_names.npy',
               gene_name_list)
        np.save(self.cfg.SRRSC_args.data_path + 'sample_names.npy', sample_name_list)

        # mouse_sample_num = len(mouse_sample_list)
        # mouse_gene_num = len(datapair['varnames_node'][0])
        sm_ = np.nan_to_num(data_initial['ov_adjs'].toarray()) > 0
        ss_ = np.nan_to_num(data_initial['oo_adjs'].toarray()) > 0
        mm_ = np.nan_to_num(data_initial['vv_adjs'].toarray()) > 0

        # Add self loop of nodes
        # This step is optional
        mm_  = mm_ + np.identity(mm_.shape[0])
        mm_ = mm_ > 0
        ss_ = ss_ + np.identity(ss_.shape[0])
        ss_ = ss_ > 0

        self.logger.info('sm_.shape:{}'.format(sm_.shape))
        self.logger.info('ss_.shape:{}'.format(ss_.shape))
        self.logger.info('mm_.shape:{}'.format(mm_.shape))
        s_sm_mat = np.concatenate((ss_, sm_), axis=1)
        m_sm_mat = np.concatenate((sm_.T, mm_), axis=1)
        sm_mat = np.concatenate((s_sm_mat, m_sm_mat), axis=0)
        ms_mat = sp.csc_matrix(sm_mat.T)
        sm_mat = sp.csr_matrix(sm_mat)

        # ss_mat = sp.csc_matrix(ss_)
        # mm_mat = sp.csc_matrix(mm_)
        s_sm_mat_mm = np.concatenate((np.zeros(ss_.shape), np.zeros(sm_.shape)), axis=1)
        m_sm_mat_mm = np.concatenate((np.zeros(sm_.shape).T, mm_), axis=1)
        mm_mat = np.concatenate((s_sm_mat_mm, m_sm_mat_mm), axis=0)
        mm_mat = sp.csr_matrix(mm_mat)

        s_sm_mat_ss = np.concatenate((ss_, np.zeros(sm_.shape)), axis=1)
        m_sm_mat_ss = np.concatenate((np.zeros(sm_.shape).T, np.zeros(mm_.shape)), axis=1)
        ss_mat = np.concatenate((s_sm_mat_ss, m_sm_mat_ss), axis=0)
        ss_mat = sp.csr_matrix(ss_mat)

        edges_data = {'s-m': sm_mat, 'm-s': ms_mat, 's-s': ss_mat, 'm-m': mm_mat}
        f = open(self.cfg.SRRSC_args.data_path + "edges.pkl", "wb")
        pickle.dump(edges_data, f)

        feats_s = data_initial['o_feat'].toarray()

        print('np.max(feats_s)', np.max(feats_s))
        print('np.min(feats_s)', np.min(feats_s))
        feats_s = np.nan_to_num(feats_s)
        feats_m = data_initial['v_feat'].toarray()
        print('np.max(feats_m)', np.max(feats_m))
        print('np.min(feats_m)', np.min(feats_m))
        feats_m = np.nan_to_num(feats_m)

        # Not pass feature of genes
        #feats_m = np.zeros(feats_m.shape)
        mh_ = mm_[0:self.cfg.BrainAlign.binary_M, self.cfg.BrainAlign.binary_M:]
        feats_m_temp = feats_m.copy()
        for i in range(self.cfg.BrainAlign.binary_M):
            h_index = [x + self.cfg.BrainAlign.binary_M for x in np.nonzero(mh_[i, :])[0]]
            if len(h_index) > 0:
                h_index.append(i)
                feat_h = feats_m[h_index, :]
                feats_m_temp[i, :] = np.mean(feat_h, axis=0)
        for j in range(self.cfg.BrainAlign.binary_H):
            m_index = list(np.nonzero(mh_[:, j])[0])
            if len(m_index) > 0:
                m_index.append(int(self.cfg.BrainAlign.binary_M + j))
                feat_g = feats_m[m_index, :]
                feats_m_temp[self.cfg.BrainAlign.binary_M + j, :] = np.mean(feat_g, axis=0)
        feats_m = feats_m_temp

        #####################################
        #feats_s = np.zeros(feats_s.shape)
        #feats_m = np.zeros(feats_m.shape)

        print('np.max(feats_m)', np.max(feats_m))
        print('np.min(feats_m)', np.min(feats_m))
        node_feats = sp.coo_matrix(np.concatenate((feats_s, feats_m), axis=0)).tocsr()
        f = open(self.cfg.SRRSC_args.data_path + "node_features.pkl", "wb")
        pickle.dump(node_feats, f)
        self.logger.info('Input graph data for BrainAlign is saved.')

        return None


    def run_BrainAlign(self):
        self.logger.info('---------------Running BrainAlign for {}--------------'.format(','.join(list(self.cfg.BrainAlign.species_list))))

        main_sr_rsc.run_srrsc(self.cfg, logger=self.logger)

        return None

    def load_brainalign_embeddings(self):
        self.logger.info('Loading output embeddings...')

        f = open(self.cfg.SRRSC_args.save_path + "/embeds/" + 'node.pkl', "rb")
        all_embeddings = pickle.load(f)
        self.logger.info('All embedding shape: {}'.format(all_embeddings.shape))

        with open(self.cfg.BrainAlign.DATA_PATH + 'data_init.pickle', 'rb') as handle:
            data_initial = pickle.load(handle)

        S_embeddings_num_list = data_initial['o_list']
        S = np.sum(S_embeddings_num_list)
        S_nodes_num_cumu_list = [int(np.sum(np.array(S_embeddings_num_list[0:i + 1]))) for i in range(self.species_num)]
        S_nodes_num_cumu_list = [int(0)] + S_nodes_num_cumu_list
        M_embeddings_num_list = data_initial['v_list']
        M = np.sum(M_embeddings_num_list)
        M_nodes_num_cumu_list = [int(np.sum(np.array(M_embeddings_num_list[0:i + 1]))) for i in range(self.species_num)]
        M_nodes_num_cumu_list = [int(0)] + M_nodes_num_cumu_list

        S_name_list = data_initial['o_name']
        M_name_list = data_initial['v_name']


        obs_df = self.adatas[0].obs
        for i in range(1, self.species_num):
            obs_df = pd.concat((obs_df, self.adatas[i].obs), axis=0)

        sample_species_name = []
        gene_species_name = []
        for i in range(self.species_num):
            sample_species_name = sample_species_name + [self.cfg.BrainAlign.species_list[i]] * S_embeddings_num_list[i]
            gene_species_name = gene_species_name + [self.cfg.BrainAlign.species_list[i]] * M_embeddings_num_list[i]

        self.logger.info('Loading sample embeddings.')
        S_embeddings = all_embeddings[0:S,:]
        adata_sample = ad.AnnData(S_embeddings)
        adata_sample.obs = obs_df
        adata_sample.obs_names = S_name_list
        adata_sample.obs['Species'] = sample_species_name

        self.logger.info('Loading gene embeddings.')
        M_embeddings = all_embeddings[S:(S+M),:]
        adata_gene = ad.AnnData(M_embeddings)
        adata_gene.obs_names = M_name_list
        adata_gene.obs['Species'] = gene_species_name

        if not os.path.exists(self.cfg.BrainAlign.embeddings_file_path):
            os.makedirs(self.cfg.BrainAlign.embeddings_file_path)
        adata_sample.write_h5ad(self.cfg.BrainAlign.embeddings_file_path + '/s_embeddings.h5ad')

        adata_gene.write_h5ad(self.cfg.BrainAlign.embeddings_file_path + '/m_embeddings.h5ad')



    def run_analysis(self):
        self.logger.info('----------------Running embeddings analysis------------------')
        sns.set(style='white')
        TINY_SIZE = 22  # 39
        SMALL_SIZE = 22  # 42
        MEDIUM_SIZE = 28  # 46
        BIGGER_SIZE = 28  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']
        sys.setrecursionlimit(100000)

        alignment_STs_analysis_obj = alignment_STs_analysis(self.cfg, logger=self.logger)
        alignment_STs_analysis_obj.experiment_1_cross_species_clustering()
        alignment_STs_analysis_obj.experiment_2_alignment_score_evaluation()
        alignment_STs_analysis_obj.experiment_4_gene_clustering()




    def init_var_features(self, ov_adj_list, obs_feature_list, pca_dim=128, if_pca=False):
        """
        Compute variable (gene) features via object (sample) features and sample*gene adjacent matrix
        :param ov_adj_list: sample*gene adjacent matrix
        :param obs_feature_list: sample feature for each species
        :param pca_dim: PCA dimension reduction dimension
        :param if_pca: whether do pca
        :return: variable (gene) features for each species
        """
        var_feature_init_list = []
        obs_num_list = [obs_feature.shape[0] for obs_feature in obs_feature_list]
        obs_num_cumu_list = [int(np.sum(np.array(obs_num_list[0:i + 1]))) for i in range(len(obs_feature_list))]
        obs_num_cumu_list = [int(0)] + obs_num_cumu_list
        if not if_pca:
            for i in range(len(obs_feature_list)):
                var_feature = transform_pca_embedding_np(ov_adj_list[i], obs_feature_list[i])
                var_feature_init_list.append(var_feature)
            return obs_feature_list, var_feature_init_list
        else:
            feature_all = vstack(obs_feature_list)
            feature_all_adata = sc.AnnData(feature_all)
            sc.tl.pca(feature_all_adata, svd_solver='arpack', n_comps=pca_dim)
            feature_all_X = feature_all_adata.obsm['X_pca']
            obs_feature_list_pca = []
            for i in range(len(obs_feature_list)):
                obs_feature_list_pca.append(feature_all_X[obs_num_cumu_list[i]:obs_num_cumu_list[i + 1], :])
                var_feature = transform_pca_embedding_np(ov_adj_list[i], obs_feature_list_pca[i])
                var_feature_init_list.append(var_feature)
            return obs_feature_list_pca,  var_feature_init_list
        #return var_feature_init_list

    def get_varname_as_nodes(self, species_order, df_varmap_list_binary):
        r'''
        :param species_order:
        :param df_varmap_list_binary:
        :return:
        '''
        gvar_as_nodes_set = set()
        # Compute aligned features for each species
        vars_all1 = self.vars_all_list[species_order]
        vars_nodes1 = self.vars_as_nodes[species_order]
        # vars_use1 = self.vars_feat_list[species_order]
        # obs1 = self.obs_list[species_order]
        adata_raw1 = self.adata_raw_list[species_order]
        adata1 = self.adatas[species_order]

        # feature_list = []
        # vnames_feat_list = []
        # vnames_species0_all = []

        r = list(range(self.species_num))
        r.remove(species_order) # [1, 2, 3]

        for i in r:
            vars_nodes2 = self.vars_as_nodes[i]
            vars_use2 = self.vars_feat_list[i]
            obs2 = self.obs_list[i]
            vars_all2 = self.vars_all_list[i]
            adata_raw2 = self.adata_raw_list[i]
            adata2 = self.adatas[i]

            df_varmap = df_varmap_list_binary[i]

            submaps = pp.subset_matches(df_varmap, vars_nodes1, vars_nodes2,
                                        union=self.cfg.Preprocess.union_var_nodes)
            submaps = pp.subset_matches(submaps, vars_all1, vars_all2, union=False)

            if self.cfg.Preprocess.with_single_vnodes:
                vv_adj, vnodes1, vnodes2 = pp.make_bipartite_adj(
                    submaps, vars_nodes1, vars_nodes2,
                    with_singleton=True, symmetric=True,
                )
            else:
                vv_adj, vnodes1, vnodes2 = pp.make_bipartite_adj(
                    submaps, with_singleton=False, symmetric=True,
                )
                # --- ov_adjs (unaligned features, for making `ov_adj`)
            print('vv_adj.shape', vv_adj.shape)
            print('len of vars_nodes1', len(vars_nodes1))
            print('len of vars_nodes2', len(vars_nodes2))
            ov_adjs1 = adata_raw1[:, vnodes1].X  # for ov_adj
            ov_adjs2 = adata_raw2[:, vnodes2].X
            print('ov_adjs1.shape', ov_adjs1.shape)
            print('ov_adjs2.shape', ov_adjs2.shape)
            # print(adata_raw1.var)
            # print(adata1.var)
            var1 = adata1.var.copy().loc[vnodes1, :]
            print('len of var1', len(var1))
            var2 = adata2.var.copy().loc[vnodes2, :]
            print('len of var2', len(var2))
            gvar_as_nodes_set.update(vnodes1)

        gvar_as_nodes = list(gvar_as_nodes_set)#adata1.var.copy().loc[list(gvar_as_nodes_set), :]
        return gvar_as_nodes


    def forward(self):

        self.preprocess_unaligned()
        self.get_brainalign_input()
        self.run_BrainAlign()
        self.load_brainalign_embeddings()
        self.run_analysis()
        #return feature_list, vnames_feat_list


def get_spatial_relation_multiple(cfg, S_cumu_list):
    if cfg.BrainAlign.dataset == 'sa_2020' and len(cfg.BrainAlign.species_list)==2:
        expression_specie1_path = cfg.BrainAlign.path_rawdata_list[0]
        adata_specie1 = sc.read_h5ad(expression_specie1_path)
        print(adata_specie1)
        x = adata_specie1.obs['x_grid']
        y = adata_specie1.obs['y_grid']
        z = adata_specie1.obs['z_grid']
        xyz_arr1 = np.array([x, y, z]).T  # N row, each row is in 3 dimension
        ss_1 = kneighbors_graph(xyz_arr1, 1 + cfg.BrainAlign.spatial_node_neighbor, mode='connectivity', include_self=False) > 0

        expression_specie2_path = cfg.BrainAlign.path_rawdata_list[1]
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
        ss_ = sp.coo_matrix((ss_data, (ss_row, ss_col)), shape=(cfg.BrainAlign.S, cfg.BrainAlign.S))
        # print(adata_human)
        #coordinates_3d_specie1 = adata_sm.obs
    else:
        adata_specie_list = [sc.read_h5ad(exp_path) for exp_path in cfg.BrainAlign.path_rawdata_list]
        ss_list = []
        ss_row = []
        ss_col = []
        ss_data = []
        for i in range(len(cfg.BrainAlign.species_list)):
            row_col_add = S_cumu_list[i]
            adata_specie = adata_specie_list[i]
            x = adata_specie.obs['x']
            y = adata_specie.obs['y']
            z = adata_specie.obs['z']
            xyz_arri = np.array([x, y, z]).T  # N row, each row is in 3 dimension
            ss_i = kneighbors_graph(xyz_arri, 1 + cfg.BrainAlign.spatial_node_neighbor, mode='connectivity',
                                    include_self=False) > 0
            ss_i = sp.csr_matrix(ss_i).tocoo()
            ss_row = np.concatenate((ss_row, ss_i.row + row_col_add[i]))
            ss_col = np.concatenate((ss_col, ss_i.col + row_col_add[i]))
            ss_data = np.concatenate((ss_data, ss_i.data))
        ss_ = sp.coo_matrix((ss_data, (ss_row, ss_col)), shape=(cfg.BrainAlign.S, cfg.BrainAlign.S))

    return ss_



def make_id_name_maps_simple(nodes1, nodes2):
    name_srs1 = pd.Series(list(nodes1))
    name_srs2 = pd.Series(list(nodes2))
    n2i_dict1 = reverse_dict(name_srs1)
    n2i_dict2 = reverse_dict(name_srs2)
    return n2i_dict1, n2i_dict2


def sub_df_map(df_map, nodes1, nodes2):
    return df_map[df_map[df_map.columns[0]].isin(nodes1) & df_map[df_map.columns[1]].isin(nodes2)]



def make_bipartite_adj_simple(df_map: pd.DataFrame,
                       nodes1=None, nodes2=None,
                       key_data: Optional[str] = None):
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
    name2i_dict1, name2i_dict2 = make_id_name_maps_simple(nodes1, nodes2)

    if key_data is None:
        data = np.ones(df_map.shape[0], dtype=int)
    else:
        data = df_map[key_data].values
    # sparse adjacent matrix construction:
    # ids representing nodes from left-nodes (ii) and right-nodes (jj)
    ii = change_names(df_map.iloc[:, 0], lambda x: name2i_dict1[x])
    jj = change_names(df_map.iloc[:, 1], lambda x: name2i_dict2[x])
    bi_adj = sparse.coo_matrix(
        (data, (ii, jj)),
        shape=(len(nodes1), len(nodes2)))

    return bi_adj


def find_match_indices(reference_list, query_list, pair_reference_list):
    res = []
    pair_value_list = []
    set_ref = set(reference_list)
    for i in range(len(query_list)):
        if query_list[i] in set_ref:
            index = reference_list.index(query_list[i])
            res.append(index)
            pair_value_list.append(pair_reference_list[index])
    return res, pair_value_list


def load_sample_embedding_concat(mouse_sample_list, human_sample_list, cfg):

    # No label order version
    '''
    Step 1: Compute average embedding of every region in two species, use two dict to store;
    Step 2: Compute similarity matrix, use np array to store;
    Step 3: Heatmap.

    '''

    path_human = cfg.CAME.ROOT + 'adt_hidden_cell.h5ad'

    adata_human = sc.read_h5ad(cfg.CAME.path_rawdata2)

    adata_mouse = sc.read_h5ad(cfg.CAME.path_rawdata1)

    adata_human_mouse = sc.read_h5ad(path_human)
    #print(adata_human_mouse)

    # Step 1
    #embedding_len = 128

    # ---------------------------human------------------------------------

    # assign embedding to original adata
    adata_human_mouse.obs_names = adata_human_mouse.obs['original_name']

    adata_human_embedding = adata_human_mouse[adata_human.obs_names]
    adata_human_embedding.obs['region_name'] = adata_human.obs['region_name']
    for sample_id in adata_human_embedding.obs_names:
        adata_human_embedding[sample_id].obs['region_name'] = adata_human[sample_id].obs['region_name']

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


    diff = False
    for k, v in zip(mouse_sample_list, list(adata_mouse_embedding.obs_names)):
        if k != v:
            print("The is difference!")
            diff = True
    if diff == False:
        print('They are identical!')

    print(type(adata_mouse_embedding.X))
    concat_X  = np.concatenate((adata_mouse_embedding.X, adata_human_embedding.X), axis=0)
    print(concat_X.shape)

    obs_names = adata_mouse_embedding.obs_names + adata_human_embedding.obs_names
    print(len(obs_names))


def load_gene_embedding(mouse_gene_list, human_gene_list, cfg):
    path_gene_mouse = cfg.CAME.ROOT + 'adt_hidden_gene1.h5ad'
    adata_gene_mouse = sc.read_h5ad(path_gene_mouse)
    print(adata_gene_mouse)

    sp.save_npz(cfg.HECO.DATA_PATH + 'm_feat.npz', sp.csr_matrix(adata_gene_mouse.X))
    mouse_gene_df = pd.DataFrame(adata_gene_mouse.X, index=adata_gene_mouse.obs_names)
    #mouse_gene_df.to_csv('./mouse_gene.csv')
    print(mouse_gene_df)

    diff = False
    for k, v in zip(mouse_gene_list, list(adata_gene_mouse.obs_names)):
        if k != v:
            print("The is difference!")
            diff = True
    if diff == False:
        print('They are identical!')


    path_gene_human = cfg.CAME.ROOT + 'adt_hidden_gene2.h5ad'
    adata_gene_human = sc.read_h5ad(path_gene_human)
    print(adata_gene_human)

    sp.save_npz(cfg.HECO.DATA_PATH + 'h_feat.npz', sp.csr_matrix(adata_gene_human.X))
    human_gene_df = pd.DataFrame(adata_gene_human.X, index=adata_gene_human.obs_names)
    #human_gene_df.to_csv('./human_gene.csv')
    print(human_gene_df)

    diff = False
    for k, v in zip(human_gene_list, list(adata_gene_human.obs_names)):
        if k != v:
            print("The is difference!")
            diff = True
    if diff == False:
        print('They are identical!')


def load_embeddings_came(cfg):
    #cfg = heco_config._C
    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)
    print(datapair)
    print(datapair['obs_dfs'][0].shape)
    print(datapair['obs_dfs'][1].shape)

    mouse_sample_list = list(datapair['obs_dfs'][0].index)
    # print(mouse_sample_list)
    print(len(mouse_sample_list))

    human_sample_list = list(datapair['obs_dfs'][1].index)
    # print(human_sample_list)
    print(len(human_sample_list))
    # mouse_sample_list = obs_dfs
    load_sample_embedding_concat(mouse_sample_list, human_sample_list)

    mouse_gene_list = datapair['varnames_node'][0]
    human_gene_list = datapair['varnames_node'][1]

    load_gene_embedding(mouse_gene_list, human_gene_list)



if __name__ == '__main__':
    reference_list = ['a', 'b', 'c', 'd']
    query_list = ['c', 'b', 'a']
    pair_reference_list = ['a1', 'b1', 'c1', 'd1']
    res, pair_value_list = find_match_indices(reference_list, query_list, pair_reference_list)
    print(pair_value_list)