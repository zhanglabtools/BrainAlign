# -- coding: utf-8 --
# @Time : 2023/5/13 13:28
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : pipline_analysis_alignment.py
# @Description: This file is used to ...


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
from BrainAlign.code.utils import set_params
from BrainAlign.brain_analysis.configs import heco_config
from BrainAlign.brain_analysis.data_utils import plot_marker_selection_umap, plot_marker_selection_umap_embedding
from BrainAlign.brain_analysis.analysis_utils import gen_mpl_labels
from BrainAlign.brain_analysis.process import normalize_before_pruning
from scipy.spatial.distance import squareform, pdist
sys.path.append('../')
#import ./came
import BrainAlign.came as came
from BrainAlign.came import pipeline, pp, pl, make_abstracted_graph, weight_linked_vars

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from matplotlib import rcParams
from matplotlib.patches import Patch
from matplotlib_venn import venn2
from matplotlib.patches import Rectangle


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from scipy.spatial.distance import pdist, squareform

from scipy import stats

# call marker gene
from scGeneFit.functions import *

# color map
import colorcet as cc

from collections import Counter

# get gene ontology
import gseapy

from imblearn.over_sampling import RandomOverSampler
from statannot import add_stat_annotation
import logging

try:
    import matplotlib as mpl
    mpl.use('agg')
except Exception as e:
    print(f"An error occurred when setting matplotlib backend ({e})")

from BrainAlign.brain_analysis.metrics import seurat_alignment_score

from BrainAlign.brain_analysis.analysis_utils import get_common_special_gene_list, \
    get_homologous_mat, \
    get_homologous_gene_list, \
    average_expression, \
    gene_module_abstract_graph

class alignment_STs_analysis():
    def __init__(self, cfg, logger):
        self.logger = logger

        self.cfg = cfg
        self.fig_format = cfg.BrainAlign.fig_format  # the figure save format
        self.fig_dpi = cfg.BrainAlign.fig_dpi

        self.adata_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + '/s_embeddings.h5ad')

        with open(self.cfg.BrainAlign.DATA_PATH + 'data_init.pickle', 'rb') as handle:
            self.data_initial = pickle.load(handle)

        self.cfg.BrainAlign.S_list = self.data_initial['o_list']
        self.cfg.BrainAlign.M_list = self.data_initial['v_list']

        obs_color = []
        for i in range(len(self.cfg.BrainAlign.species_list)):
            obs_color = obs_color + [self.cfg.BrainAlign.species_color[i]] * self.cfg.BrainAlign.S_list[i]
        self.adata_embedding.obs['sample_color'] = obs_color

        self.adata_gene_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + '/m_embeddings.h5ad')
        obs_color = []
        for i in range(len(self.cfg.BrainAlign.species_list)):
            obs_color = obs_color + [self.cfg.BrainAlign.species_color[i]] * self.cfg.BrainAlign.M_list[i]
        self.adata_gene_embedding.obs['sample_color'] = obs_color

        self.species_num = len(self.cfg.BrainAlign.species_list)
        self.gene_name_list = self.data_initial['v_name']

        self.save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.palette = {k: v for k, v in zip(self.cfg.BrainAlign.species_list, self.cfg.BrainAlign.species_color)}

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

    def experiment_1_cross_species_clustering(self):

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/1_experiment_cross_species_clustering/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

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

        cfg = self.cfg
        fig_format = cfg.BrainAlign.fig_format
        sys.setrecursionlimit(100000)
        # No label order version
        '''
        Basic UMAP of sample embeddings
        '''
        self.logger.info('Basic UMAP of sample embeddings.')
        sc.pp.neighbors(self.adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(self.adata_embedding, min_dist=cfg.ANALYSIS.min_dist)

        with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.cfg.BrainAlign.fig_dpi)}):
            fg = sc.pl.umap(self.adata_embedding, color=['Species'], return_fig=True, legend_loc=None, palette=self.palette)
            plt.title('')
            fg.savefig(save_path + 'umap_dataset.' + fig_format, format=fig_format)
        rcParams["figure.subplot.right"] = 0.66
        with plt.rc_context({"figure.figsize": (12, 8), "figure.dpi": (self.cfg.BrainAlign.fig_dpi)}):
            fg = sc.pl.umap(self.adata_embedding, color=['Species'], return_fig=True, legend_loc='right margin',
                            palette=self.palette)
            plt.title('')
            fg.savefig(save_path + 'umap_dataset_right_margin.' + fig_format, format=fig_format)


        # clustering
        row_colors = self.adata_embedding.obs['sample_color'].to_numpy()

        rcParams["figure.subplot.top"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.3
        with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
            g = sns.clustermap(self.adata_embedding.X, row_colors=row_colors, dendrogram_ratio=(.3, .2), cmap='YlGnBu_r')
            g.ax_heatmap.tick_params(tick2On=False, labelsize=False)
            g.ax_heatmap.set(xlabel='Embedding features', ylabel='Samples')
            handles = [Patch(facecolor=self.palette[name]) for name in self.palette]
            plt.legend(handles, self.palette, #title='',
                       bbox_to_anchor=(1, 1),
                       bbox_transform=plt.gcf().transFigure,
                       loc='upper right', frameon=False)
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.05
        plt.savefig(save_path + 'Hiercluster_sample_embedding_concate.png', dpi=self.fig_dpi) # + fig_format, format=fig_format
        plt.show()
        rcParams["figure.subplot.right"] = 0.9


    def experiment_2_alignment_score_evaluation(self):
        r"""
               Umap and evaluate the embeddings across species before and after integration.
               * 1. Umap of embeddings of each species.
               * 2. PCA separately and UMAP together.
               * 3. Compute the seurat alignment score for unligned data.
               * 4. Compute the seurat alignment score for ligned data.
               * 5. plot the box plots.
               :return:None
               """
        cfg = self.cfg
        fig_format = cfg.BrainAlign.fig_format

        sns.set(style='white')
        TINY_SIZE = 28  # 39
        SMALL_SIZE = 28  # 42
        MEDIUM_SIZE = 32  # 46
        BIGGER_SIZE = 32  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/2_experiment_alignment_score_evaluation/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.55

        for i in range(self.species_num):
            species = self.cfg.BrainAlign.species_list[i]
            adata_embedding_species = self.adata_embedding[self.adata_embedding.obs['Species'].isin([species])]
            with plt.rc_context({"figure.figsize": (12, 6), "figure.dpi": (self.fig_dpi)}):
                sc.pp.neighbors(adata_embedding_species, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine',
                                use_rep='X')
                sc.tl.umap(adata_embedding_species)
                sc.pl.umap(adata_embedding_species, color=[self.cfg.BrainAlign.key_class_list[i]], return_fig=True,
                           legend_loc='right margin').savefig(
                    save_path + 'umap_{}.'.format(species) + fig_format, format=fig_format)

        rcParams["figure.subplot.left"] = 0.2
        rcParams["figure.subplot.right"] = 0.9

        # umap together before integration
        # pca separately and do dimension reduction

        gene_num_list = self.data_initial['v_list']
        var_nodes_num_cumu_list = [int(np.sum(np.array(gene_num_list[0:i + 1]))) for i in range(self.species_num)]
        var_nodes_num_cumu_list = [int(0)] + var_nodes_num_cumu_list

        adata_selected_list = [sc.read_h5ad(atd_path) for atd_path in self.cfg.BrainAlign.path_rawdata_list]
        for i in range(self.species_num):
            adata_selected_list[i] = adata_selected_list[i][:, self.gene_name_list[var_nodes_num_cumu_list[i]:var_nodes_num_cumu_list[i+1]]]

            adata = adata_selected_list[i]
            sc.tl.pca(adata, svd_solver='arpack', n_comps=30)
            sc.pp.neighbors(adata, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X_pca')
            sc.tl.umap(adata)
            adata_selected_list[i] = adata
            adata_selected_list[i].obs['Species'] = self.cfg.BrainAlign.species_list[i]


        adata_expression = ad.concat(adata_selected_list)

        rcParams["figure.subplot.left"] = 0.1
        with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(adata_expression, color=['Species'], return_fig=True, legend_loc=None, title='', #size=15,
                            palette=self.palette)
            plt.title('')
            fg.savefig(save_path + 'umap_dataset_before_integration_ondata.' + fig_format, format=fig_format)

        rcParams["figure.subplot.right"] = 0.66

        with plt.rc_context({"figure.figsize": (12, 8), "figure.dpi": (self.fig_dpi)}):
            # plt.figure(figsize=(8,8), dpi=self.fig_dpi)
            fg = sc.pl.umap(adata_expression, color=['Species'], return_fig=True, legend_loc=None, title='', #size=15,
                            palette=self.palette)
            plt.title('')
            fg.savefig(save_path + 'umap_dataset_before_integration_rightmargin.' + fig_format, format=fig_format)

        align_score_dict = {'Method': [], 'alignment score': []}
        # compute alignment score for unaligned data
        X = adata_expression.obsm['X_pca']
        species_label_list = []
        for i in range(self.species_num):
            species_label_list.append(np.ones((adata_selected_list[i].n_obs, 1))*i)
        Y = np.concatenate(species_label_list, axis=0)
        unaligned_score = seurat_alignment_score(X, Y)
        align_score_dict['Method'].append('Unaligned')
        align_score_dict['alignment score'].append(unaligned_score)

        # after integration

        sc.tl.pca(self.adata_embedding, svd_solver='arpack', n_comps=30)
        sc.pp.neighbors(self.adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine',
                        use_rep='X')
        sc.tl.umap(self.adata_embedding)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.6545
        with plt.rc_context({"figure.figsize": (11, 8), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(self.adata_embedding, color=['Species'], return_fig=True, legend_loc='right margin', title='',
                            palette=self.palette) #size=15,
            plt.title('')
            fg.savefig(save_path + 'umap_dataset_after_integration_rightmargin.' + fig_format, format=fig_format)

        rcParams["figure.subplot.right"] = 0.9
        # rcParams["figure.subplot.bottom"] = 0.1
        with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(self.adata_embedding, color=['Species'], return_fig=True, legend_loc=None, title='', #size=15,
                            palette=self.palette)
            plt.title('')
            fg.savefig(save_path + 'umap_dataset_after_integration.' + fig_format, format=fig_format)
        # compute alignment score for aligned data
        X = self.adata_embedding.obsm['X_umap']
        Y = Y
        aligned_score = seurat_alignment_score(X, Y)
        align_score_dict['Method'].append('BrainAlign')
        align_score_dict['alignment score'].append(aligned_score)
        # CAME results
        # align_score_dict['Method'].append('CAME')
        # align_score_dict['alignment score'].append(0.020107865877827872)
        print(align_score_dict)
        align_score_df = pd.DataFrame.from_dict(align_score_dict)
        # plot bar plot of alignment score
        # my_pal = {"Unaligned": '#CD4537', "CAME":'#28AF60', "BrainAlign": '#2C307A'}
        my_pal = {"Unaligned": '#CD4537', "BrainAlign": '#2C307A'}
        rcParams["figure.subplot.left"] = 0.2
        rcParams["figure.subplot.right"] = 0.90
        rcParams["figure.subplot.bottom"] = 0.1
        plt.figure(figsize=(8, 8), dpi=self.fig_dpi)
        # sns.set(rc={"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)})
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        sns.barplot(x="Method", y="alignment score", data=align_score_df, order=["Unaligned", "BrainAlign"],
                    palette=my_pal, width=0.618)  #
        plt.xlabel('')
        plt.savefig(save_path + 'seurate_alignment_score.' + fig_format, format=fig_format)
        align_score_df.to_csv(save_path + 'seurat_alignment_score.csv')

        rcParams["figure.subplot.left"] = 0.05

    def experiment_4_gene_clustering(self):

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/4_experiment_gene_clustering/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

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

        cfg = self.cfg
        fig_format = cfg.BrainAlign.fig_format
        sys.setrecursionlimit(100000)
        # No label order version
        '''
        Basic UMAP of gene embeddings
        '''
        self.logger.info('Basic UMAP of gene embeddings.')
        sc.pp.neighbors(self.adata_gene_embedding, n_neighbors=cfg.ANALYSIS.genes_umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(self.adata_gene_embedding, min_dist=cfg.ANALYSIS.genes_min_dist)

        with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.cfg.BrainAlign.fig_dpi)}):
            fg = sc.pl.umap(self.adata_gene_embedding, color=['Species'], return_fig=True, legend_loc=None,
                            palette=self.palette)
            plt.title('')
            fg.savefig(save_path + 'umap_dataset.' + fig_format, format=fig_format)
        rcParams["figure.subplot.right"] = 0.66
        with plt.rc_context({"figure.figsize": (12, 8), "figure.dpi": (self.cfg.BrainAlign.fig_dpi)}):
            fg = sc.pl.umap(self.adata_gene_embedding, color=['Species'], return_fig=True, legend_loc='right margin',
                            palette=self.palette)
            plt.title('')
            fg.savefig(save_path + 'umap_dataset_right_margin.' + fig_format, format=fig_format)

        # clustering
        row_colors = self.adata_gene_embedding.obs['sample_color'].to_numpy()

        rcParams["figure.subplot.top"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.3
        with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
            g = sns.clustermap(self.adata_gene_embedding.X, row_colors=row_colors, dendrogram_ratio=(.3, .2),
                               cmap='YlGnBu_r')
            g.ax_heatmap.tick_params(tick2On=False, labelsize=False)
            g.ax_heatmap.set(xlabel='Embedding features', ylabel='Samples')
            handles = [Patch(facecolor=self.palette[name]) for name in self.palette]
            plt.legend(handles, self.palette,  # title='',
                       bbox_to_anchor=(1, 1),
                       bbox_transform=plt.gcf().transFigure,
                       loc='upper right', frameon=False)
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.05
        plt.savefig(save_path + 'Hiercluster_gene_embedding_concate.png',
                    dpi=self.fig_dpi)  # + fig_format, format=fig_format
        plt.show()
        rcParams["figure.subplot.right"] = 0.9
