# -- coding: utf-8 --
# @Time : 2023/6/1 11:47
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : analysis_genomic.py
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
sys.path.append('../code/')
from BrainAlign.code.utils import set_params
from BrainAlign.brain_analysis.configs import heco_config
from BrainAlign.brain_analysis.data_utils import get_correlation, \
    Dict2Mapping, \
    plot_marker_selection_umap, \
    plot_marker_selection_umap_embedding
# from BrainAlign.brain_analysis.analysis_utils import gen_mpl_labels
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
from matplotlib import colors

from collections import abc

from BrainAlign.came.utils import preprocess

from colormap import Colormap

try:
    import matplotlib as mpl
    mpl.use('agg')
except Exception as e:
    print(f"An error occurred when setting matplotlib backend ({e})")

from BrainAlign.brain_analysis.metrics import seurat_alignment_score

from BrainAlign.brain_analysis.analysis_utils import get_common_special_gene_list, \
    get_homologous_mat, \
    get_common_special_gene_list_homologous, \
    average_expression, \
    gene_module_abstract_graph

plt.rcParams['figure.edgecolor'] = 'black'

def deg_plot(adata, ntop_genes, ntop_genes_visual, save_path_parent, groupby, save_pre, ifmouse=True, fig_format='png'):
    TINY_SIZE = 24  # 39
    SMALL_SIZE = 24  # 42
    MEDIUM_SIZE = 28  # 46
    BIGGER_SIZE = 28  # 46

    plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    sc.pp.filter_cells(adata, min_counts=2)

    cluster_counts = adata.obs[groupby].value_counts()
    keep = cluster_counts.index[cluster_counts >= 2]
    adata = adata[adata.obs[groupby].isin(keep)].copy()

    sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added="wilcoxon")
    sc.pl.rank_genes_groups(adata, n_genes=ntop_genes, sharey=False, key="wilcoxon", show=False)
    if ifmouse:
        sc.tl.filter_rank_genes_groups(adata, key="wilcoxon", key_added="wilcoxon",
                                       min_in_group_fraction=0.1,
                                       max_out_group_fraction=0.9,
                                       min_fold_change=5)
    else:
        # pass
        sc.tl.filter_rank_genes_groups(adata, key="wilcoxon", key_added="wilcoxon_filtered",
                                       min_fold_change=2)
    plt.savefig(save_path_parent + save_pre + 'degs.' + fig_format, format=fig_format)

    rcParams["figure.subplot.top"] = 0.8
    rcParams["figure.subplot.bottom"] = 0.2
    rcParams["figure.subplot.left"] = 0.2
    # with plt.rc_context({"figure.figsize": (12, 16)}):
    sc.pl.rank_genes_groups_heatmap(adata, n_genes=ntop_genes_visual, key="wilcoxon",
                                    groupby=groupby, show_gene_labels=True, show=False, dendrogram=True,
                                    figsize=(20, 24))
    plt.savefig(save_path_parent + save_pre + '_heatmap.' + fig_format, format=fig_format)
    # with plt.rc_context({"figure.figsize": (12, 12)}):
    sc.pl.rank_genes_groups_dotplot(adata, n_genes=ntop_genes_visual, key="wilcoxon",
                                    groupby=groupby, show=False, dendrogram=True, figsize=(20, 20))
    plt.savefig(save_path_parent + save_pre + '_dotplot.' + fig_format, format=fig_format)
    # with plt.rc_context({"figure.figsize": (12, 12)}):
    sc.pl.rank_genes_groups_stacked_violin(adata, n_genes=ntop_genes_visual, key="wilcoxon",
                                           groupby=groupby, show=False, dendrogram=True, figsize=(20, 20))
    plt.savefig(save_path_parent + save_pre + '_violin.' + fig_format, format=fig_format)

    # with plt.rc_context({"figure.figsize": (12, 12)}):
    sc.pl.rank_genes_groups_matrixplot(adata, n_genes=ntop_genes_visual, key="wilcoxon",
                                       groupby=groupby, show=False, dendrogram=True, figsize=(20, 20))
    plt.savefig(save_path_parent + save_pre + '_matrixplot.' + fig_format, format=fig_format)

class genomic_analysis():
    def __init__(self, cfg):
        self.cfg = cfg
        self.fig_format = cfg.BrainAlign.fig_format  # the figure save format
        self.fig_dpi = cfg.BrainAlign.fig_dpi

        self.mouse_color = '#5D8AEF'  # '#4472C4'
        self.human_color = '#FE1613'  # '#C94799'#'#ED7D31'

        self.save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # read labels data including acronym, color and parent region name
        self.mouse_64_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)
        self.mouse_64_labels_list = list(self.mouse_64_labels['region_name'])
        self.mouse_64_acronym_dict = {k: v for k, v in
                                      zip(self.mouse_64_labels['region_name'], self.mouse_64_labels['acronym'])}
        self.mouse_64_color_dict = {k: v for k, v in
                                    zip(self.mouse_64_labels['region_name'], self.mouse_64_labels['color_hex_triplet'])}
        self.mouse_64_acronym_color_dict = {k: v for k, v in
                                            zip(self.mouse_64_labels['acronym'],
                                                self.mouse_64_labels['color_hex_triplet'])}
        self.mouse_64_parent_region_dict = {k: v for k, v in
                                            zip(self.mouse_64_labels['region_name'],
                                                self.mouse_64_labels['parent_region_name'])}
        self.mouse_15_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.parent_labels_mouse_file)
        self.mouse_15_labels_list = list(self.mouse_15_labels['region_name'])
        self.mouse_15_acronym_dict = {k: v for k, v in
                                      zip(self.mouse_15_labels['region_name'], self.mouse_15_labels['acronym'])}
        self.mouse_15_color_dict = {k: v for k, v in
                                    zip(self.mouse_15_labels['region_name'], self.mouse_15_labels['color_hex_triplet'])}
        self.mouse_15_acronym_color_dict = {k: v for k, v in
                                            zip(self.mouse_15_labels['acronym'],
                                                self.mouse_15_labels['color_hex_triplet'])}

        self.mouse_15_acronym_list = list(self.mouse_15_labels['acronym'])

        self.human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
        self.human_88_labels_list = list(self.human_88_labels['region_name'])
        self.human_88_acronym_dict = {k: v for k, v in
                                      zip(self.human_88_labels['region_name'], self.human_88_labels['acronym'])}
        self.human_88_color_dict = {k: v for k, v in
                                    zip(self.human_88_labels['region_name'], self.human_88_labels['color_hex_triplet'])}
        self.human_88_acronym_color_dict = {k: v for k, v in
                                            zip(self.human_88_labels['acronym'],
                                                self.human_88_labels['color_hex_triplet'])}
        self.human_88_parent_region_dict = {k: v for k, v in
                                            zip(self.human_88_labels['region_name'],
                                                self.human_88_labels['parent_region_name'])}
        self.human_16_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.parent_labels_human_file)
        self.human_16_labels_list = list(self.human_16_labels['region_name'])
        self.human_16_acronym_dict = {k: v for k, v in
                                      zip(self.human_16_labels['region_name'], self.human_16_labels['acronym'])}
        self.human_16_color_dict = {k: v for k, v in
                                    zip(self.human_16_labels['region_name'], self.human_16_labels['color_hex_triplet'])}
        self.human_16_acronym_color_dict = {k: v for k, v in
                                            zip(self.human_16_labels['acronym'],
                                                self.human_16_labels['color_hex_triplet'])}
        self.human_16_acronym_list = list(self.human_16_labels['acronym'])

        self.acronym_color_dict = {}
        for k, v in self.mouse_64_acronym_color_dict.items():
            self.acronym_color_dict.update({'M-' + k: v})
        for k, v in self.human_88_acronym_color_dict.items():
            self.acronym_color_dict.update({('H-' + k): v})
        #print(self.acronym_color_dict)

        self.parent_acronym_color_dict = {k: v for k, v in self.mouse_15_acronym_color_dict.items()}
        self.parent_acronym_color_dict.update({k: v for k, v in self.human_16_acronym_color_dict.items()})
        #print(self.parent_acronym_color_dict)

        self.color_dict = self.mouse_64_color_dict
        self.color_dict.update(self.human_88_color_dict)
        #print(self.color_dict)

        self.parent_color_dict = self.mouse_15_color_dict
        self.parent_color_dict.update(self.human_16_color_dict)
        #print(self.parent_color_dict)

        path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
        path_datapiar_file = open(path_datapiar, 'rb')
        datapair = pickle.load(path_datapiar_file)

        mouse_gene_list = datapair['varnames_node'][0]
        human_gene_list = datapair['varnames_node'][1]

        self.adata_mouse_exp = sc.read_h5ad(cfg.CAME.path_rawdata1)[:, mouse_gene_list]
        self.adata_human_exp = sc.read_h5ad(cfg.CAME.path_rawdata2)[:, human_gene_list]

        self.adata_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_embedding.h5ad')

        self.adata_gene_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_gene_embedding.h5ad')
        self.adata_mouse_gene_embedding = self.adata_gene_embedding[self.adata_gene_embedding.obs['dataset'].isin(['Mouse'])]
        self.adata_human_gene_embedding = self.adata_gene_embedding[
            self.adata_gene_embedding.obs['dataset'].isin(['Human'])]

        self.adata_mouse_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + '/s_embeddings.h5ad')
        self.adata_human_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + '/v_embeddings.h5ad')

        self.cluster_name_unique = sorted(Counter(self.adata_embedding.obs['cluster_name_acronym']).keys(),
                                     key=lambda t: int(t.split('-')[0]))

        palette = sns.color_palette(cc.glasbey, n_colors=len(self.cluster_name_unique))

        self.cluster_color_dict = {k: v for k, v in zip(self.cluster_name_unique, palette)}


    def deg_plot(self, adata, ntop_genes, ntop_genes_visual, save_path_parent, groupby, save_pre, ifmouse=True):

        sc.pp.filter_cells(adata, min_counts=2)

        cluster_counts = adata.obs[groupby].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        adata = adata[adata.obs[groupby].isin(keep)].copy()

        sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added="wilcoxon")
        sc.pl.rank_genes_groups(adata, n_genes=ntop_genes, sharey=False, key="wilcoxon", show=False)
        if ifmouse:
            sc.tl.filter_rank_genes_groups(adata, key="wilcoxon", key_added="wilcoxon",
                                           min_in_group_fraction=0.1,
                                           max_out_group_fraction=0.9,
                                           min_fold_change=5)
        else:
            # pass
            sc.tl.filter_rank_genes_groups(adata, key="wilcoxon", key_added="wilcoxon_filtered",
                                           min_fold_change=2)
        plt.savefig(save_path_parent + save_pre + 'degs.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)

        rcParams["figure.subplot.top"] = 0.8
        rcParams["figure.subplot.bottom"] = 0.2
        rcParams["figure.subplot.left"] = 0.2
        # with plt.rc_context({"figure.figsize": (12, 16)}):
        sc.pl.rank_genes_groups_heatmap(adata, n_genes=ntop_genes_visual, key="wilcoxon",
                                        groupby=groupby, show_gene_labels=True, show=False, dendrogram=True,
                                        figsize=(20, 24), dpi=self.fig_dpi)
        plt.savefig(save_path_parent + save_pre + '_heatmap.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_dotplot(adata, n_genes=ntop_genes_visual, key="wilcoxon",
                                        groupby=groupby, show=False, dendrogram=True, figsize=(20, 20))
        plt.savefig(save_path_parent + save_pre + '_dotplot.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_stacked_violin(adata, n_genes=ntop_genes_visual, key="wilcoxon",
                                               groupby=groupby, show=False, dendrogram=True, figsize=(20, 20))
        plt.savefig(save_path_parent + save_pre + '_violin.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)

        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_matrixplot(adata, n_genes=ntop_genes_visual, key="wilcoxon",
                                           groupby=groupby, show=False, dendrogram=True, figsize=(20, 20))
        plt.savefig(save_path_parent + save_pre + '_matrixplot.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)

    def experiment_1_th_clusters_deg(self):

        sns.set(style='white')
        TINY_SIZE = 14  # 39
        SMALL_SIZE = 14  # 42
        MEDIUM_SIZE = 26  # 46
        BIGGER_SIZE = 26  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/1_experiment_th_clusters_deg/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        #adata_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_embedding.h5ad')

        TH_cluster_list = []
        cluster_name_unique = sorted(Counter(self.adata_embedding.obs['cluster_name_acronym']).keys(),
                                     key=lambda t: int(t.split('-')[0]))
        for cluster_name in cluster_name_unique:
            if cluster_name.split('-')[1] == 'TH' and cluster_name.split('-')[2] == 'TH':
                TH_cluster_list.append(cluster_name)

        adata_embedding_TH = self.adata_embedding[self.adata_embedding.obs['cluster_name_acronym'].isin(TH_cluster_list)]
        adata_embedding_TH_mouse = adata_embedding_TH[adata_embedding_TH.obs['dataset'].isin(['Mouse'])]
        adata_embedding_TH_human = adata_embedding_TH[adata_embedding_TH.obs['dataset'].isin(['Human'])]

        adata_exp_TH_mouse = self.adata_mouse_exp[adata_embedding_TH_mouse.obs_names, :]
        adata_exp_TH_human = self.adata_human_exp[adata_embedding_TH_human.obs_names, :]

        # mouse degs

        groupby = 'cluster_name_acronym'
        adata = adata_exp_TH_mouse
        #sc.pp.filter_cells(adata, min_counts=2)
        ntop_genes = 50
        ntop_genes_visual = 3
        save_pre = 'mouse'
        key = 'wilcoxon'  # 't-test_overestim_var'#'wilcoxon'#'t-test'
        key_filtered = 'wilcoxon'  # 't-test_overestim_var'# 'wilcoxon-filtered'#'t-test-filtered'

        cluster_counts =  adata.obs[groupby].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        adata = adata[adata.obs[groupby].isin(keep)].copy()

        sc.tl.rank_genes_groups(adata, groupby, method='t-test', key_added="t-test")
        sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added="wilcoxon")
        sc.tl.rank_genes_groups(adata, groupby, method='t-test_overestim_var', key_added="t-test_overestim_var")
        sc.tl.rank_genes_groups(adata, groupby, method='logreg', key_added="logreg")

        sc.pl.rank_genes_groups(adata, n_genes=ntop_genes, sharey=False, key=key, show=False)
        # sc.tl.filter_rank_genes_groups(adata, key=key, key_added=key_filtered,
        #                                #min_in_group_fraction=0.2,
        #                                #max_out_group_fraction=0.5,
        #                                min_fold_change=3)
        # else:
        #     # pass
        #     sc.tl.filter_rank_genes_groups(adata, key="wilcoxon", key_added="wilcoxon_filtered",
        #                                    min_fold_change=2)
        plt.savefig(save_path + save_pre + 'degs.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)

        rcParams["figure.subplot.top"] = 0.7
        rcParams["figure.subplot.bottom"] = 0.25
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.85
        # with plt.rc_context({"figure.figsize": (12, 16)}):

        sc.pl.rank_genes_groups_heatmap(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                        groupby=groupby, show_gene_labels=True, show=False, dendrogram=True,
                                        figsize=(6, 8))
        plt.savefig(save_path + save_pre + '_heatmap.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (12, 12)}):

        var_group_positions = []
        for i in range(len(TH_cluster_list)):
            var_group_positions.append((i*ntop_genes_visual+1, (i+1)*ntop_genes_visual+1))

        dp = sc.pl.rank_genes_groups_dotplot(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                        groupby=groupby, show=False,
                                        dendrogram=True,
                                        figsize=(14, 5),
                                        smallest_dot=0,
                                        #var_group_positions=var_group_positions,
                                        #values_to_plot="logfoldchanges",
                                        #colorbar_title='log fold change',
                                        #colorbar_title='Mean expression in group',
                                        #size_title='Fraction of cells in group (%)',
                                        return_fig=True,
                                        standard_scale='var') #, values_to_plot="logfoldchanges" , cmap='bwr'

        categories_order = dp.categories_order
        dp.savefig(save_path + save_pre + '_dotplot.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_stacked_violin(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                               groupby=groupby, show=False, dendrogram=True, figsize=(8, 6))
        plt.savefig(save_path + save_pre + '_violin.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)

        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_matrixplot(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                           groupby=groupby, show=False, dendrogram=True, figsize=(8, 6))
        plt.savefig(save_path + save_pre + '_matrixplot.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)


        ######################################################################################
        groupby = 'cluster_name_acronym'
        adata = adata_exp_TH_human
        print(adata)
        #sc.pp.filter_cells(adata, min_counts=2)
        save_pre = 'human'
        key = 'wilcoxon' #'t-test_overestim_var'#'wilcoxon'#'t-test'
        key_filtered ='wilcoxon' #'t-test_overestim_var'# 'wilcoxon-filtered'#'t-test-filtered'

        cluster_counts = adata.obs[groupby].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        adata = adata[adata.obs[groupby].isin(keep)].copy()

        sc.tl.rank_genes_groups(adata, groupby, method='t-test', key_added="t-test")
        sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added="wilcoxon")
        sc.tl.rank_genes_groups(adata, groupby, method='t-test_overestim_var', key_added="t-test_overestim_var")
        sc.tl.rank_genes_groups(adata, groupby, method='logreg', key_added="logreg")

        sc.pl.rank_genes_groups(adata, n_genes=ntop_genes, sharey=False, key=key, show=False)
        # sc.tl.filter_rank_genes_groups(adata, key=key, key_added=key_filtered,
        #                                min_in_group_fraction=0.2,
        #                                max_out_group_fraction=0.5,
        #                                # min_in_group_fraction=0,
        #                                # #min_fold_change=1,
        #                                # max_out_group_fraction=1,
        #                                min_fold_change=1)
        # else:
        #     # pass
        #     sc.tl.filter_rank_genes_groups(adata, key="wilcoxon", key_added="wilcoxon_filtered",
        #                                    min_fold_change=2)
        plt.savefig(save_path + save_pre + 'degs.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)

        # rcParams["figure.subplot.top"] = 0.8
        # rcParams["figure.subplot.bottom"] = 0.2
        # rcParams["figure.subplot.left"] = 0.2
        # with plt.rc_context({"figure.figsize": (12, 16)}):
        sc.pl.rank_genes_groups_heatmap(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                        groupby=groupby, show_gene_labels=True, show=False, dendrogram=True,
                                        figsize=(6, 8))
        plt.savefig(save_path + save_pre + '_heatmap.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (12, 12)}):

        dp = sc.pl.rank_genes_groups_dotplot(adata, n_genes=ntop_genes_visual, key=key_filtered, #dot_min=0.3, dot_max=1,
                                        groupby=groupby,
                                        show=False,
                                        dendrogram=False,
                                        figsize=(14, 5),
                                        smallest_dot=0,
                                        #var_group_positions=var_group_positions,
                                        standard_scale='var',#,
                                             #colorbar_title='Mean expression in group',
                                             #size_title='Fraction of cells in group (%)',
                                        return_fig=True
                                        #values_to_plot="logfoldchanges",
                                        #colorbar_title='log fold change'
                                        )  # , values_to_plot="logfoldchanges" cmap='bwr'

        X_1 = TH_cluster_list
        print('X_1', X_1)
        X_ordered = categories_order
        print('X_ordered', X_ordered)
        Y_1 = dp.var_names
        print(Y_1)
        var_names_ordered = reorder(X_1, X_ordered, Y_1, ntop=ntop_genes_visual)
        print(var_names_ordered)
        dp.savefig(save_path + save_pre + '_dotplot.' + self.fig_format, format=self.fig_format,
                   dpi=self.fig_dpi)

        var_names_ordered_ntop = []
        for i in range(len(categories_order)):
            li = var_names_ordered[i*ntop_genes_visual:i*ntop_genes_visual+ntop_genes_visual]
            var_names_ordered_ntop.append(li)
        var_names_ordered_dict = {k: v for k, v in zip(X_ordered, var_names_ordered_ntop)}

        var_group_labels = []
        for i in range(len(categories_order)):
            var_group_labels = [categories_order[i]] * 3 + var_group_labels

        dp = sc.pl.rank_genes_groups_dotplot(adata, key=key_filtered,
                                             # dot_min=0.3, dot_max=1,
                                             groupby=groupby,
                                             show=False,
                                             dendrogram=False,
                                             figsize=(14, 5),
                                             smallest_dot=0,
                                             var_names=var_names_ordered_dict,
                                             # groups = cluster_name_unique,
                                             categories_order=categories_order,
                                             var_group_labels=var_group_labels,
                                             #colorbar_title='Mean expression in group',
                                             #size_title='Fraction of cells in group (%)',
                                             # var_group_positions=var_group_positions,
                                             standard_scale='var',  # ,
                                             return_fig=True
                                             # values_to_plot="logfoldchanges",
                                             # colorbar_title='log fold change'
                                             )  # , values_to_plot="logfoldchanges" cmap='bwr'

        dp.savefig(save_path + save_pre + '_dotplot.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_stacked_violin(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                               groupby=groupby, show=False, dendrogram=True, figsize=(8, 6))
        plt.savefig(save_path + save_pre + '_violin.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)

        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_matrixplot(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                           groupby=groupby, show=False, dendrogram=True, figsize=(8, 6))
        plt.savefig(save_path + save_pre + '_matrixplot.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)


        # homoglous to mouse marker genes
        sc.pl.rank_genes_groups_dotplot(adata,
                                        var_names=['AHI1', 'KIF5A', 'PCSK2', 'RIMS3', 'SHANK3'], # dot_min=0.3, dot_max=1,
                                        groupby=groupby,
                                        key=key_filtered,
                                        show=False,
                                        dendrogram=True,
                                        values_to_plot="logfoldchanges",
                                        figsize=(8, 4),
                                        smallest_dot=20,
                                        standard_scale='var'
                                        )  # , values_to_plot="logfoldchanges" cmap='bwr'
        plt.savefig(save_path + save_pre + '_dotplot_mouse_homologous.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)



        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.left"] = 0.1


    def experiment_2_clusters_deg(self):

        sns.set(style='white')
        TINY_SIZE = 18  # 39
        SMALL_SIZE = 18  # 42
        MEDIUM_SIZE = 18  # 46
        BIGGER_SIZE = 18  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/2_experiment_clusters_deg/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        #adata_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_embedding.h5ad')

        TH_cluster_list = []
        cluster_name_unique = sorted(Counter(self.adata_embedding.obs['cluster_name_acronym']).keys(),
                                     key=lambda t: int(t.split('-')[0]))
        for cluster_name in cluster_name_unique:
            if cluster_name.split('-')[1] == 'TH' and cluster_name.split('-')[2] == 'TH':
                TH_cluster_list.append(cluster_name)

        # adata_embedding_TH = self.adata_embedding[self.adata_embedding.obs['cluster_name_acronym'].isin(TH_cluster_list)]
        # adata_embedding_TH_mouse = adata_embedding_TH[adata_embedding_TH.obs['dataset'].isin(['Mouse'])]
        # adata_embedding_TH_human = adata_embedding_TH[adata_embedding_TH.obs['dataset'].isin(['Human'])]

        # adata_exp_mouse = self.adata_mouse_exp[adata_embedding_mouse.obs_names, :]
        # adata_exp_human = self.adata_human_exp[adata_embedding_human.obs_names, :]

        figsize = (21, 21)

        ######################################################################################
        groupby = 'cluster_name_acronym'
        adata = self.adata_human_exp
        print(adata)
        sc.pp.filter_cells(adata, min_counts=2)
        ntop_genes = 100
        ntop_genes_visual = 1
        save_pre = 'human'
        key = 'wilcoxon' #'t-test_overestim_var'#'wilcoxon'#'t-test'
        key_filtered ='wilcoxon' #'t-test_overestim_var'# 'wilcoxon-filtered'#'t-test-filtered'

        cluster_counts = adata.obs[groupby].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        adata = adata[adata.obs[groupby].isin(keep)].copy()

        #sc.tl.rank_genes_groups(adata, groupby, method='t-test', key_added="t-test")
        sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added="wilcoxon")
        #sc.tl.rank_genes_groups(adata, groupby, method='t-test_overestim_var', key_added="t-test_overestim_var")
        #sc.tl.rank_genes_groups(adata, groupby, method='logreg', key_added="logreg")

        sc.pl.rank_genes_groups(adata, n_genes=ntop_genes, sharey=False, key=key, show=False)
        # sc.tl.filter_rank_genes_groups(adata, key=key, key_added=key_filtered,
        #                                min_in_group_fraction=0.2,
        #                                max_out_group_fraction=0.5,
        #                                # min_in_group_fraction=0,
        #                                # #min_fold_change=1,
        #                                # max_out_group_fraction=1,
        #                                min_fold_change=1)
        # else:
        #     # pass
        #     sc.tl.filter_rank_genes_groups(adata, key="wilcoxon", key_added="wilcoxon_filtered",
        #                                    min_fold_change=2)
        plt.savefig(save_path + save_pre + 'degs.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)

        # rcParams["figure.subplot.top"] = 0.8
        # rcParams["figure.subplot.bottom"] = 0.2
        # rcParams["figure.subplot.left"] = 0.2
        # with plt.rc_context({"figure.figsize": (12, 16)}):

        sc.pl.rank_genes_groups_heatmap(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                        groupby=groupby, show_gene_labels=True, show=False, dendrogram=True,
                                        figsize=figsize)
        plt.savefig(save_path + save_pre + '_heatmap.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (12, 12)}):

        c = Colormap()
        color_map = c.cmap_linear('white', 'white', 'red')

        dp = sc.pl.rank_genes_groups_dotplot(adata, n_genes=ntop_genes_visual, key=key_filtered, #dot_min=0.3, dot_max=1,
                                        groupby=groupby,
                                        show=False,
                                        dendrogram=True,
                                        #categories_order=categories_order,
                                        figsize= (24, 21),
                                        smallest_dot=0,
                                        standard_scale='var',
                                        cmap=color_map, return_fig=True,
                                        edgecolors='black'
                                        #values_to_plot="logfoldchanges",
                                        #colorbar_title='log fold change'
                                        )  # , values_to_plot="logfoldchanges"

        categories_order = dp.categories_order
        dp.add_totals(color=[self.cluster_color_dict[x] for x in categories_order])  # .style(dot_edge_color='black', dot_edge_lw=0.5)
        dp.legend(width=3)

        #var_group_labels = dp.var_group_labels
        #dp.legend(width=3, colorbar_title='log fold change')

        dp.savefig(save_path + save_pre + '_dotplot.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)

        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_stacked_violin(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                               groupby=groupby, show=False, dendrogram=True, figsize=figsize)
        plt.savefig(save_path + save_pre + '_violin.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)

        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_matrixplot(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                           groupby=groupby, show=False, dendrogram=True, figsize=figsize)
        plt.savefig(save_path + save_pre + '_matrixplot.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)


        # # homoglous to mouse marker genes
        # sc.pl.rank_genes_groups_dotplot(adata,
        #                                 var_names=['AHI1', 'KIF5A', 'PCSK2', 'RIMS3', 'SHANK3'], # dot_min=0.3, dot_max=1,
        #                                 groupby=groupby,
        #                                 key=key_filtered,
        #                                 show=False,
        #                                 dendrogram=True,
        #                                 values_to_plot="logfoldchanges",
        #                                 figsize=(8, 4),
        #                                 smallest_dot=20,
        #                                 standard_scale='var',
        #                                 cmap='bwr')  # , values_to_plot="logfoldchanges"
        # plt.savefig(save_path + save_pre + '_dotplot_mouse_homologous.' + self.fig_format, format=self.fig_format,
        #             dpi=self.fig_dpi)

        ##########################
        # mouse degs

        groupby = 'cluster_name_acronym'
        adata = self.adata_mouse_exp
        sc.pp.filter_cells(adata, min_counts=2)
        ntop_genes = 100
        ntop_genes_visual = 1
        save_pre = 'mouse'
        key = 'wilcoxon'  # 't-test_overestim_var'#'wilcoxon'#'t-test'
        key_filtered = 'wilcoxon'  # 't-test_overestim_var'# 'wilcoxon-filtered'#'t-test-filtered'

        cluster_counts =  adata.obs[groupby].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        adata = adata[adata.obs[groupby].isin(keep)].copy()

        #sc.tl.rank_genes_groups(adata, groupby, method='t-test', key_added="t-test")
        sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added="wilcoxon")
        #sc.tl.rank_genes_groups(adata, groupby, method='t-test_overestim_var', key_added="t-test_overestim_var")
        #sc.tl.rank_genes_groups(adata, groupby, method='logreg', key_added="logreg")
        sc.pl.rank_genes_groups(adata, n_genes=ntop_genes, sharey=False, key=key, show=False)
        # sc.tl.filter_rank_genes_groups(adata, key=key, key_added=key_filtered,
        #                                #min_in_group_fraction=0.2,
        #                                #max_out_group_fraction=0.5,
        #                                min_fold_change=3)
        # else:
        #     # pass
        #     sc.tl.filter_rank_genes_groups(adata, key="wilcoxon", key_added="wilcoxon_filtered",
        #                                    min_fold_change=2)
        plt.savefig(save_path + save_pre + 'degs.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)

        rcParams["figure.subplot.top"] = 0.8
        rcParams["figure.subplot.bottom"] = 0.15
        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.right"] = 0.9
        # with plt.rc_context({"figure.figsize": (12, 16)}):



        sc.pl.rank_genes_groups_heatmap(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                        groupby=groupby, show_gene_labels=True, show=False, dendrogram=True,
                                        figsize=figsize)
        plt.savefig(save_path + save_pre + '_heatmap.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (12, 12)}):
        dp = sc.pl.rank_genes_groups_dotplot(adata,  key=key_filtered, n_genes=ntop_genes_visual,
                                        groupby=groupby, show=False,
                                        dendrogram=False,
                                        #groups = cluster_name_unique,
                                        #categories_order=categories_order,
                                        #var_group_labels = var_group_labels,
                                        figsize=figsize,
                                        smallest_dot=0,
                                        return_fig=True,
                                        standard_scale='var',
                                         cmap=color_map,
                                         dot_max=1,
                                             edgecolors='black'
                                         #values_to_plot="logfoldchanges"
                                         #colorbar_title='log fold change'
                                         ) #, values_to_plot="logfoldchanges" , dot_max=0.1
        X_1 = cluster_name_unique
        print('X_1', X_1)
        X_ordered = categories_order
        print('X_ordered', X_ordered)
        Y_1 = dp.var_names
        print('Y_1', Y_1)
        var_names_ordered = reorder(X_1, X_ordered, Y_1)
        var_names_ordered_dict = {k:v for k,v in zip(X_ordered, var_names_ordered)}
        #categories_order = dp.categories_order
        dp.add_totals()#.style(dot_edge_color='black', dot_edge_lw=0.5)
        #dp.legend(width=3, colorbar_title='log fold change')
        dp.legend(width=3)

        dp.savefig(save_path + save_pre + '_dotplot.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)


        dp = sc.pl.rank_genes_groups_dotplot(adata, key=key_filtered, # n_genes=ntop_genes_visual,
                                             groupby=groupby, show=False,
                                             dendrogram=False,
                                             var_names=var_names_ordered_dict,
                                             # groups = cluster_name_unique,
                                             categories_order=categories_order,
                                             var_group_labels = categories_order,
                                             figsize=figsize,
                                             smallest_dot=0,
                                             return_fig=True,
                                             standard_scale='var',
                                             cmap=color_map,
                                             dot_max=1,
                                             edgecolors='black'
                                             # values_to_plot="logfoldchanges"
                                             # colorbar_title='log fold change'
                                             )  # , values_to_plot="logfoldchanges" , dot_max=0.1

        # categories_order = dp.categories_order
        dp.add_totals(color=[self.cluster_color_dict[x] for x in categories_order])
        # dp.legend(width=3, colorbar_title='log fold change')
        dp.legend(width=3)
        dp.savefig(save_path + save_pre + '_dotplot.' + self.fig_format, format=self.fig_format,
                   dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_stacked_violin(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                               groupby=groupby, show=False, dendrogram=True, figsize=figsize)
        plt.savefig(save_path + save_pre + '_violin.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)

        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_matrixplot(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                           groupby=groupby, show=False, dendrogram=True, figsize=figsize)
        plt.savefig(save_path + save_pre + '_matrixplot.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)


        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.left"] = 0.1

    def experiment_2_1_homoregions_deg(self):

        sns.set(style='white')
        TINY_SIZE = 22  # 39
        SMALL_SIZE = 22 # 42
        MEDIUM_SIZE = 22  # 46
        BIGGER_SIZE = 22  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/2_1_experiment_homoregions_deg/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        adata_mouse_gene_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'm_embeddings.h5ad')
        adata_human_gene_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'h_embeddings.h5ad')

        mouse_gene_list = adata_mouse_gene_embedding.obs_names.to_list()
        human_gene_list = adata_human_gene_embedding.obs_names.to_list()

        # adata_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_embedding.h5ad')

        # TH_cluster_list = []
        # cluster_name_unique = sorted(Counter(self.adata_embedding.obs['cluster_name_acronym']).keys(),
        #                              key=lambda t: int(t.split('-')[0]))
        # for cluster_name in cluster_name_unique:
        #     if cluster_name.split('-')[1] == 'TH' and cluster_name.split('-')[2] == 'TH':
        #         TH_cluster_list.append(cluster_name)

        # homologous regions
        human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        homo_region_dict = OrderedDict()
        for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
            homo_region_dict[x] = y
        mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        mouse_homo_region_list = list(human_mouse_homo_region['Mouse'].values)
        human_homo_set = set(human_mouse_homo_region['Human'].values)
        human_homo_region_list = list(human_mouse_homo_region['Human'].values)

        figsize = (10, 8)
        ######################################################################################
        groupby = 'region_name'

        adata = self.adata_human_exp[self.adata_human_exp.obs['region_name'].isin(human_homo_region_list)]
        #adata = adata[:, human_gene_list]
        #print(adata)
        #sc.pp.filter_cells(adata, min_counts=2)
        ntop_genes = 20
        ntop_genes_visual = 1
        save_pre = 'human'
        key = 't-test'#'wilcoxon'  # 't-test_overestim_var'#'wilcoxon'#'t-test'
        key_filtered = 't-test'#'wilcoxon'  # 't-test_overestim_var'# 'wilcoxon-filtered'#'t-test-filtered'

        rcParams["figure.subplot.top"] = 0.8
        rcParams["figure.subplot.bottom"] = 0.15
        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.right"] = 0.9

        cluster_counts = adata.obs[groupby].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        adata = adata[adata.obs[groupby].isin(keep)].copy()

        #adata = adata

        sc.tl.rank_genes_groups(adata, groupby, method='t-test', key_added="t-test")
        #sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added="wilcoxon") #, groups=human_homo_region_list
        # sc.tl.rank_genes_groups(adata, groupby, method='t-test_overestim_var', key_added="t-test_overestim_var")
        # sc.tl.rank_genes_groups(adata, groupby, method='logreg', key_added="logreg")

        sc.pl.rank_genes_groups(adata, n_genes=ntop_genes, sharey=False, key=key, show=False) #, groups=human_homo_region_list

        plt.savefig(save_path + save_pre + 'degs.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)

        # rcParams["figure.subplot.top"] = 0.8
        # rcParams["figure.subplot.bottom"] = 0.2
        # rcParams["figure.subplot.left"] = 0.2
        # with plt.rc_context({"figure.figsize": (12, 16)}):

        c = Colormap()
        color_map = c.cmap_linear('white', 'white', 'red')

        sc.pl.rank_genes_groups_heatmap(adata, n_genes=ntop_genes_visual, key=key_filtered, groups=human_homo_region_list,
                                        groupby=groupby, show_gene_labels=True, show=False, dendrogram=True,
                                        figsize=figsize)
        plt.savefig(save_path + save_pre + '_heatmap.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (12, 12)}):
        #fig, ax = plt.subplots(figsize=(5, 4), dpi=self.fig_dpi)

        dp = sc.pl.rank_genes_groups_dotplot(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                             # dot_min=0.3, dot_max=1,
                                             groups=human_homo_region_list,
                                             groupby=groupby,
                                             show=False,
                                             dendrogram=True,
                                             figsize=figsize,
                                             #ax=ax,
                                             #var_group_labels=None,
                                             # categories_order=categories_order,
                                             smallest_dot=0, dot_max=0.03,
                                             standard_scale='var',
                                             cmap=color_map, return_fig=True, edgecolors='black'
                                             # values_to_plot="logfoldchanges",
                                             # colorbar_title='log fold change'
                                             )  # , values_to_plot="logfoldchanges"
        #print(l.get_text())
        categories_order = dp.categories_order
        dp.add_totals(color=[self.human_88_color_dict[x] for x in categories_order])  # .style(dot_edge_color='black', dot_edge_lw=0.5)
        dp.legend(width=3)
        print(dp.get_axes())
        #print(dp)
        #plt.rc('xtick', labelsize=MEDIUM_SIZE, labelstyle='italic')  # fontsize of the tick labels
        # var_group_labels = dp.var_group_labels
        # dp.legend(width=3, colorbar_title='log fold change')
        for l in dp.get_axes()['mainplot_ax'].get_xticklabels():
            l.set_fontstyle('italic')
        # xtick_labels = plt.gca().get_xticklabels()
        # for label in xtick_labels:
        #     label.set_fontstyle('italic')
        dp.savefig(save_path + save_pre + '_dotplot.' + self.fig_format, format=self.fig_format,
                   dpi=self.fig_dpi)

        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_stacked_violin(adata, n_genes=ntop_genes_visual, key=key_filtered, groups=human_homo_region_list,
                                               groupby=groupby, show=False, dendrogram=True, figsize=figsize)
        plt.savefig(save_path + save_pre + '_violin.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)

        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_matrixplot(adata, n_genes=ntop_genes_visual, key=key_filtered, groups=human_homo_region_list,
                                           groupby=groupby, show=False, dendrogram=True, figsize=figsize)
        plt.savefig(save_path + save_pre + '_matrixplot.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)


        ##########################
        # mouse degs

        groupby = 'region_name'
        adata = self.adata_mouse_exp[self.adata_mouse_exp.obs['region_name'].isin(mouse_homo_region_list)]
        #adata = adata[:, mouse_gene_list]
        sc.pp.filter_cells(adata, min_counts=2)
        ntop_genes = 20
        ntop_genes_visual = 1
        save_pre = 'mouse'
        key ='t-test'# 'wilcoxon'  # 't-test_overestim_var'#'wilcoxon'#'t-test'
        key_filtered = 't-test'# 'wilcoxon'  # 't-test_overestim_var'# 'wilcoxon-filtered'#'t-test-filtered'

        # cluster_counts = adata.obs[groupby].value_counts()
        # keep = cluster_counts.index[cluster_counts >= 2]
        # adata = adata[adata.obs[groupby].isin(keep)].copy()

        sc.tl.rank_genes_groups(adata, groupby, method='t-test', key_added="t-test")
        #sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added="wilcoxon") #, groups=mouse_homo_region_list
        # sc.tl.rank_genes_groups(adata, groupby, method='t-test_overestim_var', key_added="t-test_overestim_var")
        # sc.tl.rank_genes_groups(adata, groupby, method='logreg', key_added="logreg")
        sc.pl.rank_genes_groups(adata, n_genes=ntop_genes, sharey=False, key=key, show=False) #, groups=mouse_homo_region_list
        # sc.tl.filter_rank_genes_groups(adata, key=key, key_added=key_filtered,
        #                                #min_in_group_fraction=0.2,
        #                                #max_out_group_fraction=0.5,
        #                                min_fold_change=3)
        # else:
        #     # pass
        #     sc.tl.filter_rank_genes_groups(adata, key="wilcoxon", key_added="wilcoxon_filtered",
        #                                    min_fold_change=2)
        plt.savefig(save_path + save_pre + 'degs.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)

        # rcParams["figure.subplot.top"] = 0.8
        # rcParams["figure.subplot.bottom"] = 0.15
        # rcParams["figure.subplot.left"] = 0.15
        # rcParams["figure.subplot.right"] = 0.9
        # with plt.rc_context({"figure.figsize": (12, 16)}):

        sc.pl.rank_genes_groups_heatmap(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                        groupby=groupby, show_gene_labels=True, show=False, dendrogram=True,
                                        figsize=figsize)
        plt.savefig(save_path + save_pre + '_heatmap.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (12, 12)}):
        dp = sc.pl.rank_genes_groups_dotplot(adata, key=key_filtered, n_genes=ntop_genes_visual,
                                             groupby=groupby, show=False, groups=mouse_homo_region_list,
                                             dendrogram=False,
                                             # groups = cluster_name_unique,
                                             # categories_order=categories_order,
                                             # var_group_labels = var_group_labels,
                                             figsize=figsize,
                                             smallest_dot=0,
                                             return_fig=True,
                                             standard_scale='var',
                                             cmap=color_map,
                                             dot_max=0.03, edgecolors='black'
                                             # values_to_plot="logfoldchanges"
                                             # colorbar_title='log fold change'
                                             )  # , values_to_plot="logfoldchanges" , dot_max=0.1
        X_1 = human_homo_region_list
        print('X_1', X_1)
        X_ordered = categories_order
        print('X_ordered', X_ordered)
        Y_1 = dp.var_names
        print('Y_1', Y_1)
        categories_order_mouse= reorder(X_1, X_ordered, mouse_homo_region_list)
        var_names_ordered = reorder(X_1, X_ordered, Y_1)
        var_names_ordered_dict = {k: v for k, v in zip(categories_order_mouse, var_names_ordered)}
        # categories_order = dp.categories_order
        dp.add_totals(color=[self.mouse_64_color_dict[x] for x in categories_order_mouse])   # .style(dot_edge_color='black', dot_edge_lw=0.5)
        # dp.legend(width=3, colorbar_title='log fold change')
        dp.legend(width=3)

        dp.savefig(save_path + save_pre + '_dotplot.' + self.fig_format, format=self.fig_format,
                   dpi=self.fig_dpi)

        dp = sc.pl.rank_genes_groups_dotplot(adata, key=key_filtered,  # n_genes=ntop_genes_visual,
                                             groupby=groupby, show=False,
                                             dendrogram=False,
                                             var_names=var_names_ordered_dict,
                                             groups = categories_order_mouse,
                                             categories_order=categories_order_mouse,
                                             var_group_labels=categories_order_mouse,
                                             figsize=figsize,
                                             smallest_dot=0,
                                             return_fig=True,
                                             standard_scale='var',
                                             cmap=color_map,
                                             dot_max=0.05,edgecolors='black'
                                             # values_to_plot="logfoldchanges"
                                             # colorbar_title='log fold change'
                                             )  # , values_to_plot="logfoldchanges" , dot_max=0.1

        # categories_order = dp.categories_order
        dp.add_totals(color=[self.mouse_64_color_dict[x] for x in categories_order_mouse]) # .style(dot_edge_color='black', dot_edge_lw=0.5)
        # dp.legend(width=3, colorbar_title='log fold change')
        dp.legend(width=3)
        #ax = dp.mainplot_ax
        # Loop through ticklabels and make them italic
        #ax =
        # Loop through ticklabels and make them italic
        for l in dp.get_axes()['mainplot_ax'].get_xticklabels():
            l.set_style('italic')

        dp.savefig(save_path + save_pre + '_dotplot.' + self.fig_format, format=self.fig_format,
                   dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_stacked_violin(adata,  key=key_filtered, var_names=var_names_ordered_dict, #n_genes=ntop_genes_visual,
                                             groups = categories_order_mouse,
                                               groupby=groupby, show=False, dendrogram=True, figsize=figsize)
        plt.savefig(save_path + save_pre + '_violin.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)

        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_matrixplot(adata,  key=key_filtered, var_names=var_names_ordered_dict, #n_genes=ntop_genes_visual,
                                             groups = categories_order_mouse,
                                           groupby=groupby, show=False, dendrogram=True, figsize=figsize)
        plt.savefig(save_path + save_pre + '_matrixplot.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)

        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.left"] = 0.1


    def experiment_2_3_homoregions_deg_proportion(self):

        '''


        :return:
        '''

        sns.set(style='white')
        TINY_SIZE = 22  # 39
        SMALL_SIZE = 22  # 42
        MEDIUM_SIZE = 22  # 46
        BIGGER_SIZE = 22  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/2_3_experiment_homoregions_deg_proportion/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        adata_mouse_gene_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'm_embeddings.h5ad')
        adata_human_gene_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'h_embeddings.h5ad')

        mouse_gene_list = adata_mouse_gene_embedding.obs_names.to_list()
        human_gene_list = adata_human_gene_embedding.obs_names.to_list()

        # adata_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_embedding.h5ad')

        # TH_cluster_list = []
        # cluster_name_unique = sorted(Counter(self.adata_embedding.obs['cluster_name_acronym']).keys(),
        #                              key=lambda t: int(t.split('-')[0]))
        # for cluster_name in cluster_name_unique:
        #     if cluster_name.split('-')[1] == 'TH' and cluster_name.split('-')[2] == 'TH':
        #         TH_cluster_list.append(cluster_name)

        # homologous regions
        human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        homo_region_dict = OrderedDict()
        for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
            homo_region_dict[x] = y
        mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        mouse_homo_region_list = list(human_mouse_homo_region['Mouse'].values)
        human_homo_set = set(human_mouse_homo_region['Human'].values)
        human_homo_region_list = list(human_mouse_homo_region['Human'].values)

        figsize = (10, 8)
        ######################################################################################
        groupby = 'region_name'

        adata = self.adata_human_exp[self.adata_human_exp.obs['region_name'].isin(human_homo_region_list)]
        #adata = adata[:, human_gene_list]
        #print(adata)
        #sc.pp.filter_cells(adata, min_counts=2)
        ntop_genes = 100
        ntop_genes_visual = 1
        save_pre = 'human'
        key = 'wilcoxon'  # 't-test_overestim_var'#'wilcoxon'#'t-test'
        key_filtered = 'wilcoxon'  # 't-test_overestim_var'# 'wilcoxon-filtered'#'t-test-filtered'

        rcParams["figure.subplot.top"] = 0.8
        rcParams["figure.subplot.bottom"] = 0.15
        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.right"] = 0.9

        cluster_counts = adata.obs[groupby].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        adata = adata[adata.obs[groupby].isin(keep)].copy()

        #adata = adata

        # sc.tl.rank_genes_groups(adata, groupby, method='t-test', key_added="t-test")
        sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added="wilcoxon") #, groups=human_homo_region_list
        # sc.tl.rank_genes_groups(adata, groupby, method='t-test_overestim_var', key_added="t-test_overestim_var")
        # sc.tl.rank_genes_groups(adata, groupby, method='logreg', key_added="logreg")

        #sc.pl.rank_genes_groups(adata, n_genes=ntop_genes, sharey=False, key=key, show=False) #, groups=human_homo_region_list

        c = Colormap()
        color_map = c.cmap_linear('white', 'white', 'red')



        groupby = 'region_name'
        adata = self.adata_mouse_exp[self.adata_mouse_exp.obs['region_name'].isin(mouse_homo_region_list)]
        #adata = adata[:, mouse_gene_list]
        sc.pp.filter_cells(adata, min_counts=2)
        ntop_genes = 100
        ntop_genes_visual = 1
        save_pre = 'mouse'
        key = 'wilcoxon'  # 't-test_overestim_var'#'wilcoxon'#'t-test'
        key_filtered = 'wilcoxon'  # 't-test_overestim_var'# 'wilcoxon-filtered'#'t-test-filtered'

        # cluster_counts = adata.obs[groupby].value_counts()
        # keep = cluster_counts.index[cluster_counts >= 2]
        # adata = adata[adata.obs[groupby].isin(keep)].copy()

        # sc.tl.rank_genes_groups(adata, groupby, method='t-test', key_added="t-test")
        sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added="wilcoxon") #, groups=mouse_homo_region_list
        # sc.tl.rank_genes_groups(adata, groupby, method='t-test_overestim_var', key_added="t-test_overestim_var")
        # sc.tl.rank_genes_groups(adata, groupby, method='logreg', key_added="logreg")


        mouse_pval_cutoff = 0.001
        human_pval_cutoff = 0.001

        for m_region in mouse_homo_region_list:
            for h_region in human_homo_region_list:

                mouse_degs_list = sc.get.rank_genes_groups_df(self.adata_mouse_exp,
                                                              group=m_region,
                                                              key='wilcoxon',
                                                              # log2fc_min=0.25,
                                                              pval_cutoff=mouse_pval_cutoff)['names'].squeeze().str.strip().tolist()
                human_degs_list = sc.get.rank_genes_groups_df(self.adata_human_exp,
                                                              group=h_region,
                                                              key='wilcoxon',
                                                              # log2fc_min=0.25,
                                                              pval_cutoff=human_pval_cutoff)['names'].squeeze().str.strip().tolist()




        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.left"] = 0.1


    def experiment_2_1_homoregions_deg_homo(self):

        sns.set(style='white')
        TINY_SIZE = 22  # 39
        SMALL_SIZE = 22  # 42
        MEDIUM_SIZE = 22  # 46
        BIGGER_SIZE = 22  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/2_1_experiment_homoregions_deg_homo/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        adata_mouse_gene_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'm_embeddings.h5ad')
        adata_human_gene_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'h_embeddings.h5ad')

        mouse_gene_list = adata_mouse_gene_embedding.obs_names.to_list()
        human_gene_list = adata_human_gene_embedding.obs_names.to_list()

        # adata_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_embedding.h5ad')

        # TH_cluster_list = []
        # cluster_name_unique = sorted(Counter(self.adata_embedding.obs['cluster_name_acronym']).keys(),
        #                              key=lambda t: int(t.split('-')[0]))
        # for cluster_name in cluster_name_unique:
        #     if cluster_name.split('-')[1] == 'TH' and cluster_name.split('-')[2] == 'TH':
        #         TH_cluster_list.append(cluster_name)

        # homologous regions
        human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        homo_region_dict = OrderedDict()
        for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
            homo_region_dict[x] = y
        mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        mouse_homo_region_list = list(human_mouse_homo_region['Mouse'].values)
        human_homo_set = set(human_mouse_homo_region['Human'].values)
        human_homo_region_list = list(human_mouse_homo_region['Human'].values)

        figsize = (10, 8)
        ######################################################################################
        groupby = 'region_name'

        adata = self.adata_human_exp[self.adata_human_exp.obs['region_name'].isin(human_homo_region_list)]
        adata = adata[:, human_gene_list]
        # print(adata)
        # sc.pp.filter_cells(adata, min_counts=2)
        ntop_genes = 20
        ntop_genes_visual = 1
        save_pre = 'human'
        key = 't-test'#'wilcoxon'  # 't-test_overestim_var'#'wilcoxon'#'t-test'
        key_filtered = 't-test'#'wilcoxon'  # 't-test_overestim_var'# 'wilcoxon-filtered'#'t-test-filtered'

        rcParams["figure.subplot.top"] = 0.8
        rcParams["figure.subplot.bottom"] = 0.15
        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.right"] = 0.9

        cluster_counts = adata.obs[groupby].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        adata = adata[adata.obs[groupby].isin(keep)].copy()

        # adata = adata

        sc.tl.rank_genes_groups(adata, groupby, method='t-test', key_added="t-test")
        #sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon',
        #                        key_added="wilcoxon")  # , groups=human_homo_region_list
        # sc.tl.rank_genes_groups(adata, groupby, method='t-test_overestim_var', key_added="t-test_overestim_var")
        # sc.tl.rank_genes_groups(adata, groupby, method='logreg', key_added="logreg")

        sc.pl.rank_genes_groups(adata, n_genes=ntop_genes, sharey=False, key=key,
                                show=False)  # , groups=human_homo_region_list

        plt.savefig(save_path + save_pre + 'degs.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)

        # rcParams["figure.subplot.top"] = 0.8
        # rcParams["figure.subplot.bottom"] = 0.2
        # rcParams["figure.subplot.left"] = 0.2
        # with plt.rc_context({"figure.figsize": (12, 16)}):

        c = Colormap()
        color_map = c.cmap_linear('white', 'white', 'red')

        sc.pl.rank_genes_groups_heatmap(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                        groups=human_homo_region_list,
                                        groupby=groupby, show_gene_labels=True, show=False, dendrogram=True,
                                        figsize=figsize)
        plt.savefig(save_path + save_pre + '_heatmap.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (12, 12)}):
        # fig, ax = plt.subplots(figsize=(5, 4), dpi=self.fig_dpi)

        dp = sc.pl.rank_genes_groups_dotplot(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                             # dot_min=0.3, dot_max=1,
                                             groups=human_homo_region_list,
                                             groupby=groupby,
                                             show=False,
                                             dendrogram=True,
                                             figsize=figsize,
                                             # ax=ax,
                                             # var_group_labels=None,
                                             # categories_order=categories_order,
                                             smallest_dot=0, dot_max=0.05,
                                             standard_scale='var',
                                             cmap=color_map, return_fig=True, edgecolors='black'
                                             # values_to_plot="logfoldchanges",
                                             # colorbar_title='log fold change'
                                             )  # , values_to_plot="logfoldchanges"
        # print(l.get_text())
        categories_order = dp.categories_order
        dp.add_totals(color=[self.human_88_color_dict[x] for x in
                             categories_order])  # .style(dot_edge_color='black', dot_edge_lw=0.5)
        dp.legend(width=3)
        print(dp.get_axes())
        # print(dp)
        # plt.rc('xtick', labelsize=MEDIUM_SIZE, labelstyle='italic')  # fontsize of the tick labels
        # var_group_labels = dp.var_group_labels
        # dp.legend(width=3, colorbar_title='log fold change')
        for l in dp.get_axes()['mainplot_ax'].get_xticklabels():
            l.set_fontstyle('italic')
        # xtick_labels = plt.gca().get_xticklabels()
        # for label in xtick_labels:
        #     label.set_fontstyle('italic')
        dp.savefig(save_path + save_pre + '_dotplot.' + self.fig_format, format=self.fig_format,
                   dpi=self.fig_dpi)

        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_stacked_violin(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                               groups=human_homo_region_list,
                                               groupby=groupby, show=False, dendrogram=True, figsize=figsize)
        plt.savefig(save_path + save_pre + '_violin.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)

        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_matrixplot(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                           groups=human_homo_region_list,
                                           groupby=groupby, show=False, dendrogram=True, figsize=figsize)
        plt.savefig(save_path + save_pre + '_matrixplot.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)

        ##########################
        # mouse degs

        groupby = 'region_name'
        adata = self.adata_mouse_exp[self.adata_mouse_exp.obs['region_name'].isin(mouse_homo_region_list)]
        adata = adata[:, mouse_gene_list]
        sc.pp.filter_cells(adata, min_counts=2)
        ntop_genes = 20
        ntop_genes_visual = 1
        save_pre = 'mouse'
        key = 't-test'#'wilcoxon'  # 't-test_overestim_var'#'wilcoxon'#'t-test'
        key_filtered = 't-test'#'wilcoxon'  # 't-test_overestim_var'# 'wilcoxon-filtered'#'t-test-filtered'

        # cluster_counts = adata.obs[groupby].value_counts()
        # keep = cluster_counts.index[cluster_counts >= 2]
        # adata = adata[adata.obs[groupby].isin(keep)].copy()

        sc.tl.rank_genes_groups(adata, groupby, method='t-test', key_added="t-test")
        #sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon',
        #                        key_added="wilcoxon")  # , groups=mouse_homo_region_list
        # sc.tl.rank_genes_groups(adata, groupby, method='t-test_overestim_var', key_added="t-test_overestim_var")
        # sc.tl.rank_genes_groups(adata, groupby, method='logreg', key_added="logreg")
        sc.pl.rank_genes_groups(adata, n_genes=ntop_genes, sharey=False, key=key,
                                show=False)  # , groups=mouse_homo_region_list
        # sc.tl.filter_rank_genes_groups(adata, key=key, key_added=key_filtered,
        #                                #min_in_group_fraction=0.2,
        #                                #max_out_group_fraction=0.5,
        #                                min_fold_change=3)
        # else:
        #     # pass
        #     sc.tl.filter_rank_genes_groups(adata, key="wilcoxon", key_added="wilcoxon_filtered",
        #                                    min_fold_change=2)
        plt.savefig(save_path + save_pre + 'degs.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)

        # rcParams["figure.subplot.top"] = 0.8
        # rcParams["figure.subplot.bottom"] = 0.15
        # rcParams["figure.subplot.left"] = 0.15
        # rcParams["figure.subplot.right"] = 0.9
        # with plt.rc_context({"figure.figsize": (12, 16)}):

        sc.pl.rank_genes_groups_heatmap(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                        groupby=groupby, show_gene_labels=True, show=False, dendrogram=True,
                                        figsize=figsize)
        plt.savefig(save_path + save_pre + '_heatmap.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (12, 12)}):
        dp = sc.pl.rank_genes_groups_dotplot(adata, key=key_filtered, n_genes=ntop_genes_visual,
                                             groupby=groupby, show=False, groups=mouse_homo_region_list,
                                             dendrogram=False,
                                             # groups = cluster_name_unique,
                                             # categories_order=categories_order,
                                             # var_group_labels = var_group_labels,
                                             figsize=figsize,
                                             smallest_dot=0,
                                             return_fig=True,
                                             standard_scale='var',
                                             cmap=color_map,
                                             dot_max=0.05, edgecolors='black'
                                             # values_to_plot="logfoldchanges"
                                             # colorbar_title='log fold change'
                                             )  # , values_to_plot="logfoldchanges" , dot_max=0.1
        X_1 = human_homo_region_list
        print('X_1', X_1)
        X_ordered = categories_order
        print('X_ordered', X_ordered)
        Y_1 = dp.var_names
        print('Y_1', Y_1)
        categories_order_mouse = reorder(X_1, X_ordered, mouse_homo_region_list)
        var_names_ordered = reorder(X_1, X_ordered, Y_1)
        var_names_ordered_dict = {k: v for k, v in zip(categories_order_mouse, var_names_ordered)}
        # categories_order = dp.categories_order
        dp.add_totals(color=[self.mouse_64_color_dict[x] for x in
                             categories_order_mouse])  # .style(dot_edge_color='black', dot_edge_lw=0.5)
        # dp.legend(width=3, colorbar_title='log fold change')
        dp.legend(width=3)

        dp.savefig(save_path + save_pre + '_dotplot.' + self.fig_format, format=self.fig_format,
                   dpi=self.fig_dpi)

        dp = sc.pl.rank_genes_groups_dotplot(adata, key=key_filtered,  # n_genes=ntop_genes_visual,
                                             groupby=groupby, show=False,
                                             dendrogram=False,
                                             var_names=var_names_ordered_dict,
                                             groups=categories_order_mouse,
                                             categories_order=categories_order_mouse,
                                             var_group_labels=categories_order_mouse,
                                             figsize=figsize,
                                             smallest_dot=0,
                                             return_fig=True,
                                             standard_scale='var',
                                             cmap=color_map,
                                             dot_max=0.05, edgecolors='black'
                                             # values_to_plot="logfoldchanges"
                                             # colorbar_title='log fold change'
                                             )  # , values_to_plot="logfoldchanges" , dot_max=0.1

        # categories_order = dp.categories_order
        dp.add_totals(color=[self.mouse_64_color_dict[x] for x in
                             categories_order_mouse])  # .style(dot_edge_color='black', dot_edge_lw=0.5)
        # dp.legend(width=3, colorbar_title='log fold change')
        dp.legend(width=3)
        # ax = dp.mainplot_ax
        # Loop through ticklabels and make them italic
        # ax =
        # Loop through ticklabels and make them italic
        for l in dp.get_axes()['mainplot_ax'].get_xticklabels():
            l.set_style('italic')

        dp.savefig(save_path + save_pre + '_dotplot.' + self.fig_format, format=self.fig_format,
                   dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_stacked_violin(adata, key=key_filtered, var_names=var_names_ordered_dict,
                                               # n_genes=ntop_genes_visual,
                                               groups=categories_order_mouse,
                                               groupby=groupby, show=False, dendrogram=True, figsize=figsize)
        plt.savefig(save_path + save_pre + '_violin.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)

        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_matrixplot(adata, key=key_filtered, var_names=var_names_ordered_dict,
                                           # n_genes=ntop_genes_visual,
                                           groups=categories_order_mouse,
                                           groupby=groupby, show=False, dendrogram=True, figsize=figsize)
        plt.savefig(save_path + save_pre + '_matrixplot.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)

        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.left"] = 0.1


    def experiment_2_2_homoregions_multiple_deg(self):

        #sns.set(style='white')
        TINY_SIZE = 22  # 39
        SMALL_SIZE = 22  # 42
        MEDIUM_SIZE = 22  # 46
        BIGGER_SIZE = 22  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        adata_mouse_gene_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'm_embeddings.h5ad')
        adata_human_gene_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'h_embeddings.h5ad')

        mouse_gene_list = adata_mouse_gene_embedding.obs_names.to_list()
        human_gene_list = adata_human_gene_embedding.obs_names.to_list()

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/2_2_experiment_homoregions_multiple_deg/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # homologous regions
        human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        homo_region_dict = OrderedDict()
        for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
            homo_region_dict[x] = y
        mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        mouse_homo_region_list = list(human_mouse_homo_region['Mouse'].values)
        human_homo_set = set(human_mouse_homo_region['Human'].values)
        human_homo_region_list = list(human_mouse_homo_region['Human'].values)

        figsize = (10, 8)
        ######################################################################################
        groupby = 'region_name'
        #adata = self.adata_human_exp[adata.var_names.isin(human_gene_list)]
        adata = self.adata_human_exp[self.adata_human_exp.obs['region_name'].isin(human_homo_region_list)]
        adata = adata[:, human_gene_list]

        #print(adata)
        #sc.pp.filter_cells(adata, min_counts=2)
        ntop_genes = 20
        ntop_genes_visual = 1
        save_pre = 'human'
        key = 't-test'#'wilcoxon'  # 't-test_overestim_var'#'wilcoxon'#'t-test'
        key_filtered = 't-test'#'wilcoxon'  # 't-test_overestim_var'# 'wilcoxon-filtered'#'t-test-filtered'

        rcParams["figure.subplot.top"] = 0.8
        rcParams["figure.subplot.bottom"] = 0.15
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.95

        cluster_counts = adata.obs[groupby].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        adata = adata[adata.obs[groupby].isin(keep)].copy()

        #adata = adata

        sc.tl.rank_genes_groups(adata, groupby, method='t-test', key_added="t-test")
        #sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added="wilcoxon") #, groups=human_homo_region_list
        # sc.tl.rank_genes_groups(adata, groupby, method='t-test_overestim_var', key_added="t-test_overestim_var")
        # sc.tl.rank_genes_groups(adata, groupby, method='logreg', key_added="logreg")

        sc.pl.rank_genes_groups(adata, n_genes=ntop_genes, sharey=False, key=key, show=False) #, groups=human_homo_region_list

        plt.savefig(save_path + save_pre + 'degs.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)

        c = Colormap()
        color_map = c.cmap_linear('white', 'white', 'red')

        sc.pl.rank_genes_groups_heatmap(adata, n_genes=ntop_genes_visual, key=key_filtered, groups=human_homo_region_list,
                                        groupby=groupby, show_gene_labels=True, show=False, dendrogram=True,
                                        figsize=figsize)
        plt.savefig(save_path + save_pre + '_heatmap.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)

        dp = sc.pl.rank_genes_groups_dotplot(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                             # dot_min=0.3, dot_max=1,
                                             groups=human_homo_region_list,
                                             groupby=groupby,
                                             show=False,
                                             dendrogram=True,
                                             figsize=figsize,
                                             #ax=ax,
                                             #var_group_labels=None,
                                             # categories_order=categories_order,
                                             smallest_dot=0, dot_max=0.05,
                                             standard_scale='var',
                                             cmap=color_map, return_fig=True, edgecolors='black'
                                             # values_to_plot="logfoldchanges",
                                             # colorbar_title='log fold change'
                                             )  # , values_to_plot="logfoldchanges"
        #print(l.get_text())
        categories_order = dp.categories_order
        categories_order_human = dp.categories_order
        dp.add_totals(color=[self.human_88_color_dict[x] for x in categories_order])  # .style(dot_edge_color='black', dot_edge_lw=0.5)
        dp.legend(width=3)
        print(dp.get_axes())
        #print(dp)
        #plt.rc('xtick', labelsize=MEDIUM_SIZE, labelstyle='italic')  # fontsize of the tick labels
        # var_group_labels = dp.var_group_labels
        # dp.legend(width=3, colorbar_title='log fold change')
        human_marker_gene_ntop = dp.var_names


        print('human_marker_gene_ntop', human_marker_gene_ntop)
        for l in dp.get_axes()['mainplot_ax'].get_xticklabels():
            l.set_fontstyle('italic')
        # xtick_labels = plt.gca().get_xticklabels()
        # for label in xtick_labels:
        #     label.set_fontstyle('italic')
        dp.savefig(save_path + save_pre + '_dotplot.' + self.fig_format, format=self.fig_format,
                   dpi=self.fig_dpi)

        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_stacked_violin(adata, n_genes=ntop_genes_visual, key=key_filtered, groups=human_homo_region_list,
                                               groupby=groupby, show=False, dendrogram=True, figsize=figsize)
        plt.savefig(save_path + save_pre + '_violin.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)

        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_matrixplot(adata, n_genes=ntop_genes_visual, key=key_filtered, groups=human_homo_region_list,
                                           groupby=groupby, show=False, dendrogram=True, figsize=figsize)
        plt.savefig(save_path + save_pre + '_matrixplot.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)
        ##########################
        # mouse degs
        groupby = 'region_name'
        adata = self.adata_mouse_exp[self.adata_mouse_exp.obs['region_name'].isin(mouse_homo_region_list)]
        adata = adata[:, mouse_gene_list]

        sc.pp.filter_cells(adata, min_counts=2)
        ntop_genes = 20
        ntop_genes_visual = 1
        save_pre = 'mouse'
        key = 't-test'#'wilcoxon'  # 't-test_overestim_var'#'wilcoxon'#'t-test'
        key_filtered = 't-test' #'wilcoxon'  # 't-test_overestim_var'# 'wilcoxon-filtered'#'t-test-filtered'

        # cluster_counts = adata.obs[groupby].value_counts()
        # keep = cluster_counts.index[cluster_counts >= 2]
        # adata = adata[adata.obs[groupby].isin(keep)].copy()

        sc.tl.rank_genes_groups(adata, groupby, method='t-test', key_added="t-test")
        #sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added="wilcoxon") #, groups=mouse_homo_region_list
        # sc.tl.rank_genes_groups(adata, groupby, method='t-test_overestim_var', key_added="t-test_overestim_var")
        # sc.tl.rank_genes_groups(adata, groupby, method='logreg', key_added="logreg")
        sc.pl.rank_genes_groups(adata, n_genes=ntop_genes, sharey=False, key=key, show=False) #, groups=mouse_homo_region_list
        # sc.tl.filter_rank_genes_groups(adata, key=key, key_added=key_filtered,
        #                                #min_in_group_fraction=0.2,
        #                                #max_out_group_fraction=0.5,
        #                                min_fold_change=3)
        # else:
        #     # pass
        #     sc.tl.filter_rank_genes_groups(adata, key="wilcoxon", key_added="wilcoxon_filtered",
        #                                    min_fold_change=2)
        plt.savefig(save_path + save_pre + 'degs.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)

        sc.pl.rank_genes_groups_heatmap(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                        groupby=groupby, show_gene_labels=True, show=False, dendrogram=True,
                                        figsize=figsize)
        plt.savefig(save_path + save_pre + '_heatmap.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (12, 12)}):
        dp = sc.pl.rank_genes_groups_dotplot(adata, key=key_filtered, n_genes=ntop_genes_visual,
                                             groupby=groupby, show=False, groups=mouse_homo_region_list,
                                             dendrogram=False,
                                             # groups = cluster_name_unique,
                                             # categories_order=categories_order,
                                             # var_group_labels = var_group_labels,
                                             figsize=figsize,
                                             smallest_dot=0,
                                             return_fig=True,
                                             standard_scale='var',
                                             cmap=color_map,
                                             dot_max=0.05, edgecolors='black'
                                             # values_to_plot="logfoldchanges"
                                             # colorbar_title='log fold change'
                                             )  # , values_to_plot="logfoldchanges" , dot_max=0.1
        X_1 = human_homo_region_list
        print('X_1', X_1)
        X_ordered = categories_order
        print('X_ordered', X_ordered)
        Y_1 = dp.var_names
        print('Y_1', Y_1)
        categories_order_mouse= reorder(X_1, X_ordered, mouse_homo_region_list)
        var_names_ordered = reorder(X_1, X_ordered, Y_1, ntop=ntop_genes_visual)

        #var_names_ordered_dict = {k: v for k, v in zip(categories_order_mouse, var_names_ordered)}
        var_names_ordered_dict = {}
        for k in range(len(categories_order_mouse)):
            var_names_ordered_k = [var_names_ordered[x] for x in range(ntop_genes_visual * k, ntop_genes_visual * (k+1))]
            var_names_ordered_dict[categories_order_mouse[k]] = var_names_ordered_k
        # categories_order = dp.categories_order
        dp.add_totals(color=[self.mouse_64_color_dict[x] for x in categories_order_mouse])   # .style(dot_edge_color='black', dot_edge_lw=0.5)
        # dp.legend(width=3, colorbar_title='log fold change')
        dp.legend(width=3)

        dp.savefig(save_path + save_pre + '_dotplot.' + self.fig_format, format=self.fig_format,
                   dpi=self.fig_dpi)

        dp = sc.pl.rank_genes_groups_dotplot(adata, key=key_filtered,  #n_genes=ntop_genes_visual,
                                             groupby=groupby, show=False,
                                             dendrogram=False,
                                             var_names=var_names_ordered_dict,
                                             groups = categories_order_mouse,
                                             categories_order=categories_order_mouse,
                                             var_group_labels=categories_order_mouse,
                                             figsize=figsize,
                                             smallest_dot=0,
                                             return_fig=True,
                                             standard_scale='var',
                                             cmap=color_map,
                                             dot_max=0.05,edgecolors='black'
                                             # values_to_plot="logfoldchanges"
                                             # colorbar_title='log fold change'
                                             )  # , values_to_plot="logfoldchanges" , dot_max=0.1

        # categories_order = dp.categories_order
        mouse_marker_gene_ntop = var_names_ordered
        dp.add_totals(color=[self.mouse_64_color_dict[x] for x in categories_order_mouse]) # .style(dot_edge_color='black', dot_edge_lw=0.5)
        # dp.legend(width=3, colorbar_title='log fold change')
        dp.legend(width=3)
        #ax = dp.mainplot_ax
        # Loop through ticklabels and make them italic
        #ax =
        # Loop through ticklabels and make them italic
        for l in dp.get_axes()['mainplot_ax'].get_xticklabels():
            l.set_style('italic')

        dp.savefig(save_path + save_pre + '_dotplot.' + self.fig_format, format=self.fig_format,
                   dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_stacked_violin(adata,  key=key_filtered, var_names=var_names_ordered_dict, #n_genes=ntop_genes_visual,
                                             groups = categories_order_mouse,
                                               groupby=groupby, show=False, dendrogram=True, figsize=figsize)
        plt.savefig(save_path + save_pre + '_violin.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)

        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_matrixplot(adata,  key=key_filtered, var_names=var_names_ordered_dict, #n_genes=ntop_genes_visual,
                                             groups = categories_order_mouse,
                                           groupby=groupby, show=False, dendrogram=True, figsize=figsize)
        plt.savefig(save_path + save_pre + '_matrixplot.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)



        path_datapiar = self.cfg.CAME.ROOT + 'datapair_init.pickle'
        path_datapiar_file = open(path_datapiar, 'rb')
        datapair = pickle.load(path_datapiar_file)
        mh_mat = datapair['vv_adj'].toarray()[0:self.cfg.BrainAlign.binary_M, self.cfg.BrainAlign.binary_M:] > 0

        mouse_gene2id = {g:id for g, id in zip(mouse_gene_list, range(len(mouse_gene_list)))}
        human_gene2id = {g:id for g, id in zip(human_gene_list, range(len(human_gene_list)))}

        #var_names_ordered_dict
        #human_marker_gene_ntop =
        #var_names_ordered = reorder(X_1, human_marker_gene_ntop, Y_1, ntop=ntop_genes_visual

        #mouse_marker_gene_ntop = var_names_ordered#human_marker_gene_ntop
        for i, h_gene in zip(range(len(human_marker_gene_ntop)), human_marker_gene_ntop):
            h_g_id = human_gene2id[h_gene]
            for m_g_id in range(len(mouse_gene2id)):
                if mh_mat[m_g_id, h_g_id] != 0:
                    mouse_marker_gene_ntop[i] = mouse_gene_list[m_g_id]

        X_1 = human_homo_region_list
        X_ordered = categories_order_human
        Y_1 = mouse_marker_gene_ntop
        print('Y_1', Y_1)
        categories_order_mouse = reorder(X_1, X_ordered, mouse_homo_region_list)
        var_names_ordered = reorder(X_1, X_ordered, Y_1, ntop=ntop_genes_visual)
        #var_names_ordered = mouse_marker_gene_ntop

        #var_names_ordered = reorder(X_1, X_ordered, Y_1, ntop=ntop_genes_visual)

        print('var_names_ordered', var_names_ordered)

        var_names_ordered_dict = {}
        for k in range(len(categories_order_mouse)):
            var_names_ordered_k = [var_names_ordered[x] for x in
                                   range(ntop_genes_visual * k, ntop_genes_visual * (k + 1))]
            var_names_ordered_dict[categories_order_mouse[k]] = var_names_ordered_k

        dp = sc.pl.rank_genes_groups_dotplot(adata, key=key_filtered,  # n_genes=ntop_genes_visual,
                                             groupby=groupby, show=False,
                                             dendrogram=False,
                                             var_names=var_names_ordered_dict,
                                             groups=categories_order_mouse,
                                             categories_order=categories_order_mouse,
                                             var_group_labels=categories_order_mouse,
                                             figsize=figsize,
                                             smallest_dot=0,
                                             return_fig=True,
                                             standard_scale='var',
                                             cmap=color_map,
                                             dot_max=0.05, edgecolors='black'
                                             # values_to_plot="logfoldchanges"
                                             # colorbar_title='log fold change'
                                             )  # , values_to_plot="logfoldchanges" , dot_max=0.1

        # categories_order = dp.categories_order
        #human_marker_gene_ntop = dp.var_names
        dp.add_totals(color=[self.mouse_64_color_dict[x] for x in
                             categories_order_mouse])  # .style(dot_edge_color='black', dot_edge_lw=0.5)
        # dp.legend(width=3, colorbar_title='log fold change')
        dp.legend(width=3)
        # ax = dp.mainplot_ax
        # Loop through ticklabels and make them italic
        # ax =
        # Loop through ticklabels and make them italic
        for l in dp.get_axes()['mainplot_ax'].get_xticklabels():
            l.set_style('italic')

        dp.savefig(save_path + save_pre + '_dotplot_homologous_mouse.' + self.fig_format, format=self.fig_format,
                   dpi=self.fig_dpi)



        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.left"] = 0.1


    def experiment_2_2_regions_deg(self):

        sns.set(style='white')
        TINY_SIZE = 18  # 39
        SMALL_SIZE = 18  # 42
        MEDIUM_SIZE = 18  # 46
        BIGGER_SIZE = 18  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/2_2_experiment_regions_deg/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        figsize = (30, 30)

        ######################################################################################
        groupby = 'region_name'
        adata = self.adata_human_exp#[self.adata_human_exp.obs['region_name'].isin(human_homo_region_list)]
        #print(adata)
        #sc.pp.filter_cells(adata, min_counts=2)
        ntop_genes = 100
        ntop_genes_visual = 1
        save_pre = 'human'
        key = 'wilcoxon'  # 't-test_overestim_var'#'wilcoxon'#'t-test'
        key_filtered = 'wilcoxon'  # 't-test_overestim_var'# 'wilcoxon-filtered'#'t-test-filtered'

        rcParams["figure.subplot.top"] = 0.8
        rcParams["figure.subplot.bottom"] = 0.15
        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.right"] = 0.9

        cluster_counts = adata.obs[groupby].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        adata = adata[adata.obs[groupby].isin(keep)].copy()

        #adata = adata

        c = Colormap()
        color_map = c.cmap_linear('white', 'white', 'red')

        # sc.tl.rank_genes_groups(adata, groupby, method='t-test', key_added="t-test")
        sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added="wilcoxon") #, groups=human_homo_region_list
        # sc.tl.rank_genes_groups(adata, groupby, method='t-test_overestim_var', key_added="t-test_overestim_var")
        # sc.tl.rank_genes_groups(adata, groupby, method='logreg', key_added="logreg")

        sc.pl.rank_genes_groups(adata, n_genes=ntop_genes, sharey=False, key=key, show=False) #, groups=human_homo_region_list

        plt.savefig(save_path + save_pre + 'degs.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)

        # rcParams["figure.subplot.top"] = 0.8
        # rcParams["figure.subplot.bottom"] = 0.2
        # rcParams["figure.subplot.left"] = 0.2
        # with plt.rc_context({"figure.figsize": (12, 16)}):

        # sc.pl.rank_genes_groups_heatmap(adata, n_genes=ntop_genes_visual, key=key_filtered, groups=human_homo_region_list,
        #                                 groupby=groupby, show_gene_labels=True, show=False, dendrogram=True,
        #                                 figsize=figsize)
        # plt.savefig(save_path + save_pre + '_heatmap.' + self.fig_format, format=self.fig_format,
        #             dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (12, 12)}):


        dp = sc.pl.rank_genes_groups_dotplot(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                             # dot_min=0.3, dot_max=1,
                                             #groups=human_homo_region_list,
                                             groupby=groupby,
                                             show=False,
                                             dendrogram=True,
                                             figsize=figsize,
                                             # categories_order=categories_order,
                                             smallest_dot=0,
                                             standard_scale='var',
                                             cmap=color_map, return_fig=True
                                             # values_to_plot="logfoldchanges",
                                             # colorbar_title='log fold change'
                                             )  # , values_to_plot="logfoldchanges"
        categories_order = dp.categories_order
        dp.add_totals(color=[self.human_88_color_dict[x] for x in categories_order])  # .style(dot_edge_color='black', dot_edge_lw=0.5)
        dp.legend(width=3)


        # var_group_labels = dp.var_group_labels
        # dp.legend(width=3, colorbar_title='log fold change')

        dp.savefig(save_path + save_pre + '_dotplot.' + self.fig_format, format=self.fig_format,
                   dpi=self.fig_dpi)

        ##########################
        # mouse degs

        groupby = 'region_name'
        adata = self.adata_mouse_exp#[self.adata_mouse_exp.obs['region_name'].isin(mouse_homo_region_list)]
        sc.pp.filter_cells(adata, min_counts=2)
        ntop_genes = 100
        ntop_genes_visual = 1
        save_pre = 'mouse'
        key = 'wilcoxon'  # 't-test_overestim_var'#'wilcoxon'#'t-test'
        key_filtered = 'wilcoxon'  # 't-test_overestim_var'# 'wilcoxon-filtered'#'t-test-filtered'

        # cluster_counts = adata.obs[groupby].value_counts()
        # keep = cluster_counts.index[cluster_counts >= 2]
        # adata = adata[adata.obs[groupby].isin(keep)].copy()

        # sc.tl.rank_genes_groups(adata, groupby, method='t-test', key_added="t-test")
        sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added="wilcoxon") #, groups=mouse_homo_region_list
        # sc.tl.rank_genes_groups(adata, groupby, method='t-test_overestim_var', key_added="t-test_overestim_var")
        # sc.tl.rank_genes_groups(adata, groupby, method='logreg', key_added="logreg")
        sc.pl.rank_genes_groups(adata, n_genes=ntop_genes, sharey=False, key=key, show=False) #, groups=mouse_homo_region_list
        # sc.tl.filter_rank_genes_groups(adata, key=key, key_added=key_filtered,
        #                                #min_in_group_fraction=0.2,
        #                                #max_out_group_fraction=0.5,
        #                                min_fold_change=3)
        # else:
        #     # pass
        #     sc.tl.filter_rank_genes_groups(adata, key="wilcoxon", key_added="wilcoxon_filtered",
        #                                    min_fold_change=2)
        plt.savefig(save_path + save_pre + 'degs.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)

        # rcParams["figure.subplot.top"] = 0.8
        # rcParams["figure.subplot.bottom"] = 0.15
        # rcParams["figure.subplot.left"] = 0.15
        # rcParams["figure.subplot.right"] = 0.9
        # with plt.rc_context({"figure.figsize": (12, 16)}):

        # with plt.rc_context({"figure.figsize": (12, 12)}):
        dp = sc.pl.rank_genes_groups_dotplot(adata, key=key_filtered, n_genes=ntop_genes_visual,
                                             groupby=groupby, show=False,
                                             dendrogram=True,
                                             # groups = cluster_name_unique,
                                             # categories_order=categories_order,
                                             # var_group_labels = var_group_labels,
                                             figsize=figsize,
                                             smallest_dot=0,
                                             return_fig=True,
                                             standard_scale='var',
                                             cmap=color_map,
                                             dot_max=1
                                             # values_to_plot="logfoldchanges"
                                             # colorbar_title='log fold change'
                                             )  # , values_to_plot="logfoldchanges" , dot_max=0.1
        categories_order_mouse = dp.categories_order
        dp.add_totals(color=[self.mouse_64_color_dict[x] for x in categories_order_mouse])   # .style(dot_edge_color='black', dot_edge_lw=0.5)
        # dp.legend(width=3, colorbar_title='log fold change')
        dp.legend(width=3)

        dp.savefig(save_path + save_pre + '_dotplot.' + self.fig_format, format=self.fig_format,
                   dpi=self.fig_dpi)


        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.left"] = 0.1


    def experiment_3_regions_gene_module(self):
        """
        Analysis of homologous region pairs and clusters expression on gene modules

        - 1. Identify top 50 degs for region/cluster pairs;
        - 2. Check degs' gene modules, assign gene modules for those degs (or clustering those genes);
        - 3. For the same gene module expressing in two species, do gene enrichment analysis and check their functional invariance.

        :Figures:
        - UMAP of homologous regions or clusters;
        - volcano figure of degs;
        - Degs dot plot;
        - gene module distribution scatter;
        - GO analysis figure.

        :return: None
        """

        sns.set(style='white')
        TINY_SIZE = 15  # 39
        SMALL_SIZE = 15  # 42
        MEDIUM_SIZE = 18  # 46
        BIGGER_SIZE = 18  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        #plt.rc('title', titlesize=BIGGER_SIZE)
        #rcParams["legend.title_fontsize"]

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']


        # save_path
        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/3_experiment_gene_module/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        #-----------homologous regions----------------------------------------
        # module information
        module_name = 'module'
        module_num = len(Counter(self.adata_gene_embedding.obs[module_name]))
        gene_names = self.adata_gene_embedding.obs_names.values
        module_labels = self.adata_gene_embedding.obs[module_name].values
        print('module_labels.shape', module_labels.shape)
        gene_module_dict = {k: v for k, v in zip(gene_names, module_labels)}

        mouse_gene_names = self.adata_mouse_gene_embedding.obs_names.values
        mouse_module_labels = self.adata_mouse_gene_embedding.obs[module_name].values
        mouse_gene_module_dict = {k: v for k, v in zip(mouse_gene_names, mouse_module_labels)}
        # human gene_name to module
        human_gene_names = self.adata_human_gene_embedding.obs_names.values
        human_module_labels = self.adata_human_gene_embedding.obs[module_name].values
        human_gene_module_dict = {k: v for k, v in zip(human_gene_names, human_module_labels)}

        # homologous regions
        human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        homo_region_dict = OrderedDict()
        for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
            homo_region_dict[x] = y
        mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        mouse_homo_region_list = human_mouse_homo_region['Mouse'].values
        human_homo_set = set(human_mouse_homo_region['Human'].values)
        human_homo_region_list = human_mouse_homo_region['Human'].values



        #----------------------gene ontology analysis-------------------------------
        ntop_gene = 100
        p_cut = 1e-3
        go_p_cut = 1e-2
        top_term = 2
        color_list = list(sns.color_palette(cc.glasbey, n_colors=8))

        mouse_go_gene_sets = ['Allen_Brain_Atlas_down',
                                                'Allen_Brain_Atlas_up',
                                                'Azimuth_Cell_Types_2021',
                                                'CellMarker_Augmented_2021',
                                                'GO_Biological_Process_2021',
                                                'GO_Cellular_Component_2021',
                                                'GO_Molecular_Function_2021',
                                                'Mouse_Gene_Atlas']
        human_go_gene_sets = ['Allen_Brain_Atlas_down', 'Allen_Brain_Atlas_up',
                                                'Azimuth_Cell_Types_2021', 'CellMarker_Augmented_2021',
                                                'GO_Biological_Process_2021', 'GO_Cellular_Component_2021',
                                                'GO_Molecular_Function_2021', 'Human_Gene_Atlas']

        ####################################################


        # mouse region
        save_path_homo = save_path + 'homo_regions_mouse/'
        if not os.path.exists(save_path_homo):
            os.makedirs(save_path_homo)

        sc.tl.rank_genes_groups(self.adata_mouse_exp, groupby='region_name', method='wilcoxon',
                                key_added="wilcoxon")

        #mouse_homo_region_list = ['Field CA3', 'Field CA2', 'Field CA1']

        for region_name in mouse_homo_region_list:
            print('mouse:', region_name)
            sc.pl.rank_genes_groups(self.adata_mouse_exp, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                    show=False)
            glist_mouse_region_genes = sc.get.rank_genes_groups_df(self.adata_mouse_exp,
                                                              group=region_name,
                                                              key='wilcoxon',
                                                              #log2fc_min=0.25,
                                                              pval_cutoff=p_cut)['names'].squeeze().str.strip().tolist()
            module_list_region = [mouse_gene_module_dict[g] for g in glist_mouse_region_genes]

            module_set_unique = list(set(module_list_region))
            module_set_unique.sort()

            save_path_region = save_path_homo + '_'.join(region_name.split(' ')) + '/'
            if not os.path.exists(save_path_region):
                os.makedirs(save_path_region)

            module_genelist_dict = {}
            for g_module_id in module_set_unique:
                module_genelist_dict[g_module_id] = []
                for m, g in zip(module_list_region, glist_mouse_region_genes):
                    if m == g_module_id:
                        module_genelist_dict[g_module_id].append(g)

                go_gene_list = module_genelist_dict[g_module_id]

                enr_res = gseapy.enrichr(gene_list=go_gene_list,
                                         organism='Mouse',
                                         gene_sets=mouse_go_gene_sets,
                                         top_term=top_term) #cutoff=go_p_cut
                #fig, axes = plt.subplot()
                rcParams["figure.subplot.left"] = 0.52
                rcParams["figure.subplot.right"] = 0.70
                rcParams["figure.subplot.top"] = 0.9
                rcParams["figure.subplot.bottom"] = 0.1

                rcParams["legend.title_fontsize"] = MEDIUM_SIZE
                rcParams["axes.titlesize"] = MEDIUM_SIZE
                fig, ax = gseapy.barplot(enr_res.results, column="Adjusted P-value", group='Gene_set', color=color_list,#color=self.mouse_color,
                               # set group, so you could do a multi-sample/library comparsion
                               size=5, title=f"GO of Mouse {region_name} module {g_module_id} DEGs", figsize=(15, 8), top_term=top_term) # cutoff=go_p_cut
                ax.get_legend().set_title("Gene sets")
                #ax.get_legend().set_fontsize(MEDIUM_SIZE)
                #ax.get_title().set_fontsize(MEDIUM_SIZE)
                #ax.get_legend().set_loc('lower center')
                #fig.axes.append(ax)
                fig.savefig(save_path_region + f'module_{g_module_id}_barplot.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)
                # rcParams["figure.subplot.left"] = 0.1
                # rcParams["figure.subplot.right"] = 0.9
                rcParams["figure.subplot.left"] = 0.6
                rcParams["figure.subplot.right"] = 0.88
                rcParams["figure.subplot.bottom"] = 0.25#0.15
                rcParams["figure.subplot.top"] = 0.95

                fig, ax = gseapy.dotplot(enr_res.results,
                               column="Adjusted P-value",
                               x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion
                               size=10,
                               # top_term=5,  #
                               figsize=(15, 10),
                               title=f"GO of Mouse {region_name} module {g_module_id} DEGs",
                               xticklabels_rot=45,
                               # rotate xtick labels show_ring=True, set to False to revmove outer ring
                               marker='o',
                               cmap=plt.cm.winter_r,
                               format=self.fig_format, top_term=top_term) #cutoff=go_p_cut
                #cbar = fig.colorbar(ax=ax)
                # im = ax.images
                # print(len(im))
                # print(im)
                # # Assume colorbar was plotted last one plotted last
                # cbar = im.colorbar
                # cbar.set_label(size=MEDIUM_SIZE, labelpad=15)
                ax.set_xlabel('')
                fig.savefig(save_path_region + f'module_{g_module_id}_botplot.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)

        # human region ----------------------------------------------------------------------------
        save_path_homo = save_path + 'homo_regions_human/'
        if not os.path.exists(save_path_homo):
            os.makedirs(save_path_homo)

        # filtering out region with less than 2 samples
        adata_human_exp = self.adata_human_exp.copy()
        cluster_counts = adata_human_exp.obs['region_name'].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        adata_human_exp = adata_human_exp[adata_human_exp.obs['region_name'].isin(keep)].copy()

        sc.tl.rank_genes_groups(adata_human_exp, groupby='region_name', method='wilcoxon',
                                key_added="wilcoxon")

        for region_name in human_homo_region_list:
            print('human:', region_name)
            sc.pl.rank_genes_groups(adata_human_exp, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                    show=False)
            glist_human_region_genes = sc.get.rank_genes_groups_df(adata_human_exp,
                                                                   group=region_name,
                                                                   key='wilcoxon',
                                                                   # log2fc_min=0.25,
                                                                   pval_cutoff=p_cut)[
                'names'].squeeze().str.strip().tolist()
            module_list_region = [human_gene_module_dict[g] for g in glist_human_region_genes]

            module_set_unique = list(set(module_list_region))
            module_set_unique.sort()

            save_path_region = save_path_homo + '_'.join(region_name.split(' ')) + '/'
            if not os.path.exists(save_path_region):
                os.makedirs(save_path_region)

            module_genelist_dict = {}
            for g_module_id in module_set_unique:
                module_genelist_dict[g_module_id] = []
                for m, g in zip(module_list_region, glist_human_region_genes):
                    if m == g_module_id:
                        module_genelist_dict[g_module_id].append(g)

                go_gene_list = module_genelist_dict[g_module_id]

                enr_res = gseapy.enrichr(gene_list=go_gene_list,
                                         organism='Human',
                                         gene_sets=human_go_gene_sets, top_term=top_term
                                         )  # cutoff=go_p_cut
                # fig, axes = plt.subplot()
                rcParams["figure.subplot.left"] = 0.52
                rcParams["figure.subplot.right"] = 0.70
                rcParams["figure.subplot.top"] = 0.9
                rcParams["figure.subplot.bottom"] = 0.1
                # rcParams["figure.subplot.right"] = 0.9
                rcParams["legend.title_fontsize"] = MEDIUM_SIZE
                rcParams["axes.titlesize"] = MEDIUM_SIZE
                fig, ax = gseapy.barplot(enr_res.results, column="Adjusted P-value", group='Gene_set',
                                         color=color_list,
                                         # color=self.mouse_color,
                                         # set group, so you could do a multi-sample/library comparsion
                                         size=5, title=f"GO of Human {region_name} module {g_module_id} DEGs",
                                         figsize=(15, 8), top_term=top_term)  # cutoff=go_p_cut
                ax.get_legend().set_title("Gene sets")
                # ax.get_legend().set_fontsize(MEDIUM_SIZE)
                # ax.get_title().set_fontsize(MEDIUM_SIZE)
                # ax.get_legend().set_loc('lower center')
                # fig.axes.append(ax)
                fig.savefig(save_path_region + f'module_{g_module_id}_barplot.' + self.fig_format,
                            format=self.fig_format, dpi=self.fig_dpi)

                rcParams["figure.subplot.left"] = 0.6
                rcParams["figure.subplot.right"] = 0.88
                rcParams["figure.subplot.bottom"] = 0.25  # 0.15
                rcParams["figure.subplot.top"] = 0.95

                fig, ax = gseapy.dotplot(enr_res.results,
                                         column="Adjusted P-value",
                                         x='Gene_set',
                                         # set x axis, so you could do a multi-sample/library comparsion
                                         size=10,
                                         # top_term=5,  #
                                         figsize=(15, 10),
                                         title=f"GO of Human {region_name} module {g_module_id} DEGs",
                                         xticklabels_rot=45,
                                         # rotate xtick labels show_ring=True, set to False to revmove outer ring
                                         marker='o',
                                         cmap=plt.cm.winter_r,
                                         # ofname=save_path_region + f'module_{g_module_id}_botplot.' + self.fig_format,
                                         format=self.fig_format, top_term=top_term)  # cutoff=go_p_cut
                # cbar.set_label(size=MEDIUM_SIZE, labelpad=15)
                ax.set_xlabel('')
                fig.savefig(save_path_region + f'module_{g_module_id}_botplot.' + self.fig_format,
                            format=self.fig_format, dpi=self.fig_dpi)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1
        return None


    def experiment_3_1_regions_gene_module_analysis(self):
        """
        Analysis of homologous region pairs and clusters expression on gene modules

        - 1. Identify top 50 degs for region/cluster pairs;
        - 2. Check degs' gene modules, assign gene modules for those degs (or clustering those genes);
        - 3. For the same gene module expressing in two species, do gene enrichment analysis and check their functional invariance.

        :Figures:
        - UMAP of homologous regions or clusters;
        - volcano figure of degs;
        - Degs dot plot;
        - gene module distribution scatter;
        - GO analysis figure.

        :return: None
        """

        sns.set(style='white')
        TINY_SIZE = 15  # 39
        SMALL_SIZE = 15  # 42
        MEDIUM_SIZE = 18  # 46
        BIGGER_SIZE = 18  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        #plt.rc('title', titlesize=BIGGER_SIZE)
        #rcParams["legend.title_fontsize"]

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']


        # save_path
        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/3_1_experiment_gene_module_analysis/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        #-----------homologous regions----------------------------------------
        # module information
        module_name = 'module'
        module_num = len(Counter(self.adata_gene_embedding.obs[module_name]))
        gene_names = self.adata_gene_embedding.obs_names.values
        module_labels = self.adata_gene_embedding.obs[module_name].values
        print('module_labels.shape', module_labels.shape)
        gene_module_dict = {k: v for k, v in zip(gene_names, module_labels)}

        mouse_gene_names = self.adata_mouse_gene_embedding.obs_names.values
        mouse_module_labels = self.adata_mouse_gene_embedding.obs[module_name].values
        mouse_gene_module_dict = {k: v for k, v in zip(mouse_gene_names, mouse_module_labels)}
        # human gene_name to module
        human_gene_names = self.adata_human_gene_embedding.obs_names.values
        human_module_labels = self.adata_human_gene_embedding.obs[module_name].values
        human_gene_module_dict = {k: v for k, v in zip(human_gene_names, human_module_labels)}

        # homologous regions
        human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        homo_region_dict = OrderedDict()
        for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
            homo_region_dict[x] = y
        mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        mouse_homo_region_list = human_mouse_homo_region['Mouse'].values
        human_homo_set = set(human_mouse_homo_region['Human'].values)
        human_homo_region_list = human_mouse_homo_region['Human'].values



        #----------------------gene ontology analysis-------------------------------
        ntop_gene = 100
        p_cut = 1e-3
        go_p_cut = 1e-2
        top_term = 2
        color_list = list(sns.color_palette(cc.glasbey, n_colors=8))

        mouse_go_gene_sets = ['Allen_Brain_Atlas_down',
                                                'Allen_Brain_Atlas_up',
                                                'Azimuth_Cell_Types_2021',
                                                'CellMarker_Augmented_2021',
                                                'GO_Biological_Process_2021',
                                                'GO_Cellular_Component_2021',
                                                'GO_Molecular_Function_2021',
                                                'Mouse_Gene_Atlas']
        human_go_gene_sets = ['Allen_Brain_Atlas_down', 'Allen_Brain_Atlas_up',
                                                'Azimuth_Cell_Types_2021', 'CellMarker_Augmented_2021',
                                                'GO_Biological_Process_2021', 'GO_Cellular_Component_2021',
                                                'GO_Molecular_Function_2021', 'Human_Gene_Atlas']

        ####################################################


        # mouse region
        save_path_homo = save_path + 'homo_regions_mouse/'
        if not os.path.exists(save_path_homo):
            os.makedirs(save_path_homo)

        sc.tl.rank_genes_groups(self.adata_mouse_exp, groupby='region_name', method='wilcoxon',
                                key_added="wilcoxon")

        mouse_homo_region_list = ['Field CA2']

        for region_name in mouse_homo_region_list:
            sc.pl.rank_genes_groups(self.adata_mouse_exp, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                    show=False)
            glist_mouse_region_genes = sc.get.rank_genes_groups_df(self.adata_mouse_exp,
                                                              group=region_name,
                                                              key='wilcoxon',
                                                              #log2fc_min=0.25,
                                                              pval_cutoff=p_cut)['names'].squeeze().str.strip().tolist()
            enr_res_mouse = gseapy.enrichr(gene_list=glist_mouse_region_genes,
                                     organism='Mouse',
                                     gene_sets=mouse_go_gene_sets,
                                     top_term=top_term)  # cutoff=go_p_cut


        # human region ----------------------------------------------------------------------------
        save_path_homo = save_path + 'homo_regions_human/'
        if not os.path.exists(save_path_homo):
            os.makedirs(save_path_homo)

        # filtering out region with less than 2 samples
        adata_human_exp = self.adata_human_exp.copy()
        cluster_counts = adata_human_exp.obs['region_name'].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        adata_human_exp = adata_human_exp[adata_human_exp.obs['region_name'].isin(keep)].copy()

        sc.tl.rank_genes_groups(adata_human_exp, groupby='region_name', method='wilcoxon',
                                key_added="wilcoxon")

        human_homo_region_list = ['CA2 field']

        for region_name in human_homo_region_list:
            sc.pl.rank_genes_groups(adata_human_exp, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                    show=False)
            glist_human_region_genes = sc.get.rank_genes_groups_df(adata_human_exp,
                                                                   group=region_name,
                                                                   key='wilcoxon',
                                                                   # log2fc_min=0.25,
                                                                   pval_cutoff=p_cut)['names'].squeeze().str.strip().tolist()

            enr_res_human = gseapy.enrichr(gene_list=glist_human_region_genes,
                                         organism='Human',
                                         gene_sets=human_go_gene_sets, top_term=top_term
                                         )  # cutoff=go_p_cut

        enr_res_mouse_results = enr_res_mouse.results[enr_res_mouse.results['Adjusted P-value'] <= go_p_cut]#.columns
        print(enr_res_mouse_results)
        enr_res_human_results = enr_res_human.results[enr_res_human.results['Adjusted P-value'] <= go_p_cut]#.columns
        print(enr_res_human_results)

        print('enr_res_mouse go term numbers = ', enr_res_mouse_results.shape[0])
        print('enr_res_human go term numbers = ', enr_res_human_results.shape[0])

        Overlapped_term_num = 0

        enr_res_mouse_results_term = enr_res_mouse_results['Term'].tolist()
        enr_res_human_results_term = enr_res_human_results['Term'].tolist()

        Overlapped_term_set = list()
        for i in range(enr_res_mouse_results.shape[0]):
            for j in range(enr_res_human_results.shape[0]):
                if enr_res_mouse_results_term[i] == enr_res_human_results_term[j]:
                    print(enr_res_mouse_results_term[i], enr_res_human_results_term[j])
                    Overlapped_term_num += 1
        print(Overlapped_term_num)



        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1
        return None


    def experiment_4_clusters_gene_module(self):

        """
        Analysis of homologous region pairs and clusters expression on gene modules

        - 1. Identify top 50 degs for region/cluster pairs;
        - 2. Check degs' gene modules, assign gene modules for those degs (or clustering those genes);
        - 3. For the same gene module expressing in two species, do gene enrichment analysis and check their functional invariance.

        :Figures:
        - UMAP of homologous regions or clusters;
        - volcano figure of degs;
        - Degs dot plot;
        - gene module distribution scatter;
        - GO analysis figure.

        :return: None
        """

        sns.set(style='white')
        TINY_SIZE = 15  # 39
        SMALL_SIZE = 15  # 42
        MEDIUM_SIZE = 18  # 46
        BIGGER_SIZE = 18  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        # plt.rc('title', titlesize=BIGGER_SIZE)
        # rcParams["legend.title_fontsize"]

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        # save_path
        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/4_experiment_clusters_gene_module/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # -----------homologous regions----------------------------------------
        # module information
        module_name = 'module'
        module_num = len(Counter(self.adata_gene_embedding.obs[module_name]))
        gene_names = self.adata_gene_embedding.obs_names.values
        module_labels = self.adata_gene_embedding.obs[module_name].values
        print('module_labels.shape', module_labels.shape)
        gene_module_dict = {k: v for k, v in zip(gene_names, module_labels)}

        mouse_gene_names = self.adata_mouse_gene_embedding.obs_names.values
        mouse_module_labels = self.adata_mouse_gene_embedding.obs[module_name].values
        mouse_gene_module_dict = {k: v for k, v in zip(mouse_gene_names, mouse_module_labels)}
        # human gene_name to module
        human_gene_names = self.adata_human_gene_embedding.obs_names.values
        human_module_labels = self.adata_human_gene_embedding.obs[module_name].values
        human_gene_module_dict = {k: v for k, v in zip(human_gene_names, human_module_labels)}

        # # homologous regions
        # human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        # homo_region_dict = OrderedDict()
        # for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
        #     homo_region_dict[x] = y
        # mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        # mouse_homo_region_list = human_mouse_homo_region['Mouse'].values
        # human_homo_set = set(human_mouse_homo_region['Human'].values)
        # human_homo_region_list = human_mouse_homo_region['Human'].values

        # clusters name
        cluster_name_unique = sorted(Counter(self.adata_embedding.obs['cluster_name_acronym']).keys(),
                                     key=lambda t: int(t.split('-')[0]))


        # ----------------------gene ontology analysis-------------------------------
        ntop_gene = 100
        p_cut = 1e-3
        go_p_cut = 1e-3
        top_term = 2
        color_list = list(sns.color_palette(cc.glasbey, n_colors=8))

        mouse_go_gene_sets = ['Allen_Brain_Atlas_down',
                              'Allen_Brain_Atlas_up',
                              'Azimuth_Cell_Types_2021',
                              'CellMarker_Augmented_2021',
                              'GO_Biological_Process_2021',
                              'GO_Cellular_Component_2021',
                              'GO_Molecular_Function_2021',
                              'Mouse_Gene_Atlas']
        human_go_gene_sets = ['Allen_Brain_Atlas_down', 'Allen_Brain_Atlas_up',
                              'Azimuth_Cell_Types_2021', 'CellMarker_Augmented_2021',
                              'GO_Biological_Process_2021', 'GO_Cellular_Component_2021',
                              'GO_Molecular_Function_2021', 'Human_Gene_Atlas']

        ####################################################
        # human region ----------------------------------------------------------------------------
        save_path_homo = save_path + 'clusters_human/'
        if not os.path.exists(save_path_homo):
            os.makedirs(save_path_homo)

        sc.tl.rank_genes_groups(self.adata_human_exp, groupby='cluster_name_acronym', method='wilcoxon',
                                key_added="wilcoxon")

        for c_name in cluster_name_unique:
            sc.pl.rank_genes_groups(self.adata_human_exp, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                    show=False)
            glist_human_region_genes = sc.get.rank_genes_groups_df(self.adata_human_exp,
                                                                   group=c_name,
                                                                   key='wilcoxon',
                                                                   # log2fc_min=0.25,
                                                                   pval_cutoff=p_cut)['names'].squeeze().str.strip().tolist()
            module_list_region = [human_gene_module_dict[g] for g in glist_human_region_genes]

            module_set_unique = list(set(module_list_region))
            module_set_unique.sort()

            save_path_region = save_path_homo + '_'.join(c_name.split(' ')) + '/'
            if not os.path.exists(save_path_region):
                os.makedirs(save_path_region)

            module_genelist_dict = {}
            for g_module_id in module_set_unique:
                module_genelist_dict[g_module_id] = []
                for m, g in zip(module_list_region, glist_human_region_genes):
                    if m == g_module_id:
                        module_genelist_dict[g_module_id].append(g)

                go_gene_list = module_genelist_dict[g_module_id]

                enr_res = gseapy.enrichr(gene_list=go_gene_list,
                                         organism='Human',
                                         gene_sets=human_go_gene_sets, top_term=top_term
                                         ) #cutoff=go_p_cut
                # fig, axes = plt.subplot()
                rcParams["figure.subplot.left"] = 0.52
                rcParams["figure.subplot.right"] = 0.70
                rcParams["figure.subplot.top"] = 0.9
                rcParams["figure.subplot.bottom"] = 0.1
                # rcParams["figure.subplot.right"] = 0.9
                rcParams["legend.title_fontsize"] = MEDIUM_SIZE
                rcParams["axes.titlesize"] = MEDIUM_SIZE
                fig, ax = gseapy.barplot(enr_res.results, column="Adjusted P-value", group='Gene_set', color=color_list,
                                         # color=self.mouse_color,
                                         # set group, so you could do a multi-sample/library comparsion
                                         size=5, title=f"GO of Human {c_name} module {g_module_id} DEGs",
                                         figsize=(15, 8), top_term=top_term) #cutoff=go_p_cut
                ax.get_legend().set_title("Gene sets")
                # ax.get_legend().set_fontsize(MEDIUM_SIZE)
                # ax.get_title().set_fontsize(MEDIUM_SIZE)
                # ax.get_legend().set_loc('lower center')
                # fig.axes.append(ax)
                fig.savefig(save_path_region + f'module_{g_module_id}_barplot.' + self.fig_format,
                            format=self.fig_format, dpi=self.fig_dpi)

                rcParams["figure.subplot.left"] = 0.6
                rcParams["figure.subplot.right"] = 0.88
                rcParams["figure.subplot.bottom"] = 0.25
                rcParams["figure.subplot.top"] = 0.95

                fig, ax = gseapy.dotplot(enr_res.results,
                                         column="Adjusted P-value",
                                         x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion
                                         size=10,
                                         # top_term=5,  #
                                         figsize=(15, 10),
                                         title=f"GO of Human {c_name} module {g_module_id} DEGs",
                                         xticklabels_rot=45,
                                         # rotate xtick labels show_ring=True, set to False to revmove outer ring
                                         marker='o',
                                         cmap=plt.cm.winter_r,
                                         # ofname=save_path_region + f'module_{g_module_id}_botplot.' + self.fig_format,
                                         format=self.fig_format, top_term=top_term) #cutoff=go_p_cut
                # cbar.set_label(size=MEDIUM_SIZE, labelpad=15)
                ax.set_xlabel('')
                fig.savefig(save_path_region + f'module_{g_module_id}_botplot.' + self.fig_format,
                            format=self.fig_format, dpi=self.fig_dpi)

            # mouse region
        save_path_homo = save_path + 'clusters_mouse/'
        if not os.path.exists(save_path_homo):
            os.makedirs(save_path_homo)

        sc.tl.rank_genes_groups(self.adata_mouse_exp, groupby='cluster_name_acronym', method='wilcoxon',
                                key_added="wilcoxon")

        for c_name in cluster_name_unique:
            sc.pl.rank_genes_groups(self.adata_mouse_exp, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                    show=False)
            glist_mouse_cluster_genes = sc.get.rank_genes_groups_df(self.adata_mouse_exp,
                                                                    group=c_name,
                                                                    key='wilcoxon',
                                                                    # log2fc_min=0.25,
                                                                    pval_cutoff=p_cut)[
                'names'].squeeze().str.strip().tolist()
            module_list_cluster = [mouse_gene_module_dict[g] for g in glist_mouse_cluster_genes]

            module_set_unique = list(set(module_list_cluster))
            module_set_unique.sort()

            save_path_region = save_path_homo + '_'.join(c_name.split(' ')) + '/'
            if not os.path.exists(save_path_region):
                os.makedirs(save_path_region)

            module_genelist_dict = {}
            for g_module_id in module_set_unique:
                module_genelist_dict[g_module_id] = []
                for m, g in zip(module_list_cluster, glist_mouse_cluster_genes):
                    if m == g_module_id:
                        module_genelist_dict[g_module_id].append(g)

                go_gene_list = module_genelist_dict[g_module_id]

                enr_res = gseapy.enrichr(gene_list=go_gene_list,
                                         organism='Mouse',
                                         gene_sets=mouse_go_gene_sets,
                                         top_term=top_term)  # cutoff=go_p_cut,
                # fig, axes = plt.subplot()
                rcParams["figure.subplot.left"] = 0.52
                rcParams["figure.subplot.right"] = 0.70
                rcParams["figure.subplot.top"] = 0.9
                rcParams["figure.subplot.bottom"] = 0.1

                rcParams["legend.title_fontsize"] = MEDIUM_SIZE
                rcParams["axes.titlesize"] = MEDIUM_SIZE
                # print(enr_res.results.columns)
                fig, ax = gseapy.barplot(enr_res.results, column="Adjusted P-value", group='Gene_set',
                                         color=color_list,
                                         # color=self.mouse_color,
                                         # set group, so you could do a multi-sample/library comparsion
                                         size=5, title=f"GO of Mouse {c_name} module {g_module_id} DEGs",
                                         figsize=(15, 8), top_term=top_term)  # cutoff=go_p_cut
                ax.get_legend().set_title("Gene sets")
                # ax.get_legend().set_fontsize(MEDIUM_SIZE)
                # ax.get_title().set_fontsize(MEDIUM_SIZE)
                # ax.get_legend().set_loc('lower center')
                # fig.axes.append(ax)
                fig.savefig(save_path_region + f'module_{g_module_id}_barplot.' + self.fig_format,
                            format=self.fig_format, dpi=self.fig_dpi)
                # rcParams["figure.subplot.left"] = 0.1
                # rcParams["figure.subplot.right"] = 0.9
                rcParams["figure.subplot.left"] = 0.6
                rcParams["figure.subplot.right"] = 0.88
                rcParams["figure.subplot.bottom"] = 0.25
                rcParams["figure.subplot.top"] = 0.95

                fig, ax = gseapy.dotplot(enr_res.results,
                                         column="Adjusted P-value",
                                         x='Gene_set',
                                         # set x axis, so you could do a multi-sample/library comparsion
                                         size=10,
                                         # top_term=5,  #
                                         figsize=(15, 10),
                                         title=f"GO of Mouse {c_name} module {g_module_id} DEGs",
                                         xticklabels_rot=45,
                                         # rotate xtick labels show_ring=True, set to False to revmove outer ring
                                         marker='o',
                                         cmap=plt.cm.winter_r,
                                         format=self.fig_format, top_term=top_term)  # cutoff=go_p_cut
                # cbar = fig.colorbar(ax=ax)
                # im = ax.images
                # print(len(im))
                # print(im)
                # # Assume colorbar was plotted last one plotted last
                # cbar = im.colorbar
                # cbar.set_label(size=MEDIUM_SIZE, labelpad=15)
                ax.set_xlabel('')
                fig.savefig(save_path_region + f'module_{g_module_id}_botplot.' + self.fig_format,
                            format=self.fig_format, dpi=self.fig_dpi)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1

        return None


    def experiment_5_1_gene_module_homogenes_ratio(self):
        '''
        Get bar plot of homologous genes region
        :return:
        '''

        sns.set(style='white')
        TINY_SIZE = 12  # 39
        SMALL_SIZE = 12  # 42
        MEDIUM_SIZE = 12  # 46
        BIGGER_SIZE = 12  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        # save_path
        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/5_1_experiment_gene_module_homogenes_ratio/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # module information
        module_name = 'module'
        module_num = len(Counter(self.adata_gene_embedding.obs[module_name]))
        gene_names = self.adata_gene_embedding.obs_names.values
        module_labels = self.adata_gene_embedding.obs[module_name].values
        print('module_labels.shape', module_labels.shape)
        gene_module_dict = {k: v for k, v in zip(gene_names, module_labels)}

        gene_module_palette = sns.color_palette('tab20b', module_num+1)

        module_num_dict = Counter(self.adata_gene_embedding.obs[module_name])

        module_name_list = sorted(Counter(self.adata_gene_embedding.obs[module_name]).keys(), key=lambda t: int(t))

        cfg = self.cfg
        path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
        path_datapiar_file = open(path_datapiar, 'rb')
        datapair = pickle.load(path_datapiar_file)
        mh_mat = datapair['vv_adj'].toarray()[0:cfg.BrainAlign.binary_M, cfg.BrainAlign.binary_M:] > 0

        mouse_gene_list = self.adata_mouse_gene_embedding.obs_names.to_list()
        human_gene_list = self.adata_human_gene_embedding.obs_names.to_list()

        if os.path.exists(save_path + 'module_homo_ratio_dict.npz'):
            module_homo_ratio_dict = dict(np.load(save_path + 'module_homo_ratio_dict.npz', allow_pickle=True))
            print(module_homo_ratio_dict)
        else:

            ## Init dict to save counts
            module_homo_ratio_dict = Counter(self.adata_gene_embedding.obs[module_name])

            module_homo_ratio_dict = {key: value for key, value in
                                      sorted(module_homo_ratio_dict.items(), key=lambda item: int(item[0]))}

            module_homo_ratio_dict['None'] = 0

            ## Compute gene module homologous gene number
            for gene_module_name in module_name_list:
                mouse_genes = self.adata_mouse_gene_embedding[self.adata_mouse_gene_embedding.obs[module_name]
                    .isin([gene_module_name])].obs_names.to_list()
                human_genes = self.adata_human_gene_embedding[self.adata_human_gene_embedding.obs[module_name]
                        .isin([gene_module_name])].obs_names.to_list()

                mouse_gene_set = []

                human_gene_set = []

                for m_g in mouse_genes:
                    for h_g in human_genes:
                        ori_m_index = mouse_gene_list.index(m_g)
                        ori_h_index = human_gene_list.index(h_g)
                        if mh_mat[ori_m_index, ori_h_index] != 0:
                            mouse_gene_set.append(m_g)
                            human_gene_set.append(h_g)

                mouse_gene_set = list(set(mouse_gene_set))
                human_gene_set = list(set(human_gene_set))

                module_homo_ratio_dict[gene_module_name] = (len(mouse_gene_set) + len(human_gene_set)) / (len(mouse_genes) + len(human_genes))


            module_none_homo_sum = 0
            module_none_homo_homo = 0
            for mouse_gene_module_name in module_name_list:
                for human_gene_module_name in module_name_list:
                    if mouse_gene_module_name != human_gene_module_name:

                        mouse_genes = self.adata_mouse_gene_embedding[self.adata_mouse_gene_embedding.obs[module_name]
                            .isin([mouse_gene_module_name])].obs_names.to_list()
                        human_genes = self.adata_human_gene_embedding[self.adata_human_gene_embedding.obs[module_name]
                            .isin([human_gene_module_name])].obs_names.to_list()

                        mouse_gene_set = []

                        human_gene_set = []

                        for m_g in mouse_genes:
                            for h_g in human_genes:
                                ori_m_index = mouse_gene_list.index(m_g)
                                ori_h_index = human_gene_list.index(h_g)
                                if mh_mat[ori_m_index, ori_h_index] != 0:
                                    mouse_gene_set.append(m_g)
                                    human_gene_set.append(h_g)

                        mouse_gene_set = list(set(mouse_gene_set))
                        human_gene_set = list(set(human_gene_set))

                        module_none_homo_sum += len(mouse_genes) + len(human_genes)
                        module_none_homo_homo += len(mouse_gene_set) + len(human_gene_set)

            module_homo_ratio_dict['None'] = module_none_homo_homo / module_none_homo_sum


            print(module_homo_ratio_dict)

            np.savez(save_path + 'module_homo_ratio_dict.npz', **module_homo_ratio_dict)


        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.2
        rcParams["figure.subplot.bottom"] = 0.2
        rcParams["figure.subplot.top"] = 0.9

        color_pal = {k: v for k, v in zip(module_name_list, gene_module_palette)}
        color_pal['None'] = 'grey'

        Barplot_dict = {'Gene module': list(module_homo_ratio_dict.keys()), 'Homologs ratio': list(module_homo_ratio_dict.values())}
        Barplot_df = pd.DataFrame.from_dict(Barplot_dict)

        plt.figure(figsize=(5, 2.5), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        # sns.set(rc={"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)})
        ax = sns.barplot(x="Gene module", y='Homologs ratio', data=Barplot_df, order=module_name_list + ['None'], palette=color_pal,
                         width=0.75)  #
        plt.title('')
        plt.ylabel('Homologs ratio')
        plt.xlabel('')
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.savefig(save_path + 'gene_module_homologs_ratio.' + self.fig_format, format=self.fig_format)
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.9

        return None


    def experiment_5_deg_homo_distribution(self):
        TINY_SIZE = 10  # 39
        SMALL_SIZE = 10  # 42
        MEDIUM_SIZE = 12  # 46
        BIGGER_SIZE = 12 # 46

        plt.rc('font', size=12)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/5_experiment_deg_homo_distribution/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        path_datapiar = self.cfg.CAME.ROOT + 'datapair_init.pickle'
        path_datapiar_file = open(path_datapiar, 'rb')
        datapair = pickle.load(path_datapiar_file)

        mouse_gene_list = datapair['varnames_node'][0]
        human_gene_list = datapair['varnames_node'][1]

        mouse_gene_set = set(mouse_gene_list)
        human_gene_set = set(human_gene_list)

        mh_mat = datapair['vv_adj'].toarray()[0:self.cfg.BrainAlign.binary_M, self.cfg.BrainAlign.binary_M:]

        # mouse cluster
        ntop_gene = 100
        cluster_name_acronym_list = list(set(self.adata_mouse_exp.obs['cluster_name_acronym'].tolist()))
        cluster_name_acronym_list.sort(key=lambda x:int(x.split('-')[0]))

        parent_cluster_dict = {k: v for k, v in zip(self.adata_human_exp.obs['parent_cluster_name'].tolist(),
                                                    [x.replace(' ', '_') for x in self.adata_human_exp.obs[
                                                        'parent_cluster_name_acronym'].tolist()])}

        #
        sc.tl.rank_genes_groups(self.adata_mouse_exp, groupby='cluster_name_acronym', method='wilcoxon', n_genes=ntop_gene,
                                key_added="wilcoxon")
        sc.tl.rank_genes_groups(self.adata_human_exp, groupby='cluster_name_acronym', method='wilcoxon', n_genes=ntop_gene,
                                key_added="wilcoxon")
        # gene_name_mouse_list = adata_mouse_exp.var_names.tolist()
        # gene_name_human_list = adata_human_exp.var_names.tolist()

        gene_stacked_df = {}
        gene_num_stacked_df = {'Mouse specialized':[], 'Mouse homologous':[], 'Human homologous':[], 'Human specialized':[], 'Cluster':[]}

        for cluster_name_acronym in cluster_name_acronym_list:
            gene_stacked_df[cluster_name_acronym] = {}
            mouse_degs_list = sc.get.rank_genes_groups_df(self.adata_mouse_exp,
                                                          group=cluster_name_acronym,
                                                          key='wilcoxon',
                                                          #log2fc_min=0.25,
                                                          pval_cutoff=None)['names'].squeeze().str.strip().tolist()
            mouse_degs_list = list(mouse_gene_set.intersection(set(mouse_degs_list)))
            human_degs_list = sc.get.rank_genes_groups_df(self.adata_human_exp,
                                                          group=cluster_name_acronym,
                                                          key='wilcoxon',
                                                          #log2fc_min=0.25,
                                                          pval_cutoff=None)['names'].squeeze().str.strip().tolist()
            human_degs_list = list(human_gene_set.intersection(set(human_degs_list)))
            gene_list_mouse_special, gene_list_mouse_common, gene_list_human_common, gene_list_human_special = \
                get_common_special_gene_list_homologous(mouse_degs_list, human_degs_list,mouse_gene_list, human_gene_list,
                                     mh_mat)
            gene_stacked_df[cluster_name_acronym]['Mouse specialized'] = gene_list_mouse_special
            gene_stacked_df[cluster_name_acronym]['Mouse homologous'] = gene_list_mouse_common
            gene_stacked_df[cluster_name_acronym]['Human homologous'] = gene_list_human_common
            gene_stacked_df[cluster_name_acronym]['Human specialized'] = gene_list_human_special

            # update stacked bar data
            gene_num_stacked_df['Mouse specialized'].append(len(gene_list_mouse_special))
            gene_num_stacked_df['Mouse homologous'].append(len(gene_list_mouse_common))
            gene_num_stacked_df['Human homologous'].append(len(gene_list_human_common))
            gene_num_stacked_df['Human specialized'].append(len(gene_list_human_special))
            gene_num_stacked_df['Cluster'].append(cluster_name_acronym)
        #print(adata_mouse_expression.uns)
        #sc.pl.rank_genes_groups(adata_mouse_expression, n_genes=ntop_genes, sharey=False, key="wilcoxon", show=False)
        gene_stacked_df = pd.DataFrame.from_dict(gene_stacked_df)
        gene_stacked_df.to_csv(save_path + 'gene_stacked_special_common.csv')
        gene_num_stacked_df = pd.DataFrame.from_dict(gene_num_stacked_df)
        gene_stacked_df.to_csv(save_path + 'gene_stacked_special_common_number.csv')
        colors = [self.mouse_color, '#8B1C62', '#008B00', self.human_color]
        rcParams["figure.subplot.bottom"] = 0.53
        rcParams["figure.subplot.right"] = 0.98
        rcParams["figure.subplot.left"] = 0.08
        #rcParams["figure.subplot.top"] = 0.7
        plt.rcParams["figure.dpi"] = self.fig_dpi

        from matplotlib.ticker import MaxNLocator

        # Sort the data according the homologous genes number
        gene_num_stacked_df = gene_num_stacked_df.sort_values('Mouse homologous', ascending=False)
        print(gene_num_stacked_df.head(5))

        ax=gene_num_stacked_df.plot(x='Cluster', kind='bar', stacked=True, title=None,#title='DEGs comparison of clusters',
                                 color=colors, figsize=(8.5, 3), rot=45, width=0.9)
        ha_adjust = 'right'
        plt.setp(ax.xaxis.get_majorticklabels(), ha=ha_adjust, rotation_mode="anchor")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=len(cluster_name_acronym_list)/1.5))
        plt.legend(loc='upper center', bbox_to_anchor=(0.45, 1.3), frameon=False, ncol=4)
        plt.ylabel('Gene number')
        plt.xlabel('')
        plt.savefig(save_path + 'gene_stacked_special_common_number.'+self.fig_format, format=self.fig_format, dpi=self.fig_dpi)

        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.1


    def experiment_6_regions_gene_module_average_exp(self):
        """
        Compute everage expression of degs on gene modules, to check which regions/clusters is specialized expressed by which module.
        And then check Gene ontology analysis.

        For each species:
        1. Identify DEGs for each region and cluster, e.g., 50 DEGs;
        2. Map DEGs on gene modules, and compute average gene expressions;
        3. Plot regions x gene module heatmap.

        :return: None
        """
        TINY_SIZE = 18  # 39
        SMALL_SIZE = 18  # 42
        MEDIUM_SIZE = 26  # 46
        BIGGER_SIZE = 26  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/6_regions_gene_module_average_exp/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # -----------homologous regions----------------------------------------
        # module information
        module_name = 'module'
        module_num = len(Counter(self.adata_gene_embedding.obs[module_name]))
        gene_names = self.adata_gene_embedding.obs_names.values
        module_labels = self.adata_gene_embedding.obs[module_name].values

        module_labels_unique = list(set(self.adata_gene_embedding.obs[module_name].values))
        module_labels_unique.sort(key=lambda x:int(x))
        #print('module_labels.shape', module_labels.shape)
        gene_module_dict = {k: v for k, v in zip(gene_names, module_labels)}

        mouse_gene_names = self.adata_mouse_gene_embedding.obs_names.values
        mouse_module_labels = self.adata_mouse_gene_embedding.obs[module_name].values
        mouse_gene_module_dict = {k: v for k, v in zip(mouse_gene_names, mouse_module_labels)}
        # human gene_name to module
        human_gene_names = self.adata_human_gene_embedding.obs_names.values
        human_module_labels = self.adata_human_gene_embedding.obs[module_name].values
        human_gene_module_dict = {k: v for k, v in zip(human_gene_names, human_module_labels)}

        # homologous regions
        human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        homo_region_dict = OrderedDict()
        for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
            homo_region_dict[x] = y
        mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        mouse_homo_region_list = human_mouse_homo_region['Mouse'].values
        mouse_homo_region_acronym_list = [self.mouse_64_acronym_dict[x] for x in mouse_homo_region_list]
        human_homo_set = set(human_mouse_homo_region['Human'].values)
        human_homo_region_list = human_mouse_homo_region['Human'].values
        human_homo_region_acronym_list = [self.human_88_acronym_dict[x] for x in human_homo_region_list]

        # ----------------------gene ontology analysis-------------------------------
        ntop_gene = 50
        p_cut = 1e-3

        # # clusters name
        # cluster_name_unique = sorted(Counter(self.adata_embedding.obs['cluster_name_acronym']).keys(),
        #                              key=lambda t: int(t.split('-')[0]))

        # ----------------------mouse----------------------------------
        sc.tl.rank_genes_groups(self.adata_mouse_exp, groupby='region_name', method='wilcoxon',
                                key_added="wilcoxon")

        mouse_cluster_gene_module_exp_array = np.zeros((len(mouse_homo_region_list), len(module_labels_unique)))

        for i in range(len(mouse_homo_region_list)):
            region_name = mouse_homo_region_list[i]
            sc.pl.rank_genes_groups(self.adata_mouse_exp, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                    show=False)
            glist_mouse_region_genes = sc.get.rank_genes_groups_df(self.adata_mouse_exp,
                                                                   group=region_name,
                                                                   key='wilcoxon',
                                                                   # log2fc_min=0.25,
                                                                   pval_cutoff=p_cut)['names'].squeeze().str.strip().tolist()

            module_list_region = [mouse_gene_module_dict[g] for g in glist_mouse_region_genes]

            module_set_unique = list(set(module_list_region))
            module_set_unique.sort()

            # save_path_region = save_path_homo + '_'.join(region_name.split(' ')) + '/'
            # if not os.path.exists(save_path_region):
            #     os.makedirs(save_path_region)

            module_genelist_dict = {}
            for j in range(len(module_set_unique)):
                g_module_id = module_set_unique[j]
            #for g_module_id in module_set_unique:
                module_genelist_dict[g_module_id] = []
                for m, g in zip(module_list_region, glist_mouse_region_genes):
                    if m == g_module_id:
                        module_genelist_dict[g_module_id].append(g)

                glist = module_genelist_dict[g_module_id]
                if len(module_genelist_dict[g_module_id]) > 0:
                    adata_exp_temp = self.adata_mouse_exp[self.adata_mouse_exp.obs['region_name'].isin([region_name])][:, glist]
                    mouse_cluster_gene_module_exp_array[i, j] = np.mean(adata_exp_temp.X)
                    #print(mouse_cluster_gene_module_exp_array[i, j])

        mouse_cluster_gene_module_exp_array = mouse_cluster_gene_module_exp_array - np.reshape(np.mean(mouse_cluster_gene_module_exp_array, axis=1), (-1, 1))

        mouse_cluster_gene_module_exp_array = mouse_cluster_gene_module_exp_array - np.reshape(np.mean(mouse_cluster_gene_module_exp_array, axis=0), (1, -1))

        mouse_cluster_gene_module_exp_array = preprocess.zscore(mouse_cluster_gene_module_exp_array)

        #mouse_cluster_gene_module_exp_array = preprocess.normalize_row(mouse_cluster_gene_module_exp_array)
        #mouse_cluster_gene_module_exp_array = preprocess.normalize_col(mouse_cluster_gene_module_exp_array)

        mouse_cluster_gene_module_exp_df = pd.DataFrame(mouse_cluster_gene_module_exp_array.T, index=module_labels_unique, columns=mouse_homo_region_list)
        print(mouse_cluster_gene_module_exp_df)
        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.5
        fig, ax = plt.subplots(figsize=(10, 7), dpi=self.fig_dpi)

        c = Colormap()
        color_map = c.cmap_linear('white', 'white', 'darkgreen')

        hm = sns.heatmap(mouse_cluster_gene_module_exp_df,
                         square=False,
                         cbar_kws={'location': 'right'},
                         cmap=color_map,#'Greens', #"YlGnBu",
                         ax=ax, #Spectral_r
                         xticklabels=True, yticklabels=True, linewidths=0.5, linecolor='lightgrey')
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.legend([],[], frameon=False)
        # plt.xticks(rotation=45)
        plt.ylabel('Gene module')
        plt.savefig(save_path + 'mouse_homoregions_gene_module_heatmap.' + self.fig_format, format=self.fig_format)


        # -------------------human-----------------------
        adata_human_exp = self.adata_human_exp.copy()
        cluster_counts = adata_human_exp.obs['region_name'].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        adata_human_exp = adata_human_exp[adata_human_exp.obs['region_name'].isin(keep)].copy()

        sc.tl.rank_genes_groups(adata_human_exp, groupby='region_name', method='wilcoxon',
                                key_added="wilcoxon")

        human_cluster_gene_module_exp_array = np.zeros((len(human_homo_region_list), len(module_labels_unique)))

        for i in range(len(human_homo_region_list)):
            region_name = human_homo_region_list[i]
            sc.pl.rank_genes_groups(adata_human_exp, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                    show=False)
            glist_human_region_genes = sc.get.rank_genes_groups_df(adata_human_exp,
                                                                   group=region_name,
                                                                   key='wilcoxon',
                                                                   # log2fc_min=0.25,
                                                                   pval_cutoff=p_cut)['names'].squeeze().str.strip().tolist()

            module_list_region = [human_gene_module_dict[g] for g in glist_human_region_genes]

            module_set_unique = list(set(module_list_region))
            module_set_unique.sort()

            # save_path_region = save_path_homo + '_'.join(region_name.split(' ')) + '/'
            # if not os.path.exists(save_path_region):
            #     os.makedirs(save_path_region)

            module_genelist_dict = {}
            for j in range(len(module_set_unique)):
                g_module_id = module_set_unique[j]
                # for g_module_id in module_set_unique:
                module_genelist_dict[g_module_id] = []
                for m, g in zip(module_list_region, glist_human_region_genes):
                    if m == g_module_id:
                        module_genelist_dict[g_module_id].append(g)

                glist = module_genelist_dict[g_module_id]
                if len(module_genelist_dict[g_module_id]) > 0:
                    adata_exp_temp = adata_human_exp[adata_human_exp.obs['region_name'].isin([region_name])][
                                     :, glist]
                    human_cluster_gene_module_exp_array[i, j] = np.mean(adata_exp_temp.X)
                    # print(mouse_cluster_gene_module_exp_array[i, j])

        human_cluster_gene_module_exp_array = human_cluster_gene_module_exp_array - np.reshape(
            np.mean(human_cluster_gene_module_exp_array, axis=1), (-1, 1))

        human_cluster_gene_module_exp_array = human_cluster_gene_module_exp_array - np.reshape(
            np.mean(human_cluster_gene_module_exp_array, axis=0), (1, -1))
        human_cluster_gene_module_exp_array = preprocess.zscore(human_cluster_gene_module_exp_array)
        #human_cluster_gene_module_exp_array = preprocess.normalize_row(human_cluster_gene_module_exp_array)
        #human_cluster_gene_module_exp_array = preprocess.normalize_col(human_cluster_gene_module_exp_array)

        human_cluster_gene_module_exp_df = pd.DataFrame(human_cluster_gene_module_exp_array.T,
                                                        index=module_labels_unique, columns=human_homo_region_list)
        print(human_cluster_gene_module_exp_df)
        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.5
        fig, ax = plt.subplots(figsize=(10, 7), dpi=self.fig_dpi)

        hm = sns.heatmap(human_cluster_gene_module_exp_df, square=False, cbar_kws={'location': 'right'}, cmap=color_map, #"YlGnBu", 'Spectral_r'
                         ax=ax,
                         xticklabels=True, yticklabels=True, linewidths=0.5, linecolor='lightgrey')
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.legend([],[], frameon=False)
        # plt.xticks(rotation=45)
        plt.ylabel('Gene module')
        plt.savefig(save_path + 'human_homoregions_gene_module_heatmap.' + self.fig_format, format=self.fig_format)


        #-------------------------------------------------------------------------------------#
        # Absolute difference between these two heatmaps

        diff_gene_module_exp_df = pd.DataFrame(np.abs(mouse_cluster_gene_module_exp_df.values - human_cluster_gene_module_exp_df.values),
                                                        index=module_labels_unique,
                                               columns=[x+'-'+y for x,y in zip(mouse_homo_region_acronym_list, human_homo_region_acronym_list)])

        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.5
        fig, ax = plt.subplots(figsize=(10, 7), dpi=self.fig_dpi)

        #c = Colormap()
        #color_map = c.cmap_linear('white', 'white', 'blue')

        hm = sns.heatmap(diff_gene_module_exp_df, square=False, cbar_kws={'location': 'right'},
                         cmap='Blues',  # "YlGnBu",
                         ax=ax,
                         xticklabels=True, yticklabels=True, linewidths=0.5, linecolor='lightgrey')
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.legend([],[], frameon=False)
        # plt.xticks(rotation=45)
        plt.ylabel('Gene module')
        plt.savefig(save_path + 'homoregions_gene_module_difference_heatmap.' + self.fig_format, format=self.fig_format)


        return None

    def experiment_6_1_regions_gene_module_cross(self):
        """
        Compute everage expression of degs on gene modules, to check which regions/clusters is specialized expressed by which module.
        And then check Gene ontology analysis.

        For each species:
        1. Identify DEGs for each region and cluster, e.g., 50 DEGs;
        2. Map DEGs on gene modules, and compute average gene expressions;
        3. Plot regions x gene module heatmap.

        :return: None
        """
        TINY_SIZE = 18  # 39
        SMALL_SIZE = 18  # 42
        MEDIUM_SIZE = 26  # 46
        BIGGER_SIZE = 26  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/6_1_regions_gene_module_cross/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # -----------homologous regions----------------------------------------
        # module information
        module_name = 'module'
        module_num = len(Counter(self.adata_gene_embedding.obs[module_name]))
        gene_names = self.adata_gene_embedding.obs_names.values
        module_labels = self.adata_gene_embedding.obs[module_name].values

        module_labels_unique = list(set(self.adata_gene_embedding.obs[module_name].values))
        module_labels_unique.sort(key=lambda x:int(x))
        #print('module_labels.shape', module_labels.shape)
        gene_module_dict = {k: v for k, v in zip(gene_names, module_labels)}

        mouse_gene_names = self.adata_mouse_gene_embedding.obs_names.values
        mouse_module_labels = self.adata_mouse_gene_embedding.obs[module_name].values
        mouse_gene_module_dict = {k: v for k, v in zip(mouse_gene_names, mouse_module_labels)}
        # human gene_name to module
        human_gene_names = self.adata_human_gene_embedding.obs_names.values
        human_module_labels = self.adata_human_gene_embedding.obs[module_name].values
        human_gene_module_dict = {k: v for k, v in zip(human_gene_names, human_module_labels)}

        # homologous regions
        human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        homo_region_dict = OrderedDict()
        for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
            homo_region_dict[x] = y
        mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        mouse_homo_region_list = human_mouse_homo_region['Mouse'].values
        mouse_homo_region_acronym_list = [self.mouse_64_acronym_dict[x] for x in mouse_homo_region_list]
        human_homo_set = set(human_mouse_homo_region['Human'].values)
        human_homo_region_list = human_mouse_homo_region['Human'].values
        human_homo_region_acronym_list = [self.human_88_acronym_dict[x] for x in human_homo_region_list]

        mouse_homo_region_color_list = [self.mouse_64_color_dict[m_r] for m_r in
                                        list(human_mouse_homo_region['Mouse'].values)]
        human_homo_region_color_list = [self.human_88_color_dict[h_r] for h_r in
                                        list(human_mouse_homo_region['Human'].values)]

        # ----------------------gene ontology analysis-------------------------------
        ntop_gene = 50
        p_cut = 1e-3

        # # clusters name
        # cluster_name_unique = sorted(Counter(self.adata_embedding.obs['cluster_name_acronym']).keys(),
        #                              key=lambda t: int(t.split('-')[0]))

        # ----------------------mouse----------------------------------
        sc.tl.rank_genes_groups(self.adata_mouse_exp, groupby='region_name', method='wilcoxon',
                                key_added="wilcoxon")

        mouse_cluster_gene_module_exp_array = np.zeros((len(mouse_homo_region_list), len(module_labels_unique)))

        for i in range(len(mouse_homo_region_list)):
            region_name = mouse_homo_region_list[i]
            sc.pl.rank_genes_groups(self.adata_mouse_exp, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                    show=False)
            glist_mouse_region_genes = sc.get.rank_genes_groups_df(self.adata_mouse_exp,
                                                                   group=region_name,
                                                                   key='wilcoxon',
                                                                   # log2fc_min=0.25,
                                                                   pval_cutoff=p_cut)['names'].squeeze().str.strip().tolist()

            module_list_region = [mouse_gene_module_dict[g] for g in glist_mouse_region_genes]

            module_set_unique = list(set(module_list_region))
            module_set_unique.sort()

            # save_path_region = save_path_homo + '_'.join(region_name.split(' ')) + '/'
            # if not os.path.exists(save_path_region):
            #     os.makedirs(save_path_region)

            module_genelist_dict = {}
            for j in range(len(module_set_unique)):
                g_module_id = module_set_unique[j]
            #for g_module_id in module_set_unique:
                module_genelist_dict[g_module_id] = []
                for m, g in zip(module_list_region, glist_mouse_region_genes):
                    if m == g_module_id:
                        module_genelist_dict[g_module_id].append(g)

                glist = module_genelist_dict[g_module_id]
                if len(module_genelist_dict[g_module_id]) > 0:
                    adata_exp_temp = self.adata_mouse_exp[self.adata_mouse_exp.obs['region_name'].isin([region_name])][:, glist]
                    mouse_cluster_gene_module_exp_array[i, j] = np.mean(adata_exp_temp.X)
                    #print(mouse_cluster_gene_module_exp_array[i, j])

        # mouse_cluster_gene_module_exp_array = mouse_cluster_gene_module_exp_array - np.reshape(np.mean(mouse_cluster_gene_module_exp_array, axis=1), (-1, 1))
        #
        # mouse_cluster_gene_module_exp_array = mouse_cluster_gene_module_exp_array - np.reshape(np.mean(mouse_cluster_gene_module_exp_array, axis=0), (1, -1))
        #
        mouse_cluster_gene_module_exp_array = preprocess.zscore(mouse_cluster_gene_module_exp_array)

        #mouse_cluster_gene_module_exp_array = preprocess.normalize_row(mouse_cluster_gene_module_exp_array)
        #mouse_cluster_gene_module_exp_array = preprocess.normalize_col(mouse_cluster_gene_module_exp_array)

        mouse_cluster_gene_module_exp_df = pd.DataFrame(mouse_cluster_gene_module_exp_array.T, index=module_labels_unique, columns=mouse_homo_region_list)



        # -------------------human-----------------------
        adata_human_exp = self.adata_human_exp.copy()
        cluster_counts = adata_human_exp.obs['region_name'].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        adata_human_exp = adata_human_exp[adata_human_exp.obs['region_name'].isin(keep)].copy()

        sc.tl.rank_genes_groups(adata_human_exp, groupby='region_name', method='wilcoxon',
                                key_added="wilcoxon")

        human_cluster_gene_module_exp_array = np.zeros((len(human_homo_region_list), len(module_labels_unique)))

        for i in range(len(human_homo_region_list)):
            region_name = human_homo_region_list[i]
            sc.pl.rank_genes_groups(adata_human_exp, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                    show=False)
            glist_human_region_genes = sc.get.rank_genes_groups_df(adata_human_exp,
                                                                   group=region_name,
                                                                   key='wilcoxon',
                                                                   # log2fc_min=0.25,
                                                                   pval_cutoff=p_cut)['names'].squeeze().str.strip().tolist()

            module_list_region = [human_gene_module_dict[g] for g in glist_human_region_genes]

            module_set_unique = list(set(module_list_region))
            module_set_unique.sort()

            # save_path_region = save_path_homo + '_'.join(region_name.split(' ')) + '/'
            # if not os.path.exists(save_path_region):
            #     os.makedirs(save_path_region)

            module_genelist_dict = {}
            for j in range(len(module_set_unique)):
                g_module_id = module_set_unique[j]
                # for g_module_id in module_set_unique:
                module_genelist_dict[g_module_id] = []
                for m, g in zip(module_list_region, glist_human_region_genes):
                    if m == g_module_id:
                        module_genelist_dict[g_module_id].append(g)

                glist = module_genelist_dict[g_module_id]
                if len(module_genelist_dict[g_module_id]) > 0:
                    adata_exp_temp = adata_human_exp[adata_human_exp.obs['region_name'].isin([region_name])][
                                     :, glist]
                    human_cluster_gene_module_exp_array[i, j] = np.mean(adata_exp_temp.X)
                    # print(mouse_cluster_gene_module_exp_array[i, j])

        # human_cluster_gene_module_exp_array = human_cluster_gene_module_exp_array - np.reshape(
        #     np.mean(human_cluster_gene_module_exp_array, axis=1), (-1, 1))
        #
        # human_cluster_gene_module_exp_array = human_cluster_gene_module_exp_array - np.reshape(
        #     np.mean(human_cluster_gene_module_exp_array, axis=0), (1, -1))
        human_cluster_gene_module_exp_array = preprocess.zscore(human_cluster_gene_module_exp_array)
        #human_cluster_gene_module_exp_array = preprocess.normalize_row(human_cluster_gene_module_exp_array)
        #human_cluster_gene_module_exp_array = preprocess.normalize_col(human_cluster_gene_module_exp_array)

        human_cluster_gene_module_exp_df = pd.DataFrame(human_cluster_gene_module_exp_array.T,
                                                        index=module_labels_unique, columns=human_homo_region_list)

        result = pd.concat([human_cluster_gene_module_exp_df, mouse_cluster_gene_module_exp_df], axis=1).corr()
        Var_Corr = result[mouse_cluster_gene_module_exp_df.columns].loc[human_cluster_gene_module_exp_df.columns]

        c = Colormap()
        color_map = c.cmap_linear('white', 'white', 'green')

        #Var_Corr_homo = Var_Corr.iloc[0:20, 0:20]
        rcParams["figure.subplot.right"] = 0.85
        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.bottom"] = 0.04
        hm = sns.clustermap(Var_Corr, cmap=color_map,
                            row_colors=human_homo_region_color_list,
                            col_colors=mouse_homo_region_color_list,
                            yticklabels=True,
                            row_cluster=False, col_cluster=False,
                            xticklabels=True, figsize=(12, 12), linewidth=0.02, linecolor='grey',
                            cbar_pos=(0.05, 0.8, 0.04, 0.12))  # cmap=color_map, center=0.6
        # ax = hm.ax_heatmap
        # lw_v = 10
        # ax.add_patch(Rectangle((0, 0), 20, 20, fill=False, edgecolor='cyan', lw=lw_v))
        # plt.setp(hm.ax_heatmap.get_xticklabels(), rotation=30)
        for item in hm.ax_heatmap.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.ax_heatmap.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.subplots_adjust(bottom=0.26)

        plt.savefig(save_path + 'homo_region_mean_exp_on_modules.' + self.fig_format, format=self.fig_format)



        #-------------------------------------------------------------------------------------#
        # Absolute difference between these two heatmaps

        diff_gene_module_exp_df = pd.DataFrame(np.abs(mouse_cluster_gene_module_exp_df.values - human_cluster_gene_module_exp_df.values),
                                                        index=module_labels_unique,
                                               columns=[x+'-'+y for x,y in zip(mouse_homo_region_acronym_list, human_homo_region_acronym_list)])

        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.5
        fig, ax = plt.subplots(figsize=(10, 7), dpi=self.fig_dpi)

        #c = Colormap()
        #color_map = c.cmap_linear('white', 'white', 'blue')

        hm = sns.heatmap(diff_gene_module_exp_df, square=False, cbar_kws={'location': 'right'},
                         cmap='Blues',  # "YlGnBu",
                         ax=ax,
                         xticklabels=True, yticklabels=True, linewidths=0.5, linecolor='lightgrey')
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.legend([],[], frameon=False)
        # plt.xticks(rotation=45)
        plt.ylabel('Gene module')
        plt.savefig(save_path + 'homoregions_gene_module_difference_heatmap.' + self.fig_format, format=self.fig_format)


        return None

    def experiment_7_clusters_gene_module_average_exp(self):
        """
        Compute everage expression of degs on gene modules, to check which regions/clusters is specialized expressed by which module.
        And then check Gene ontology analysis.

        For each species:
        1. Identify DEGs for each region and cluster, e.g., 50 DEGs;
        2. Map DEGs on gene modules, and compute average gene expressions;
        3. Plot regions x gene module heatmap.

        :return: None
        """
        TINY_SIZE = 14  # 39
        SMALL_SIZE = 16  # 42
        MEDIUM_SIZE = 22  # 46
        BIGGER_SIZE = 22  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/7_clusters_gene_module_average_exp/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # -----------homologous regions----------------------------------------
        # module information
        module_name = 'module'
        module_num = len(Counter(self.adata_gene_embedding.obs[module_name]))
        gene_names = self.adata_gene_embedding.obs_names.values
        module_labels = self.adata_gene_embedding.obs[module_name].values

        module_labels_unique = list(set(self.adata_gene_embedding.obs[module_name].values))
        module_labels_unique.sort(key=lambda x:int(x))
        # print('module_labels.shape', module_labels.shape)
        gene_module_dict = {k: v for k, v in zip(gene_names, module_labels)}

        mouse_gene_names = self.adata_mouse_gene_embedding.obs_names.values
        mouse_module_labels = self.adata_mouse_gene_embedding.obs[module_name].values
        mouse_gene_module_dict = {k: v for k, v in zip(mouse_gene_names, mouse_module_labels)}
        # human gene_name to module
        human_gene_names = self.adata_human_gene_embedding.obs_names.values
        human_module_labels = self.adata_human_gene_embedding.obs[module_name].values
        human_gene_module_dict = {k: v for k, v in zip(human_gene_names, human_module_labels)}

        # # homologous regions
        # human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        # homo_region_dict = OrderedDict()
        # for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
        #     homo_region_dict[x] = y
        # mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        # mouse_homo_region_list = human_mouse_homo_region['Mouse'].values
        # human_homo_set = set(human_mouse_homo_region['Human'].values)
        # human_homo_region_list = human_mouse_homo_region['Human'].values

        # ----------------------gene ontology analysis-------------------------------
        ntop_gene = 50
        p_cut = 1e-2

        # # clusters name
        cluster_name_unique = sorted(Counter(self.adata_embedding.obs['cluster_name_acronym']).keys(),
                                      key=lambda t: int(t.split('-')[0]))

        # ----------------------mouse----------------------------------
        sc.tl.rank_genes_groups(self.adata_mouse_exp, groupby='cluster_name_acronym', method='wilcoxon',
                                key_added="wilcoxon")

        mouse_cluster_gene_module_exp_array = np.zeros((len(cluster_name_unique), len(module_labels_unique)))

        for i in range(len(cluster_name_unique)):
            c_name = cluster_name_unique[i]
            sc.pl.rank_genes_groups(self.adata_mouse_exp, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                    show=False)
            glist_mouse_region_genes = sc.get.rank_genes_groups_df(self.adata_mouse_exp,
                                                                   group=c_name,
                                                                   key='wilcoxon', #ntop_gene=ntop_gene
                                                                   # log2fc_min=0.25,  pval_cutoff=p_cut
                                                                  )['names'].squeeze().str.strip().tolist()
            #print(len(glist_mouse_region_genes))
            glist_mouse_region_genes = glist_mouse_region_genes[0:ntop_gene]

            module_list_region = [mouse_gene_module_dict[g] for g in glist_mouse_region_genes]

            module_set_unique = list(set(module_list_region))
            module_set_unique.sort()

            # save_path_region = save_path_homo + '_'.join(region_name.split(' ')) + '/'
            # if not os.path.exists(save_path_region):
            #     os.makedirs(save_path_region)

            module_genelist_dict = {}
            for j in range(len(module_set_unique)):
                g_module_id = module_set_unique[j]
                # for g_module_id in module_set_unique:
                module_genelist_dict[g_module_id] = []
                for m, g in zip(module_list_region, glist_mouse_region_genes):
                    if m == g_module_id:
                        module_genelist_dict[g_module_id].append(g)

                glist = module_genelist_dict[g_module_id]
                if len(module_genelist_dict[g_module_id]) > 0:
                    adata_exp_temp = self.adata_mouse_exp[self.adata_mouse_exp.obs['cluster_name_acronym'].isin([c_name])][
                                     :, glist]
                    mouse_cluster_gene_module_exp_array[i, j] = np.mean(adata_exp_temp.X)
                    # print(mouse_cluster_gene_module_exp_array[i, j])

        mouse_cluster_gene_module_exp_array = mouse_cluster_gene_module_exp_array - np.reshape(
            np.mean(mouse_cluster_gene_module_exp_array, axis=1), (-1, 1))

        mouse_cluster_gene_module_exp_array = mouse_cluster_gene_module_exp_array - np.reshape(
            np.mean(mouse_cluster_gene_module_exp_array, axis=0), (1, -1))

        mouse_cluster_gene_module_exp_array = preprocess.zscore(mouse_cluster_gene_module_exp_array)

        mouse_cluster_gene_module_exp_df = pd.DataFrame(mouse_cluster_gene_module_exp_array.T,
                                                        index=module_labels_unique, columns=cluster_name_unique)
        print(mouse_cluster_gene_module_exp_df)

        rcParams["figure.subplot.left"] = 0.03
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.top"] = 0.98
        rcParams["figure.subplot.bottom"] = 0.4

        #c = Colormap()
        #color_map = c.cmap_linear('white', 'white', 'darkgreen')

        fig, ax = plt.subplots(figsize=(30, 6), dpi=self.fig_dpi)

        hm = sns.heatmap(mouse_cluster_gene_module_exp_df, square=False, cbar_kws={'location': 'right'}, cmap='Greens', #"YlGnBu",
                         ax=ax,
                         xticklabels=True, yticklabels=True, linewidths=0.5, linecolor='lightgrey')
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.legend([],[], frameon=False)
        # plt.xticks(rotation=45)
        plt.ylabel('Gene module')
        plt.savefig(save_path + 'mouse_clusters_gene_module_heatmap.' + self.fig_format, format=self.fig_format)

        # -------------------human-----------------------
        # adata_human_exp = self.adata_human_exp.copy()
        # cluster_counts = adata_human_exp.obs['region_name'].value_counts()
        # keep = cluster_counts.index[cluster_counts >= 2]
        # adata_human_exp = adata_human_exp[adata_human_exp.obs['region_name'].isin(keep)].copy()

        sc.tl.rank_genes_groups(self.adata_human_exp, groupby='cluster_name_acronym', method='wilcoxon',
                                key_added="wilcoxon")

        human_cluster_gene_module_exp_array = np.zeros((len(cluster_name_unique), len(module_labels_unique)))

        for i in range(len(cluster_name_unique)):
            region_name = cluster_name_unique[i]
            sc.pl.rank_genes_groups(self.adata_human_exp, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                    show=False)
            glist_human_region_genes = sc.get.rank_genes_groups_df(self.adata_human_exp,
                                                                   group=region_name,
                                                                   key='wilcoxon', #ntop_gene=ntop_gene
                                                                   # log2fc_min=0.25, pval_cutoff=p_cut
                                                                   )['names'].squeeze().str.strip().tolist()
            #print(len(glist_human_region_genes))
            glist_human_region_genes = glist_human_region_genes[0:ntop_gene]

            module_list_region = [human_gene_module_dict[g] for g in glist_human_region_genes]

            module_set_unique = list(set(module_list_region))
            module_set_unique.sort()

            # save_path_region = save_path_homo + '_'.join(region_name.split(' ')) + '/'
            # if not os.path.exists(save_path_region):
            #     os.makedirs(save_path_region)

            module_genelist_dict = {}
            for j in range(len(module_set_unique)):
                g_module_id = module_set_unique[j]
                # for g_module_id in module_set_unique:
                module_genelist_dict[g_module_id] = []
                for m, g in zip(module_list_region, glist_human_region_genes):
                    if m == g_module_id:
                        module_genelist_dict[g_module_id].append(g)

                glist = module_genelist_dict[g_module_id]
                if len(module_genelist_dict[g_module_id]) > 0:
                    adata_exp_temp = self.adata_human_exp[self.adata_human_exp.obs['cluster_name_acronym'].isin([region_name])][:, glist]
                    human_cluster_gene_module_exp_array[i, j] = np.mean(adata_exp_temp.X)
                    # print(mouse_cluster_gene_module_exp_array[i, j])

        human_cluster_gene_module_exp_array = human_cluster_gene_module_exp_array - np.reshape(
            np.mean(human_cluster_gene_module_exp_array, axis=1), (-1, 1))

        human_cluster_gene_module_exp_array = human_cluster_gene_module_exp_array - np.reshape(
            np.mean(human_cluster_gene_module_exp_array, axis=0), (1, -1))

        human_cluster_gene_module_exp_array = preprocess.zscore(human_cluster_gene_module_exp_array)


        human_cluster_gene_module_exp_df = pd.DataFrame(human_cluster_gene_module_exp_array.T,
                                                        index=module_labels_unique, columns=cluster_name_unique)
        print(human_cluster_gene_module_exp_df)

        rcParams["figure.subplot.left"] = 0.03
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.top"] = 0.98
        rcParams["figure.subplot.bottom"] = 0.4

        fig, ax = plt.subplots(figsize=(30, 6), dpi=self.fig_dpi)
        hm = sns.heatmap(human_cluster_gene_module_exp_df, square=False, cbar_kws={'location': 'right'}, cmap='Greens', #"YlGnBu",
                         ax=ax,
                         xticklabels=True, yticklabels=True, linewidths=0.5, linecolor='lightgrey')
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.legend([],[], frameon=False)
        # plt.xticks(rotation=45)
        plt.ylabel('Gene module')
        plt.savefig(save_path + 'human_clusters_gene_module_heatmap.' + self.fig_format, format=self.fig_format)


        # -------------------------------------------------------------------------------------#
        # Absolute difference between these two heatmaps

        diff_gene_module_exp_df = pd.DataFrame(
            np.abs(mouse_cluster_gene_module_exp_df.values - human_cluster_gene_module_exp_df.values),
            index=module_labels_unique,
            columns=cluster_name_unique)

        rcParams["figure.subplot.left"] = 0.03
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.top"] = 0.98
        rcParams["figure.subplot.bottom"] = 0.4

        fig, ax = plt.subplots(figsize=(30, 6), dpi=self.fig_dpi)

        hm = sns.heatmap(diff_gene_module_exp_df, square=False, cbar_kws={'location': 'right'},
                         cmap='Blues',  # "YlGnBu",
                         ax=ax,
                         xticklabels=True, yticklabels=True, linewidths=0.5, linecolor='lightgrey')
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.legend([],[], frameon=False)
        # plt.xticks(rotation=45)
        plt.ylabel('Gene module')
        plt.savefig(save_path + 'clusters_gene_module_difference_heatmap.' + self.fig_format, format=self.fig_format)

        return None


    def experiment_7_1_clusters_gene_module_cross(self):
        """
        Compute everage expression of degs on gene modules, to check which regions/clusters is specialized expressed by which module.
        And then check Gene ontology analysis.

        For each species:
        1. Identify DEGs for each region and cluster, e.g., 50 DEGs;
        2. Map DEGs on gene modules, and compute average gene expressions;
        3. Plot regions x gene module heatmap.

        :return: None
        """
        TINY_SIZE = 14  # 39
        SMALL_SIZE = 16  # 42
        MEDIUM_SIZE = 22  # 46
        BIGGER_SIZE = 22  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/7_1_clusters_gene_module_cross/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # -----------homologous regions----------------------------------------
        # module information
        module_name = 'module'
        module_num = len(Counter(self.adata_gene_embedding.obs[module_name]))
        gene_names = self.adata_gene_embedding.obs_names.values
        module_labels = self.adata_gene_embedding.obs[module_name].values

        module_labels_unique = list(set(self.adata_gene_embedding.obs[module_name].values))
        module_labels_unique.sort(key=lambda x:int(x))
        # print('module_labels.shape', module_labels.shape)
        gene_module_dict = {k: v for k, v in zip(gene_names, module_labels)}

        mouse_gene_names = self.adata_mouse_gene_embedding.obs_names.values
        mouse_module_labels = self.adata_mouse_gene_embedding.obs[module_name].values
        mouse_gene_module_dict = {k: v for k, v in zip(mouse_gene_names, mouse_module_labels)}
        # human gene_name to module
        human_gene_names = self.adata_human_gene_embedding.obs_names.values
        human_module_labels = self.adata_human_gene_embedding.obs[module_name].values
        human_gene_module_dict = {k: v for k, v in zip(human_gene_names, human_module_labels)}



        # # homologous regions
        # human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        # homo_region_dict = OrderedDict()
        # for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
        #     homo_region_dict[x] = y
        # mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        # mouse_homo_region_list = human_mouse_homo_region['Mouse'].values
        # human_homo_set = set(human_mouse_homo_region['Human'].values)
        # human_homo_region_list = human_mouse_homo_region['Human'].values

        # ----------------------gene ontology analysis-------------------------------
        ntop_gene = 100
        p_cut = 1e-2

        # # clusters name
        cluster_name_unique = sorted(Counter(self.adata_embedding.obs['cluster_name_acronym']).keys(),
                                      key=lambda t: int(t.split('-')[0]))

        # ----------------------mouse----------------------------------
        sc.tl.rank_genes_groups(self.adata_mouse_exp, groupby='cluster_name_acronym', method='wilcoxon',
                                key_added="wilcoxon")

        mouse_cluster_gene_module_exp_array = np.zeros((len(cluster_name_unique), len(module_labels_unique)))

        for i in range(len(cluster_name_unique)):
            c_name = cluster_name_unique[i]
            sc.pl.rank_genes_groups(self.adata_mouse_exp, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                    show=False)
            glist_mouse_region_genes = sc.get.rank_genes_groups_df(self.adata_mouse_exp,
                                                                   group=c_name,
                                                                   key='wilcoxon', #ntop_gene=ntop_gene
                                                                   # log2fc_min=0.25,  pval_cutoff=p_cut
                                                                  )['names'].squeeze().str.strip().tolist()
            #print(len(glist_mouse_region_genes))
            glist_mouse_region_genes = glist_mouse_region_genes[0:ntop_gene]

            module_list_region = [mouse_gene_module_dict[g] for g in glist_mouse_region_genes]

            module_set_unique = list(set(module_list_region))
            module_set_unique.sort()

            # save_path_region = save_path_homo + '_'.join(region_name.split(' ')) + '/'
            # if not os.path.exists(save_path_region):
            #     os.makedirs(save_path_region)

            module_genelist_dict = {}
            for j in range(len(module_set_unique)):
                g_module_id = module_set_unique[j]
                # for g_module_id in module_set_unique:
                module_genelist_dict[g_module_id] = []
                for m, g in zip(module_list_region, glist_mouse_region_genes):
                    if m == g_module_id:
                        module_genelist_dict[g_module_id].append(g)

                glist = module_genelist_dict[g_module_id]
                if len(module_genelist_dict[g_module_id]) > 0:
                    adata_exp_temp = self.adata_mouse_exp[self.adata_mouse_exp.obs['cluster_name_acronym'].isin([c_name])][
                                     :, glist]
                    mouse_cluster_gene_module_exp_array[i, j] = np.mean(adata_exp_temp.X)
                    # print(mouse_cluster_gene_module_exp_array[i, j])

        mouse_cluster_gene_module_exp_array = mouse_cluster_gene_module_exp_array - np.reshape(
            np.mean(mouse_cluster_gene_module_exp_array, axis=1), (-1, 1))

        mouse_cluster_gene_module_exp_array = mouse_cluster_gene_module_exp_array - np.reshape(
            np.mean(mouse_cluster_gene_module_exp_array, axis=0), (1, -1))

        mouse_cluster_gene_module_exp_array = preprocess.zscore(mouse_cluster_gene_module_exp_array)

        mouse_cluster_gene_module_exp_df = pd.DataFrame(mouse_cluster_gene_module_exp_array.T,
                                                        index=module_labels_unique, columns=['mouse_'+x for x in cluster_name_unique])

        # -------------------human-----------------------
        # adata_human_exp = self.adata_human_exp.copy()
        # cluster_counts = adata_human_exp.obs['region_name'].value_counts()
        # keep = cluster_counts.index[cluster_counts >= 2]
        # adata_human_exp = adata_human_exp[adata_human_exp.obs['region_name'].isin(keep)].copy()

        sc.tl.rank_genes_groups(self.adata_human_exp, groupby='cluster_name_acronym', method='wilcoxon',
                                key_added="wilcoxon")

        human_cluster_gene_module_exp_array = np.zeros((len(cluster_name_unique), len(module_labels_unique)))

        for i in range(len(cluster_name_unique)):
            region_name = cluster_name_unique[i]
            sc.pl.rank_genes_groups(self.adata_human_exp, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                    show=False)
            glist_human_region_genes = sc.get.rank_genes_groups_df(self.adata_human_exp,
                                                                   group=region_name,
                                                                   key='wilcoxon', #ntop_gene=ntop_gene
                                                                   # log2fc_min=0.25, pval_cutoff=p_cut
                                                                   )['names'].squeeze().str.strip().tolist()
            #print(len(glist_human_region_genes))
            glist_human_region_genes = glist_human_region_genes[0:ntop_gene]

            module_list_region = [human_gene_module_dict[g] for g in glist_human_region_genes]

            module_set_unique = list(set(module_list_region))
            module_set_unique.sort()

            # save_path_region = save_path_homo + '_'.join(region_name.split(' ')) + '/'
            # if not os.path.exists(save_path_region):
            #     os.makedirs(save_path_region)

            module_genelist_dict = {}
            for j in range(len(module_set_unique)):
                g_module_id = module_set_unique[j]
                # for g_module_id in module_set_unique:
                module_genelist_dict[g_module_id] = []
                for m, g in zip(module_list_region, glist_human_region_genes):
                    if m == g_module_id:
                        module_genelist_dict[g_module_id].append(g)

                glist = module_genelist_dict[g_module_id]
                if len(module_genelist_dict[g_module_id]) > 0:
                    adata_exp_temp = self.adata_human_exp[self.adata_human_exp.obs['cluster_name_acronym'].isin([region_name])][:, glist]
                    human_cluster_gene_module_exp_array[i, j] = np.mean(adata_exp_temp.X)
                    # print(mouse_cluster_gene_module_exp_array[i, j])

        human_cluster_gene_module_exp_array = human_cluster_gene_module_exp_array - np.reshape(
            np.mean(human_cluster_gene_module_exp_array, axis=1), (-1, 1))

        human_cluster_gene_module_exp_array = human_cluster_gene_module_exp_array - np.reshape(
            np.mean(human_cluster_gene_module_exp_array, axis=0), (1, -1))

        human_cluster_gene_module_exp_array = preprocess.zscore(human_cluster_gene_module_exp_array)


        human_cluster_gene_module_exp_df = pd.DataFrame(human_cluster_gene_module_exp_array.T,
                                                        index=module_labels_unique, columns=['human_'+x for x in cluster_name_unique])

        #print()

        result = pd.concat([human_cluster_gene_module_exp_df, mouse_cluster_gene_module_exp_df], axis=1).corr()
        #Var_Corr = result[mouse_cluster_gene_module_exp_df.columns].loc[human_cluster_gene_module_exp_df.columns]
        Var_Corr = result[mouse_cluster_gene_module_exp_df.columns].loc[human_cluster_gene_module_exp_df.columns]

        Var_Corr.columns = cluster_name_unique
        Var_Corr.index = cluster_name_unique

        print(Var_Corr.shape)

        c = Colormap()
        color_map = c.cmap_linear('white', 'white', 'green')

        # Var_Corr_homo = Var_Corr.iloc[0:20, 0:20]
        rcParams["figure.subplot.right"] = 0.85
        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.bottom"] = 0.04
        print(len([self.cluster_color_dict[x] for x in cluster_name_unique]))
        hm = sns.clustermap(Var_Corr, cmap=color_map,
                            row_colors=[self.cluster_color_dict[x] for x in cluster_name_unique],
                            col_colors=[self.cluster_color_dict[x] for x in cluster_name_unique],
                            yticklabels=True,
                            row_cluster=False, col_cluster=False,
                            xticklabels=True, figsize=(30, 30), linewidth=0.02, linecolor='grey',
                            cbar_pos=(0.05, 0.8, 0.04, 0.12))  # cmap=color_map, center=0.6
        # ax = hm.ax_heatmap
        # lw_v = 10
        # ax.add_patch(Rectangle((0, 0), 20, 20, fill=False, edgecolor='cyan', lw=lw_v))
        # plt.setp(hm.ax_heatmap.get_xticklabels(), rotation=30)
        for item in hm.ax_heatmap.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.ax_heatmap.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.subplots_adjust(bottom=0.26)

        plt.savefig(save_path + 'homo_region_mean_exp_on_modules.' + self.fig_format, format=self.fig_format)


        return None


    def experiment_8_homoregions_go_terms_overlap(self):
        """
        Check the overlap of GO terms for two species on gene modules.

        1. Acquire DEGs for each region on gene modules for each species;
        2. Get the GO terms below a p value threshold;
        3. Get the overlap set of GO terms for each pair of (mouse region, human region, gene module);
            Overlap ratio = (Mouse Unique GO terms in the overlap set / Unique Mouse GO terms with Adjusted P < p-threshold)  +
             (Unique GO terms in the overlap set / Unique Human GO terms with Adjusted P < p-threshold) / 2
            Overlap ratio mouse = Mouse Unique GO terms in the overlap set / Unique Mouse GO terms with Adjusted P < p-threshold
        4.  Plot the heatmap of overlap ratio: x-regions, y-gene module, color-ratio values
        5. Plot the bar or dot figure of the most significant common GO terms.
        :return:None
        """

        sns.set(style='white')
        TINY_SIZE = 18  # 39
        SMALL_SIZE = 18  # 42
        MEDIUM_SIZE = 26  # 46
        BIGGER_SIZE = 26  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        # save_path
        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/8_experiment_homoregions_go_terms_overlap/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # -----------homologous regions----------------------------------------
        # module information
        module_name = 'module'
        module_num = len(Counter(self.adata_gene_embedding.obs[module_name]))
        gene_names = self.adata_gene_embedding.obs_names.values
        module_labels = self.adata_gene_embedding.obs[module_name].values

        module_labels_unique = list(set(self.adata_gene_embedding.obs[module_name].values))
        module_labels_unique.sort(key=lambda x: int(x))
        #print('module_labels.shape', module_labels.shape)
        gene_module_dict = {k: v for k, v in zip(gene_names, module_labels)}

        mouse_gene_names = self.adata_mouse_gene_embedding.obs_names.values
        mouse_module_labels = self.adata_mouse_gene_embedding.obs[module_name].values
        mouse_gene_module_dict = {k: v for k, v in zip(mouse_gene_names, mouse_module_labels)}
        # human gene_name to module
        human_gene_names = self.adata_human_gene_embedding.obs_names.values
        human_module_labels = self.adata_human_gene_embedding.obs[module_name].values
        human_gene_module_dict = {k: v for k, v in zip(human_gene_names, human_module_labels)}

        # homologous regions
        human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        homo_region_dict = OrderedDict()
        for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
            homo_region_dict[x] = y
        mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        mouse_homo_region_list = human_mouse_homo_region['Mouse'].values
        human_homo_set = set(human_mouse_homo_region['Human'].values)
        human_homo_region_list = human_mouse_homo_region['Human'].values

        mouse_homo_region_acronym_list = [self.mouse_64_acronym_dict[x] for x in mouse_homo_region_list]
        human_homo_region_acronym_list = [self.human_88_acronym_dict[x] for x in human_homo_region_list]

        # ----------------------gene ontology analysis-------------------------------
        ntop_gene = 150
        p_cut = 1e-3

        mouse_p_cut = 1e-2

        go_p_cut = 5e-2#5e-2
        top_term = 2
        color_list = list(sns.color_palette(cc.glasbey, n_colors=8))

        top_terms_plot = 10

        plot_size = (12, 10)

        mouse_go_gene_sets = ['Allen_Brain_Atlas_down',
                              'Allen_Brain_Atlas_up',
                              'Azimuth_Cell_Types_2021',
                              'CellMarker_Augmented_2021',
                              'GO_Biological_Process_2021',
                              'GO_Cellular_Component_2021',
                              'GO_Molecular_Function_2021',
                              'Mouse_Gene_Atlas']
        human_go_gene_sets = ['Allen_Brain_Atlas_down', 'Allen_Brain_Atlas_up',
                              'Azimuth_Cell_Types_2021', 'CellMarker_Augmented_2021',
                              'GO_Biological_Process_2021', 'GO_Cellular_Component_2021',
                              'GO_Molecular_Function_2021', 'Human_Gene_Atlas']

        sc.tl.rank_genes_groups(self.adata_mouse_exp, groupby='region_name', method='wilcoxon',
                                key_added="wilcoxon")

        adata_human_exp = self.adata_human_exp.copy()
        cluster_counts = adata_human_exp.obs['region_name'].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        adata_human_exp = adata_human_exp[adata_human_exp.obs['region_name'].isin(keep)].copy()

        sc.tl.rank_genes_groups(adata_human_exp, groupby='region_name', method='wilcoxon',
                                key_added="wilcoxon")

        # mouse_homo_region_list = ['Field CA3', 'Field CA2', 'Field CA1']
        homoregion_go_terms_overlap_array = np.zeros((len(mouse_homo_region_list), len(module_labels_unique)))

        sns.set(style='white')
        TINY_SIZE = 12  # 39
        SMALL_SIZE = 12  # 42
        MEDIUM_SIZE = 14  # 46
        BIGGER_SIZE = 14  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        # plt.rc('title', titlesize=BIGGER_SIZE)
        # rcParams["legend.title_fontsize"]

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        current_r_id = 0

        if os.path.exists(save_path + 'homoregion_go_terms_overlap_array.npz'):
            npzfile = np.load(save_path + 'homoregion_go_terms_overlap_array.npz')
            homoregion_go_terms_overlap_array = npzfile['homoregion_go_terms_overlap_array']
            current_r_id = npzfile['current_r_id']

        #mouse_homo_region_list = ['Claustrum']
        #human_homo_region_list = ['claustrum']

        original_stdout = sys.stdout  # Save a reference to the original standard output

        with open(save_path + f'mouse_human_region_GSEA.txt', 'a+') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            for r_id, m_region_name, h_region_name in zip(range(len(mouse_homo_region_list)), mouse_homo_region_list, human_homo_region_list):
                print(r_id, m_region_name, h_region_name)
                if r_id < current_r_id:
                    continue
                #print()
                sc.pl.rank_genes_groups(self.adata_mouse_exp, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                        show=False)
                glist_mouse_region_genes = sc.get.rank_genes_groups_df(self.adata_mouse_exp,
                                                                       group=m_region_name,
                                                                       key='wilcoxon',
                                                                       # log2fc_min=0.25,
                                                                       pval_cutoff=mouse_p_cut)['names'].squeeze().str.strip().tolist()
                #print(glist_mouse_region_genes)
                print('glist_mouse_region_genes number:', len(glist_mouse_region_genes))

                module_list_region_m = [mouse_gene_module_dict[g] for g in glist_mouse_region_genes]

                module_set_unique_m = list(set(module_list_region_m))
                module_set_unique_m.sort()
                print('module_set_unique_m:', module_set_unique_m)

                save_path_region_m = save_path + 'Mouse/' + '_'.join(m_region_name.split(' ')) + '/'
                if not os.path.exists(save_path_region_m):
                    os.makedirs(save_path_region_m)

                save_path_region_m_special = save_path + 'Mouse/' + '_'.join(m_region_name.split(' ')) + '_s/'
                if not os.path.exists(save_path_region_m_special):
                    os.makedirs(save_path_region_m_special)


                sc.pl.rank_genes_groups(adata_human_exp, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                        show=False)
                if h_region_name == 'piriform cortex':
                    glist_human_region_genes = sc.get.rank_genes_groups_df(adata_human_exp,
                                                                           group=h_region_name,
                                                                           key='wilcoxon',
                                                                           # log2fc_min=0.25,
                                                                           pval_cutoff=0.53)['names'].squeeze().str.strip().tolist()
                else:
                    glist_human_region_genes = sc.get.rank_genes_groups_df(adata_human_exp,
                                                                       group=h_region_name,
                                                                       key='wilcoxon',
                                                                       # log2fc_min=0.25,
                                                                       pval_cutoff=p_cut)['names'].squeeze().str.strip().tolist()

                print('glist_human_region_genes number:', len(glist_human_region_genes))

                module_list_region_h = [human_gene_module_dict[g] for g in glist_human_region_genes]

                module_set_unique_h = list(set(module_list_region_h))
                module_set_unique_h.sort()
                print('module_set_unique_h:', module_set_unique_h)

                module_set_unique = list(set(module_set_unique_m + module_set_unique_h))
                module_set_unique.sort()

                save_path_region_h = save_path + 'Human/' + '_'.join(h_region_name.split(' ')) + '/'
                if not os.path.exists(save_path_region_h):
                    os.makedirs(save_path_region_h)

                save_path_region_h_special = save_path + 'Human/' + '_'.join(h_region_name.split(' ')) + '_s/'
                if not os.path.exists(save_path_region_h_special):
                    os.makedirs(save_path_region_h_special)

                module_genelist_dict_m = {}
                module_genelist_dict_h = {}


                for module_id, g_module_id in zip(range(len(module_set_unique)), module_set_unique):

                    print('region:', r_id, m_region_name, h_region_name, module_id, 'gene module:', g_module_id)

                    module_genelist_dict_m[g_module_id] = []
                    for m, g in zip(module_list_region_m, glist_mouse_region_genes):
                        if m == g_module_id:
                            module_genelist_dict_m[g_module_id].append(g)

                    go_gene_list_m = module_genelist_dict_m[g_module_id]

                    enr_res_m = gseapy.enrichr(gene_list=go_gene_list_m,
                                               organism='Mouse',
                                               gene_sets=mouse_go_gene_sets#, cutoff=go_p_cut
                                               )  # cutoff=go_p_cut


                    module_genelist_dict_h[g_module_id] = []
                    for h, g in zip(module_list_region_h, glist_human_region_genes):
                        if h == g_module_id:
                            module_genelist_dict_h[g_module_id].append(g)

                    go_gene_list_h = module_genelist_dict_h[g_module_id]

                    enr_res_h = gseapy.enrichr(gene_list=go_gene_list_h,
                                             organism='Human',
                                             gene_sets=human_go_gene_sets#, cutoff=go_p_cut
                                             )  # top_term=top_term

                    enr_res_mouse_results = enr_res_m.results[
                        enr_res_m.results['Adjusted P-value'] <= go_p_cut]  # .columns
                    #print(enr_res_mouse_results)
                    enr_res_human_results = enr_res_h.results[
                        enr_res_h.results['Adjusted P-value'] <= go_p_cut]  # .columns
                    #print(enr_res_human_results)

                    #print('enr_res_mouse go term numbers = ', enr_res_mouse_results.shape[0])
                    #print('enr_res_human go term numbers = ', enr_res_human_results.shape[0])

                    Overlapped_term_num = 0

                    enr_res_mouse_results_term = enr_res_mouse_results['Term'].tolist()
                    enr_res_human_results_term = enr_res_human_results['Term'].tolist()

                    Overlapped_term_set = []
                    for i in range(enr_res_mouse_results.shape[0]):
                        for j in range(enr_res_human_results.shape[0]):
                            if enr_res_mouse_results_term[i] == enr_res_human_results_term[j]:
                                #print(enr_res_mouse_results_term[i], enr_res_human_results_term[j])
                                Overlapped_term_set.append(enr_res_mouse_results_term[i])
                                Overlapped_term_num += 1
                    #print(Overlapped_term_num)
                    Overlapped_term_set_list = list(Overlapped_term_set)
                    print('Overlapped_term_set_list:', Overlapped_term_set_list)
                    enr_res_mouse_results_special = enr_res_mouse_results[enr_res_mouse_results['Term'].isin(list(set(enr_res_mouse_results_term) -  set(Overlapped_term_set)))]
                    enr_res_human_results_special = enr_res_human_results[enr_res_human_results['Term'].isin(list(set(enr_res_human_results_term) -  set(Overlapped_term_set)))]

                    print('enr_res_mouse_results_special:', enr_res_mouse_results_special)
                    print('enr_res_human_results_special:', enr_res_human_results_special)

                    if len(list(set(enr_res_mouse_results_term))) != 0 and len(list(set(enr_res_human_results_term))) !=0:
                        overlap_ratio = (len(list(set(Overlapped_term_set))) / len(list(set(enr_res_mouse_results_term)))) * 1/2 +\
                                    (len(list(set(Overlapped_term_set))) / len(list(set(enr_res_human_results_term)))) * 1/2

                    elif len(list(set(enr_res_mouse_results_term))) == 0:
                        overlap_ratio = len(list(set(Overlapped_term_set))) / len(list(set(enr_res_human_results_term))) * 1/2

                    elif len(list(set(enr_res_human_results_term))) == 0:
                        overlap_ratio = len(list(set(Overlapped_term_set))) / len(list(set(enr_res_mouse_results_term))) * 1 / 2
                    else:
                        overlap_ratio = 0

                    homoregion_go_terms_overlap_array[r_id, module_id] = overlap_ratio

                    enr_res_mouse_results_top, enr_res_human_results_top, if_empty = get_top_same_terms(enr_res_mouse_results,
                                                                                              enr_res_human_results, cutoff=go_p_cut) #top_terms = top_terms_plot
                    if if_empty:
                        continue

                    # plot common GO terms
                    # fig, axes = plt.subplot()
                    rcParams["figure.subplot.left"] = 0.48
                    rcParams["figure.subplot.right"] = 0.70
                    rcParams["figure.subplot.top"] = 0.9
                    rcParams["figure.subplot.bottom"] = 0.1

                    rcParams["legend.title_fontsize"] = MEDIUM_SIZE
                    rcParams["axes.titlesize"] = MEDIUM_SIZE
                    fig, ax = gseapy.barplot(enr_res_mouse_results_top, column="Adjusted P-value", group='Gene_set', color=color_list,
                                             # color=self.mouse_color,
                                             # set group, so you could do a multi-sample/library comparsion
                                             size=5, title=f"GO of Mouse {m_region_name} module {g_module_id} DEGs",
                                             figsize=plot_size, cutoff=go_p_cut)  # top_term=top_terms_plot
                    ax.get_legend().set_title("Gene sets")
                    # ax.get_legend().set_fontsize(MEDIUM_SIZE)
                    # ax.get_title().set_fontsize(MEDIUM_SIZE)
                    # ax.get_legend().set_loc('lower center')
                    # fig.axes.append(ax)
                    fig.savefig(save_path_region_m + f'm{g_module_id}_barplot.' + self.fig_format,
                                format=self.fig_format, dpi=self.fig_dpi)
                    # rcParams["figure.subplot.left"] = 0.1
                    # rcParams["figure.subplot.right"] = 0.9
                    rcParams["figure.subplot.left"] = 0.6
                    rcParams["figure.subplot.right"] = 0.88
                    rcParams["figure.subplot.bottom"] = 0.25  # 0.15
                    rcParams["figure.subplot.top"] = 0.95

                    fig, ax = gseapy.dotplot(enr_res_mouse_results_top,
                                             column="Adjusted P-value",
                                             x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion
                                             size=10,
                                             # top_term=5,  #
                                             figsize=plot_size,
                                             title=f"GO of Mouse {m_region_name} module {g_module_id} DEGs",
                                             xticklabels_rot=45,
                                             # rotate xtick labels show_ring=True, set to False to revmove outer ring
                                             marker='o',
                                             cmap=plt.cm.winter_r,
                                             format=self.fig_format, cutoff=go_p_cut)  # top_term=top_terms_plot
                    # cbar = fig.colorbar(ax=ax)
                    # im = ax.images
                    # print(len(im))
                    # print(im)
                    # # Assume colorbar was plotted last one plotted last
                    # cbar = im.colorbar
                    # cbar.set_label(size=MEDIUM_SIZE, labelpad=15)
                    ax.set_xlabel('')
                    fig.savefig(save_path_region_m + f'm{g_module_id}_botplot.' + self.fig_format,
                                format=self.fig_format, dpi=self.fig_dpi)

                    rcParams["figure.subplot.left"] = 0.48
                    rcParams["figure.subplot.right"] = 0.70
                    rcParams["figure.subplot.top"] = 0.9
                    rcParams["figure.subplot.bottom"] = 0.1

                    # plot human GO terms
                    fig, ax = gseapy.barplot(enr_res_human_results_top, column="Adjusted P-value", group='Gene_set',
                                             color=color_list,
                                             # color=self.mouse_color,
                                             # set group, so you could do a multi-sample/library comparsion
                                             size=5, title=f"GO of Human {h_region_name} module {g_module_id} DEGs",
                                             figsize=plot_size, cutoff=go_p_cut)  # top_term=top_terms_plot
                    ax.get_legend().set_title("Gene sets")
                    # ax.get_legend().set_fontsize(MEDIUM_SIZE)
                    # ax.get_title().set_fontsize(MEDIUM_SIZE)
                    # ax.get_legend().set_loc('lower center')
                    # fig.axes.append(ax)
                    fig.savefig(save_path_region_h + f'm{g_module_id}_barplot.' + self.fig_format,
                                format=self.fig_format, dpi=self.fig_dpi)

                    rcParams["figure.subplot.left"] = 0.6
                    rcParams["figure.subplot.right"] = 0.88
                    rcParams["figure.subplot.bottom"] = 0.25  # 0.15
                    rcParams["figure.subplot.top"] = 0.95

                    fig, ax = gseapy.dotplot(enr_res_human_results_top,
                                             column="Adjusted P-value",
                                             x='Gene_set',
                                             # set x axis, so you could do a multi-sample/library comparsion
                                             size=10,
                                             # top_term=5,  #
                                             figsize=plot_size,
                                             title=f"GO of Human {h_region_name} module {g_module_id} DEGs",
                                             xticklabels_rot=45,
                                             # rotate xtick labels show_ring=True, set to False to revmove outer ring
                                             marker='o',
                                             cmap=plt.cm.winter_r,
                                             # ofname=save_path_region + f'module_{g_module_id}_botplot.' + self.fig_format,
                                             format=self.fig_format, cutoff=go_p_cut)  # cutoff=go_p_cut top_term=top_terms_plot
                    # cbar.set_label(size=MEDIUM_SIZE, labelpad=15)
                    ax.set_xlabel('')
                    fig.savefig(save_path_region_h + f'm{g_module_id}_botplot.' + self.fig_format,
                                format=self.fig_format, dpi=self.fig_dpi)


                    ##############################################################################
                    # Speciel GO terms for each species
                    # fig, axes = plt.subplot()

                    rcParams["figure.subplot.left"] = 0.48
                    rcParams["figure.subplot.right"] = 0.70
                    rcParams["figure.subplot.top"] = 0.9
                    rcParams["figure.subplot.bottom"] = 0.1

                    rcParams["legend.title_fontsize"] = MEDIUM_SIZE
                    rcParams["axes.titlesize"] = MEDIUM_SIZE
                    fig, ax = gseapy.barplot(enr_res_mouse_results_special, column="Adjusted P-value", group='Gene_set',
                                             color=color_list,
                                             # color=self.mouse_color,
                                             # set group, so you could do a multi-sample/library comparsion
                                             size=5, title=f"GO of Mouse {m_region_name} module {g_module_id} DEGs",
                                             figsize=plot_size, cutoff=go_p_cut)  # top_term=top_terms_plot
                    ax.get_legend().set_title("Gene sets")
                    # ax.get_legend().set_fontsize(MEDIUM_SIZE)
                    # ax.get_title().set_fontsize(MEDIUM_SIZE)
                    # ax.get_legend().set_loc('lower center')
                    # fig.axes.append(ax)
                    fig.savefig(save_path_region_m_special + f'm{g_module_id}_barplot.' + self.fig_format,
                                format=self.fig_format, dpi=self.fig_dpi)
                    # rcParams["figure.subplot.left"] = 0.1
                    # rcParams["figure.subplot.right"] = 0.9
                    rcParams["figure.subplot.left"] = 0.6
                    rcParams["figure.subplot.right"] = 0.88
                    rcParams["figure.subplot.bottom"] = 0.25  # 0.15
                    rcParams["figure.subplot.top"] = 0.95

                    fig, ax = gseapy.dotplot(enr_res_mouse_results_special,
                                             column="Adjusted P-value",
                                             x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion
                                             size=10,
                                             # top_term=5,  #
                                             figsize=plot_size,
                                             title=f"GO of Mouse {m_region_name} module {g_module_id} DEGs",
                                             xticklabels_rot=45,
                                             # rotate xtick labels show_ring=True, set to False to revmove outer ring
                                             marker='o',
                                             cmap=plt.cm.winter_r,
                                             format=self.fig_format, cutoff=go_p_cut)  # top_term=top_terms_plot
                    # cbar = fig.colorbar(ax=ax)
                    # im = ax.images
                    # print(len(im))
                    # print(im)
                    # # Assume colorbar was plotted last one plotted last
                    # cbar = im.colorbar
                    # cbar.set_label(size=MEDIUM_SIZE, labelpad=15)
                    ax.set_xlabel('')
                    fig.savefig(save_path_region_m_special + f'm{g_module_id}_botplot.' + self.fig_format,
                                format=self.fig_format, dpi=self.fig_dpi)

                    rcParams["figure.subplot.left"] = 0.48
                    rcParams["figure.subplot.right"] = 0.70
                    rcParams["figure.subplot.top"] = 0.9
                    rcParams["figure.subplot.bottom"] = 0.1

                    # plot human GO terms
                    fig, ax = gseapy.barplot(enr_res_human_results_special, column="Adjusted P-value", group='Gene_set',
                                             color=color_list,
                                             # color=self.mouse_color,
                                             # set group, so you could do a multi-sample/library comparsion
                                             size=5, title=f"GO of Human {h_region_name} module {g_module_id} DEGs",
                                             figsize=plot_size, cutoff=go_p_cut)  # top_term=top_terms_plot
                    ax.get_legend().set_title("Gene sets")
                    # ax.get_legend().set_fontsize(MEDIUM_SIZE)
                    # ax.get_title().set_fontsize(MEDIUM_SIZE)
                    # ax.get_legend().set_loc('lower center')
                    # fig.axes.append(ax)
                    fig.savefig(save_path_region_h_special + f'm{g_module_id}_barplot.' + self.fig_format,
                                format=self.fig_format, dpi=self.fig_dpi)

                    rcParams["figure.subplot.left"] = 0.6
                    rcParams["figure.subplot.right"] = 0.88
                    rcParams["figure.subplot.bottom"] = 0.25  # 0.15
                    rcParams["figure.subplot.top"] = 0.95

                    fig, ax = gseapy.dotplot(enr_res_human_results_special,
                                             column="Adjusted P-value",
                                             x='Gene_set',
                                             # set x axis, so you could do a multi-sample/library comparsion
                                             size=10,
                                             # top_term=5,  #
                                             figsize=plot_size,
                                             title=f"GO of Human {h_region_name} module {g_module_id} DEGs",
                                             xticklabels_rot=45,
                                             # rotate xtick labels show_ring=True, set to False to revmove outer ring
                                             marker='o',
                                             cmap=plt.cm.winter_r,
                                             # ofname=save_path_region + f'module_{g_module_id}_botplot.' + self.fig_format,
                                             format=self.fig_format,
                                             cutoff=go_p_cut)  # cutoff=go_p_cut top_term=top_terms_plot
                    # cbar.set_label(size=MEDIUM_SIZE, labelpad=15)
                    ax.set_xlabel('')
                    fig.savefig(save_path_region_h_special + f'm{g_module_id}_botplot.' + self.fig_format,
                                format=self.fig_format, dpi=self.fig_dpi)


                current_r_id = r_id
                np.savez(save_path + 'homoregion_go_terms_overlap_array.npz',
                         current_r_id=current_r_id,
                         homoregion_go_terms_overlap_array=homoregion_go_terms_overlap_array)

            sys.stdout = original_stdout  # Reset the standard output to its original value

        homoregion_go_terms_overlap_array = homoregion_go_terms_overlap_array - np.reshape(
            np.mean(homoregion_go_terms_overlap_array, axis=1), (-1, 1))

        homoregion_go_terms_overlap_array = homoregion_go_terms_overlap_array - np.reshape(
            np.mean(homoregion_go_terms_overlap_array, axis=0), (1, -1))

        homoregion_go_terms_overlap_array = preprocess.zscore(homoregion_go_terms_overlap_array)

        diff_gene_module_exp_df = pd.DataFrame(
            homoregion_go_terms_overlap_array,
            index=[x + '-' + y for x, y in zip(mouse_homo_region_acronym_list, human_homo_region_acronym_list)],
            columns=module_labels_unique)

        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.5
        fig, ax = plt.subplots(figsize=(6.5, 8), dpi=self.fig_dpi)

        hm = sns.heatmap(diff_gene_module_exp_df, square=False, cbar_kws={'location': 'right'},
                         cmap='Purples', #'Spectral_r',  # "YlGnBu",
                         ax=ax,
                         xticklabels=True, yticklabels=True, linewidths=0.5, linecolor='lightgrey')
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.legend([],[], frameon=False)
        # plt.xticks(rotation=45)
        plt.ylabel('Gene module')
        plt.savefig(save_path + 'Go_terms_overlap.' + self.fig_format, format=self.fig_format)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1


        return None


    def experiment_8_1_plot_heatmap(self):

        sns.set(style='white')
        TINY_SIZE = 10  # 39
        SMALL_SIZE = 10  # 42
        MEDIUM_SIZE = 12  # 46
        BIGGER_SIZE = 12  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        # save_path
        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/8_experiment_homoregions_go_terms_overlap/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # -----------homologous regions----------------------------------------
        # module information
        module_name = 'module'
        module_num = len(Counter(self.adata_gene_embedding.obs[module_name]))
        gene_names = self.adata_gene_embedding.obs_names.values
        module_labels = self.adata_gene_embedding.obs[module_name].values

        module_labels_unique = list(set(self.adata_gene_embedding.obs[module_name].values))
        module_labels_unique.sort(key=lambda x: int(x))
        # print('module_labels.shape', module_labels.shape)
        gene_module_dict = {k: v for k, v in zip(gene_names, module_labels)}

        mouse_gene_names = self.adata_mouse_gene_embedding.obs_names.values
        mouse_module_labels = self.adata_mouse_gene_embedding.obs[module_name].values
        mouse_gene_module_dict = {k: v for k, v in zip(mouse_gene_names, mouse_module_labels)}
        # human gene_name to module
        human_gene_names = self.adata_human_gene_embedding.obs_names.values
        human_module_labels = self.adata_human_gene_embedding.obs[module_name].values
        human_gene_module_dict = {k: v for k, v in zip(human_gene_names, human_module_labels)}

        # homologous regions
        human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        homo_region_dict = OrderedDict()
        for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
            homo_region_dict[x] = y
        mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        mouse_homo_region_list = human_mouse_homo_region['Mouse'].values
        human_homo_set = set(human_mouse_homo_region['Human'].values)
        human_homo_region_list = human_mouse_homo_region['Human'].values

        mouse_homo_region_acronym_list = [self.mouse_64_acronym_dict[x] for x in mouse_homo_region_list]
        human_homo_region_acronym_list = [self.human_88_acronym_dict[x] for x in human_homo_region_list]


        if os.path.exists(save_path + 'homoregion_go_terms_overlap_array.npz'):
            npzfile = np.load(save_path + 'homoregion_go_terms_overlap_array.npz')
            homoregion_go_terms_overlap_array = npzfile['homoregion_go_terms_overlap_array']
            current_r_id = npzfile['current_r_id']

        #homoregion_go_terms_overlap_array = preprocess.zscore(homoregion_go_terms_overlap_array)

        homoregion_go_terms_overlap_array = homoregion_go_terms_overlap_array - np.reshape(
            np.mean(homoregion_go_terms_overlap_array, axis=1), (-1, 1))
        #
        homoregion_go_terms_overlap_array = homoregion_go_terms_overlap_array - np.reshape(
            np.mean(homoregion_go_terms_overlap_array, axis=0), (1, -1))
        #
        homoregion_go_terms_overlap_array = preprocess.zscore(homoregion_go_terms_overlap_array)

        diff_gene_module_exp_df = pd.DataFrame(
            homoregion_go_terms_overlap_array,
            index=[x + '-' + y for x, y in zip(mouse_homo_region_acronym_list, human_homo_region_acronym_list)],
            columns=module_labels_unique)

        rcParams["figure.subplot.left"] = 0.45
        rcParams["figure.subplot.right"] = 0.88
        rcParams["figure.subplot.top"] = 0.97
        rcParams["figure.subplot.bottom"] = 0.15
        fig, ax = plt.subplots(figsize=(2.5, 3.6), dpi=self.fig_dpi)

        c = Colormap()
        color_map = c.cmap_linear('white', 'white', 'indigo')

        hm = sns.heatmap(diff_gene_module_exp_df, square=False, cbar_kws={'location': 'top', 'label':'GO term overlap index'},
                         cmap=color_map,#'Purples',  # 'Spectral_r',  # "YlGnBu",
                         ax=ax,
                            #col_cluster=False,
                            #figsize=(2.5, 3.5),
                         xticklabels=True, yticklabels=True, linewidths=0.2, linecolor='lightgrey')
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.legend([],[], frameon=False)
        # plt.xticks(rotation=45)
        plt.ylabel('Mouse-human region pairs')
        plt.xlabel('Gene module')
        plt.savefig(save_path + 'Go_terms_overlap.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1


    def experiment_9_clusters_go_terms_overlap(self):
        """
        Check the overlap of GO terms for two species on gene modules.

        1. Acquire DEGs for each cluster on gene modules for each species;
        2. Get the GO terms below a p value threshold;
        3. Get the overlap set of GO terms for each pair of (mouse cluster, human cluster, gene module);
            Overlap ratio = (Mouse Unique GO terms in the overlap set / Unique Mouse GO terms with Adjusted P < p-threshold)  +
             (Unique GO terms in the overlap set / Unique Human GO terms with Adjusted P < p-threshold) / 2
            Overlap ratio mouse = Mouse Unique GO terms in the overlap set / Unique Mouse GO terms with Adjusted P < p-threshold
        4.  Plot the heatmap of overlap ratio: x-regions, y-gene module, color-ratio values

        :return:None
        """


        sns.set(style='white')
        TINY_SIZE = 18  # 39
        SMALL_SIZE = 18  # 42
        MEDIUM_SIZE = 26  # 46
        BIGGER_SIZE = 26  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        # save_path
        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/9_experiment_clusters_go_terms_overlap/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # -----------homologous regions----------------------------------------
        # module information
        module_name = 'module'
        module_num = len(Counter(self.adata_gene_embedding.obs[module_name]))
        gene_names = self.adata_gene_embedding.obs_names.values
        module_labels = self.adata_gene_embedding.obs[module_name].values

        module_labels_unique = list(set(self.adata_gene_embedding.obs[module_name].values))
        module_labels_unique.sort(key=lambda x: int(x))
        #print('module_labels.shape', module_labels.shape)
        gene_module_dict = {k: v for k, v in zip(gene_names, module_labels)}

        mouse_gene_names = self.adata_mouse_gene_embedding.obs_names.values
        mouse_module_labels = self.adata_mouse_gene_embedding.obs[module_name].values
        mouse_gene_module_dict = {k: v for k, v in zip(mouse_gene_names, mouse_module_labels)}
        # human gene_name to module
        human_gene_names = self.adata_human_gene_embedding.obs_names.values
        human_module_labels = self.adata_human_gene_embedding.obs[module_name].values
        human_gene_module_dict = {k: v for k, v in zip(human_gene_names, human_module_labels)}

        # homologous regions
        human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        homo_region_dict = OrderedDict()
        for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
            homo_region_dict[x] = y
        mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        mouse_homo_region_list = human_mouse_homo_region['Mouse'].values
        human_homo_set = set(human_mouse_homo_region['Human'].values)
        human_homo_region_list = human_mouse_homo_region['Human'].values

        mouse_homo_region_acronym_list = [self.mouse_64_acronym_dict[x] for x in mouse_homo_region_list]
        human_homo_region_acronym_list = [self.human_88_acronym_dict[x] for x in human_homo_region_list]

        cluster_name_unique = sorted(Counter(self.adata_embedding.obs['cluster_name_acronym']).keys(),
                                     key=lambda t: int(t.split('-')[0]))

        # ----------------------gene ontology analysis-------------------------------
        ntop_gene = 200
        p_cut = 5e-2
        go_p_cut = 5e-2
        top_term = 2
        color_list = list(sns.color_palette(cc.glasbey, n_colors=8))

        top_terms_plot = 10

        mouse_go_gene_sets = ['Allen_Brain_Atlas_down',
                              'Allen_Brain_Atlas_up',
                              'Azimuth_Cell_Types_2021',
                              'CellMarker_Augmented_2021',
                              'GO_Biological_Process_2021',
                              'GO_Cellular_Component_2021',
                              'GO_Molecular_Function_2021',
                              'Mouse_Gene_Atlas']
        human_go_gene_sets = ['Allen_Brain_Atlas_down', 'Allen_Brain_Atlas_up',
                              'Azimuth_Cell_Types_2021', 'CellMarker_Augmented_2021',
                              'GO_Biological_Process_2021', 'GO_Cellular_Component_2021',
                              'GO_Molecular_Function_2021', 'Human_Gene_Atlas']

        sc.tl.rank_genes_groups(self.adata_mouse_exp, groupby='cluster_name_acronym', method='wilcoxon',
                                key_added="wilcoxon")

        adata_human_exp = self.adata_human_exp.copy()
        cluster_counts = adata_human_exp.obs['cluster_name_acronym'].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        adata_human_exp = adata_human_exp[adata_human_exp.obs['cluster_name_acronym'].isin(keep)].copy()

        sc.tl.rank_genes_groups(adata_human_exp, groupby='cluster_name_acronym', method='wilcoxon',
                                key_added="wilcoxon")

        # mouse_homo_region_list = ['Field CA3', 'Field CA2', 'Field CA1']
        homoregion_go_terms_overlap_array = np.zeros((len(cluster_name_unique), len(module_labels_unique)))

        sns.set(style='white')
        TINY_SIZE = 15  # 39
        SMALL_SIZE = 15  # 42
        MEDIUM_SIZE = 18  # 46
        BIGGER_SIZE = 18  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        # plt.rc('title', titlesize=BIGGER_SIZE)
        # rcParams["legend.title_fontsize"]

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        current_r_id = 0

        if os.path.exists(save_path + 'homoregion_go_terms_overlap_array.npz'):
            npzfile = np.load(save_path + 'homoregion_go_terms_overlap_array.npz')
            homoregion_go_terms_overlap_array = npzfile['homoregion_go_terms_overlap_array']
            current_r_id = npzfile['current_r_id']

        original_stdout = sys.stdout  # Save a reference to the original standard output

        with open(save_path + f'mouse_human_region_GSEA.txt', 'w') as f:
            sys.stdout = f  # Change the standard output to the file we created.

            for c_id, cluster_name in zip(range(len(cluster_name_unique)), cluster_name_unique):

                if c_id < current_r_id:
                    continue

                sc.pl.rank_genes_groups(self.adata_mouse_exp, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                        show=False)
                glist_mouse_region_genes = sc.get.rank_genes_groups_df(self.adata_mouse_exp,
                                                                       group=cluster_name,
                                                                       key='wilcoxon',
                                                                       # log2fc_min=0.25,
                                                                       pval_cutoff=p_cut)['names'].squeeze().str.strip().tolist()
                print('length of glist_mouse_region_genes', len(glist_mouse_region_genes))

                module_list_region_m = [mouse_gene_module_dict[g] for g in glist_mouse_region_genes]

                module_set_unique_m = list(set(module_list_region_m))
                module_set_unique_m.sort()

                save_path_region_m = save_path + 'Mouse/' + '_'.join(cluster_name.split(' ')) + '/'
                if not os.path.exists(save_path_region_m):
                    os.makedirs(save_path_region_m)

                save_path_region_m_special = save_path + 'Mouse/' + '_'.join(cluster_name.split(' ')) + '_special/'
                if not os.path.exists(save_path_region_m_special):
                    os.makedirs(save_path_region_m_special)



                sc.pl.rank_genes_groups(adata_human_exp, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                        show=False)
                glist_human_region_genes = sc.get.rank_genes_groups_df(adata_human_exp,
                                                                       group=cluster_name,
                                                                       key='wilcoxon',
                                                                       # log2fc_min=0.25,
                                                                       pval_cutoff=p_cut)[
                    'names'].squeeze().str.strip().tolist()
                print('length of glist_human_region_genes', len(glist_human_region_genes))
                module_list_region_h = [human_gene_module_dict[g] for g in glist_human_region_genes]

                module_set_unique = list(set(module_list_region_h))
                module_set_unique.sort()

                save_path_region_h = save_path + 'Human/' + '_'.join(cluster_name.split(' ')) + '/'
                if not os.path.exists(save_path_region_h):
                    os.makedirs(save_path_region_h)

                save_path_region_h_special = save_path + 'Human/' + '_'.join(cluster_name.split(' ')) + '_special/'
                if not os.path.exists(save_path_region_h_special):
                    os.makedirs(save_path_region_h_special)

                module_genelist_dict_m = {}
                module_genelist_dict_h = {}
                for module_id, g_module_id in zip(range(len(module_set_unique)), module_set_unique):

                    print('cluster:', c_id, cluster_name, 'gene module:', module_id, g_module_id)

                    module_genelist_dict_m[g_module_id] = []
                    for m, g in zip(module_list_region_m, glist_mouse_region_genes):
                        if m == g_module_id:
                            module_genelist_dict_m[g_module_id].append(g)

                    go_gene_list_m = module_genelist_dict_m[g_module_id]

                    enr_res_m = gseapy.enrichr(gene_list=go_gene_list_m,
                                               organism='Mouse',
                                               gene_sets=mouse_go_gene_sets#, cutoff=go_p_cut
                                               )  # cutoff=go_p_cut


                    module_genelist_dict_h[g_module_id] = []
                    for h, g in zip(module_list_region_h, glist_human_region_genes):
                        if h == g_module_id:
                            module_genelist_dict_h[g_module_id].append(g)

                    go_gene_list_h = module_genelist_dict_h[g_module_id]

                    enr_res_h = gseapy.enrichr(gene_list=go_gene_list_h,
                                             organism='Human',
                                             gene_sets=human_go_gene_sets#, cutoff=go_p_cut
                                             )  # top_term=top_term

                    enr_res_mouse_results = enr_res_m.results[
                        enr_res_m.results['Adjusted P-value'] <= go_p_cut]  # .columns
                    #print(enr_res_mouse_results)
                    enr_res_human_results = enr_res_h.results[
                        enr_res_h.results['Adjusted P-value'] <= go_p_cut]  # .columns
                    #print(enr_res_human_results)

                    #print('enr_res_mouse go term numbers = ', enr_res_mouse_results.shape[0])
                    #print('enr_res_human go term numbers = ', enr_res_human_results.shape[0])

                    Overlapped_term_num = 0

                    enr_res_mouse_results_term = enr_res_mouse_results['Term'].tolist()
                    enr_res_human_results_term = enr_res_human_results['Term'].tolist()

                    Overlapped_term_set = []
                    for i in range(enr_res_mouse_results.shape[0]):
                        for j in range(enr_res_human_results.shape[0]):
                            if enr_res_mouse_results_term[i] == enr_res_human_results_term[j]:
                                #print(enr_res_mouse_results_term[i], enr_res_human_results_term[j])
                                Overlapped_term_set.append(enr_res_mouse_results_term[i])
                                Overlapped_term_num += 1
                    #print(Overlapped_term_num)
                    Overlapped_term_set_list = list(Overlapped_term_set)
                    print('Overlapped_term_set_list', Overlapped_term_set_list)
                    enr_res_mouse_results_special = enr_res_mouse_results[enr_res_mouse_results['Term'].isin(list(set(enr_res_mouse_results_term) -  set(Overlapped_term_set)))]
                    enr_res_human_results_special = enr_res_human_results[enr_res_human_results['Term'].isin(list(set(enr_res_human_results_term) -  set(Overlapped_term_set)))]

                    if len(list(set(enr_res_mouse_results_term))) != 0 and len(list(set(enr_res_human_results_term))) != 0:
                        overlap_ratio = (len(list(set(Overlapped_term_set))) / len(
                            list(set(enr_res_mouse_results_term)))) * 1 / 2 + \
                                        (len(list(set(Overlapped_term_set))) / len(
                                            list(set(enr_res_human_results_term)))) * 1 / 2

                    elif len(list(set(enr_res_mouse_results_term))) == 0:
                        overlap_ratio = len(list(set(Overlapped_term_set))) / len(
                            list(set(enr_res_human_results_term))) * 1 / 2

                    elif len(list(set(enr_res_human_results_term))) == 0:
                        overlap_ratio = len(list(set(Overlapped_term_set))) / len(
                            list(set(enr_res_mouse_results_term))) * 1 / 2
                    else:
                        overlap_ratio = 0

                    homoregion_go_terms_overlap_array[c_id, module_id] = overlap_ratio



                    enr_res_mouse_results_top, enr_res_human_results_top, if_empty = get_top_same_terms(enr_res_mouse_results,
                                                                                              enr_res_human_results, cutoff=go_p_cut) #top_terms = top_terms_plot
                    if if_empty:
                        continue

                    # plot common GO terms
                    # fig, axes = plt.subplot()
                    rcParams["figure.subplot.left"] = 0.52
                    rcParams["figure.subplot.right"] = 0.70
                    rcParams["figure.subplot.top"] = 0.9
                    rcParams["figure.subplot.bottom"] = 0.1

                    rcParams["legend.title_fontsize"] = MEDIUM_SIZE
                    rcParams["axes.titlesize"] = MEDIUM_SIZE
                    fig, ax = gseapy.barplot(enr_res_mouse_results_top, column="Adjusted P-value", group='Gene_set', color=color_list,
                                             # color=self.mouse_color,
                                             # set group, so you could do a multi-sample/library comparsion
                                             size=5, title=f"GO of Mouse {cluster_name} module {g_module_id} DEGs",
                                             figsize=(15, 8), cutoff=go_p_cut)  # top_term=top_terms_plot
                    ax.get_legend().set_title("Gene sets")
                    # ax.get_legend().set_fontsize(MEDIUM_SIZE)
                    # ax.get_title().set_fontsize(MEDIUM_SIZE)
                    # ax.get_legend().set_loc('lower center')
                    # fig.axes.append(ax)
                    fig.savefig(save_path_region_m + f'module_{g_module_id}_barplot.' + self.fig_format,
                                format=self.fig_format, dpi=self.fig_dpi)
                    # rcParams["figure.subplot.left"] = 0.1
                    # rcParams["figure.subplot.right"] = 0.9
                    rcParams["figure.subplot.left"] = 0.6
                    rcParams["figure.subplot.right"] = 0.88
                    rcParams["figure.subplot.bottom"] = 0.25  # 0.15
                    rcParams["figure.subplot.top"] = 0.95

                    fig, ax = gseapy.dotplot(enr_res_mouse_results_top,
                                             column="Adjusted P-value",
                                             x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion
                                             size=10,
                                             # top_term=5,  #
                                             figsize=(15, 10),
                                             title=f"GO of Mouse {cluster_name} module {g_module_id} DEGs",
                                             xticklabels_rot=45,
                                             # rotate xtick labels show_ring=True, set to False to revmove outer ring
                                             marker='o',
                                             cmap=plt.cm.winter_r,
                                             format=self.fig_format, cutoff=go_p_cut)  # top_term=top_terms_plot
                    # cbar = fig.colorbar(ax=ax)
                    # im = ax.images
                    # print(len(im))
                    # print(im)
                    # # Assume colorbar was plotted last one plotted last
                    # cbar = im.colorbar
                    # cbar.set_label(size=MEDIUM_SIZE, labelpad=15)
                    ax.set_xlabel('')
                    fig.savefig(save_path_region_m + f'module_{g_module_id}_botplot.' + self.fig_format,
                                format=self.fig_format, dpi=self.fig_dpi)

                    rcParams["figure.subplot.left"] = 0.52
                    rcParams["figure.subplot.right"] = 0.70
                    rcParams["figure.subplot.top"] = 0.9
                    rcParams["figure.subplot.bottom"] = 0.1

                    # plot human GO terms
                    fig, ax = gseapy.barplot(enr_res_human_results_top, column="Adjusted P-value", group='Gene_set',
                                             color=color_list,
                                             # color=self.mouse_color,
                                             # set group, so you could do a multi-sample/library comparsion
                                             size=5, title=f"GO of Human {cluster_name} module {g_module_id} DEGs",
                                             figsize=(15, 8), cutoff=go_p_cut)  # top_term=top_terms_plot
                    ax.get_legend().set_title("Gene sets")
                    # ax.get_legend().set_fontsize(MEDIUM_SIZE)
                    # ax.get_title().set_fontsize(MEDIUM_SIZE)
                    # ax.get_legend().set_loc('lower center')
                    # fig.axes.append(ax)
                    fig.savefig(save_path_region_h + f'module_{g_module_id}_barplot.' + self.fig_format,
                                format=self.fig_format, dpi=self.fig_dpi)

                    rcParams["figure.subplot.left"] = 0.6
                    rcParams["figure.subplot.right"] = 0.88
                    rcParams["figure.subplot.bottom"] = 0.25  # 0.15
                    rcParams["figure.subplot.top"] = 0.95

                    fig, ax = gseapy.dotplot(enr_res_human_results_top,
                                             column="Adjusted P-value",
                                             x='Gene_set',
                                             # set x axis, so you could do a multi-sample/library comparsion
                                             size=10,
                                             # top_term=5,  #
                                             figsize=(15, 10),
                                             title=f"GO of Human {cluster_name} module {g_module_id} DEGs",
                                             xticklabels_rot=45,
                                             # rotate xtick labels show_ring=True, set to False to revmove outer ring
                                             marker='o',
                                             cmap=plt.cm.winter_r,
                                             # ofname=save_path_region + f'module_{g_module_id}_botplot.' + self.fig_format,
                                             format=self.fig_format, cutoff=go_p_cut)  # cutoff=go_p_cut top_term=top_terms_plot
                    # cbar.set_label(size=MEDIUM_SIZE, labelpad=15)
                    ax.set_xlabel('')
                    fig.savefig(save_path_region_h + f'module_{g_module_id}_botplot.' + self.fig_format,
                                format=self.fig_format, dpi=self.fig_dpi)


                    ##############################################################################
                    # Speciel GO terms for each species
                    # fig, axes = plt.subplot()

                    rcParams["figure.subplot.left"] = 0.52
                    rcParams["figure.subplot.right"] = 0.70
                    rcParams["figure.subplot.top"] = 0.9
                    rcParams["figure.subplot.bottom"] = 0.1

                    rcParams["legend.title_fontsize"] = MEDIUM_SIZE
                    rcParams["axes.titlesize"] = MEDIUM_SIZE
                    fig, ax = gseapy.barplot(enr_res_mouse_results_special, column="Adjusted P-value", group='Gene_set',
                                             color=color_list,
                                             # color=self.mouse_color,
                                             # set group, so you could do a multi-sample/library comparsion
                                             size=5, title=f"GO of Mouse {cluster_name} module {g_module_id} DEGs",
                                             figsize=(15, 8), cutoff=go_p_cut)  # top_term=top_terms_plot
                    ax.get_legend().set_title("Gene sets")
                    # ax.get_legend().set_fontsize(MEDIUM_SIZE)
                    # ax.get_title().set_fontsize(MEDIUM_SIZE)
                    # ax.get_legend().set_loc('lower center')
                    # fig.axes.append(ax)
                    ##fig.savefig(save_path_region_m_special + f'module_{g_module_id}_barplot.' + self.fig_format,
                    ##            format=self.fig_format, dpi=self.fig_dpi)
                    # rcParams["figure.subplot.left"] = 0.1
                    # rcParams["figure.subplot.right"] = 0.9
                    rcParams["figure.subplot.left"] = 0.6
                    rcParams["figure.subplot.right"] = 0.88
                    rcParams["figure.subplot.bottom"] = 0.25  # 0.15
                    rcParams["figure.subplot.top"] = 0.95

                    fig, ax = gseapy.dotplot(enr_res_mouse_results_special,
                                             column="Adjusted P-value",
                                             x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion
                                             size=10,
                                             # top_term=5,  #
                                             figsize=(15, 10),
                                             title=f"GO of Mouse {cluster_name} module {g_module_id} DEGs",
                                             xticklabels_rot=45,
                                             # rotate xtick labels show_ring=True, set to False to revmove outer ring
                                             marker='o',
                                             cmap=plt.cm.winter_r,
                                             format=self.fig_format, cutoff=go_p_cut)  # top_term=top_terms_plot
                    # cbar = fig.colorbar(ax=ax)
                    # im = ax.images
                    # print(len(im))
                    # print(im)
                    # # Assume colorbar was plotted last one plotted last
                    # cbar = im.colorbar
                    # cbar.set_label(size=MEDIUM_SIZE, labelpad=15)
                    ax.set_xlabel('')
                    ##fig.savefig(save_path_region_m_special + f'module_{g_module_id}_botplot.' + self.fig_format,
                    ##            format=self.fig_format, dpi=self.fig_dpi)

                    rcParams["figure.subplot.left"] = 0.52
                    rcParams["figure.subplot.right"] = 0.70
                    rcParams["figure.subplot.top"] = 0.9
                    rcParams["figure.subplot.bottom"] = 0.1

                    # plot human GO terms
                    fig, ax = gseapy.barplot(enr_res_human_results_special, column="Adjusted P-value", group='Gene_set',
                                             color=color_list,
                                             # color=self.mouse_color,
                                             # set group, so you could do a multi-sample/library comparsion
                                             size=5, title=f"GO of Human {cluster_name} module {g_module_id} DEGs",
                                             figsize=(15, 8), cutoff=go_p_cut)  # top_term=top_terms_plot
                    ax.get_legend().set_title("Gene sets")
                    # ax.get_legend().set_fontsize(MEDIUM_SIZE)
                    # ax.get_title().set_fontsize(MEDIUM_SIZE)
                    # ax.get_legend().set_loc('lower center')
                    # fig.axes.append(ax)
                    ##fig.savefig(save_path_region_h_special + f'module_{g_module_id}_barplot.' + self.fig_format,
                    ##            format=self.fig_format, dpi=self.fig_dpi)

                    rcParams["figure.subplot.left"] = 0.6
                    rcParams["figure.subplot.right"] = 0.88
                    rcParams["figure.subplot.bottom"] = 0.25  # 0.15
                    rcParams["figure.subplot.top"] = 0.95

                    fig, ax = gseapy.dotplot(enr_res_human_results_special,
                                             column="Adjusted P-value",
                                             x='Gene_set',
                                             # set x axis, so you could do a multi-sample/library comparsion
                                             size=10,
                                             # top_term=5,  #
                                             figsize=(15, 10),
                                             title=f"GO of Human {cluster_name} module {g_module_id} DEGs",
                                             xticklabels_rot=45,
                                             # rotate xtick labels show_ring=True, set to False to revmove outer ring
                                             marker='o',
                                             cmap=plt.cm.winter_r,
                                             # ofname=save_path_region + f'module_{g_module_id}_botplot.' + self.fig_format,
                                             format=self.fig_format,
                                             cutoff=go_p_cut)  # cutoff=go_p_cut top_term=top_terms_plot
                    # cbar.set_label(size=MEDIUM_SIZE, labelpad=15)
                    ax.set_xlabel('')
                    ##fig.savefig(save_path_region_h_special + f'module_{g_module_id}_botplot.' + self.fig_format,
                    ##            format=self.fig_format, dpi=self.fig_dpi)


                current_r_id = c_id
                np.savez(save_path + 'homoregion_go_terms_overlap_array.npz',
                         current_r_id=current_r_id,
                         homoregion_go_terms_overlap_array=homoregion_go_terms_overlap_array)

            sys.stdout = original_stdout  # Reset the standard output to its original value

        homoregion_go_terms_overlap_array = homoregion_go_terms_overlap_array - np.reshape(
            np.mean(homoregion_go_terms_overlap_array, axis=1), (-1, 1))

        homoregion_go_terms_overlap_array = homoregion_go_terms_overlap_array - np.reshape(
            np.mean(homoregion_go_terms_overlap_array, axis=0), (1, -1))

        homoregion_go_terms_overlap_array = preprocess.zscore(homoregion_go_terms_overlap_array)

        diff_gene_module_exp_df = pd.DataFrame(
            homoregion_go_terms_overlap_array.T,
            index=module_labels_unique,
            columns=cluster_name_unique)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.3
        fig, ax = plt.subplots(figsize=(25, 7), dpi=self.fig_dpi)

        c = Colormap()
        color_map = c.cmap_linear('white', 'white', 'indigo')

        hm = sns.heatmap(diff_gene_module_exp_df, square=False, cbar_kws={'location': 'right'},
                         cmap=color_map, #'Spectral_r',  # "YlGnBu",
                         ax=ax,
                         xticklabels=True, yticklabels=True, linewidths=0.5, linecolor='lightgrey')
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.legend([],[], frameon=False)
        # plt.xticks(rotation=45)
        plt.ylabel('Gene module')
        plt.savefig(save_path + 'Go_terms_overlap.' + self.fig_format, format=self.fig_format)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1

        return None


    def experiment_9_1_plot_heatmap(self):

        sns.set(style='white')
        TINY_SIZE = 10  # 39
        SMALL_SIZE = 10  # 42
        MEDIUM_SIZE = 12  # 46
        BIGGER_SIZE = 12  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        # save_path
        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/9_experiment_clusters_go_terms_overlap/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # -----------homologous regions----------------------------------------
        # module information
        module_name = 'module'
        module_num = len(Counter(self.adata_gene_embedding.obs[module_name]))
        gene_names = self.adata_gene_embedding.obs_names.values
        module_labels = self.adata_gene_embedding.obs[module_name].values

        module_labels_unique = list(set(self.adata_gene_embedding.obs[module_name].values))
        module_labels_unique.sort(key=lambda x: int(x))
        # print('module_labels.shape', module_labels.shape)
        gene_module_dict = {k: v for k, v in zip(gene_names, module_labels)}

        mouse_gene_names = self.adata_mouse_gene_embedding.obs_names.values
        mouse_module_labels = self.adata_mouse_gene_embedding.obs[module_name].values
        mouse_gene_module_dict = {k: v for k, v in zip(mouse_gene_names, mouse_module_labels)}
        # human gene_name to module
        human_gene_names = self.adata_human_gene_embedding.obs_names.values
        human_module_labels = self.adata_human_gene_embedding.obs[module_name].values
        human_gene_module_dict = {k: v for k, v in zip(human_gene_names, human_module_labels)}

        # homologous regions
        human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        homo_region_dict = OrderedDict()
        for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
            homo_region_dict[x] = y
        mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        mouse_homo_region_list = human_mouse_homo_region['Mouse'].values
        human_homo_set = set(human_mouse_homo_region['Human'].values)
        human_homo_region_list = human_mouse_homo_region['Human'].values

        mouse_homo_region_acronym_list = [self.mouse_64_acronym_dict[x] for x in mouse_homo_region_list]
        human_homo_region_acronym_list = [self.human_88_acronym_dict[x] for x in human_homo_region_list]

        cluster_name_unique = sorted(Counter(self.adata_embedding.obs['cluster_name_acronym']).keys(),
                                     key=lambda t: int(t.split('-')[0]))


        if os.path.exists(save_path + 'homoregion_go_terms_overlap_array.npz'):
            npzfile = np.load(save_path + 'homoregion_go_terms_overlap_array.npz')
            homoregion_go_terms_overlap_array = npzfile['homoregion_go_terms_overlap_array']
            current_r_id = npzfile['current_r_id']

        homoregion_go_terms_overlap_array = homoregion_go_terms_overlap_array - np.reshape(
            np.mean(homoregion_go_terms_overlap_array, axis=1), (-1, 1))

        homoregion_go_terms_overlap_array = homoregion_go_terms_overlap_array - np.reshape(
            np.mean(homoregion_go_terms_overlap_array, axis=0), (1, -1))

        homoregion_go_terms_overlap_array = preprocess.zscore(homoregion_go_terms_overlap_array)

        diff_gene_module_exp_df = pd.DataFrame(
            homoregion_go_terms_overlap_array.T,
            index=module_labels_unique,
            columns=cluster_name_unique)

        rcParams["figure.subplot.left"] = 0.03
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.top"] = 0.98
        rcParams["figure.subplot.bottom"] = 0.5

        # c = Colormap()
        c = Colormap()
        color_map = c.cmap_linear('white', 'white', 'indigo')

        fig, ax = plt.subplots(figsize=(18, 3.5), dpi=self.fig_dpi)

        hm = sns.heatmap(diff_gene_module_exp_df, square=False, cbar_kws={'location': 'right'},
                         cmap=color_map,  # 'Spectral_r',  # "YlGnBu",
                         ax=ax,
                         xticklabels=True, yticklabels=True, linewidths=0.2, linecolor='lightgrey')
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.legend([],[], frameon=False)
        # plt.xticks(rotation=45)
        plt.tight_layout()
        plt.ylabel('Gene module')
        plt.savefig(save_path + 'Go_terms_overlap.' + self.fig_format, format=self.fig_format)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1


    def experiment_10_binary_clustering(self):
        from collections import Counter

        sns.set(style='white')
        TINY_SIZE = 10  # 39
        SMALL_SIZE = 10  # 42
        MEDIUM_SIZE = 12  # 46
        BIGGER_SIZE = 12  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        # save_path
        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/10_experiment_binary_clustering/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        sc.pp.neighbors(self.adata_gene_embedding, n_neighbors=self.cfg.ANALYSIS.genes_umap_neighbor, metric='cosine',
                        use_rep='X')

        sc.tl.leiden(self.adata_gene_embedding, resolution=0.1, key_added='binary_leiden')

        print(Counter(self.adata_gene_embedding.obs['binary_leiden']))

        gene_cluster_name_unique = sorted(Counter(self.adata_gene_embedding.obs['binary_leiden']).keys(),
                                     key=lambda t: int(t.split('-')[0]))

        ntop_gene = 100
        p_cut = 1e-3
        go_p_cut = 5e-2
        top_term = 2
        color_list = list(sns.color_palette(cc.glasbey, n_colors=1))

        mouse_go_gene_sets = ['Mouse_Gene_Atlas']
        human_go_gene_sets = ['Human_Gene_Atlas']


        for species in ['Mouse', 'Human']:
            for module in gene_cluster_name_unique:
                adata_embeds_module = self.adata_gene_embedding[self.adata_gene_embedding.obs['dataset'].isin([species]) & self.adata_gene_embedding.obs['binary_leiden'].isin([module])]
                gene_list = list(adata_embeds_module.obs_names)

                if species == 'Mouse':
                    go_gene_sets = mouse_go_gene_sets
                else:
                    go_gene_sets = human_go_gene_sets

                #if species == 'Mouse':
                enr_res = gseapy.enrichr(gene_list=gene_list,
                                         organism=species,
                                         gene_sets=go_gene_sets, cutoff=go_p_cut)
                                         #top_term=top_term)  # cutoff=go_p_cut
                # fig, axes = plt.subplot()
                rcParams["figure.subplot.left"] = 0.4
                rcParams["figure.subplot.right"] = 0.7
                rcParams["figure.subplot.top"] = 0.9
                rcParams["figure.subplot.bottom"] = 0.1

                rcParams["legend.title_fontsize"] = MEDIUM_SIZE
                rcParams["axes.titlesize"] = MEDIUM_SIZE
                fig, ax = gseapy.barplot(enr_res.results, column="Adjusted P-value", group='Gene_set', color=color_list,
                                         # color=self.mouse_color,
                                         # set group, so you could do a multi-sample/library comparsion
                                         size=5, title=f"GO terms of {species} gene module {module}",
                                         figsize=(6, 5), cutoff=go_p_cut)  # cutoff=go_p_cut , top_term=top_term
                ax.get_legend().set_title("Gene sets")
                # ax.get_legend().set_fontsize(MEDIUM_SIZE)
                # ax.get_title().set_fontsize(MEDIUM_SIZE)
                # ax.get_legend().set_loc('lower center')
                # fig.axes.append(ax)
                fig.savefig(save_path + f'GO terms of {species} gene module {module}_barplot.' + self.fig_format,
                            format=self.fig_format, dpi=self.fig_dpi)
                # rcParams["figure.subplot.left"] = 0.1
                # rcParams["figure.subplot.right"] = 0.9
                rcParams["figure.subplot.left"] = 0.5
                rcParams["figure.subplot.right"] = 0.7
                rcParams["figure.subplot.bottom"] = 0.25  # 0.15
                rcParams["figure.subplot.top"] = 0.95

                fig, ax = gseapy.dotplot(enr_res.results,
                                         column="Adjusted P-value",
                                         x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion
                                         size=10,
                                         # top_term=5,  #
                                         figsize=(6, 5),
                                         title=f"GO terms of {species} gene module {module}",
                                         xticklabels_rot=45,
                                         # rotate xtick labels show_ring=True, set to False to revmove outer ring
                                         marker='o',
                                         cmap=plt.cm.winter_r,
                                         format=self.fig_format, cutoff=go_p_cut)  # cutoff=go_p_cut top_term=top_term
                # cbar = fig.colorbar(ax=ax)
                # im = ax.images
                # print(len(im))
                # print(im)
                # # Assume colorbar was plotted last one plotted last
                # cbar = im.colorbar
                # cbar.set_label(size=MEDIUM_SIZE, labelpad=15)
                ax.set_xlabel('')
                fig.savefig(save_path + f'GO terms of {species} gene module {module}_botplot.' + self.fig_format,
                            format=self.fig_format, dpi=self.fig_dpi)



        # Mouse genes

        # for g_module_id in gene_cluster_name_unique:
        #
        #     enr_res = gseapy.enrichr(gene_list=go_gene_list,
        #                              organism='Mouse',
        #                              gene_sets=mouse_go_gene_sets,
        #                              top_term=top_term)  # cutoff=go_p_cut
        #     # fig, axes = plt.subplot()
        #     rcParams["figure.subplot.left"] = 0.52
        #     rcParams["figure.subplot.right"] = 0.70
        #     rcParams["figure.subplot.top"] = 0.9
        #     rcParams["figure.subplot.bottom"] = 0.1
        #
        #     rcParams["legend.title_fontsize"] = MEDIUM_SIZE
        #     rcParams["axes.titlesize"] = MEDIUM_SIZE
        #     fig, ax = gseapy.barplot(enr_res.results, column="Adjusted P-value", group='Gene_set', color=color_list,
        #                              # color=self.mouse_color,
        #                              # set group, so you could do a multi-sample/library comparsion
        #                              size=5, title=f"GO of Mouse cluster {c_name} {g_module_id} DEGs",
        #                              figsize=(15, 8), top_term=top_term)  # cutoff=go_p_cut
        #     ax.get_legend().set_title("Gene sets")
        #     # ax.get_legend().set_fontsize(MEDIUM_SIZE)
        #     # ax.get_title().set_fontsize(MEDIUM_SIZE)
        #     # ax.get_legend().set_loc('lower center')
        #     # fig.axes.append(ax)
        #     fig.savefig(save_path_region + f'module_{g_module_id}_barplot.' + self.fig_format,
        #                 format=self.fig_format, dpi=self.fig_dpi)
        #     # rcParams["figure.subplot.left"] = 0.1
        #     # rcParams["figure.subplot.right"] = 0.9
        #     rcParams["figure.subplot.left"] = 0.6
        #     rcParams["figure.subplot.right"] = 0.88
        #     rcParams["figure.subplot.bottom"] = 0.25  # 0.15
        #     rcParams["figure.subplot.top"] = 0.95
        #
        #     fig, ax = gseapy.dotplot(enr_res.results,
        #                              column="Adjusted P-value",
        #                              x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion
        #                              size=10,
        #                              # top_term=5,  #
        #                              figsize=(15, 10),
        #                              title=f"GO of Mouse cluster {c_name} {g_module_id} DEGs",
        #                              xticklabels_rot=45,
        #                              # rotate xtick labels show_ring=True, set to False to revmove outer ring
        #                              marker='o',
        #                              cmap=plt.cm.winter_r,
        #                              format=self.fig_format, top_term=top_term)  # cutoff=go_p_cut
        #     # cbar = fig.colorbar(ax=ax)
        #     # im = ax.images
        #     # print(len(im))
        #     # print(im)
        #     # # Assume colorbar was plotted last one plotted last
        #     # cbar = im.colorbar
        #     # cbar.set_label(size=MEDIUM_SIZE, labelpad=15)
        #     ax.set_xlabel('')
        #     fig.savefig(save_path_region + f'module_{g_module_id}_botplot.' + self.fig_format,
        #                 format=self.fig_format, dpi=self.fig_dpi)

        return None


    def experiment_11_module_go_terms(self):
        from collections import Counter
        sns.set(style='white')
        TINY_SIZE = 10  # 39
        SMALL_SIZE = 10  # 42
        MEDIUM_SIZE = 12  # 46
        BIGGER_SIZE = 12  # 46
        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']
        # save_path
        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/11_experiment_module_go_terms/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        module_name = 'module'
        module_num = len(Counter(self.adata_gene_embedding.obs[module_name]))
        gene_names = self.adata_gene_embedding.obs_names.values
        module_labels = self.adata_gene_embedding.obs[module_name].values

        module_labels_unique = list(set(self.adata_gene_embedding.obs[module_name].values))
        module_labels_unique.sort(key=lambda x: int(x))
        # print('module_labels.shape', module_labels.shape)
        gene_module_dict = {k: v for k, v in zip(gene_names, module_labels)}

        mouse_gene_names = self.adata_mouse_gene_embedding.obs_names.values
        mouse_module_labels = self.adata_mouse_gene_embedding.obs[module_name].values
        mouse_gene_module_dict = {k: v for k, v in zip(mouse_gene_names, mouse_module_labels)}
        # human gene_name to module
        human_gene_names = self.adata_human_gene_embedding.obs_names.values
        human_module_labels = self.adata_human_gene_embedding.obs[module_name].values
        human_gene_module_dict = {k: v for k, v in zip(human_gene_names, human_module_labels)}

        gene_cluster_name_unique = sorted(Counter(self.adata_gene_embedding.obs[module_name]).keys(),
                                     key=lambda t: int(t.split('-')[0]))
        ntop_gene = 100
        p_cut = 1e-3
        go_p_cut = 5e-1
        top_term = 5
        fig_size = (5.2, 2.5)
        color_list = list(sns.color_palette(cc.glasbey, n_colors=1))
        mouse_go_gene_sets = ['Mouse_Gene_Atlas']
        human_go_gene_sets = ['Human_Gene_Atlas']
        for species in ['Mouse', 'Human']:
            for module in gene_cluster_name_unique:
                adata_embeds_module = self.adata_gene_embedding[self.adata_gene_embedding.obs['dataset'].isin([species])
                                                                & self.adata_gene_embedding.obs['module'].isin([module])]
                gene_list = list(adata_embeds_module.obs_names)
                if species == 'Mouse':
                    go_gene_sets = mouse_go_gene_sets
                else:
                    go_gene_sets = human_go_gene_sets
                enr_res = gseapy.enrichr(gene_list=gene_list,
                                         organism=species,
                                         gene_sets=go_gene_sets, #cutoff=go_p_cut
                                         top_term=top_term, cutoff=go_p_cut)  # cutoff=go_p_cut
                rcParams["figure.subplot.left"] = 0.4
                rcParams["figure.subplot.right"] = 0.6
                rcParams["figure.subplot.top"] = 0.8
                rcParams["figure.subplot.bottom"] = 0.25
                rcParams["legend.title_fontsize"] = MEDIUM_SIZE
                rcParams["axes.titlesize"] = MEDIUM_SIZE
                fig, ax = gseapy.barplot(enr_res.results, column="Adjusted P-value", group='Gene_set', color=color_list,
                                         size=5, title=f"{species} module {module}",figsize=fig_size, cutoff=go_p_cut, top_term=top_term)#, cutoff=go_p_cut)  # cutoff=go_p_cut , top_term=top_term
                ax.get_legend().set_title("Gene sets")
                # ax.get_legend().set_fontsize(MEDIUM_SIZE)
                # ax.get_title().set_fontsize(MEDIUM_SIZE)
                # ax.get_legend().set_loc('lower center')
                # fig.axes.append(ax)
                fig.savefig(save_path + f'GO terms of {species} gene module {module}_barplot.' + self.fig_format,
                            format=self.fig_format, dpi=self.fig_dpi)
                # rcParams["figure.subplot.left"] = 0.1
                # rcParams["figure.subplot.right"] = 0.9
                rcParams["figure.subplot.left"] = 0.4
                rcParams["figure.subplot.right"] = 0.6
                rcParams["figure.subplot.bottom"] = 0.28  # 0.15
                rcParams["figure.subplot.top"] = 0.8
                fig, ax = gseapy.dotplot(enr_res.results,
                                         column="Adjusted P-value",
                                         x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion
                                         size=12,
                                         # top_term=5,  #
                                         figsize=fig_size ,
                                         title=f"{species} module {module}",
                                         #xticklabels_rot=45,
                                         # rotate xtick labels show_ring=True, set to False to revmove outer ring
                                         marker='o',
                                         cmap=plt.cm.winter_r,
                                         format=self.fig_format, cutoff=go_p_cut, top_term=top_term)#cutoff=go_p_cut)  # cutoff=go_p_cut top_term=top_term
                ax.set_xlabel('')
                fig.savefig(save_path + f'GO terms of {species} gene module {module}_botplot.' + self.fig_format,
                            format=self.fig_format, dpi=self.fig_dpi)

        return None


    def experiment_10_1_binary_clustering_multiple_genesets(self):

        from collections import Counter

        sns.set(style='white')
        TINY_SIZE = 10  # 39
        SMALL_SIZE = 10  # 42
        MEDIUM_SIZE = 12  # 46
        BIGGER_SIZE = 12  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        # save_path
        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/3_genomic_analysis/10_1_experiment_binary_clustering_multiple_genesets/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        sc.pp.neighbors(self.adata_gene_embedding, n_neighbors=self.cfg.ANALYSIS.genes_umap_neighbor, metric='cosine',
                        use_rep='X')

        sc.tl.leiden(self.adata_gene_embedding, resolution=0.1, key_added='binary_leiden')

        print(Counter(self.adata_gene_embedding.obs['binary_leiden']))

        gene_cluster_name_unique = sorted(Counter(self.adata_gene_embedding.obs['binary_leiden']).keys(),
                                     key=lambda t: int(t.split('-')[0]))

        ntop_gene = 50
        p_cut = 1e-3
        go_p_cut = 1e-3
        top_term = 2
        color_list = list(sns.color_palette(cc.glasbey, n_colors=8))

        mouse_go_gene_sets = ['Allen_Brain_Atlas_down',
                              'Allen_Brain_Atlas_up',
                              'Azimuth_Cell_Types_2021',
                              'CellMarker_Augmented_2021',
                              'GO_Biological_Process_2021',
                              'GO_Cellular_Component_2021',
                              'GO_Molecular_Function_2021',
                              'Mouse_Gene_Atlas']
        human_go_gene_sets = ['Allen_Brain_Atlas_down', 'Allen_Brain_Atlas_up',
                              'Azimuth_Cell_Types_2021', 'CellMarker_Augmented_2021',
                              'GO_Biological_Process_2021', 'GO_Cellular_Component_2021',
                              'GO_Molecular_Function_2021', 'Human_Gene_Atlas']


        for species in ['Mouse', 'Human']:
            for module in gene_cluster_name_unique:
                adata_embeds_module = self.adata_gene_embedding[self.adata_gene_embedding.obs['dataset'].isin([species]) & self.adata_gene_embedding.obs['binary_leiden'].isin([module])]
                gene_list = list(adata_embeds_module.obs_names)

                if species == 'Mouse':
                    go_gene_sets = mouse_go_gene_sets
                else:
                    go_gene_sets = human_go_gene_sets

                #if species == 'Mouse':
                enr_res = gseapy.enrichr(gene_list=gene_list,
                                         organism=species,
                                         gene_sets=go_gene_sets, cutoff=go_p_cut)
                                         #top_term=top_term)  # cutoff=go_p_cut
                # fig, axes = plt.subplot()
                rcParams["figure.subplot.left"] = 0.6
                rcParams["figure.subplot.right"] = 0.85
                rcParams["figure.subplot.top"] = 0.98
                rcParams["figure.subplot.bottom"] = 0.02

                rcParams["legend.title_fontsize"] = MEDIUM_SIZE
                rcParams["axes.titlesize"] = MEDIUM_SIZE
                fig, ax = gseapy.barplot(enr_res.results, column="Adjusted P-value", group='Gene_set', color=color_list,
                                         # color=self.mouse_color,
                                         # set group, so you could do a multi-sample/library comparsion
                                         size=5, title=f"GO terms of {species} gene module {module}",
                                         figsize=(15, 15), cutoff=go_p_cut)  # cutoff=go_p_cut , top_term=top_term
                ax.get_legend().set_title("Gene sets")
                # ax.get_legend().set_fontsize(MEDIUM_SIZE)
                # ax.get_title().set_fontsize(MEDIUM_SIZE)
                # ax.get_legend().set_loc('lower center')
                # fig.axes.append(ax)
                fig.savefig(save_path + f'GO terms of {species} gene module {module}_barplot.' + self.fig_format,
                            format=self.fig_format, dpi=self.fig_dpi)
                # rcParams["figure.subplot.left"] = 0.1
                # rcParams["figure.subplot.right"] = 0.9
                rcParams["figure.subplot.left"] = 0.6
                rcParams["figure.subplot.right"] = 0.95
                rcParams["figure.subplot.bottom"] = 0.15  # 0.15
                rcParams["figure.subplot.top"] = 0.98

                fig, ax = gseapy.dotplot(enr_res.results,
                                         column="Adjusted P-value",
                                         x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion
                                         size=10,
                                         # top_term=5,  #
                                         figsize=(18, 15),
                                         title=f"GO terms of {species} gene module {module}",
                                         xticklabels_rot=45,
                                         # rotate xtick labels show_ring=True, set to False to revmove outer ring
                                         marker='o',
                                         cmap=plt.cm.winter_r,
                                         format=self.fig_format, cutoff=go_p_cut)  # cutoff=go_p_cut top_term=top_term
                # cbar = fig.colorbar(ax=ax)
                # im = ax.images
                # print(len(im))
                # print(im)
                # # Assume colorbar was plotted last one plotted last
                # cbar = im.colorbar
                # cbar.set_label(size=MEDIUM_SIZE, labelpad=15)
                ax.set_xlabel('')
                fig.savefig(save_path + f'GO terms of {species} gene module {module}_botplot.' + self.fig_format,
                            format=self.fig_format, dpi=self.fig_dpi)



        # Mouse genes

        # for g_module_id in gene_cluster_name_unique:
        #
        #     enr_res = gseapy.enrichr(gene_list=go_gene_list,
        #                              organism='Mouse',
        #                              gene_sets=mouse_go_gene_sets,
        #                              top_term=top_term)  # cutoff=go_p_cut
        #     # fig, axes = plt.subplot()
        #     rcParams["figure.subplot.left"] = 0.52
        #     rcParams["figure.subplot.right"] = 0.70
        #     rcParams["figure.subplot.top"] = 0.9
        #     rcParams["figure.subplot.bottom"] = 0.1
        #
        #     rcParams["legend.title_fontsize"] = MEDIUM_SIZE
        #     rcParams["axes.titlesize"] = MEDIUM_SIZE
        #     fig, ax = gseapy.barplot(enr_res.results, column="Adjusted P-value", group='Gene_set', color=color_list,
        #                              # color=self.mouse_color,
        #                              # set group, so you could do a multi-sample/library comparsion
        #                              size=5, title=f"GO of Mouse cluster {c_name} {g_module_id} DEGs",
        #                              figsize=(15, 8), top_term=top_term)  # cutoff=go_p_cut
        #     ax.get_legend().set_title("Gene sets")
        #     # ax.get_legend().set_fontsize(MEDIUM_SIZE)
        #     # ax.get_title().set_fontsize(MEDIUM_SIZE)
        #     # ax.get_legend().set_loc('lower center')
        #     # fig.axes.append(ax)
        #     fig.savefig(save_path_region + f'module_{g_module_id}_barplot.' + self.fig_format,
        #                 format=self.fig_format, dpi=self.fig_dpi)
        #     # rcParams["figure.subplot.left"] = 0.1
        #     # rcParams["figure.subplot.right"] = 0.9
        #     rcParams["figure.subplot.left"] = 0.6
        #     rcParams["figure.subplot.right"] = 0.88
        #     rcParams["figure.subplot.bottom"] = 0.25  # 0.15
        #     rcParams["figure.subplot.top"] = 0.95
        #
        #     fig, ax = gseapy.dotplot(enr_res.results,
        #                              column="Adjusted P-value",
        #                              x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion
        #                              size=10,
        #                              # top_term=5,  #
        #                              figsize=(15, 10),
        #                              title=f"GO of Mouse cluster {c_name} {g_module_id} DEGs",
        #                              xticklabels_rot=45,
        #                              # rotate xtick labels show_ring=True, set to False to revmove outer ring
        #                              marker='o',
        #                              cmap=plt.cm.winter_r,
        #                              format=self.fig_format, top_term=top_term)  # cutoff=go_p_cut
        #     # cbar = fig.colorbar(ax=ax)
        #     # im = ax.images
        #     # print(len(im))
        #     # print(im)
        #     # # Assume colorbar was plotted last one plotted last
        #     # cbar = im.colorbar
        #     # cbar.set_label(size=MEDIUM_SIZE, labelpad=15)
        #     ax.set_xlabel('')
        #     fig.savefig(save_path_region + f'module_{g_module_id}_botplot.' + self.fig_format,
        #                 format=self.fig_format, dpi=self.fig_dpi)

        return None




def get_top_same_terms(enr_res_mouse_results, enr_res_human_results, cutoff=0.05): #, top_terms = 10
    """

    Acquire the most significant same terms for two species.

    :param enr_res_mouse_results: the GO term reuslts of Mouse
    :param enr_res_human_results: the GO term reuslts of Human
    :return: the two dataframes of the top GO terms information
    """
    enr_res_mouse_results = enr_res_mouse_results[enr_res_mouse_results['Adjusted P-value'] <= cutoff]
    enr_res_human_results = enr_res_human_results[enr_res_human_results['Adjusted P-value'] <= cutoff]

    enr_res_mouse_results = enr_res_mouse_results.sort_values('Adjusted P-value')
    enr_res_human_results = enr_res_human_results.sort_values('Adjusted P-value')

    enr_res_mouse_results_term = enr_res_mouse_results['Term'].tolist()
    enr_res_human_results_term = enr_res_human_results['Term'].tolist()

    enr_res_mouse_index = []
    enr_res_human_index = []
    for i in range(enr_res_mouse_results.shape[0]):
        for j in range(enr_res_human_results.shape[0]):
            if enr_res_mouse_results_term[i] == enr_res_human_results_term[j]:
                # print(enr_res_mouse_results_term[i], enr_res_human_results_term[j])
                enr_res_mouse_index.append(enr_res_mouse_results.index[i])
                enr_res_human_index.append(enr_res_human_results.index[j])
                continue


    enr_res_mouse_results_top = enr_res_mouse_results.loc[enr_res_mouse_index, :]
    enr_res_human_results_top = enr_res_human_results.loc[enr_res_human_index, :]

    enr_res_mouse_results_top = enr_res_mouse_results_top.sort_values(by=['Term'])
    enr_res_mouse_results_top.index = np.array(range(0, len(enr_res_mouse_index)))
    enr_res_human_results_top = enr_res_human_results_top.sort_values(by=['Term'])
    enr_res_human_results_top.index = np.array(range(0, len(enr_res_human_index)))

    if_empty = False
    if len(enr_res_mouse_index) <= 0:
        if_empty = True

    return enr_res_mouse_results_top, enr_res_human_results_top, if_empty


def reorder(X_1, X_ordered, Y_1, ntop=1):
    Y_ordered = []
    if ntop == 1:
        for i in range(len(X_1)):
            X_ele_index = X_1.index(X_ordered[i])
            Y_ordered.append(Y_1[X_ele_index])
    else:
        for i in range(len(X_1)):
            X_ele_index = X_1.index(X_ordered[i]) * ntop
            for k in range(ntop):
                #print(X_ele_index + k)
                Y_ordered.append(Y_1[X_ele_index + k])

    return Y_ordered

if __name__ == '__main__':
    X_1 = ['a', 'b', 'c']
    X_ordered = ['c', 'a', 'b']
    Y_1 = ['1', '2', '3', '4', '5', '6']
    print(reorder(X_1, X_ordered, Y_1, ntop=2))
    #
    # X_1 = [1, 2, 3, 5]
    # X_2 = [2, 3, 4, 5]
    # for i in X_1:
    #     for j in X_2:
    #         if i == j:
    #             print(i)
    #             continue


    # import seaborn as sns
    # sns.set_theme(style="whitegrid")
    # penguins = sns.load_dataset("penguins")
    # print(penguins)
    # data_dict = {'Species': ['Mouse', 'Human', 'Mouse', 'Human', 'Mouse', 'Human'],
    #              'Homologous type': ['All', 'All', 'Many-to-Many', 'Many-to-Many', 'One-to-one', 'One-to-one'],
    #              'Gene number': [3000, 4000, 2100, 2300, 1800, 1800]}
    #
    # data_df = pd.DataFrame.from_dict(data_dict)
    # print(data_df)
    # # Draw a nested barplot by species and sex
    # g = sns.catplot(
    #     data=data_df, kind="bar",
    #     x="Homologous type", y="Gene number", hue="Species",
    #     palette="dark", alpha=.6, height=6
    # )
    # g.despine(left=True)
    # #g.set_axis_labels("", "Body mass (g)")
    # g.legend.set_title("")

    #plt.show()

    '''
    def plot_colormap(cmap):
        fig, ax = plt.subplots(figsize=(6, 2))
        #cmap = mpl.cm.get_cmap(cmap_name)
        colors = cmap(np.linspace(0, 1, cmap.N))
        ax.imshow([colors], extent=[0, 10, 0, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.set_title(cmap_name)


    from colormap import Colormap
    import matplotlib

    c = Colormap()
    mycmap = c.cmap_linear('#3288BD', 'white', '#D53E4F')
    plot_colormap(mycmap)
    import matplotlib.pyplot as plt

    c.test_colormap(mycmap)

    plt.show()
    '''