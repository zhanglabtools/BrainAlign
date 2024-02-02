# -- coding: utf-8 --
# @Time : 2023/3/9 15:36
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : analysis_main.py
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
from matplotlib import colors

from collections import abc
from colormap import Colormap


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



plt.rcParams['figure.edgecolor'] = 'black'


# Systemly analysis in the manuscript:
# 1. Alignment of whole brain spatial transcriptomics;
# 2. Anatomical and spatial structure comparison across species;
# 3. Gene expression comparative analysis;
# 4. Local regions property conservation analysis.

def alignment_STs(cfg):
    # Alignment of whole brain spatial transcriptomics;
    # load embeddings and assign acronym, color and parent region name
    fig_format = cfg.BrainAlign.fig_format  # the figure save format

    save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # read labels data including acronym, color and parent region name
    mouse_64_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)
    mouse_64_labels_list = list(mouse_64_labels['region_name'])
    mouse_64_acronym_dict = {k:v for k,v in zip(mouse_64_labels['region_name'], mouse_64_labels['acronym'])}
    mouse_64_color_dict = {k:v for k,v in zip(mouse_64_labels['region_name'], mouse_64_labels['color_hex_triplet'])}
    mouse_64_parent_region_dict = {k: v for k, v in zip(mouse_64_labels['region_name'], mouse_64_labels['parent_region_name'])}
    mouse_15_labels =pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.parent_labels_mouse_file)
    mouse_15_acronym_dict =  {k:v for k,v in zip(mouse_15_labels['region_name'], mouse_15_labels['acronym'])}
    mouse_15_color_dict = {k: v for k, v in zip(mouse_15_labels['region_name'], mouse_15_labels['color_hex_triplet'])}

    human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
    human_88_labels_list = list(human_88_labels['region_name'])
    human_88_acronym_dict = {k: v for k, v in zip(human_88_labels['region_name'], human_88_labels['acronym'])}
    human_88_color_dict = {k: v for k, v in zip(human_88_labels['region_name'], human_88_labels['color_hex_triplet'])}
    human_88_parent_region_dict = {k: v for k, v in
                                   zip(human_88_labels['region_name'], human_88_labels['parent_region_name'])}
    human_16_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.parent_labels_human_file)
    human_16_acronym_dict = {k: v for k, v in zip(human_16_labels['region_name'], human_16_labels['acronym'])}
    human_16_color_dict = {k: v for k, v in zip(human_16_labels['region_name'], human_16_labels['color_hex_triplet'])}

    # load embeddings
    adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
    adata_mouse_embedding.obs['dataset'] = 'mouse'
    adata_human_embedding.obs['dataset'] = 'human'

    # assign attributes for mouse and save
    adata_mouse_embedding.obs['acronym'] = [mouse_64_acronym_dict[r] for r in adata_mouse_embedding.obs['region_name']]
    adata_mouse_embedding.obs['color_hex_triplet'] = [mouse_64_color_dict[r] for r in adata_mouse_embedding.obs['region_name']]
    adata_mouse_embedding.obs['parent_region_name'] = [mouse_64_parent_region_dict[r] for r in adata_mouse_embedding.obs['region_name']]
    adata_mouse_embedding.obs['parent_acronym'] = [mouse_15_acronym_dict[r] for r in adata_mouse_embedding.obs['parent_region_name']]
    adata_mouse_embedding.obs['parent_color_hex_triplet'] = [mouse_15_color_dict[r] for r in adata_mouse_embedding.obs['parent_region_name']]
    adata_mouse_embedding.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/s_embeddings.h5ad')

    # assign attibutes for human and save
    adata_human_embedding.obs['acronym'] = [human_88_acronym_dict[r] for r in adata_human_embedding.obs['region_name']]
    adata_human_embedding.obs['color_hex_triplet'] = [human_88_color_dict[r] for r in
                                                      adata_human_embedding.obs['region_name']]
    adata_human_embedding.obs['parent_region_name'] = [human_88_parent_region_dict[r] for r in
                                                       adata_human_embedding.obs['region_name']]
    adata_human_embedding.obs['parent_acronym'] = [human_16_acronym_dict[r] for r in
                                                   adata_human_embedding.obs['parent_region_name']]
    adata_human_embedding.obs['parent_color_hex_triplet'] = [human_16_color_dict[r] for r in
                                                             adata_human_embedding.obs['parent_region_name']]
    adata_human_embedding.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/v_embeddings.h5ad')



    return None


class alignment_STs_analysis():
    def __init__(self, cfg):
        self.cfg = cfg
        self.fig_format = cfg.BrainAlign.fig_format  # the figure save format
        self.fig_dpi = cfg.BrainAlign.fig_dpi

        self.mouse_color = '#5D8AEF'#'#4472C4'
        self.human_color = '#FE1613'#'#C94799'#'#ED7D31'


        self.save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # read labels data including acronym, color and parent region name
        self.mouse_64_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)
        self.mouse_64_labels_list = list(self.mouse_64_labels['region_name'])
        self.mouse_64_acronym_dict = {k: v for k, v in zip(self.mouse_64_labels['region_name'], self.mouse_64_labels['acronym'])}
        self.mouse_64_color_dict = {k: v for k, v in
                               zip(self.mouse_64_labels['region_name'], self.mouse_64_labels['color_hex_triplet'])}
        self.mouse_64_acronym_color_dict = {k: v for k, v in
                               zip(self.mouse_64_labels['acronym'], self.mouse_64_labels['color_hex_triplet'])}
        self.mouse_64_parent_region_dict = {k: v for k, v in
                                       zip(self.mouse_64_labels['region_name'], self.mouse_64_labels['parent_region_name'])}
        self.mouse_15_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.parent_labels_mouse_file)
        self.mouse_15_labels_list = list(self.mouse_15_labels['region_name'])
        self.mouse_15_acronym_dict = {k: v for k, v in zip(self.mouse_15_labels['region_name'], self.mouse_15_labels['acronym'])}
        self.mouse_15_color_dict = {k: v for k, v in
                               zip(self.mouse_15_labels['region_name'], self.mouse_15_labels['color_hex_triplet'])}
        self.mouse_15_acronym_color_dict = {k: v for k, v in
                                    zip(self.mouse_15_labels['acronym'], self.mouse_15_labels['color_hex_triplet'])}

        self.mouse_15_acronym_list = list(self.mouse_15_labels['acronym'])

        self.human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
        self.human_88_labels_list = list(self.human_88_labels['region_name'])
        self.human_88_acronym_dict = {k: v for k, v in zip(self.human_88_labels['region_name'], self.human_88_labels['acronym'])}
        self.human_88_color_dict = {k: v for k, v in
                               zip(self.human_88_labels['region_name'],self.human_88_labels['color_hex_triplet'])}
        self.human_88_acronym_color_dict = {k: v for k, v in
                                    zip(self.human_88_labels['acronym'], self.human_88_labels['color_hex_triplet'])}
        self.human_88_parent_region_dict = {k: v for k, v in
                                       zip(self.human_88_labels['region_name'], self.human_88_labels['parent_region_name'])}
        self.human_16_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.parent_labels_human_file)
        self.human_16_labels_list = list(self.human_16_labels['region_name'])
        self.human_16_acronym_dict = {k: v for k, v in zip(self.human_16_labels['region_name'], self.human_16_labels['acronym'])}
        self.human_16_color_dict = {k: v for k, v in
                               zip(self.human_16_labels['region_name'], self.human_16_labels['color_hex_triplet'])}
        self.human_16_acronym_color_dict = {k: v for k, v in
                                    zip(self.human_16_labels['acronym'], self.human_16_labels['color_hex_triplet'])}
        self.human_16_acronym_list = list(self.human_16_labels['acronym'])

        self.acronym_color_dict = {}
        for k,v in self.mouse_64_acronym_color_dict.items():
            self.acronym_color_dict.update({'M-'+k:v})
        for k, v in self.human_88_acronym_color_dict.items():
            self.acronym_color_dict.update({('H-'+k):v})
        print(self.acronym_color_dict)

        self.parent_acronym_color_dict = {k:v for k,v in self.mouse_15_acronym_color_dict.items()}
        self.parent_acronym_color_dict.update({k:v for k,v in self.human_16_acronym_color_dict.items()})
        print(self.parent_acronym_color_dict)

        self.color_dict = self.mouse_64_color_dict
        self.color_dict.update(self.human_88_color_dict)
        print(self.color_dict)

        self.parent_color_dict = self.mouse_15_color_dict
        self.parent_color_dict.update(self.human_16_color_dict)
        print(self.parent_color_dict)



    def init_data(self):
        cfg = self.cfg

        fig_format = cfg.BrainAlign.fig_format  # the figure save format

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # read labels data including acronym, color and parent region name
        mouse_64_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)
        mouse_64_labels_list = list(mouse_64_labels['region_name'])
        mouse_64_acronym_dict = {k: v for k, v in zip(mouse_64_labels['region_name'], mouse_64_labels['acronym'])}
        mouse_64_color_dict = {k: v for k, v in
                               zip(mouse_64_labels['region_name'], mouse_64_labels['color_hex_triplet'])}
        mouse_64_parent_region_dict = {k: v for k, v in
                                       zip(mouse_64_labels['region_name'], mouse_64_labels['parent_region_name'])}
        mouse_15_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.parent_labels_mouse_file)
        mouse_15_labels_list = list(mouse_15_labels['region_name'])
        mouse_15_acronym_dict = {k: v for k, v in zip(mouse_15_labels['region_name'], mouse_15_labels['acronym'])}
        mouse_15_color_dict = {k: v for k, v in
                               zip(mouse_15_labels['region_name'], mouse_15_labels['color_hex_triplet'])}

        human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
        human_88_labels_list = list(human_88_labels['region_name'])
        human_88_acronym_dict = {k: v for k, v in zip(human_88_labels['region_name'], human_88_labels['acronym'])}
        human_88_color_dict = {k: v for k, v in
                               zip(human_88_labels['region_name'], human_88_labels['color_hex_triplet'])}
        human_88_parent_region_dict = {k: v for k, v in
                                       zip(human_88_labels['region_name'], human_88_labels['parent_region_name'])}
        human_16_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.parent_labels_human_file)
        human_16_labels_list = list(human_16_labels['region_name'])
        human_16_acronym_dict = {k: v for k, v in zip(human_16_labels['region_name'], human_16_labels['acronym'])}
        human_16_color_dict = {k: v for k, v in
                               zip(human_16_labels['region_name'], human_16_labels['color_hex_triplet'])}

        # load embeddings
        adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
        adata_mouse_embedding.obs['dataset'] = 'mouse'
        adata_human_embedding.obs['dataset'] = 'human'

        # assign attributes for mouse and save
        adata_mouse_embedding.obs['acronym'] = [mouse_64_acronym_dict[r] for r in
                                                adata_mouse_embedding.obs['region_name']]
        adata_mouse_embedding.obs['color_hex_triplet'] = [mouse_64_color_dict[r] for r in
                                                          adata_mouse_embedding.obs['region_name']]
        adata_mouse_embedding.obs['parent_region_name'] = [mouse_64_parent_region_dict[r] for r in
                                                           adata_mouse_embedding.obs['region_name']]
        adata_mouse_embedding.obs['parent_acronym'] = [mouse_15_acronym_dict[r] for r in
                                                       adata_mouse_embedding.obs['parent_region_name']]
        adata_mouse_embedding.obs['parent_color_hex_triplet'] = [mouse_15_color_dict[r] for r in
                                                                 adata_mouse_embedding.obs['parent_region_name']]
        adata_mouse_embedding.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/s_embeddings.h5ad')

        # assign attibutes for human and save
        adata_human_embedding.obs['acronym'] = [human_88_acronym_dict[r] for r in
                                                adata_human_embedding.obs['region_name']]
        adata_human_embedding.obs['color_hex_triplet'] = [human_88_color_dict[r] for r in
                                                          adata_human_embedding.obs['region_name']]
        adata_human_embedding.obs['parent_region_name'] = [human_88_parent_region_dict[r] for r in
                                                           adata_human_embedding.obs['region_name']]
        adata_human_embedding.obs['parent_acronym'] = [human_16_acronym_dict[r] for r in
                                                       adata_human_embedding.obs['parent_region_name']]
        adata_human_embedding.obs['parent_color_hex_triplet'] = [human_16_color_dict[r] for r in
                                                                 adata_human_embedding.obs['parent_region_name']]
        adata_human_embedding.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/v_embeddings.h5ad')


    def experiment_1_cross_species_clustering(self):
        sns.set(style='white')
        TINY_SIZE = 20  # 39
        SMALL_SIZE = 20  # 42
        MEDIUM_SIZE = 20  # 46
        BIGGER_SIZE = 20  # 46

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
        Step 1: Compute average embedding of every region in two species, use two dict to store;
        Step 2: Compute similarity matrix, use np array to store;
        Step 3: Heatmap.
        '''
        adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')

        custom_palette = sns.color_palette("Paired", 2)
        print(custom_palette)
        # lut = dict(zip(['Human', 'Mouse'], custom_palette))
        lut = {"Human": self.human_color, "Mouse": self.mouse_color}
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

        rcParams["figure.subplot.top"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.3
        with plt.rc_context({"figure.figsize": (5, 5), "figure.dpi": (self.fig_dpi)}):
            g = sns.clustermap(np.concatenate((adata_human_embedding.X, adata_mouse_embedding.X[rints, :]), axis=0),
                           row_colors=row_colors, dendrogram_ratio=(.3, .2), figsize=(5, 5), cbar_pos=(0.82, 0.8, 0.04, 0.18), cmap='seismic') #'YlGnBu_r'

            #s.set(xlabel='Embedding features', ylabel='Samples')
            g.ax_heatmap.tick_params(tick2On=False, labelsize=False)
            g.ax_heatmap.set(xlabel='Embedding features', ylabel='Samples')
            #plt.ylabel('Samples')
            #plt.xlabel('Embedding features')
            #plt.yticks()
            handles = [Patch(facecolor=lut[name]) for name in lut]
            plt.legend(handles, lut, #title='',
                       bbox_to_anchor=(0.35, 1.02),
                       bbox_transform=plt.gcf().transFigure,
                       loc='upper right', frameon=False)
            #rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.05

        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        # sns.clustermap(Var_Corr, cmap=color_map, center=0.6)
        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/1_experiment_cross_species_clustering/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + 'Hiercluster_heco_embedding_concate.png', dpi=self.fig_dpi) # + fig_format, format=fig_format
        plt.show()

        adata_mouse_embedding.obs['dataset'] = 'Mouse'
        adata_human_embedding.obs['dataset'] = 'Human'

        adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
        print(adata_embedding)

        # Umap of the whole dataset
        # sc.pp.neighbors(adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
        # sc.tl.umap(adata_embedding)
        sc.pp.neighbors(adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(adata_embedding)

        palette = {'Mouse':self.mouse_color, 'Human':self.human_color}

        with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(adata_embedding, color=['dataset'], return_fig=True, legend_loc=None, palette=palette, size=15)
            plt.title('')
            # handles = [Patch(facecolor=lut[name]) for name in lut]
            # plt.legend(handles, lut, #title='Species',
            #            bbox_to_anchor=(1, 1),
            #            bbox_transform=plt.gcf().transFigure,
            #            loc='upper center', frameon=False)
            fg.savefig(
                save_path + 'umap_dataset.' + fig_format, format=fig_format)
        rcParams["figure.subplot.right"] = 0.66
        with plt.rc_context({"figure.figsize": (12, 8), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(adata_embedding, color=['dataset'], return_fig=True, legend_loc='right margin', palette=palette, size=15)
            plt.title('')
            fg.savefig(
                save_path + 'umap_dataset_right_margin.' + fig_format, format=fig_format)

            # plt.subplots_adjust(right=0.3)

        # sc.set_figure_params(dpi_save=200)

        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.27
        rcParams["figure.subplot.bottom"] = 0.15
        rcParams["figure.subplot.top"] = 0.9

        # if cfg.BrainAlign.homo_region_num >= 10:
        human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)
        homo_region_list = list(human_mouse_homo_region['Mouse'].values) + list(human_mouse_homo_region['Human'].values)
        homo_region_color_list = [self.mouse_64_color_dict[m_r] for m_r in list(human_mouse_homo_region['Mouse'].values)] + \
                                 [self.human_88_color_dict[h_r] for h_r in list(human_mouse_homo_region['Human'].values)]
        print(homo_region_list)
        adata_mouse_embedding_temp = adata_embedding[adata_embedding.obs['dataset'].isin(['Mouse'])]
        adata_mouse_embedding_temp.obs['region_name'] = ['Mouse ' + x for x in adata_mouse_embedding_temp.obs['region_name']]
        adata_human_embedding_temp = adata_embedding[adata_embedding.obs['dataset'].isin(['Human'])]
        adata_human_embedding_temp.obs['region_name'] = ['Human ' + x for x in
                                                         adata_human_embedding_temp.obs['region_name']]
        adata_embedding_temp = ad.concat([adata_mouse_embedding_temp, adata_human_embedding_temp])

        homo_region_list_temp = list(['Mouse ' + x for x in human_mouse_homo_region['Mouse'].values]) \
                                + list(['Human ' + x for x in human_mouse_homo_region['Human'].values])
        adata_embedding_homo = adata_embedding[
            adata_embedding.obs['region_name'].isin(homo_region_list)]
        adata_embedding_homo_temp = adata_embedding_temp[adata_embedding_temp.obs['region_name'].isin(homo_region_list_temp)]
        homo_region_palette = {k:v for k,v in zip(homo_region_list, homo_region_color_list)}
        homo_region_palette_temp = {k: v for k, v in zip(homo_region_list_temp, homo_region_color_list)}

        plt.rc('font', size=TINY_SIZE-3)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE-3)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE-3)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE-3)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE-3)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE-3)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE-3)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']
        with plt.rc_context({"figure.figsize": (18.5, 5), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(adata_embedding_homo_temp, color=['region_name'], return_fig=True,
                            legend_loc='right margin', palette=homo_region_palette_temp, size=20)
            plt.title('')
            fg.savefig(
                save_path + 'umap_types.' + fig_format, format=fig_format)

        homo_region_color_list_mouse = sns.color_palette(cc.glasbey, n_colors=len(list(['Mouse ' + x for x in human_mouse_homo_region['Mouse'].values])))
        homo_region_color_list_mouse = homo_region_color_list_mouse + homo_region_color_list_mouse
        homo_region_list_temp = list(['Mouse ' + x for x in human_mouse_homo_region['Mouse'].values]) \
                                + list(['Human ' + x for x in human_mouse_homo_region['Human'].values])
        homo_region_palette_mouse = {k: v for k, v in zip(homo_region_list_temp, homo_region_color_list_mouse)}
        with plt.rc_context({"figure.figsize": (18.5, 5), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(adata_embedding_homo_temp, color=['region_name'], return_fig=True,
                            legend_loc='right margin', palette=homo_region_palette_mouse, size=20)
            plt.title('')
            fg.savefig(
                save_path + 'umap_types_samecolor.' + fig_format, format=fig_format)


        with plt.rc_context({"figure.figsize": (18.5, 5), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(adata_embedding_homo_temp, color=['dataset'], return_fig=True,
                            legend_loc='right margin', palette={'Mouse':self.mouse_color, 'Human':self.human_color}, size=20)
            plt.title('')
            fg.savefig(
                save_path + 'umap_types_mouse_human.' + fig_format, format=fig_format)



        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']


        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.95
        with plt.rc_context({"figure.figsize": (7, 7), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(adata_embedding_homo, color=['region_name'], return_fig=True, legend_loc='on data', legend_fontsize='xx-small',
                      palette=homo_region_palette, size=20) #, legend_fontweight='medium'
            # gen_mpl_labels(
            #     adata_embedding_homo,
            #     'region_name',
            #     exclude=("None",),  # This was before we had the `nan` behaviour
            #     ax=fg,
            #     adjust_kwargs=dict(arrowprops=dict(arrowstyle='-', color='black'))
            # )
            plt.title('')
            fg.savefig(
                save_path + 'umap_types_ondata.' + fig_format, format=fig_format)


        with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(adata_embedding_homo, color=['dataset'], return_fig=True, legend_loc='on data', palette=palette, size=15)
            plt.title('')
            fg.savefig(
                save_path + 'umap_dataset_homo_regions_ondata.' + fig_format, format=fig_format)


        rcParams["figure.subplot.right"] = 0.5
        with plt.rc_context({"figure.figsize": (12, 6), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(adata_embedding_homo, color=['dataset'], return_fig=True, legend_loc='right margin', palette=palette,
                       size=15)
            plt.title('')
            fg.savefig(
                save_path + 'umap_dataset_homo_regions_rightmargin.' + fig_format, format=fig_format)


        rcParams["figure.subplot.right"] = 0.9
        with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(adata_embedding_homo, color=['dataset'], return_fig=True, legend_loc=None, palette=palette, size=15)
            plt.title('')
            fg.savefig(
                save_path + 'umap_dataset_homo_regions.' + fig_format, format=fig_format)

        # plt.subplots_adjust(right=0.3)

        # rcParams["figure.subplot.left"] = 0.125
        # rcParams["figure.subplot.right"] = 0.9
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
            with plt.rc_context({"figure.figsize": (12, 6), "figure.dpi": (self.fig_dpi)}):
                sc.pl.umap(adata_embedding, color=[k_num], return_fig=True, size=15).savefig(
                    save_path + 'umap_pca_{}.'.format(k_num) + fig_format, format=fig_format)
        human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)
        homo_region_list = list(human_mouse_homo_region['Mouse'].values) + list(human_mouse_homo_region['Human'].values)
        print(homo_region_list)
        adata_embedding_homo = adata_embedding[adata_embedding.obs['region_name'].isin(homo_region_list)]
        for k_num in kmeans_list:
            with plt.rc_context({"figure.figsize": (12, 6), "figure.dpi": (self.fig_dpi)}):
                sc.pl.umap(adata_embedding_homo, color=[k_num], return_fig=True).savefig(
                    save_path + 'umap_pca_homoregions_{}.'.format(k_num) + fig_format, format=fig_format)
        rcParams["figure.subplot.right"] = 0.9

    # def experiment_2_umap_evaluation(self):
    #     r"""
    #     Umap and evaluate the embeddings across species before and after integration.
    #     * 1. Umap of embeddings of each species.
    #     * 2. PCA separately and UMAP together.
    #     * 3. Compute the seurat alignment score for unligned data.
    #     * 4. Compute the seurat alignment score for ligned data.
    #     * 5. plot the box plots.
    #     :return:None
    #     """
    #     cfg = self.cfg
    #     fig_format = cfg.BrainAlign.fig_format
    #
    #     sns.set(style='white')
    #     TINY_SIZE = 24  # 39
    #     SMALL_SIZE = 28  # 42
    #     MEDIUM_SIZE = 32  # 46
    #     BIGGER_SIZE = 36  # 46
    #
    #     plt.rc('font', size=28)  # 35 controls default text sizes
    #     plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    #     plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    #     plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    #     plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    #     plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    #     plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    #
    #     rcParams['font.family'] = 'sans-serif'
    #     rcParams['font.sans-serif'] = ['Arial']
    #
    #     adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    #     adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
    #
    #     sc.pp.neighbors(adata_mouse_embedding, n_neighbors=cfg.ANALYSIS.mouse_umap_neighbor, metric='cosine',
    #                     use_rep='X')
    #     sc.tl.umap(adata_mouse_embedding)
    #
    #     save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/2_experiment_umap_seperate/'
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #
    #
    #     rcParams["figure.subplot.left"] = 0.05
    #     rcParams["figure.subplot.right"] = 0.55
    #
    #     # plt.tight_layout()
    #     # sc.set_figure_params(dpi_save=200)
    #     sc.pl.umap(adata_mouse_embedding, color=['region_name'], return_fig=True, legend_loc='right margin').savefig(
    #         save_path + 'umap_mouse.' + fig_format, format=fig_format)
    #     # plt.subplots_adjust(left = 0.1, right=5)
    #     with plt.rc_context({"figure.figsize": (12, 6)}):
    #         sc.pp.neighbors(adata_human_embedding, n_neighbors=cfg.ANALYSIS.human_umap_neighbor, metric='cosine',
    #                         use_rep='X')
    #         sc.tl.umap(adata_human_embedding)
    #
    #     with plt.rc_context({"figure.figsize": (12, 6)}):
    #
    #         # sc.set_figure_params(dpi_save=200)
    #         sc.pl.umap(adata_human_embedding, color=['region_name'], return_fig=True, size=30,
    #                    legend_loc='right margin').savefig(
    #             save_path + 'umap_human.' + fig_format, format=fig_format)
    #     # plt.subplots_adjust(left=0.1, right=5)
    #     # plt.tight_layout()
    #     rcParams["figure.subplot.left"] = 0.125
    #     rcParams["figure.subplot.right"] = 0.9
    #
    #
    #     # umap together before integration
    #     # pca separately and do dimension reduction
    #     adata_mouse_expression = sc.read_h5ad(cfg.CAME.path_rawdata1)
    #     sc.tl.pca(adata_mouse_expression, svd_solver='arpack', n_comps=30)
    #     sc.pp.neighbors(adata_mouse_expression, n_neighbors=cfg.ANALYSIS.mouse_umap_neighbor, metric='cosine',
    #                     use_rep='X_pca')
    #     sc.tl.umap(adata_mouse_expression)
    #
    #     adata_human_expression = sc.read_h5ad(cfg.CAME.path_rawdata2)
    #     sc.tl.pca(adata_human_expression, svd_solver='arpack', n_comps=30)
    #     sc.pp.neighbors(adata_human_expression, n_neighbors=cfg.ANALYSIS.human_umap_neighbor, metric='cosine',
    #                     use_rep='X_pca')
    #     sc.tl.umap(adata_human_expression)
    #
    #     adata_mouse_expression.obs['dataset'] = 'mouse'
    #     adata_human_expression.obs['dataset'] = 'human'
    #
    #     adata_expression = ad.concat([adata_mouse_expression, adata_human_expression])
    #     with plt.rc_context({"figure.figsize": (8, 8)}):
    #         sc.pl.umap(adata_expression, color=['dataset'], return_fig=True, legend_loc='on data').savefig(
    #             save_path + 'umap_dataset_before_integration_ondata.' + fig_format, format=fig_format)
    #
    #     align_score_dict = {'Method':[], 'alignment score':[]}
    #     # compute alignment score for unaligned data
    #     X = adata_expression.obsm['X_pca']
    #     Y = np.concatenate([np.zeros((adata_mouse_expression.n_obs, 1)), np.ones((adata_human_expression.n_obs, 1))], axis=0)
    #     unaligned_score = seurat_alignment_score(X, Y)
    #     align_score_dict['Method'].append('Unaligned')
    #     align_score_dict['alignment score'].append(unaligned_score)
    #
    #     # after integration
    #     sc.tl.pca(adata_mouse_embedding, svd_solver='arpack', n_comps=30)
    #     sc.pp.neighbors(adata_mouse_embedding, n_neighbors=cfg.ANALYSIS.mouse_umap_neighbor, metric='cosine',
    #                     use_rep='X_pca')
    #     sc.tl.umap(adata_mouse_embedding)
    #
    #
    #     sc.tl.pca(adata_human_embedding, svd_solver='arpack', n_comps=30)
    #     sc.pp.neighbors(adata_human_embedding, n_neighbors=cfg.ANALYSIS.human_umap_neighbor, metric='cosine',
    #                     use_rep='X_pca')
    #     sc.tl.umap(adata_human_embedding)
    #
    #     adata_mouse_embedding.obs['dataset'] = 'mouse'
    #     adata_human_embedding.obs['dataset'] = 'human'
    #
    #     adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
    #     with plt.rc_context({"figure.figsize": (8, 8)}):
    #         sc.pl.umap(adata_embedding, color=['dataset'], return_fig=True, legend_loc='on data').savefig(
    #             save_path + 'umap_dataset_after_integration_ondata.' + fig_format, format=fig_format)
    #     # compute alignment score for aligned data
    #     X = adata_embedding.obsm['X_pca']
    #     Y = np.concatenate([np.zeros((adata_mouse_embedding.n_obs, 1)), np.ones((adata_human_embedding.n_obs, 1))],
    #                        axis=0)
    #     aligned_score = seurat_alignment_score(X, Y)
    #     align_score_dict['Method'].append('Aligned')
    #     align_score_dict['alignment score'].append(aligned_score)
    #     print(align_score_dict)
    #     align_score_df  = pd.DataFrame.from_dict(align_score_dict)
    #     # plot bar plot of alignment score
    #     my_pal = {"Unaligned": '#CD4537', "Aligned": '#2C307A'}
    #     # sns.set_theme(style="whitegrid")
    #     # tips = sns.load_dataset("tips")
    #     plt.figure(figsize=(8, 8))
    #     ax = sns.barplot(x="Method", y="alignment score", data=align_score_df, order=["Unaligned", "Aligned"], palette=my_pal)
    #     plt.xlabel('')
    #     plt.savefig(save_path + 'seurate_alignment_score.svg')

    def experiment_2_umap_evaluation(self):
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
        TINY_SIZE = 11 # 39
        SMALL_SIZE = 11  # 42
        MEDIUM_SIZE = 12  # 46
        BIGGER_SIZE = 12  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')

        sc.pp.neighbors(adata_mouse_embedding, n_neighbors=cfg.ANALYSIS.mouse_umap_neighbor, metric='cosine',
                        use_rep='X')
        sc.tl.umap(adata_mouse_embedding)

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/2_experiment_umap_seperate/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)


        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.55

        palette = {'Mouse':self.mouse_color, 'Human':self.human_color}

        umap_size_dict = {'Mouse':cfg.ANALYSIS.mouse_umap_marker_size, 'Human':cfg.ANALYSIS.human_umap_marker_size}

        # plt.tight_layout()
        # sc.set_figure_params(dpi_save=200)
        with plt.rc_context({"figure.figsize": (4, 2), "figure.dpi": (self.fig_dpi)}):
            sc.pl.umap(adata_mouse_embedding, color=['region_name'], return_fig=True, legend_loc='right margin').savefig(
                save_path + 'umap_mouse.' + fig_format, format=fig_format)
        # plt.subplots_adjust(left = 0.1, right=5)
        #
        sc.pp.neighbors(adata_human_embedding, n_neighbors=cfg.ANALYSIS.human_umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(adata_human_embedding)

        with plt.rc_context({"figure.figsize": (4, 2), "figure.dpi": (self.fig_dpi)}):
            # sc.set_figure_params(dpi_save=200)
            sc.pl.umap(adata_human_embedding, color=['region_name'], return_fig=True, size=30,
                       legend_loc='right margin').savefig(
                save_path + 'umap_human.' + fig_format, format=fig_format)
        # plt.subplots_adjust(left=0.1, right=5)
        # plt.tight_layout()
        rcParams["figure.subplot.left"] = 0.2
        rcParams["figure.subplot.right"] = 0.9


        # umap together before integration
        # pca separately and do dimension reduction
        adata_mouse_expression = sc.read_h5ad(cfg.CAME.path_rawdata1)
        sc.tl.pca(adata_mouse_expression, svd_solver='arpack', n_comps=30)
        sc.pp.neighbors(adata_mouse_expression, n_neighbors=cfg.ANALYSIS.mouse_umap_neighbor, metric='cosine',
                        use_rep='X_pca')
        sc.tl.umap(adata_mouse_expression)

        adata_human_expression = sc.read_h5ad(cfg.CAME.path_rawdata2)
        sc.tl.pca(adata_human_expression, svd_solver='arpack', n_comps=30)
        sc.pp.neighbors(adata_human_expression, n_neighbors=cfg.ANALYSIS.human_umap_neighbor, metric='cosine',
                        use_rep='X_pca')
        sc.tl.umap(adata_human_expression)

        adata_mouse_expression.obs['dataset'] = 'Mouse'
        adata_human_expression.obs['dataset'] = 'Human'

        adata_expression = ad.concat([adata_mouse_expression, adata_human_expression])

        adata_umap_size_list = [umap_size_dict[x] for x in adata_expression.obs['dataset']]

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.bottom"] = 0.12
        with plt.rc_context({"figure.figsize": (3, 3), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(adata_expression, color=['dataset'], return_fig=True, legend_loc=None, title='', size=adata_umap_size_list, palette=palette)
            plt.title('')
            fg.savefig(
                save_path + 'umap_dataset_before_integration_ondata.' + fig_format, format=fig_format)


        rcParams["figure.subplot.right"] = 0.66

        with plt.rc_context({"figure.figsize": (4, 3), "figure.dpi": (self.fig_dpi)}):
        #plt.figure(figsize=(8,8), dpi=self.fig_dpi)
            fg = sc.pl.umap(adata_expression, color=['dataset'], return_fig=True, legend_loc=None, title='', size=adata_umap_size_list, palette=palette)
            plt.title('')
            fg.savefig(
                save_path + 'umap_dataset_before_integration_rightmargin.' + fig_format, format=fig_format)

        align_score_dict = {'Method':[], 'Alignment score':[]}
        # compute alignment score for unaligned data
        X = adata_expression.obsm['X_pca']
        Y = np.concatenate([np.zeros((adata_mouse_expression.n_obs, 1)), np.ones((adata_human_expression.n_obs, 1))], axis=0)
        unaligned_score = seurat_alignment_score(X, Y)
        align_score_dict['Method'].append('Unaligned')
        align_score_dict['Alignment score'].append(unaligned_score)

        # after integration

        adata_mouse_embedding.obs['dataset'] = 'Mouse'
        adata_human_embedding.obs['dataset'] = 'Human'

        adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])

        adata_umap_size_list = [umap_size_dict[x] for x in adata_embedding.obs['dataset']]

        sc.tl.pca(adata_embedding, svd_solver='arpack', n_comps=30)
        sc.pp.neighbors(adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine',
                        use_rep='X')
        sc.tl.umap(adata_embedding)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.68#0.6545
        with plt.rc_context({"figure.figsize": (4, 3), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(adata_embedding, color=['dataset'], return_fig=True, legend_loc='right margin', title='', size= adata_umap_size_list, palette=palette)
            plt.title('')
            fg.savefig(save_path + 'umap_dataset_after_integration_rightmargin.' + fig_format, format=fig_format)


        rcParams["figure.subplot.right"] = 0.9
        #rcParams["figure.subplot.bottom"] = 0.1
        with plt.rc_context({"figure.figsize": (3, 3), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(adata_embedding, color=['dataset'], return_fig=True, legend_loc=None, title='', size= adata_umap_size_list, palette=palette)
            plt.title('')
            fg.savefig(save_path + 'umap_dataset_after_integration.' + fig_format, format=fig_format)
        # compute alignment score for aligned data
        X = adata_embedding.obsm['X_umap']
        Y = np.concatenate([np.zeros((adata_mouse_embedding.n_obs, 1)), np.ones((adata_human_embedding.n_obs, 1))],
                           axis=0)
        aligned_score = seurat_alignment_score(X, Y)
        align_score_dict['Method'].append('BrainAlign')
        align_score_dict['Alignment score'].append(aligned_score)
        # CAME results
        #align_score_dict['Method'].append('CAME')
        #align_score_dict['alignment score'].append(0.020107865877827872)
        print(align_score_dict)
        align_score_df  = pd.DataFrame.from_dict(align_score_dict)
        # plot bar plot of alignment score
        #my_pal = {"Unaligned": '#CD4537', "CAME":'#28AF60', "BrainAlign": '#2C307A'}
        #my_pal = {"Unaligned": '#CD4537', "BrainAlign": '#2C307A'}
        my_pal = {"Unaligned": '#F4D03E', "BrainAlign": '#3A97D2'}
        rcParams["figure.subplot.left"] = 0.3
        rcParams["figure.subplot.right"] = 0.90
        rcParams["figure.subplot.bottom"] = 0.25
        plt.figure(figsize=(2, 2.5), dpi=self.fig_dpi)
        #sns.set(rc={"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)})
        #with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        ax = sns.barplot(x="Method", y="Alignment score", data=align_score_df, order=["Unaligned", "BrainAlign"], palette=my_pal, width=0.618) #
        plt.xlabel('')
        for item in ax.get_xticklabels():
            item.set_rotation(20)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")

        plt.savefig(save_path + 'seurate_alignment_score.' + fig_format, format=fig_format, dpi=self.fig_dpi)
        align_score_df.to_csv(save_path + 'seurat_alignment_score.csv')

        adata_mouse_embedding.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/s_embeddings.h5ad')
        adata_human_embedding.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/v_embeddings.h5ad')
        rcParams["figure.subplot.left"] = 0.05

    def experiment_2_umap_diagram(self):

        sns.set(style='white')
        TINY_SIZE = 28#24  # 39
        SMALL_SIZE = 28#28  # 42
        MEDIUM_SIZE = 32#32  # 46
        BIGGER_SIZE = 32#36  # 46

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
        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/2_experiment_umap_diagram/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # No label order version
        '''
        Step 1: Compute average embedding of every region in two species, use two dict to store;
        Step 2: Compute similarity matrix, use np array to store;
        Step 3: Heatmap.
        '''
        adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')

        adata_mouse_embedding.obs['dataset'] = 'mouse'
        adata_human_embedding.obs['dataset'] = 'human'

        adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
        print(adata_embedding)

        # Umap of the whole dataset
        # sc.pp.neighbors(adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
        # sc.tl.umap(adata_embedding)
        sc.pp.neighbors(adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(adata_embedding)

        with plt.rc_context({"figure.figsize": (8, 8)}):
            sc.pl.umap(adata_embedding, color=['dataset'], return_fig=True, legend_loc=None, size=15, title='').savefig(
                save_path + 'umap_together_marker15.' + fig_format, format=fig_format)
            #plt.title('')

        sc.tl.umap(adata_embedding, n_components=3)
        with plt.rc_context({"figure.figsize": (8, 8)}):
            sc.pl.umap(adata_embedding, color=['dataset'], return_fig=True, legend_loc='on data', size=15, title='').savefig( #, projection='3d', components=['1,2,3']
                save_path + 'umap_together_3d_marker15.' + fig_format, format=fig_format)

    def experiment_2_umap_evaluation_single(self, method='CAME'):
        r"""
        Umap and evaluate the embeddings across species before and after integration.
        * 1. Umap of embeddings of each species.
        * 2. PCA separately and UMAP together.
        * 3. Compute the seurat alignment score for unaligned data.
        * 4. Compute the seurat alignment score for aligned data.
        * 5. plot the box plots.
        :return:None
        """
        cfg = self.cfg
        fig_format = cfg.BrainAlign.fig_format

        sns.set(style='white')
        TINY_SIZE = 24  # 39
        SMALL_SIZE = 28  # 42
        MEDIUM_SIZE = 32  # 46
        BIGGER_SIZE = 36  # 46

        plt.rc('font', size=28)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')

        sc.pp.neighbors(adata_mouse_embedding, n_neighbors=cfg.ANALYSIS.mouse_umap_neighbor, metric='cosine',
                        use_rep='X')
        sc.tl.umap(adata_mouse_embedding)

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/2_experiment_umap_seperate/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.55

        # plt.tight_layout()
        # sc.set_figure_params(dpi_save=200)
        sc.pl.umap(adata_mouse_embedding, color=['region_name'], return_fig=True, legend_loc='right margin').savefig(
            save_path + 'umap_mouse.' + fig_format, format=fig_format)
        # plt.subplots_adjust(left = 0.1, right=5)
        with plt.rc_context({"figure.figsize": (12, 6)}):
            sc.pp.neighbors(adata_human_embedding, n_neighbors=cfg.ANALYSIS.human_umap_neighbor, metric='cosine',
                            use_rep='X')
            sc.tl.umap(adata_human_embedding)

        with plt.rc_context({"figure.figsize": (12, 6)}):

            # sc.set_figure_params(dpi_save=200)
            sc.pl.umap(adata_human_embedding, color=['region_name'], return_fig=True, size=30,
                       legend_loc='right margin').savefig(
                save_path + 'umap_human.' + fig_format, format=fig_format)
        # plt.subplots_adjust(left=0.1, right=5)
        # plt.tight_layout()
        rcParams["figure.subplot.left"] = 0.125
        rcParams["figure.subplot.right"] = 0.9


        align_score_dict = {'Method':[], 'alignment score':[]}
        # after integration
        sc.tl.pca(adata_mouse_embedding, svd_solver='arpack', n_comps=30)
        sc.pp.neighbors(adata_mouse_embedding, n_neighbors=cfg.ANALYSIS.mouse_umap_neighbor, metric='cosine',
                        use_rep='X_pca')
        sc.tl.umap(adata_mouse_embedding)


        sc.tl.pca(adata_human_embedding, svd_solver='arpack', n_comps=30)
        sc.pp.neighbors(adata_human_embedding, n_neighbors=cfg.ANALYSIS.human_umap_neighbor, metric='cosine',
                        use_rep='X_pca')
        sc.tl.umap(adata_human_embedding)

        adata_mouse_embedding.obs['dataset'] = 'mouse'
        adata_human_embedding.obs['dataset'] = 'human'

        adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
        with plt.rc_context({"figure.figsize": (8, 8)}):
            sc.pl.umap(adata_embedding, color=['dataset'], return_fig=True, legend_loc='on data').savefig(
                save_path + 'umap_dataset_after_integration_ondata.' + fig_format, format=fig_format)
        # compute alignment score for aligned data
        X = adata_embedding.obsm['X_pca']
        Y = np.concatenate([np.zeros((adata_mouse_embedding.n_obs, 1)), np.ones((adata_human_embedding.n_obs, 1))],
                           axis=0)
        aligned_score = seurat_alignment_score(X, Y)
        align_score_dict['Method'].append(method)
        align_score_dict['alignment score'].append(aligned_score)
        print(align_score_dict)
        align_score_df  = pd.DataFrame.from_dict(align_score_dict)
        align_score_df.to_csv(save_path + 'seurat_alignment_score.csv')
        return aligned_score


    '''
    The following four functions are used to calcalute the correlations between homologous brain 
    regions and non-homologous regions. 
    For well-learned general embeddings, we expect to see homologous regions of two species, such
    as Mouse and Human, correlate significantly stronger than those between non-homologous regions.
    '''

    #############################################################################
    def homo_corr(self):
        # No label order version
        '''
        Step 1: Compute average embedding of every region in two species, use two dict to store;
        Step 2: Compute similarity matrix, use np array to store;
        Step 3: Heatmap.
        '''
        cfg = self.cfg

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

        save_path_root = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_experiment_homo_random/' #.format(distance_type)

        if not os.path.exists(save_path_root):
            os.makedirs(save_path_root)

        with open(save_path_root + 'homo_region_dict.pkl', 'wb') as f:
            pickle.dump(home_region_dict, f)

        for human_region, mouse_region in home_region_dict.items():

            save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_experiment_homo_random/human_{}_mouse_{}/'.format(
                human_region,
                mouse_region)

            save_path = save_path.replace(' ', '_')
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
            plt.savefig(save_path + 'human.png')
            # plt.show()

            print('mouse_region', mouse_region)
            adata_mouse_embedding_region = adata_mouse_embedding[
                adata_mouse_embedding.obs['region_name'] == mouse_region]
            # plt.figure(figsize=(16, 16))
            if adata_mouse_embedding_region.X.shape[0] <= 0:
                continue
            sns.clustermap(adata_mouse_embedding_region.X, metric=distance_type).fig.suptitle(
                'Mouse - {}'.format(mouse_region))
            plt.savefig(save_path + 'mouse.png')
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
            plt.savefig(save_path + 'cross.png')

            k += 1
            print('{}-th region finished!'.format(k))
            # plt.show()

        with open(save_path_root + 'human_mouse_correlation_dict.pkl', 'wb') as f:
            pickle.dump(human_mouse_correlation_dict, f)
        with open(save_path_root + 'human_correlation_dict.pkl', 'wb') as f:
            pickle.dump(human_correlation_dict, f)
        with open(save_path_root + 'mouse_correlation_dict.pkl', 'wb') as f:
            pickle.dump(mouse_correlation_dict, f)

    def random_corr(self):
        # No label order version
        '''
        Step 1: Compute average embedding of every region in two species, use two dict to store;
        Step 2: Compute similarity matrix, use np array to store;
        Step 3: Heatmap.
        '''
        cfg = self.cfg

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
            save_path_root = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_experiment_homo_random/random_Hiercluster_{}/'.format(
                distance_type)
        else:
            save_path_root = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_experiment_homo_random/35_random_Hiercluster_{}/'.format(
                distance_type)

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
                        save_path = save_path_root + '/human_{}_mouse_{}/'.format(distance_type, human_region,
                                                                                  mouse_region)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        sns.clustermap(adata_human_embedding_region.X, metric=distance_type).fig.suptitle(
                            'Human - {}'.format(human_region))
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
                        sns.clustermap(adata_mouse_embedding_region.X, metric=distance_type).fig.suptitle(
                            'Mouse - {}'.format(mouse_region))
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

    def plot_homo_random(self):

        # No label order version
        '''
        Step 1:load human and mouse cross expression data of homologous regions, and random regions
        Step 2: plot bar, mean and std
        '''
        sns.set(style='white')
        TINY_SIZE = 10  # 39
        SMALL_SIZE = 10  # 42
        MEDIUM_SIZE = 12  # 46
        BIGGER_SIZE = 12 # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        cfg = self.cfg
        fig_format = self.fig_format

        homo_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_experiment_homo_random/'
        if not os.path.exists(homo_region_data_path):
            os.makedirs(homo_region_data_path)
        with open(homo_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
            human_mouse_correlation_dict = pickle.load(f)

        home_len = len(human_mouse_correlation_dict['mean'])
        home_random_type = ['Homologous'] * home_len
        human_mouse_correlation_dict['type'] = home_random_type
        # data_df = pd.DataFrame.from_dict(human_mouse_correlation_dict)

        if cfg.HOMO_RANDOM.random_field == 'all':
            random_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_experiment_homo_random/random_Hiercluster_correlation/'
        else:
            random_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_experiment_homo_random/35_random_Hiercluster_correlation/'
        with open(random_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
            random_human_mouse_correlation_dict = pickle.load(f)

        random_len = len(random_human_mouse_correlation_dict['mean'])
        home_random_type = ['Random'] * random_len
        random_human_mouse_correlation_dict['type'] = home_random_type
        concat_dict = {}
        for k, v in random_human_mouse_correlation_dict.items():
            concat_dict[k] = human_mouse_correlation_dict[k] + random_human_mouse_correlation_dict[k]
        data_df = pd.DataFrame.from_dict(concat_dict)

        if cfg.HOMO_RANDOM.random_field == 'all':
            save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_experiment_homo_random/homo_random/'
        else:
            save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_experiment_homo_random/35_homo_random/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # my_pal = {"homo region": (0/255,149/255,182/255), "random":(178/255, 0/255, 32/255)}
        #my_pal = {"Homologous": (0 / 255, 149 / 255, 182 / 255), "Random": (178 / 255, 0 / 255, 32 / 255)}
        my_pal = {"Homologous": '#F59E1D', "Random": '#28AF60'}
        #sns.set_theme(style="whitegrid")
        # tips = sns.load_dataset("tips")

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.32
        rcParams["figure.subplot.bottom"] = 0.25

        plt.figure(figsize=(2, 2.5), dpi=self.fig_dpi)
        #with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        #sns.set(rc={"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)})
        ax = sns.boxplot(x="type", y="mean", data=data_df, order=["Homologous", "Random"], palette=my_pal, width = 0.68) #
        add_stat_annotation(ax, data=data_df, x="type", y="mean", order=["Homologous", "Random"],
                            box_pairs=[("Homologous", "Random")],
                            test='t-test_ind', text_format='star', loc='inside', verbose=2) #text_format='star'
        for item in ax.get_xticklabels():
            item.set_rotation(20)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")

        plt.title('')
        plt.ylabel('Correlation')
        plt.xlabel('')
        plt.savefig(save_path + 'mean.'+ fig_format, format=fig_format, dpi=self.fig_dpi)
        plt.show()

        plt.figure(figsize=(2,2.5), dpi=self.fig_dpi)
        ax = sns.boxplot(x="type", y="std", data=data_df, order=["Homologous", "Random"], palette=my_pal)
        plt.title('')
        plt.savefig(save_path + 'std.'+ fig_format, format=fig_format, dpi=self.fig_dpi)
        #plt.show()

        homo_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_experiment_homo_random/'
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

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.bottom"] = 0.15

        sns.set(style='white')
        TINY_SIZE = 16  # 39
        SMALL_SIZE = 16  # 42
        MEDIUM_SIZE = 18  # 46
        BIGGER_SIZE = 18  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']


        ax = plt.figure(figsize=(4, 4), dpi=self.fig_dpi)
        #with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        g = sns.jointplot(data=pd.DataFrame.from_dict(human_mouse_dict_mean), x="Human", y="Mouse", kind="reg", height=4, color='black', ax=ax)
        plt.title('')

        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(human_mouse_dict_mean['Human']), np.array(human_mouse_dict_mean['Mouse']))
        plt.text(0.35, 0.30, f'R = 0.506, P < 0.05')
        print(f'R = {r_value}, P = {p_value}')
        plt.setp(g.ax_marg_y.patches, color=self.mouse_color)
        plt.setp(g.ax_marg_x.patches, color=self.human_color)
        plt.xlabel('Human mean correlation')
        plt.ylabel('Mouse mean correlation')
        plt.savefig(save_path + 'mean_human_mouse.'+ fig_format, format=fig_format, dpi=self.fig_dpi)
           # plt.show()

        ax = plt.figure(figsize=(4, 4), dpi=self.fig_dpi)
        sns.jointplot(data=pd.DataFrame.from_dict(human_mouse_dict_std), x="Human", y="Mouse", kind="reg", height=4, ax=ax)
        plt.title('')
        plt.savefig(save_path + 'std_human_mouse.'+ fig_format, format=fig_format, dpi=self.fig_dpi)
        #plt.show()
        rcParams["figure.subplot.left"] = 0.1

    def ttest_homo_random(self):
        cfg = self.cfg
        # No label order version
        '''
        Step 1:load human and mouse cross expression data of homologous regions, and random regions
        Step 2: plot bar, mean and std
        '''
        # Read ordered labels
        human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
        mouse_67_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)

        homo_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_experiment_homo_random/'
        with open(homo_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
            human_mouse_correlation_dict = pickle.load(f)

        home_len = len(human_mouse_correlation_dict['mean'])
        home_random_type = ['homologous'] * home_len
        human_mouse_correlation_dict['type'] = home_random_type
        # data_df = pd.DataFrame.from_dict(human_mouse_correlation_dict)

        if cfg.HOMO_RANDOM.random_field == 'all':
            random_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_experiment_homo_random/random_Hiercluster_correlation/'
        else:
            random_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_experiment_homo_random/35_random_Hiercluster_correlation/'
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
            save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_experiment_homo_random/homo_random/'
        else:
            save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_experiment_homo_random/35_homo_random/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # my_pal = {"homo region": (0/255,149/255,182/255), "random":(178/255, 0/255, 32/255)}
        my_pal = {"homologous": (0 / 255, 149 / 255, 182 / 255), "random": (178 / 255, 0 / 255, 32 / 255)}
        # sns.set_theme(style="whitegrid")
        # tips = sns.load_dataset("tips")

        #print(data_df)

        random_df = data_df[data_df['type'] == 'random']
        #print(random_df)
        mean_random_list = random_df['mean'].values
        mean_r = np.mean(mean_random_list)
        std_r = np.std(mean_random_list)

        homologous_df = data_df[data_df['type'] == 'homologous']
        #print(homologous_df)
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

        with open(save_path + 't_test_result.txt', 'w') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print(stats.ttest_ind(
                mean_homo_list,
                mean_random_list,
                equal_var=False
            ))
            sys.stdout = original_stdout  # Reset the standard output to its original value

    def experiment_3_homo_random(self):

        sns.set(style='white')
        TINY_SIZE = 28  # 24  # 39
        SMALL_SIZE = 28  # 28  # 42
        MEDIUM_SIZE = 32  # 32  # 46
        BIGGER_SIZE = 32  # 36  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']



        #self.homo_corr()
        #self.random_corr()
        self.plot_homo_random()
        self.ttest_homo_random()


    def experiment_3_1_homo_random_beforeAlign(self):

        sns.set(style='white')
        TINY_SIZE = 16  # 24  # 39
        SMALL_SIZE = 16  # 28  # 42
        MEDIUM_SIZE = 18  # 32  # 46
        BIGGER_SIZE = 18  # 36  # 46

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


        adata_mouse_expression = sc.read_h5ad(cfg.CAME.path_rawdata1)
        sc.tl.pca(adata_mouse_expression, svd_solver='arpack', n_comps=30)
        adata_mouse_embedding = ad.AnnData(adata_mouse_expression.obsm['X_pca'])
        adata_mouse_embedding.obs_names = adata_mouse_expression.obs_names
        adata_mouse_embedding.obs['region_name'] = adata_mouse_expression.obs['region_name']

        adata_human_expression = sc.read_h5ad(cfg.CAME.path_rawdata2)
        sc.tl.pca(adata_human_expression, svd_solver='arpack', n_comps=30)
        adata_human_embedding = ad.AnnData(adata_human_expression.obsm['X_pca'])
        adata_human_embedding.obs_names = adata_human_expression.obs_names
        adata_human_embedding.obs['region_name'] = adata_human_expression.obs['region_name']

        adata_mouse_embedding.obs['dataset'] = 'Mouse'
        adata_human_embedding.obs['dataset'] = 'Human'


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

        save_path_root = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_1_experiment_homo_random/'  # .format(distance_type)

        if not os.path.exists(save_path_root):
            os.makedirs(save_path_root)

        '''

        with open(save_path_root + 'homo_region_dict.pkl', 'wb') as f:
            pickle.dump(home_region_dict, f)

        for human_region, mouse_region in home_region_dict.items():

            save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_1_experiment_homo_random/human_{}_mouse_{}/'.format(
                human_region,
                mouse_region)

            save_path = save_path.replace(' ', '_')
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
            plt.savefig(save_path + 'human.png')
            # plt.show()

            print('mouse_region', mouse_region)
            adata_mouse_embedding_region = adata_mouse_embedding[
                adata_mouse_embedding.obs['region_name'] == mouse_region]
            # plt.figure(figsize=(16, 16))
            if adata_mouse_embedding_region.X.shape[0] <= 0:
                continue
            sns.clustermap(adata_mouse_embedding_region.X, metric=distance_type).fig.suptitle(
                'Mouse - {}'.format(mouse_region))
            plt.savefig(save_path + 'mouse.png')
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
            plt.savefig(save_path + 'cross.png')

            k += 1
            print('{}-th region finished!'.format(k))
            # plt.show()

        with open(save_path_root + 'human_mouse_correlation_dict.pkl', 'wb') as f:
            pickle.dump(human_mouse_correlation_dict, f)
        with open(save_path_root + 'human_correlation_dict.pkl', 'wb') as f:
            pickle.dump(human_correlation_dict, f)
        with open(save_path_root + 'mouse_correlation_dict.pkl', 'wb') as f:
            pickle.dump(mouse_correlation_dict, f)


        # random_corr()
        # Read ordered labels
        human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
        mouse_67_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)

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
            save_path_root = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_1_experiment_homo_random/random_Hiercluster_{}/'.format(
                distance_type)
        else:
            save_path_root = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_1_experiment_homo_random/35_random_Hiercluster_{}/'.format(
                distance_type)

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
                        save_path = save_path_root + '/human_{}_mouse_{}/'.format(distance_type, human_region,
                                                                                  mouse_region)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        sns.clustermap(adata_human_embedding_region.X, metric=distance_type).fig.suptitle(
                            'Human - {}'.format(human_region))
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
                        sns.clustermap(adata_mouse_embedding_region.X, metric=distance_type).fig.suptitle(
                            'Mouse - {}'.format(mouse_region))
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
            
        '''


        #####################################################################################
        # plot random corr
        # No label order version
        
        #Step 1:load human and mouse cross expression data of homologous regions, and random regions
        #Step 2: plot bar, mean and std
        
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
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        cfg = self.cfg
        fig_format = self.fig_format

        homo_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_1_experiment_homo_random/'
        with open(homo_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
            human_mouse_correlation_dict = pickle.load(f)

        home_len = len(human_mouse_correlation_dict['mean'])
        home_random_type = ['Homologous'] * home_len
        human_mouse_correlation_dict['type'] = home_random_type
        # data_df = pd.DataFrame.from_dict(human_mouse_correlation_dict)

        if cfg.HOMO_RANDOM.random_field == 'all':
            random_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_1_experiment_homo_random/random_Hiercluster_correlation/'
        else:
            random_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_1_experiment_homo_random/35_random_Hiercluster_correlation/'
        with open(random_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
            random_human_mouse_correlation_dict = pickle.load(f)

        random_len = len(random_human_mouse_correlation_dict['mean'])
        home_random_type = ['Random'] * random_len
        random_human_mouse_correlation_dict['type'] = home_random_type
        concat_dict = {}
        for k, v in random_human_mouse_correlation_dict.items():
            concat_dict[k] = human_mouse_correlation_dict[k] + random_human_mouse_correlation_dict[k]
        data_df = pd.DataFrame.from_dict(concat_dict)

        if cfg.HOMO_RANDOM.random_field == 'all':
            save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_1_experiment_homo_random/homo_random/'
        else:
            save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_1_experiment_homo_random/35_homo_random/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # my_pal = {"homo region": (0/255,149/255,182/255), "random":(178/255, 0/255, 32/255)}
        # my_pal = {"Homologous": (0 / 255, 149 / 255, 182 / 255), "Random": (178 / 255, 0 / 255, 32 / 255)}
        my_pal = {"Homologous": '#F59E1D', "Random": '#28AF60'}
        # sns.set_theme(style="whitegrid")
        # tips = sns.load_dataset("tips")

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.32
        rcParams["figure.subplot.bottom"] = 0.25

        plt.figure(figsize=(2, 2.5), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        # sns.set(rc={"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)})
        ax = sns.boxplot(x="type", y="mean", data=data_df, order=["Homologous", "Random"], palette=my_pal,
                         width=0.68)  #
        add_stat_annotation(ax, data=data_df, x="type", y="mean", order=["Homologous", "Random"],
                            box_pairs=[("Homologous", "Random")],
                            test='t-test_ind', text_format='star', loc='inside', verbose=2)  # text_format='star'
        for item in ax.get_xticklabels():
            item.set_rotation(20)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")

        plt.title('')
        plt.ylabel('Correlation')
        plt.xlabel('')
        plt.savefig(save_path + 'mean.' + fig_format, format=fig_format, dpi=self.fig_dpi)
        plt.show()

        plt.figure(figsize=(2, 2.5), dpi=self.fig_dpi)
        ax = sns.boxplot(x="type", y="std", data=data_df, order=["Homologous", "Random"], palette=my_pal)
        plt.title('')
        plt.savefig(save_path + 'std.' + fig_format, format=fig_format, dpi=self.fig_dpi)
        # plt.show()

        homo_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_1_experiment_homo_random/'
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

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.bottom"] = 0.15

        sns.set(style='white')
        TINY_SIZE = 16  # 39
        SMALL_SIZE = 16  # 42
        MEDIUM_SIZE = 18  # 46
        BIGGER_SIZE = 18  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        if cfg.HOMO_RANDOM.random_field == 'all':
            save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_1_experiment_homo_random/homo_random/'
        else:
            save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_1_experiment_homo_random/35_homo_random/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.15
        # rcParams["figure.subplot.bottom"] = 0.15
        # rcParams["figure.subplot.top"] = 0.9

        human_mouse_dict_amount = {'Human': [], 'Mouse': []}
        for human_region, mouse_region in home_region_dict.items():
            mouse_r_num = adata_mouse_embedding[adata_mouse_embedding.obs['region_name'].isin([mouse_region])].n_obs
            human_r_num = adata_human_embedding[adata_human_embedding.obs['region_name'].isin([human_region])].n_obs

            human_mouse_dict_amount['Mouse'].append(mouse_r_num)
            human_mouse_dict_amount['Human'].append(human_r_num)

        #ax = plt.figure(figsize=(4, 4), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        g = sns.jointplot(data=pd.DataFrame.from_dict(human_mouse_dict_amount), x="Human", y="Mouse",
                          kind="reg", height=4, color='black') #ax=ax, ,  marginal_kws={'color': 'black'}
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(human_mouse_dict_amount['Human']),
                                                                       np.array(human_mouse_dict_amount['Mouse']))
        plt.text(2, 3800, 'R = 0.319, P = 0.171')  # 0.31880896646406354, P = 0.1706
        print(f'R = {r_value}, P = {p_value}')
        plt.setp(g.ax_marg_y.patches, color=self.mouse_color)
        plt.setp(g.ax_marg_x.patches, color=self.human_color)
        plt.title('')
        plt.xlabel('Human spot number')
        plt.ylabel('Mouse spot number')
        plt.savefig(save_path + 'Amount_mouse_human.' + self.fig_format, format=self.fig_format, dpi=self.fig_dpi)


        #ax = plt.figure(figsize=(4, 4), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        g = sns.jointplot(data=pd.DataFrame.from_dict(human_mouse_dict_mean), x="Human", y="Mouse", kind="reg",
                      height=4, color='black') # , ax=ax
        plt.title('')

        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(human_mouse_dict_mean['Human']),
                                                                       np.array(human_mouse_dict_mean['Mouse']))
        plt.text(0.33, 0.8, f'R = 0.235, P = 0.319')
        plt.xlabel('Human mean correlation')
        plt.ylabel('Mouse mean correlation')
        print(f'R = {r_value}, P = {p_value}')
        plt.setp(g.ax_marg_y.patches, color=self.mouse_color)
        plt.setp(g.ax_marg_x.patches, color=self.human_color)
        plt.savefig(save_path + 'mean_human_mouse.' + fig_format, format=fig_format, dpi=self.fig_dpi)
        # plt.show()

        # ax = plt.figure(figsize=(4, 4), dpi=self.fig_dpi)
        # sns.jointplot(data=pd.DataFrame.from_dict(human_mouse_dict_std), x="Human", y="Mouse", kind="reg", height=4,
        #               ax=ax)
        # plt.title('')
        # plt.savefig(save_path + 'std_human_mouse.' + fig_format, format=fig_format, dpi=self.fig_dpi)
        # plt.show()
        rcParams["figure.subplot.left"] = 0.1

        # ttest_homo_random()
        #####################################################################################################
        homo_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_1_experiment_homo_random/'
        with open(homo_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
            human_mouse_correlation_dict = pickle.load(f)

        home_len = len(human_mouse_correlation_dict['mean'])
        home_random_type = ['homologous'] * home_len
        human_mouse_correlation_dict['type'] = home_random_type
        # data_df = pd.DataFrame.from_dict(human_mouse_correlation_dict)

        if cfg.HOMO_RANDOM.random_field == 'all':
            random_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_1_experiment_homo_random/random_Hiercluster_correlation/'
        else:
            random_region_data_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_1_experiment_homo_random/35_random_Hiercluster_correlation/'
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
            save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_1_experiment_homo_random/homo_random/'
        else:
            save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/3_1_experiment_homo_random/35_homo_random/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        random_df = data_df[data_df['type'] == 'random']
        # print(random_df)
        mean_random_list = random_df['mean'].values
        mean_r = np.mean(mean_random_list)
        std_r = np.std(mean_random_list)

        homologous_df = data_df[data_df['type'] == 'homologous']
        # print(homologous_df)
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

        with open(save_path + 't_test_result.txt', 'w') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print(stats.ttest_ind(
                mean_homo_list,
                mean_random_list,
                equal_var=False
            ))
            sys.stdout = original_stdout  # Reset the standard output to its original value

        #self.homo_corr()
        #self.random_corr()
        # self.plot_homo_random()
        # self.ttest_homo_random()

        # Plot homo region sample number correlation
        #############################################################################################


        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.bottom"] = 0.1


    def experiment_4_cross_species_genes_analysis(self):

        sns.set(style='white')
        TINY_SIZE = 20  # 39
        SMALL_SIZE = 20  # 42
        MEDIUM_SIZE = 20  # 46
        BIGGER_SIZE = 20  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        #sc.settings.set_figure_params(dpi=500)
        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/4_experiment_cross_species_genes_analysis/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cfg = self.cfg
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

        mouse_gene_list = adata_mouse_gene_embedding.obs_names.to_list()
        human_gene_list = adata_human_gene_embedding.obs_names.to_list()

        lut = {"Human": self.human_color, "Mouse": self.mouse_color}
        row_colors = []
        for i in range(adata_human_gene_embedding.X.shape[0]):
            row_colors.append(lut['Human'])
        ## ---------------------------------------------------------------------mouse------------------------
        for i in range(adata_human_gene_embedding.X.shape[0]):
            row_colors.append(lut['Mouse'])

        rng = np.random.default_rng(12345)
        rints = rng.integers(low=0, high=adata_mouse_gene_embedding.X.shape[0],
                             size=adata_human_gene_embedding.X.shape[0])

        rcParams["figure.subplot.top"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.3
        # rcParams["figure.subplot.left"] = 0.05
        # rcParams["figure.subplot.right"] = 0.8

        with plt.rc_context({"figure.figsize": (5, 5), "figure.dpi": (self.fig_dpi)}):
            g = sns.clustermap(np.concatenate((adata_human_gene_embedding.X, adata_mouse_gene_embedding.X[rints, :]), axis=0),
                               row_colors=row_colors, dendrogram_ratio=(.3, .2), cmap='seismic', figsize=(5, 5), cbar_pos=(0.85, 0.8, 0.04, 0.18))
            # 'YlGnBu_r'

            # s.set(xlabel='Embedding features', ylabel='Samples')
            g.ax_heatmap.tick_params(tick2On=False, labelsize=False)
            g.ax_heatmap.set(xlabel='Embedding features', ylabel='Genes')
            # plt.ylabel('Samples')
            # plt.xlabel('Embedding features')
            # plt.yticks()
            handles = [Patch(facecolor=lut[name]) for name in lut]
            plt.legend(handles, lut,  #title='',
                       bbox_to_anchor=(0.35, 1.02),
                       bbox_transform=plt.gcf().transFigure,
                       loc='upper right', frameon=False)
            # rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.05

        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        # sns.clustermap(Var_Corr, cmap=color_map, center=0.6)
       
        plt.savefig(save_path + 'Genes_Hiercluster_heco_embedding_concate.'+ fig_format, format=fig_format) #
        #plt.show()

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


        adata_mouse_gene_embedding.obs['dataset'] = 'Mouse'
        adata_human_gene_embedding.obs['dataset'] = 'Human'

        adata_gene_embedding = ad.concat([adata_mouse_gene_embedding, adata_human_gene_embedding])
        #print(adata_gene_embedding)

        #adata_gene_embedding.X = adata_gene_embedding.X + np.min(adata_gene_embedding.X)

        sc.tl.pca(adata_gene_embedding, svd_solver='arpack', n_comps=30)
        sc.pp.neighbors(adata_gene_embedding, n_neighbors=cfg.ANALYSIS.genes_umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(adata_gene_embedding, min_dist=0.5) #0.85
        print(adata_gene_embedding)

        adata_mouse_gene_embedding = adata_gene_embedding[adata_gene_embedding.obs['dataset'].isin(['Mouse'])]
        adata_human_gene_embedding = adata_gene_embedding[adata_gene_embedding.obs['dataset'].isin(['Human'])]

        palette = {'Mouse':self.mouse_color, 'Human':self.human_color}

        rcParams["figure.subplot.right"] = 0.7

        with plt.rc_context({"figure.figsize": (3.5, 2.5), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(adata_gene_embedding, color=['dataset'], return_fig=True, legend_loc=None,
                            palette=palette, size=4, edges=False, edges_width=0.1)
            plt.title('')
            handles = [Patch(facecolor=lut[name]) for name in lut]
            plt.legend(handles, lut, #title='Species',
                       bbox_to_anchor=(1, 1),
                       bbox_transform=plt.gcf().transFigure,
                       loc='upper right', frameon=False)
            fg.savefig(
                save_path + 'Genes_umap_dataset.' + fig_format, format=fig_format)
        rcParams["figure.subplot.right"] = 0.66
        rcParams["figure.subplot.bottom"] = 0.2
        rcParams["figure.subplot.left"] = 0.2
        rcParams["figure.subplot.top"] = 0.88


        with plt.rc_context({"figure.figsize": (3.8, 2.5), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(adata_gene_embedding, color=['dataset'], return_fig=True, legend_loc='right margin',
                            palette=palette, size=10, edges=False, edges_width=0.1)
            plt.title('')
            fg.savefig(
                save_path + 'Genes_umap_dataset_right_margin.' + fig_format, format=fig_format)
        rcParams["figure.subplot.right"] = 0.9
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
        mh_mat = datapair['vv_adj'].toarray()[0:cfg.BrainAlign.binary_M, cfg.BrainAlign.binary_M:] > 0

        # get genes with homologous gene relationships
        mouse_gene_list_filtered = []
        human_gene_list_filtered = []
        for m in range(len(mouse_gene_list)):
            for h in range(len(human_gene_list)):
                if mh_mat[m, h] != 0:
                    mouse_gene_list_filtered.append(mouse_gene_list[m])
                    human_gene_list_filtered.append(human_gene_list[h])
        mouse_gene_list_filtered = list(set(mouse_gene_list_filtered))
        human_gene_list_filtered = list(set(human_gene_list_filtered))

        adata_mouse_gene_embedding_filtered = adata_mouse_gene_embedding[mouse_gene_list_filtered, :]
        adata_human_gene_embedding_filtered = adata_human_gene_embedding[human_gene_list_filtered, :]

        mh_mat_filtered = np.zeros((len(mouse_gene_list_filtered), len(human_gene_list_filtered)))
        for m in range(len(mouse_gene_list_filtered)):
            m_g = mouse_gene_list_filtered[m]
            for h in range(len(human_gene_list_filtered)):
                h_g = human_gene_list_filtered[h]
                ori_m_index = mouse_gene_list.index(m_g)
                ori_h_index = human_gene_list.index(h_g)
                if mh_mat[ori_m_index, ori_h_index] != 0:
                    mh_mat_filtered[m, h] = mh_mat[ori_m_index, ori_h_index]

        #adata_mouse_gene_embedding_filtered.obs['dataset'] = 'Mouse'
        #adata_human_gene_embedding_filtered.obs['dataset'] = 'Human'

        adata_gene_embedding_filtered = ad.concat([adata_mouse_gene_embedding_filtered, adata_human_gene_embedding_filtered])
        print(adata_gene_embedding_filtered)
        #adata_gene_embedding_filtered.obs = ad.concat([adata_mouse_gene_embedding_filtered, adata_human_gene_embedding_filtered]).obs
        #print(adata_gene_embedding_filtered)

        # adata_gene_embedding =
        adata_gene_umap_mouse = adata_gene_embedding[adata_gene_embedding.obs['dataset'] == 'Mouse']
        adata_gene_umap_human = adata_gene_embedding[adata_gene_embedding.obs['dataset'] == 'Human']
        umap1_x, umap1_y = adata_gene_umap_mouse.obsm['X_umap'].toarray()[:, 0], adata_gene_umap_mouse.obsm[
                                                                                     'X_umap'].toarray()[:, 1]
        umap1_z = 1 / 4 * np.ones(umap1_x.shape) * (
                    ((np.max(umap1_x) - np.min(umap1_x)) ** 2 + (np.max(umap1_y) - np.min(umap1_y)) ** 2) ** (1 / 2))

        umap2_x, umap2_y = adata_gene_umap_human.obsm['X_umap'].toarray()[:, 0], adata_gene_umap_human.obsm[
                                                                                     'X_umap'].toarray()[:, 1]
        umap2_z = np.zeros(umap2_x.shape)

        point1_list = []
        for u1_x, u1_y, u1_z in zip(umap1_x, umap1_y, umap1_z):
            point1_list.append([u1_x, u1_y, u1_z])

        point2_list = []
        for u2_x, u2_y, u2_z in zip(umap2_x, umap2_y, umap2_z):
            point2_list.append([u2_x, u2_y, u2_z])

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

        fig = plt.figure(figsize=(3, 3), dpi=self.fig_dpi)
        axes = plt.axes(projection='3d')
        # print(type(axes))
        axes.scatter3D(umap1_x, umap1_y, umap1_z, color=self.mouse_color, s=1)
        axes.scatter3D(umap2_x, umap2_y, umap2_z, color=self.human_color, s=1)
        for i in range(len(point1_list)):
            for j in range(len(point2_list)):
                point1 = point1_list[i]
                point2 = point2_list[j]
                if mh_mat[i, j] != 0:
                    axes.plot3D([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color='gray',
                                linewidth=0.1, alpha=0.4)

        ax = plt.gca()
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)

        axes.set_xlabel('UMAP1')
        axes.set_ylabel('UMAP2')
        axes.set_zlabel('Species')
        fig.subplots_adjust(left=0, right=0.95, bottom=0.1, top=1)
        # axes.view_init(45, 215)
        plt.savefig(save_path + 'Genes_umap_plot_map.' + fig_format, dpi=self.fig_dpi)
        # ----------------------------------------------------------------------------
        #---------------------------
        ###################################################################################

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
        mh_mat = datapair['vv_adj'].toarray()[0:cfg.BrainAlign.binary_M, cfg.BrainAlign.binary_M:] > 0

        # get genes with homologous gene relationships
        mouse_gene_list_filtered = []
        human_gene_list_filtered = []
        for m in range(len(mouse_gene_list)):
            for h in range(len(human_gene_list)):
                if mh_mat[m, h] != 0:
                    mouse_gene_list_filtered.append(mouse_gene_list[m])
                    human_gene_list_filtered.append(human_gene_list[h])
        mouse_gene_list_filtered = list(set(mouse_gene_list_filtered))
        human_gene_list_filtered = list(set(human_gene_list_filtered))

        adata_mouse_gene_embedding_filtered = adata_mouse_gene_embedding[mouse_gene_list_filtered, :]
        adata_human_gene_embedding_filtered = adata_human_gene_embedding[human_gene_list_filtered, :]

        mh_mat_filtered = np.zeros((len(mouse_gene_list_filtered), len(human_gene_list_filtered)))
        for m in range(len(mouse_gene_list_filtered)):
            m_g = mouse_gene_list_filtered[m]
            for h in range(len(human_gene_list_filtered)):
                h_g = human_gene_list_filtered[h]
                ori_m_index = mouse_gene_list.index(m_g)
                ori_h_index = human_gene_list.index(h_g)
                if mh_mat[ori_m_index, ori_h_index] != 0:
                    mh_mat_filtered[m, h] = mh_mat[ori_m_index, ori_h_index]

        # adata_mouse_gene_embedding_filtered.obs['dataset'] = 'Mouse'
        # adata_human_gene_embedding_filtered.obs['dataset'] = 'Human'

        adata_gene_embedding_filtered = ad.concat(
            [adata_mouse_gene_embedding_filtered, adata_human_gene_embedding_filtered])
        print(adata_gene_embedding_filtered)
        # adata_gene_embedding_filtered.obs = ad.concat([adata_mouse_gene_embedding_filtered, adata_human_gene_embedding_filtered]).obs
        # print(adata_gene_embedding_filtered)

        # adata_gene_embedding =
        adata_gene_umap_mouse = adata_gene_embedding[adata_gene_embedding.obs['dataset'] == 'Mouse']
        adata_gene_umap_human = adata_gene_embedding[adata_gene_embedding.obs['dataset'] == 'Human']
        umap1_x, umap1_y = adata_gene_umap_mouse.obsm['X_umap'].toarray()[:, 0], adata_gene_umap_mouse.obsm[
                                                                                     'X_umap'].toarray()[:, 1]
        umap1_z = 1 / 4 * np.ones(umap1_x.shape) * (
                ((np.max(umap1_x) - np.min(umap1_x)) ** 2 + (np.max(umap1_y) - np.min(umap1_y)) ** 2) ** (1 / 2))

        umap2_x, umap2_y = adata_gene_umap_human.obsm['X_umap'].toarray()[:, 0], adata_gene_umap_human.obsm[
                                                                                     'X_umap'].toarray()[:, 1]
        umap2_z = np.zeros(umap2_x.shape)

        point1_list = []
        for u1_x, u1_y, u1_z in zip(umap1_x, umap1_y, umap1_z):
            point1_list.append([u1_x, u1_y, u1_z])

        point2_list = []
        for u2_x, u2_y, u2_z in zip(umap2_x, umap2_y, umap2_z):
            point2_list.append([u2_x, u2_y, u2_z])

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

        fig = plt.figure(figsize=(3, 3), dpi=self.fig_dpi)
        axes = plt.axes(projection='3d')
        # print(type(axes))
        axes.scatter3D(umap1_x, umap1_y, umap1_z, color=self.mouse_color, s=1)
        axes.scatter3D(umap2_x, umap2_y, umap2_z, color=self.human_color, s=1)
        for i in range(len(point1_list)):
            for j in range(len(point2_list)):
                point1 = point1_list[i]
                point2 = point2_list[j]
                if mh_mat[i, j] != 0:
                    axes.plot3D([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color='gray',
                                linewidth=0.1, alpha=0.4)

        ax = plt.gca()
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)

        axes.set_xlabel('UMAP1')
        axes.set_ylabel('UMAP2')
        axes.set_zlabel('Species')
        fig.subplots_adjust(left=0, right=0.95, bottom=0.1, top=1)
        # axes.view_init(45, 215)
        plt.savefig(save_path + 'Genes_umap_gene_module_map.' + fig_format, dpi=self.fig_dpi)





        # adata_gene_embedding =
        '''
        adata_gene_umap_mouse_filtered = adata_gene_embedding_filtered[adata_gene_embedding_filtered.obs['dataset'] == 'Mouse']
        adata_gene_umap_human_filtered = adata_gene_embedding_filtered[adata_gene_embedding_filtered.obs['dataset'] == 'Human']
        umap1_x, umap1_y = adata_gene_umap_mouse_filtered.obsm['X_umap'].toarray()[:, 0], adata_gene_umap_mouse_filtered.obsm[
                                                                                     'X_umap'].toarray()[:, 1]
        umap1_z = 1 / 4 * np.ones(umap1_x.shape) * (
                ((np.max(umap1_x) - np.min(umap1_x)) ** 2 + (np.max(umap1_y) - np.min(umap1_y)) ** 2) ** (1 / 2))

        umap2_x, umap2_y = adata_gene_umap_human_filtered.obsm['X_umap'].toarray()[:, 0], adata_gene_umap_human_filtered.obsm[
                                                                                     'X_umap'].toarray()[:, 1]
        umap2_z = np.zeros(umap2_x.shape)

        point1_list = []
        for u1_x, u1_y, u1_z in zip(umap1_x, umap1_y, umap1_z):
            point1_list.append([u1_x, u1_y, u1_z])

        point2_list = []
        for u2_x, u2_y, u2_z in zip(umap2_x, umap2_y, umap2_z):
            point2_list.append([u2_x, u2_y, u2_z])

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

        fig = plt.figure(figsize=(3, 3), dpi=500)
        axes = plt.axes(projection='3d')
        # print(type(axes))
        axes.scatter3D(umap1_x, umap1_y, umap1_z, color=self.mouse_color, s=3)
        axes.scatter3D(umap2_x, umap2_y, umap2_z, color=self.human_color, s=3)
        for i in range(len(point1_list)):
            for j in range(len(point2_list)):
                point1 = point1_list[i]
                point2 = point2_list[j]
                if mh_mat_filtered[i, j] != 0:
                    axes.plot3D([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color='gray',
                                linewidth=0.25, alpha=0.4)

        ax = plt.gca()
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)

        axes.set_xlabel('UMAP1')
        axes.set_ylabel('UMAP2')
        axes.set_zlabel('Species')
        fig.subplots_adjust(left=0, right=0.95, bottom=0, top=1)
        # axes.view_init(45, 215)
        plt.savefig(save_path + 'filtered_Genes_umap_plot_map.' + fig_format)
        '''

        # plot gene modules
        rcParams["figure.subplot.right"] = 0.68
        rcParams["figure.subplot.left"] = 0.2
        rcParams["figure.subplot.top"] = 0.88
        rcParams["figure.subplot.bottom"] = 0.2
        with plt.rc_context({"figure.figsize": (3.8, 2.5), "figure.dpi": (self.fig_dpi)}):
            sc.tl.leiden(adata_gene_embedding, resolution=1.5, key_added='module')
            #sc.tl.louvain(adata_gene_embedding, resolution=0.5, key_added='module')
            fg = sc.pl.umap(adata_gene_embedding, color='module', ncols=1, palette='tab20b', return_fig=True,
                       edges=False, edges_width=0.1, s=10)
            fg.legend('')
            plt.title('')
            fg.savefig(save_path + 'Genes_module_concat.' + fig_format, format=fig_format, dpi=self.fig_dpi)

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1

        # adata_gene_embedding.obs_names = adata_gene_embedding.obs_names.astype(str)
        gadt1, gadt2 = pp.bisplit_adata(adata_gene_embedding, 'dataset',
                                        cfg.BrainAlign.dsnames[0])  # weight_linked_vars

        color_by = 'module'
        palette = 'tab20b'
        rcParams["figure.subplot.right"] = 0.68
        with plt.rc_context({"figure.figsize": (3, 3), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(gadt1, color=color_by, s=5,  edges=False, edges_width=0.02,
                       palette=palette,
                       save=f'_{color_by}-{cfg.BrainAlign.dsnames[0]}', return_fig=True)
            fg.legend('')
            fg.savefig(
                save_path + '{}_genes_module_concat.'.format(cfg.BrainAlign.dsnames[0]) + fig_format, format=fig_format)

        rcParams["figure.subplot.right"] = 0.68
        with plt.rc_context({"figure.figsize": (3, 2), "figure.dpi": (self.fig_dpi)}):
            fg = sc.pl.umap(gadt2, color=color_by, s=5,  edges=False, edges_width=0.02,
                   palette=palette,
                   save=f'_{color_by}-{cfg.BrainAlign.dsnames[1]}', return_fig=True)
            fg.legend('')
            fg.savefig(
            save_path + '{}_genes_module_concat.'.format(cfg.BrainAlign.dsnames[1]) + fig_format, format=fig_format)

        rcParams["figure.subplot.right"] = 0.9

        path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
        path_datapiar_file = open(path_datapiar, 'rb')
        datapair = pickle.load(path_datapiar_file)
        vnode_names = datapair['varnames_node'][0] + datapair['varnames_node'][1]
        #print(len(vnode_names))
        df_var_links = weight_linked_vars(
            adata_gene_embedding.X, datapair['vv_adj'], names=vnode_names,
            matric='euclidean', index_names=cfg.BrainAlign.dsnames,
        )

        #gadt1.write(cfg.BrainAlign.embeddings_file_path + 'adt_hidden_gene1.h5ad')
        #gadt2.write(cfg.BrainAlign.embeddings_file_path + 'adt_hidden_gene2.h5ad')

        # Compute average expressions for each brain region.
        key_class1 = 'region_name'  # set by user
        key_class2 = 'region_name'  # set by user
        # averaged expressions

        adata_mouse_sample_embedding = sc.read_h5ad(cfg.CAME.path_rawdata1)
        adata_human_sample_embedding = sc.read_h5ad(cfg.CAME.path_rawdata2)
        adatas = [adata_mouse_sample_embedding, adata_human_sample_embedding]
        # compute average embedding of
        avg_expr1 = pp.group_mean_adata(
            adatas[0], groupby=key_class1, features=datapair['varnames_node'][0],
            use_raw=True)

        avg_expr2 = pp.group_mean_adata(
            adatas[1], groupby=key_class2 if key_class2 else 'predicted', features=datapair['varnames_node'][1],
            use_raw=True)

        plkwds = dict(cmap='RdYlBu_r', vmax=2.5, vmin=-1.5, do_zscore=True,
                      axscale=3, ncols=5, with_cbar=True)

        #print('gadt1', gadt1)
        #print('gadt2', gadt2)
        fig1, axs1 = pl.adata_embed_with_values(
            gadt1, avg_expr1, embed_key='UMAP',
            fp=save_path + f'umap_exprAvgs-{cfg.BrainAlign.dsnames[0]}-all.' + fig_format,
            **plkwds)
        fig2, axs2 = pl.adata_embed_with_values(
            gadt2, avg_expr2, embed_key='UMAP',
            fp=save_path + f'umap_exprAvgs-{cfg.BrainAlign.dsnames[1]}-all.' + fig_format,
            **plkwds)

        ## Abstracted graph #####################################
        norm_ov = ['max', 'zs', None][1]
        cut_ov = cfg.ANALYSIS.cut_ov

        groupby_var = 'module'
        obs_labels1, obs_labels2 = adata_mouse_sample_embedding.obs['region_name'], adata_human_sample_embedding.obs['region_name']  # adt.obs['celltype'][dpair.obs_ids1], adt.obs['celltype'][dpair.obs_ids2]
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
            g, edge_scale=5,
            figsize=(16, 10), alpha=0.5, fp=fp_abs, colors=('deepskyblue', 'limegreen', 'limegreen', 'orange'))  # nodelist=nodelist,

        # ax.figure
        came.save_pickle(g, cfg.BrainAlign.embeddings_file_path + 'abs_graph.pickle')

        # homologous regions abstracted graph
        human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)
        homo_region_list = list(human_mouse_homo_region['Mouse'].values) + list(human_mouse_homo_region['Human'].values)
        homo_region_color_list = [self.mouse_64_color_dict[m_r] for m_r in
                                  list(human_mouse_homo_region['Mouse'].values)] + \
                                 [self.human_88_color_dict[h_r] for h_r in
                                  list(human_mouse_homo_region['Human'].values)]
        #print(homo_region_list)
        #adata_embedding_homo = adata_embedding[adata_embedding.obs['region_name'].isin(homo_region_list)]
        homo_region_palette = {k: v for k, v in zip(homo_region_list, homo_region_color_list)}

        mouse_homo_region_list = list(human_mouse_homo_region['Mouse'].values)
        human_homo_region_list = list(human_mouse_homo_region['Human'].values)
        mouse_homo_region_color_list = [self.mouse_64_color_dict[m_r] for m_r in
                                  list(human_mouse_homo_region['Mouse'].values)]
        human_homo_region_color_list = [self.human_88_color_dict[h_r] for h_r in
                                  list(human_mouse_homo_region['Human'].values)]

        mouse_homo_region_palette = {k: v for k, v in zip(mouse_homo_region_list, mouse_homo_region_color_list)}
        human_homo_region_palette = {k: v for k, v in zip(human_homo_region_list, human_homo_region_color_list)}

        adata_mouse_sample_embedding = adata_mouse_sample_embedding[adata_mouse_sample_embedding.obs['region_name'].isin(mouse_homo_region_list)]
        adata_human_sample_embedding = adata_human_sample_embedding[adata_human_sample_embedding.obs['region_name'].isin(human_homo_region_list)]
        adatas = [adata_mouse_sample_embedding, adata_human_sample_embedding]

        print('3---------------------------------------------------------------------------')

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
            fp=save_path + f'homo_umap_exprAvgs-{cfg.BrainAlign.dsnames[0]}-all.' + fig_format,
            **plkwds)
        fig2, axs2 = pl.adata_embed_with_values(
            gadt2, avg_expr2, embed_key='UMAP',
            fp=save_path + f'homo_umap_exprAvgs-{cfg.BrainAlign.dsnames[1]}-all.' + fig_format,
            **plkwds)

        ## Abstracted graph #####################################
        norm_ov = ['max', 'zs', None][1]
        cut_ov = cfg.ANALYSIS.cut_ov

        groupby_var = 'module'
        obs_labels1, obs_labels2 = adata_mouse_sample_embedding.obs['region_name'], adata_human_sample_embedding.obs[
            'region_name']  # adt.obs['celltype'][dpair.obs_ids1], adt.obs['celltype'][dpair.obs_ids2]
        var_labels1, var_labels2 = gadt1.obs[groupby_var], gadt2.obs[groupby_var]

        print('2---------------------------------------------------------------')

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
        print('1---------------------------------------------------------------------------')
        ''' visualization '''
        fp_abs = save_path + f'homo_abstracted_graph-{groupby_var}-cut{cut_ov}-{norm_ov}.' + fig_format
        ax = pl.plot_multipartite_graph(
            g, edge_scale=10,
            figsize=(12, 10), alpha=0.5, fp=fp_abs, colors=('deepskyblue', 'limegreen', 'limegreen', 'orange'))  # nodelist=nodelist,

        # ax.figure
        came.save_pickle(g, cfg.BrainAlign.embeddings_file_path + 'homo_abs_graph.pickle')


        # get genes with one-to-one homologous gene relationships
        mouse_gene_list_filtered = []
        human_gene_list_filtered = []
        for m in range(len(mouse_gene_list)):
            for h in range(len(human_gene_list)):
                if mh_mat[m, h] != 0 and np.sum(mh_mat[m, :]) + np.sum(mh_mat[:, h]) <= 2:
                    mouse_gene_list_filtered.append(mouse_gene_list[m])
                    human_gene_list_filtered.append(human_gene_list[h])
        #mouse_gene_list_filtered = list(set(mouse_gene_list_filtered))
        #human_gene_list_filtered = list(set(human_gene_list_filtered))

        adata_mouse_gene_embedding_filtered = adata_mouse_gene_embedding[mouse_gene_list_filtered, :]
        adata_human_gene_embedding_filtered = adata_human_gene_embedding[human_gene_list_filtered, :]

        mh_mat_filtered = np.zeros((len(mouse_gene_list_filtered), len(human_gene_list_filtered)))
        for m, h in zip(range(len(mouse_gene_list_filtered)), range(len(human_gene_list_filtered))):
            m_g = mouse_gene_list_filtered[m]
            h_g = human_gene_list_filtered[h]
            ori_m_index = mouse_gene_list.index(m_g)
            ori_h_index = human_gene_list.index(h_g)
            if mh_mat[ori_m_index, ori_h_index] != 0:
                mh_mat_filtered[m, h] = mh_mat[ori_m_index, ori_h_index]

        adata_mouse_gene_embedding_filtered.obs['dataset'] = 'Mouse'
        adata_human_gene_embedding_filtered.obs['dataset'] = 'Human'

        adata_gene_embedding_filtered = ad.concat([adata_mouse_gene_embedding_filtered, adata_human_gene_embedding_filtered])
        print(adata_gene_embedding_filtered)

        #sc.tl.pca(adata_gene_embedding_filtered, svd_solver='arpack', n_comps=30)
        #sc.pp.neighbors(adata_gene_embedding_filtered, n_neighbors=cfg.ANALYSIS.genes_umap_neighbor, metric='euclidean',
        #                use_rep='X')
        #sc.tl.umap(adata_gene_embedding_filtered, min_dist=0.5)  # 0.85

        palette = {'Mouse': self.mouse_color, 'Human': self.human_color}

        rcParams["figure.subplot.right"] = 0.7

        # with plt.rc_context({"figure.figsize": (12, 8), "figure.dpi": (self.fig_dpi)}):
        #     fg = sc.pl.umap(adata_gene_embedding_filtered, color=['dataset'], return_fig=True, legend_loc=None,
        #                     palette=palette,
        #                     size=25, edges=True, edges_width=0.05, )
        #     plt.title('')
        #     handles = [Patch(facecolor=lut[name]) for name in lut]
        #     plt.legend(handles, lut,  # title='Species',
        #                bbox_to_anchor=(1, 1),
        #                bbox_transform=plt.gcf().transFigure,
        #                loc='upper right', frameon=False)
        #     fg.savefig(
        #         save_path + 'one2one_Genes_umap_dataset.' + fig_format, format=fig_format)
        # rcParams["figure.subplot.right"] = 0.66
        # with plt.rc_context({"figure.figsize": (12, 8), "figure.dpi": (self.fig_dpi)}):
        #     fg = sc.pl.umap(adata_gene_embedding_filtered, color=['dataset'], return_fig=True,
        #                     legend_loc='right margin',
        #                     palette=palette, size=25, edges=True, edges_width=0.05)
        #     plt.title('')
        #     fg.savefig(
        #         save_path + 'one2one_Genes_umap_dataset_right_margin.' + fig_format, format=fig_format)
        # rcParams["figure.subplot.right"] = 0.9

        # adata_gene_umap_mouse_filtered = adata_gene_embedding_filtered[
        #     adata_gene_embedding_filtered.obs['dataset'] == 'Mouse']
        # adata_gene_umap_human_filtered = adata_gene_embedding_filtered[
        #     adata_gene_embedding_filtered.obs['dataset'] == 'Human']
        '''
        adata_gene_umap_mouse_filtered = adata_mouse_gene_embedding_filtered
        adata_gene_umap_human_filtered = adata_human_gene_embedding_filtered

        umap1_x, umap1_y = adata_gene_umap_mouse_filtered.obsm['X_umap'][:, 0], \
                           adata_gene_umap_mouse_filtered.obsm[
                               'X_umap'][:, 1]
        umap1_z = 1 / 4 * np.ones(umap1_x.shape) * (
                ((np.max(umap1_x) - np.min(umap1_x)) ** 2 + (np.max(umap1_y) - np.min(umap1_y)) ** 2) ** (1 / 2))

        umap2_x, umap2_y = adata_gene_umap_human_filtered.obsm['X_umap'][:, 0], adata_gene_umap_human_filtered.obsm['X_umap'][:, 1]
        umap2_z = np.zeros(umap2_x.shape)

        point1_list = []
        for u1_x, u1_y, u1_z in zip(umap1_x, umap1_y, umap1_z):
            point1_list.append([u1_x, u1_y, u1_z])

        point2_list = []
        for u2_x, u2_y, u2_z in zip(umap2_x, umap2_y, umap2_z):
            point2_list.append([u2_x, u2_y, u2_z])

        TINY_SIZE = 12  # 39
        SMALL_SIZE = 12  # 42
        MEDIUM_SIZE = 12  # 46
        BIGGER_SIZE = 12 # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        fig = plt.figure(figsize=(3, 3), dpi=500)
        axes = plt.axes(projection='3d')
        # print(type(axes))
        axes.scatter3D(umap1_x, umap1_y, umap1_z, color=self.mouse_color, s=2)
        axes.scatter3D(umap2_x, umap2_y, umap2_z, color=self.human_color, s=2)
        for i in range(len(point1_list)):
            for j in range(len(point2_list)):
                point1 = point1_list[i]
                point2 = point2_list[j]
                if mh_mat_filtered[i, j] != 0:
                    axes.plot3D([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color='gray',
                                linewidth=0.25, alpha=0.4)

        ax = plt.gca()
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)

        axes.set_xlabel('UMAP1')
        axes.set_ylabel('UMAP2')
        axes.set_zlabel('Species')
        fig.subplots_adjust(left=0, right=0.95, bottom=0, top=1)
        # axes.view_init(45, 215)
        plt.savefig(save_path + 'one2one_Genes_umap_plot_map.' + fig_format)
        '''



    def experiment_4_1_genes_statistics(self):

        sns.set(style='white')
        TINY_SIZE = 20  # 39
        SMALL_SIZE = 20  # 42
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

        # sc.settings.set_figure_params(dpi=500)
        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/4_experiment_cross_species_genes_analysis/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cfg = self.cfg
        fig_format = cfg.BrainAlign.fig_format

        adata_mouse_gene_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'm_embeddings.h5ad')
        adata_human_gene_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'h_embeddings.h5ad')

        mouse_gene_list = adata_mouse_gene_embedding.obs_names.to_list()
        human_gene_list = adata_human_gene_embedding.obs_names.to_list()

        data_dict = {'Species': ['Mouse', 'Human', 'Mouse', 'Human', 'Mouse', 'Human'],
                     'Homologous type': ['All', 'All', 'Many-to-many', 'Many-to-many', 'One-to-one', 'One-to-one'],
                     'Gene number': []}

        data_dict['Gene number'].append(len(mouse_gene_list))
        data_dict['Gene number'].append(len(human_gene_list))


        path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
        path_datapiar_file = open(path_datapiar, 'rb')
        datapair = pickle.load(path_datapiar_file)
        mh_mat = datapair['vv_adj'].toarray()[0:cfg.BrainAlign.binary_M, cfg.BrainAlign.binary_M:] > 0

        # get genes with homologous gene relationships
        mouse_gene_list_filtered = []
        human_gene_list_filtered = []
        for m in range(len(mouse_gene_list)):
            for h in range(len(human_gene_list)):
                if mh_mat[m, h] != 0:
                    mouse_gene_list_filtered.append(mouse_gene_list[m])
                    human_gene_list_filtered.append(human_gene_list[h])
        mouse_gene_list_filtered = list(set(mouse_gene_list_filtered))
        human_gene_list_filtered = list(set(human_gene_list_filtered))

        data_dict['Gene number'].append(len(mouse_gene_list_filtered))
        data_dict['Gene number'].append(len(human_gene_list_filtered))

        # one-2-one genes
        mouse_gene_list_filtered = []
        human_gene_list_filtered = []
        for m in range(len(mouse_gene_list)):
            for h in range(len(human_gene_list)):
                if mh_mat[m, h] != 0 and np.sum(mh_mat[m, :]) + np.sum(mh_mat[:, h]) <= 2:
                    mouse_gene_list_filtered.append(mouse_gene_list[m])
                    human_gene_list_filtered.append(human_gene_list[h])
        data_dict['Gene number'].append(len(mouse_gene_list_filtered))
        data_dict['Gene number'].append(len(human_gene_list_filtered))

        data_df = pd.DataFrame.from_dict(data_dict)
        print(data_df)


        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.fig_dpi)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.4
        rcParams["figure.subplot.top"] = 0.98
        #sns.set(rc={"figure.subplot.bottom": 0.4, "figure.dpi": self.fig_dpi, "figure.figsize":(8, 6)})
        # Draw a nested barplot by species and sex
        g = sns.catplot(
            data=data_df, kind="bar",
            x="Homologous type", y="Gene number", hue="Species", legend=True,
            palette="dark",   ax=ax, legend_out=True #alpha=.6, height=6,
        )
        #plt.legend(loc='upper right')
        #g.despine(left=True)
        # g.set_axis_labels("", "Body mass (g)")
        #g.legend
        g.legend.set_title("")
        g.set_xlabels('')
        #plt.legend(loc="right", frameon=False, title=None, legend_out=True)
        g.legend.get_frame().set_linewidth(0.0)
        for ax in g.axes.ravel():
            # add annotations
            for c in ax.containers:
                # add custom labels with the labels=labels parameter if needed
                # labels = [f'{h}' if (h := v.get_height()) > 0 else '' for v in c]
                ax.bar_label(c, label_type='edge', padding=3, rotation=80) #, ha='right', rotation_mode="anchor"
            ax.margins(y=0.2)
        for item in ax.get_xticklabels():
            item.set_rotation(20)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")

        plt.savefig(save_path + 'gene_statistics.' + fig_format, format=fig_format)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.9

        return None


    def genes_homo_random_distance(self, metric_name='euclidean'):
        cfg = self.cfg
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

        homo_random_gene_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/5_experiment_genes_homo_random/'
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
        dist_matrix_all = dist_matrix[0:mouse_genes_num, mouse_genes_num:mouse_genes_num + human_genes_num]
        Var_Corr = pd.DataFrame(dist_matrix_all, columns=adata_human_gene_embedding.obs_names,
                                index=adata_mouse_gene_embedding.obs_names)
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
        my_pal = {"homologous": '#28AF60', "random": '#8A4AA2'}
        # sns.set_theme(style="whitegrid")
        # tips = sns.load_dataset("tips")
        plt.figure(figsize=(2, 2.5))
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

    def experiment_5_genes_homo_random(self):
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

        cfg = self.cfg
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

        homo_random_gene_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/5_experiment_genes_homo_random/'
        if not os.path.exists(homo_random_gene_path):
            os.makedirs(homo_random_gene_path)

        mouse_df = pd.DataFrame(adata_mouse_gene_embedding.X).T
        mouse_df.columns = ['mouse_' + x for x in adata_mouse_gene_embedding.obs_names]
        print(mouse_df.shape)
        human_df = pd.DataFrame(adata_human_gene_embedding.X).T
        human_df.columns = ['human_' + x for x in adata_human_gene_embedding.obs_names]
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

        # my_pal = {"homo region": (0/255,149/255,182/255), "random":(178/255, 0/255, 32/255)}
        my_pal = {"homologous": '#28AF60', "random": '#8A4AA2'}
        # sns.set_theme(style="whitegrid")
        # tips = sns.load_dataset("tips")

        rcParams["figure.subplot.left"] = 0.35
        rcParams["figure.subplot.right"] = 0.98
        rcParams["figure.subplot.bottom"] = 0.25
        rcParams["figure.subplot.top"] = 0.98
        plt.figure(figsize=(2, 2.5), dpi=self.fig_dpi)
        ax = sns.boxplot(x="type", y="Correlation", data=data_df, order=["homologous", "random"], palette=my_pal, width=0.618)
        add_stat_annotation(ax, data=data_df, x="type", y="Correlation", order=["homologous", "random"],
                            box_pairs=[("homologous", "random")],
                            test='t-test_ind', text_format='star', loc='inside', verbose=2)  # text_format='star'
        plt.xlabel('')
        for item in ax.get_xticklabels():
            item.set_rotation(20)
        #plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")

        plt.savefig(save_path + 'Correlation.' + fig_format)
        #plt.show()

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
        self.genes_homo_random_distance(metric_name='euclidean')
        self.genes_homo_random_distance(metric_name='cosine')
        self.genes_homo_random_distance(metric_name='chebyshev')
        self.genes_homo_random_distance(metric_name='correlation')
        self.genes_homo_random_distance(metric_name='braycurtis')
        self.genes_homo_random_distance(metric_name='jensenshannon')
        self.genes_homo_random_distance(metric_name='canberra')
        self.genes_homo_random_distance(metric_name='minkowski')

    def experiment_6_brain_region_classfier(self):
        cfg = self.cfg
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
        # print(region_index_dict)
        array_Y = np.zeros((data_Y.shape[0]))
        for i in range(data_Y.shape[0]):
            array_Y[i] = region_index_dict[data_Y[i]]

        # delete rare samples to make classification possible
        from collections import Counter
        print(Counter(array_Y))
        # define oversampling strategy
        oversample = RandomOverSampler(sampling_strategy='minority', random_state=29)  # sampling_strategy='minority'
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
        f1_micro = f1_score(test_Y, test_Y_pred, average='micro')
        f1_macro = f1_score(test_Y, test_Y_pred, average='macro')

        print('Test set accuracy = ', accuracy)
        print('Test set f1_weighted = ', f1_weighted)
        print('Test set f1_micro = ', f1_micro)
        print('Test set f1_macro = ', f1_macro)
        # roc_auc = roc_auc_score(test_Y, test_Y_pred_proba, average='weighted', multi_class='ovr')
        # print('Test set roc_auc = ', roc_auc)

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/6_experiment_brain_region_classfier/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # return test_accuracy, test_recall, test_F1
        original_stdout = sys.stdout  # Save a reference to the original standard output

        with open(save_path + 'classification_result.txt', 'w') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print('Method: xgboost')
            print('Test set accuracy = ', accuracy)
            print('Test set f1_weighted = ', f1_weighted)
            print('Test set f1_micro = ', f1_micro)
            print('Test set f1_macro = ', f1_macro)
            # print('Test set roc_auc = ', roc_auc)
            sys.stdout = original_stdout  # Reset the standard output to its original value


    def experiment_7_align_cross_species(self):
        sns.set(style='white')
        TINY_SIZE = 20#18  # 39
        SMALL_SIZE = 20#18  # 42
        MEDIUM_SIZE = 20  # 46
        BIGGER_SIZE = 20  # 46

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

        human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/7_experiment_align_cross_species/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Step 1
        human_embedding_dict = OrderedDict()
        # print(list(human_88_labels['region_name']))
        human_88_labels_list = list(human_88_labels['region_name'])
        for r_n in human_88_labels_list:
            human_embedding_dict[r_n] = None

        mouse_embedding_dict = OrderedDict()
        mouse_67_labels_list = list(mouse_67_labels['region_name'])
        for r_n in mouse_67_labels_list:
            mouse_embedding_dict[r_n] = None

        for region_name in human_88_labels_list:
            mean_embedding = np.mean(adata_human_embedding[adata_human_embedding.obs['region_name'] == region_name].X,
                                     axis=0)
            human_embedding_dict[region_name] = mean_embedding

        human_embedding_df = pd.DataFrame.from_dict(human_embedding_dict)

        for region_name in mouse_67_labels_list:
            mean_embedding = np.mean(adata_mouse_embedding[adata_mouse_embedding.obs['region_name'] == region_name].X,
                                     axis=0)
            mouse_embedding_dict[region_name] = mean_embedding
            # print(mean_embedding.shape)
        mouse_embedding_df = pd.DataFrame.from_dict(mouse_embedding_dict)

        result = pd.concat([human_embedding_df, mouse_embedding_df], axis=1).corr()
        Var_Corr = result[mouse_embedding_df.columns].loc[human_embedding_df.columns]
        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        color_map = sns.color_palette("vlag", as_cmap=True)

        cmap_used = "vlag" #"YlGnBu_r"
        # color_map = sns.color_palette("rocket_r", as_cmap=
        # rcParams["figure.subplot.left"] = 0.2
        # rcParams["figure.subplot.right"] = 1
        # rcParams["figure.subplot.bottom"] = 0.2

        sns.set(style='white')
        TINY_SIZE = 10  # 18  # 39
        SMALL_SIZE = 10  # 18  # 42
        MEDIUM_SIZE = 10  # 46
        BIGGER_SIZE = 10  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        fig, ax = plt.subplots(figsize=(14, 14), dpi=self.fig_dpi)
        rcParams["figure.subplot.right"] = 0.83
        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.bottom"] = 0.05

        hm = sns.clustermap(Var_Corr, cmap=cmap_used,
                            row_colors=list(self.human_88_labels['color_hex_triplet']),
                            col_colors=list(self.mouse_64_labels['color_hex_triplet']),
                            yticklabels=True,
                            row_cluster=False, col_cluster=False,
                            xticklabels=True, figsize=(14, 14), linewidth=0.5, linecolor='grey',
                            cbar_pos=(0.85, 0.1, 0.1, 0.03),
                            dendrogram_ratio=0.08, colors_ratio=0.01, cbar_kws={'label': 'Mean correlation', 'orientation': 'horizontal'}, tree_kws={'colors':'black'})

        for item in hm.ax_heatmap.get_xticklabels():
            item.set_rotation(45)

        human_tick_parent_color_dict = {k: self.human_16_color_dict[v] for k, v in
                                        self.human_88_parent_region_dict.items()}
        mouse_tick_parent_color_dict = {k: self.mouse_15_color_dict[v] for k, v in
                                        self.mouse_64_parent_region_dict.items()}
        for i, ticklabel in enumerate(hm.ax_heatmap.xaxis.get_ticklabels()):
            ticklabel.set_color(mouse_tick_parent_color_dict[ticklabel.get_text()])
        for i, ticklabel in enumerate(hm.ax_heatmap.yaxis.get_ticklabels()):
            ticklabel.set_color(human_tick_parent_color_dict[ticklabel.get_text()])
        plt.setp(hm.ax_heatmap.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.savefig(save_path + 'human_mouse_sim_ordered.' + fig_format, format=fig_format, dpi=self.fig_dpi)
        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        #with plt.rc_context({"figure.figsize": (32, 32), "figure.dpi": (self.fig_dpi)}):
        #fig, ax = plt.subplots(figsize=(22, 22))
        sns.set(style='white')
        TINY_SIZE = 20  # 18  # 39
        SMALL_SIZE = 20  # 18  # 42
        MEDIUM_SIZE = 20  # 46
        BIGGER_SIZE = 20  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']


        #------------------------------plot region number distribution------------------------------
        mouse_region_num_dict = Counter(adata_mouse_embedding.obs['region_name'])
        mouse_region_num_dict = {'Region name': mouse_region_num_dict.keys(), 'Sample number':mouse_region_num_dict.values()}
        mouse_region_num_df = pd.DataFrame.from_dict(mouse_region_num_dict)
        print('mouse region number:', mouse_region_num_df)
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.98
        rcParams["figure.subplot.bottom"] = 0.5
        fig, ax = plt.subplots(figsize=(24, 6.5), dpi=self.fig_dpi)
        # sns.set(rc={"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)})
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        sns.barplot(x="Region name", y="Sample number", data=mouse_region_num_df, order=mouse_67_labels_list,
                    palette=self.mouse_64_color_dict, width=0.8, ax=ax)  #
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.xlabel('')
        plt.savefig(save_path + 'mouse_region_num.' + fig_format, format=fig_format, dpi=self.fig_dpi)

        rcParams["figure.subplot.bottom"] = 0.45
        human_region_num_dict = Counter(adata_human_embedding.obs['region_name'])
        human_region_num_dict = {'Region name': human_region_num_dict.keys(),
                                 'Sample number': human_region_num_dict.values()}
        human_region_num_df = pd.DataFrame.from_dict(human_region_num_dict)
        print('human region number:', human_region_num_df)
        fig, ax = plt.subplots(figsize=(29, 7), dpi=self.fig_dpi)
        # sns.set(rc={"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)})
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        sns.barplot(x="Region name", y="Sample number", data=human_region_num_df, order=human_88_labels_list,
                    palette=self.human_88_color_dict, width=0.8, ax=ax)  #
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.xlabel('')
        plt.savefig(save_path + 'human_region_num.' + fig_format, format=fig_format, dpi=self.fig_dpi)
        #-------------------------------------------------------------------------------------------------------


        rcParams["figure.subplot.right"] = 0.85
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.bottom"] = 0.15
        rcParams["figure.subplot.top"] = 0.88
        #fig, ax = plt.subplots(figsize=(16, 16), dpi=self.fig_dpi)

        hm = sns.clustermap(Var_Corr, cmap=cmap_used,
                       row_colors=list(self.human_88_labels['color_hex_triplet']),
                       col_colors=list(self.mouse_64_labels['color_hex_triplet']),
                       yticklabels='auto',
                       xticklabels='auto', figsize=(16, 16), linewidth=0.2, linecolor='grey',
                            cbar_pos=(0.8, 0.2, 0.15, 0.03),
                            dendrogram_ratio=0.08, colors_ratio=0.01, cbar_kws={'label': 'Mean correlation', 'orientation': 'horizontal'}, tree_kws={'colors':'black'}
                            )  # cmap=color_map, center=0.6
        ax = hm.ax_heatmap
        lw_v = 2


        ax.add_patch(Rectangle((8, 0), 9, 23, fill=False, edgecolor='red', lw=lw_v))
        ax.add_patch(Rectangle((0, 23), 3, 14, fill=False, edgecolor='yellow', lw=lw_v))

        ax.add_patch(Rectangle((12, 30), 5, 5, fill=False, edgecolor='green', lw=lw_v))

        ax.add_patch(Rectangle((0, 0), 17, 37, fill=False, edgecolor='blue', lw=lw_v-2))


        ax.add_patch(Rectangle((17, 37), 19, 39, fill=False, edgecolor='purple', lw=lw_v))

        ax.add_patch(Rectangle((46, 78), 10, 6, fill=False, edgecolor='black', lw=lw_v))

        ax.add_patch(Rectangle((3, 76), 5, 2, fill=False, edgecolor='cyan', lw=lw_v))


        #plt.setp(hm.ax_heatmap.get_xticklabels(), rotation=30)
        for item in hm.ax_heatmap.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.ax_heatmap.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        #plt.subplots_adjust(bottom=0.2, left=0.05, right=0.85, top=0.9)
        human_tick_parent_color_dict ={k:self.human_16_color_dict[v] for k,v in self.human_88_parent_region_dict.items()}
        mouse_tick_parent_color_dict = {k: self.mouse_15_color_dict[v] for k, v in self.mouse_64_parent_region_dict.items()}
        for i, ticklabel in enumerate(hm.ax_heatmap.xaxis.get_majorticklabels()):
            ticklabel.set_color(mouse_tick_parent_color_dict[ticklabel.get_text()])
        for i, ticklabel in enumerate(hm.ax_heatmap.yaxis.get_majorticklabels()):
            ticklabel.set_color(human_tick_parent_color_dict[ticklabel.get_text()])
        plt.savefig(save_path + 'region_Hiercluster_human_mouse.' + fig_format, format=fig_format, dpi=self.fig_dpi)


        #-----------------------------------------
        # show all region names

        rcParams["figure.subplot.right"] = 0.85
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.bottom"] = 0.15
        rcParams["figure.subplot.top"] = 0.88
        # fig, ax = plt.subplots(figsize=(16, 16), dpi=self.fig_dpi)

        hm = sns.clustermap(Var_Corr, cmap=cmap_used,
                            row_colors=list(self.human_88_labels['color_hex_triplet']),
                            col_colors=list(self.mouse_64_labels['color_hex_triplet']),
                            yticklabels=True,
                            xticklabels=True, figsize=(25, 25), linewidth=0.5, linecolor='grey',
                            cbar_pos=(0.82, 0.15, 0.15, 0.03),
                            dendrogram_ratio=0.05, colors_ratio=0.01,
                            cbar_kws={'label': 'Mean correlation', 'orientation': 'horizontal'},
                            tree_kws={'colors': 'black'}
                            )  # cmap=color_map, center=0.6
        ax = hm.ax_heatmap
        lw_v = 2

        ax.add_patch(Rectangle((8, 0), 9, 23, fill=False, edgecolor='red', lw=lw_v))
        ax.add_patch(Rectangle((0, 23), 3, 14, fill=False, edgecolor='yellow', lw=lw_v))

        ax.add_patch(Rectangle((12, 30), 5, 5, fill=False, edgecolor='green', lw=lw_v))

        ax.add_patch(Rectangle((0, 0), 17, 37, fill=False, edgecolor='blue', lw=lw_v - 2))

        ax.add_patch(Rectangle((17, 37), 19, 39, fill=False, edgecolor='purple', lw=lw_v))

        ax.add_patch(Rectangle((46, 78), 10, 6, fill=False, edgecolor='black', lw=lw_v))

        ax.add_patch(Rectangle((3, 76), 5, 2, fill=False, edgecolor='cyan', lw=lw_v))

        # plt.setp(hm.ax_heatmap.get_xticklabels(), rotation=30)
        for item in hm.ax_heatmap.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.ax_heatmap.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.subplots_adjust(bottom=0.2, left=0.05, right=0.85, top=0.9)
        plt.savefig(save_path + 'region_Hiercluster_human_mouse_all.' + fig_format, format=fig_format, dpi=self.fig_dpi)

        # parent region
        # Step 1
        # sns.set(style='white')
        # TINY_SIZE = 32  # 24  # 39
        # SMALL_SIZE = 36  # 28  # 42
        # MEDIUM_SIZE = 36  # 32  # 46
        # BIGGER_SIZE = 40  # 36  # 46
        #
        # plt.rc('font', size=32)  # 35 controls default text sizes
        # plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        # plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        # plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        # plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        # plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        #
        # rcParams['font.family'] = 'sans-serif'
        # rcParams['font.sans-serif'] = ['Arial']

        human_embedding_dict = OrderedDict()
        # print(list(human_88_labels['region_name']))
        for r_n in self.human_16_labels_list:
            human_embedding_dict[r_n] = None

        mouse_embedding_dict = OrderedDict()
        #mouse_67_labels_list = list(mouse_67_labels['region_name'])
        for r_n in self.mouse_15_labels_list:
            mouse_embedding_dict[r_n] = None

        for region_name in self.human_16_labels_list:
            mean_embedding = np.mean(adata_human_embedding[adata_human_embedding.obs['parent_region_name'] == region_name].X,
                                     axis=0)
            human_embedding_dict[region_name] = mean_embedding

        human_embedding_df = pd.DataFrame.from_dict(human_embedding_dict)

        for region_name in self.mouse_15_labels_list:
            mean_embedding = np.mean(adata_mouse_embedding[adata_mouse_embedding.obs['parent_region_name'] == region_name].X,
                                     axis=0)
            mouse_embedding_dict[region_name] = mean_embedding
            # print(mean_embedding.shape)
        mouse_embedding_df = pd.DataFrame.from_dict(mouse_embedding_dict)

        result = pd.concat([human_embedding_df, mouse_embedding_df], axis=1).corr()
        Var_Corr = result[mouse_embedding_df.columns].loc[human_embedding_df.columns]
        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        color_map = sns.color_palette("vlag", as_cmap=True)
        # color_map = sns.color_palette("rocket_r", as_cmap=
        rcParams["figure.subplot.left"] = 0.2
        rcParams["figure.subplot.bottom"] = 0.2
        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.fig_dpi)
        hm = sns.clustermap(Var_Corr, cmap=cmap_used,
                            row_colors=list(self.human_16_labels['color_hex_triplet']),
                            col_colors=list(self.mouse_15_labels['color_hex_triplet']),
                            yticklabels=True,
                            xticklabels=True, figsize=(9, 9), linewidth=0.2, linecolor='gray',
                            row_cluster=False, col_cluster=False,
                            cbar_pos=(0.75, 0.25, 0.04, 0.10)
                            )  # cmap=color_map, center=0.6
        for item in hm.ax_heatmap.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.ax_heatmap.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.savefig(save_path + 'parent_human_mouse_sim_ordered.' + fig_format, format=fig_format)
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.bottom"] = 0.05
        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        #with plt.rc_context({"figure.figsize": (16, 16), "figure.dpi": (self.fig_dpi)}):
        hm = sns.clustermap(Var_Corr, cmap=cmap_used,
                       row_colors=list(self.human_16_labels['color_hex_triplet']),
                       col_colors=list(self.mouse_15_labels['color_hex_triplet']),
                       yticklabels=True,
                       xticklabels=True, figsize=(9, 9), linewidth=0.2, linecolor='gray', cbar_pos=(0.85, 0.85, 0.04, 0.10)
                       )  # cmap=color_map, center=0.6
        for item in hm.ax_heatmap.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.ax_heatmap.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.savefig(save_path + 'parent_region_Hiercluster_human_mouse.' + fig_format, format=fig_format)

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.bottom"] = 0.1


    def experiment_7_2_align_cross_species_alignementscore(self):
        sns.set(style='white')
        TINY_SIZE = 24#18  # 39
        SMALL_SIZE = 24#18  # 42
        MEDIUM_SIZE = 24  # 46
        BIGGER_SIZE = 24  # 46

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
        '''
            Step 1: Compute average embedding of every region in two species, use two dict to store;
            Step 2: Compute alignemt score matrix, use np array to store;
            Step 3: Heatmap.
            '''
        # Read ordered labels
        fig_format = cfg.BrainAlign.fig_format

        human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
        mouse_64_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)

        adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')

        human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/7_2_experiment_align_cross_species_alignmentscore/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Step 1
        human_embedding_dict = OrderedDict()
        # print(list(human_88_labels['region_name']))
        human_88_labels_list = list(human_88_labels['region_name'])
        human_88_labels_list.remove('paraterminal gyrus')
        for r_n in human_88_labels_list:
            human_embedding_dict[r_n] = None

        mouse_embedding_dict = OrderedDict()
        mouse_64_labels_list = list(mouse_64_labels['region_name'])

        print(human_88_labels_list)


        for r_n in mouse_64_labels_list:
            mouse_embedding_dict[r_n] = None

        for region_name in human_88_labels_list:
            mean_embedding = np.mean(adata_human_embedding[adata_human_embedding.obs['region_name'] == region_name].X,
                                     axis=0)
            human_embedding_dict[region_name] = mean_embedding

        human_embedding_df = pd.DataFrame.from_dict(human_embedding_dict)

        for region_name in mouse_64_labels_list:
            mean_embedding = np.mean(adata_mouse_embedding[adata_mouse_embedding.obs['region_name'] == region_name].X,
                                     axis=0)
            mouse_embedding_dict[region_name] = mean_embedding
            # print(mean_embedding.shape)
        mouse_embedding_df = pd.DataFrame.from_dict(mouse_embedding_dict)

        result = pd.concat([human_embedding_df, mouse_embedding_df], axis=1).corr()
        Var_Corr = result[mouse_embedding_df.columns].loc[human_embedding_df.columns]

        for h_region_name in human_88_labels_list:
            for m_region_name in mouse_64_labels_list:
                m_embedding = adata_human_embedding[adata_human_embedding.obs['region_name'] == h_region_name].X
                print('m_embedding', m_embedding.shape)
                h_embedding = adata_mouse_embedding[adata_mouse_embedding.obs['region_name'] == m_region_name].X
                print('h_embedding', h_embedding.shape)
                X = np.concatenate((m_embedding, h_embedding), axis=0)
                print('X.shape', X.shape)
                Y = np.concatenate(
                    [np.zeros((m_embedding.shape[0], 1)), np.ones((h_embedding.shape[0], 1))],
                    axis=0)
                aligned_score = seurat_alignment_score(X, Y)

                Var_Corr[m_region_name].loc[h_region_name] = aligned_score

        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        color_map = sns.color_palette("vlag", as_cmap=True)

        cmap_used = "Reds"#"vlag" #"YlGnBu_r"

        sns.set(style='white')
        TINY_SIZE = 18  # 18  # 39
        SMALL_SIZE = 18  # 18  # 42
        MEDIUM_SIZE = 18  # 46
        BIGGER_SIZE = 18  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']


        #------------------------------plot region number distribution------------------------------
        mouse_region_num_dict = Counter(adata_mouse_embedding.obs['region_name'])
        mouse_region_num_dict = {'Region name': mouse_region_num_dict.keys(), 'Sample number':mouse_region_num_dict.values()}
        mouse_region_num_df = pd.DataFrame.from_dict(mouse_region_num_dict)


        rcParams["figure.subplot.bottom"] = 0.45
        human_region_num_dict = Counter(adata_human_embedding.obs['region_name'])
        human_region_num_dict = {'Region name': human_region_num_dict.keys(),
                                 'Sample number': human_region_num_dict.values()}
        human_region_num_df = pd.DataFrame.from_dict(human_region_num_dict)

        #-------------------------------------------------------------------------------------------------------


        rcParams["figure.subplot.right"] = 0.85
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.bottom"] = 0.15
        rcParams["figure.subplot.top"] = 0.88
        #fig, ax = plt.subplots(figsize=(16, 16), dpi=self.fig_dpi)
        row_colors_list = [self.human_88_color_dict[x] for x in human_88_labels_list]

        hm = sns.clustermap(Var_Corr, cmap=cmap_used,
                       row_colors=row_colors_list,
                       col_colors=list(self.mouse_64_labels['color_hex_triplet']),
                       yticklabels='auto',
                       xticklabels='auto', figsize=(13, 13), linewidth=0.2, linecolor='grey',
                            cbar_pos=(0.78, 0.27, 0.15, 0.03),
                            dendrogram_ratio=0.08, colors_ratio=0.01, cbar_kws={'label': 'Mean correlation', 'orientation': 'horizontal'}, tree_kws={'colors':'black'}
                            ) # cmap=color_map, center=0.6
        ax = hm.ax_heatmap
        lw_v = 2

        edge_color = 'blue'

        ax.add_patch(Rectangle((0, 8), 7, 20, fill=False, edgecolor='blue', lw=lw_v))

        ax.add_patch(Rectangle((17, 0), 22, 8, fill=False, edgecolor='cyan', lw=lw_v))

        ax.add_patch(Rectangle((22, 0), 4, 28, fill=False, edgecolor='green', lw=lw_v))

        ax.add_patch(Rectangle((57, 48), 7, 39, fill=False, edgecolor='black', lw=lw_v))

        ax.add_patch(Rectangle((0, 0), 1, 87, fill=False, edgecolor='grey', lw=lw_v))

        ax.add_patch(Rectangle((7, 48), 10, 47, fill=False, edgecolor='purple', lw=lw_v))

        ax.add_patch(Rectangle((7, 83), 45, 4, fill=False, edgecolor='yellow', lw=lw_v))

        ax.add_patch(Rectangle((41, 0), 17, 28, fill=False, edgecolor='red', lw=lw_v))

        ax.add_patch(Rectangle((51, 30), 6, 18, fill=False, edgecolor='orange', lw=lw_v))

        ax.add_patch(Rectangle((7, 43), 50, 44, fill=False, edgecolor='grey', lw=lw_v))

        #
        # ax.add_patch(Rectangle((12, 30), 5, 5, fill=False, edgecolor=edge_color, lw=lw_v))
        #
        # ax.add_patch(Rectangle((0, 0), 17, 37, fill=False, edgecolor='black', lw=lw_v))
        #
        #
        # ax.add_patch(Rectangle((17, 37), 19, 39, fill=False, edgecolor=edge_color, lw=lw_v))
        #
        # ax.add_patch(Rectangle((46, 78), 10, 6, fill=False, edgecolor=edge_color, lw=lw_v))
        #
        # ax.add_patch(Rectangle((3, 76), 5, 2, fill=False, edgecolor=edge_color, lw=lw_v))


        #plt.setp(hm.ax_heatmap.get_xticklabels(), rotation=30)
        for item in hm.ax_heatmap.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.ax_heatmap.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        #plt.subplots_adjust(bottom=0.2, left=0.05, right=0.85, top=0.9)
        plt.savefig(save_path + 'region_Hiercluster_human_mouse_alignmentscore.' + fig_format, format=fig_format, dpi=self.fig_dpi)

        human_embedding_dict = OrderedDict()
        # print(list(human_88_labels['region_name']))
        for r_n in self.human_16_labels_list:
            human_embedding_dict[r_n] = None

        mouse_embedding_dict = OrderedDict()
        #mouse_67_labels_list = list(mouse_67_labels['region_name'])
        for r_n in self.mouse_15_labels_list:
            mouse_embedding_dict[r_n] = None

        for region_name in self.human_16_labels_list:
            mean_embedding = np.mean(adata_human_embedding[adata_human_embedding.obs['parent_region_name'] == region_name].X,
                                     axis=0)
            human_embedding_dict[region_name] = mean_embedding

        human_embedding_df = pd.DataFrame.from_dict(human_embedding_dict)

        for region_name in self.mouse_15_labels_list:
            mean_embedding = np.mean(adata_mouse_embedding[adata_mouse_embedding.obs['parent_region_name'] == region_name].X,
                                     axis=0)
            mouse_embedding_dict[region_name] = mean_embedding
            # print(mean_embedding.shape)
        mouse_embedding_df = pd.DataFrame.from_dict(mouse_embedding_dict)

        result = pd.concat([human_embedding_df, mouse_embedding_df], axis=1).corr()
        Var_Corr = result[mouse_embedding_df.columns].loc[human_embedding_df.columns]
        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        color_map = sns.color_palette("vlag", as_cmap=True)
        # color_map = sns.color_palette("rocket_r", as_cmap=
        rcParams["figure.subplot.left"] = 0.2
        rcParams["figure.subplot.bottom"] = 0.2
        fig, ax = plt.subplots(figsize=(8, 8), dpi=self.fig_dpi)
        hm = sns.clustermap(Var_Corr, cmap=cmap_used,
                            row_colors=list(self.human_16_labels['color_hex_triplet']),
                            col_colors=list(self.mouse_15_labels['color_hex_triplet']),
                            yticklabels=True, row_cluster=False, col_cluster=False,
                            xticklabels=True, figsize=(8, 8), linewidth=0.2, linecolor='gray',
                            cbar_pos=(0.85, 0.85, 0.04, 0.12)
                            )  # cmap=color_map, center=0.6
        for item in hm.ax_heatmap.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.ax_heatmap.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.savefig(save_path + 'parent_human_mouse_sim_ordered.' + fig_format, format=fig_format)
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.bottom"] = 0.05
        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        #with plt.rc_context({"figure.figsize": (16, 16), "figure.dpi": (self.fig_dpi)}):
        hm = sns.clustermap(Var_Corr, cmap=cmap_used,
                       row_colors=list(self.human_16_labels['color_hex_triplet']),
                       col_colors=list(self.mouse_15_labels['color_hex_triplet']),
                       yticklabels=True,
                       xticklabels=True, figsize=(8, 8), linewidth=0.2, linecolor='gray', cbar_pos=(0.85, 0.85, 0.04, 0.12)
                       )  # cmap=color_map, center=0.6
        for item in hm.ax_heatmap.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.ax_heatmap.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.savefig(save_path + 'parent_region_Hiercluster_human_mouse.' + fig_format, format=fig_format)

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.bottom"] = 0.1


    def experiment_7_1_align_cross_species_split(self):
        sns.set(style='white')
        TINY_SIZE = 18  # 39
        SMALL_SIZE = 18  # 42
        MEDIUM_SIZE = 24  # 46
        BIGGER_SIZE = 24  # 46

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
        '''
            Step 1: Compute average embedding of every region in two species, use two dict to store;
            Step 2: Compute similarity matrix, use np array to store;
            Step 3: Heatmap.
            '''
        # Read ordered labels
        fig_format = cfg.BrainAlign.fig_format

        human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
        mouse_64_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)

        adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')


        human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)
        homo_region_list = list(human_mouse_homo_region['Mouse'].values) + list(human_mouse_homo_region['Human'].values)
        homo_region_color_list = [self.mouse_64_color_dict[m_r] for m_r in
                                  list(human_mouse_homo_region['Mouse'].values)] + \
                                 [self.human_88_color_dict[h_r] for h_r in
                                  list(human_mouse_homo_region['Human'].values)]
        # print(homo_region_list)
        # adata_embedding_homo = adata_embedding[adata_embedding.obs['region_name'].isin(homo_region_list)]
        homo_region_palette = {k: v for k, v in zip(homo_region_list, homo_region_color_list)}

        mouse_homo_region_list = list(human_mouse_homo_region['Mouse'].values)
        human_homo_region_list = list(human_mouse_homo_region['Human'].values)
        mouse_homo_region_color_list = [self.mouse_64_color_dict[m_r] for m_r in
                                        list(human_mouse_homo_region['Mouse'].values)]
        human_homo_region_color_list = [self.human_88_color_dict[h_r] for h_r in
                                        list(human_mouse_homo_region['Human'].values)]


        human_region_list = human_homo_region_list
        mouse_region_list = mouse_homo_region_list

        human_color_list = human_homo_region_color_list
        mouse_color_list = mouse_homo_region_color_list

        for i in range(len(human_88_labels['region_name'])):
            h_r = human_88_labels['region_name'][i]
            if h_r not in set(human_homo_region_list):
                human_region_list.append(h_r)
                human_color_list.append(self.human_88_color_dict[h_r])


        for i in range(len(mouse_64_labels['region_name'])):
            m_r = mouse_64_labels['region_name'][i]
            if m_r not in set(mouse_homo_region_list):
                mouse_region_list.append(m_r)
                mouse_color_list.append(self.mouse_64_color_dict[m_r])


        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/7_1_experiment_align_cross_species/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Step 1
        human_embedding_dict = OrderedDict()
        # print(list(human_88_labels['region_name']))
        human_88_labels_list = human_region_list
        for r_n in human_88_labels_list:
            human_embedding_dict[r_n] = None

        mouse_embedding_dict = OrderedDict()
        mouse_64_labels_list = mouse_region_list
        for r_n in mouse_64_labels_list:
            mouse_embedding_dict[r_n] = None

        for region_name in human_88_labels_list:
            mean_embedding = np.mean(adata_human_embedding[adata_human_embedding.obs['region_name'] == region_name].X,
                                     axis=0)
            human_embedding_dict[region_name] = mean_embedding

        human_embedding_df = pd.DataFrame.from_dict(human_embedding_dict)

        for region_name in mouse_64_labels_list:
            mean_embedding = np.mean(adata_mouse_embedding[adata_mouse_embedding.obs['region_name'] == region_name].X,
                                     axis=0)
            mouse_embedding_dict[region_name] = mean_embedding
            # print(mean_embedding.shape)
        mouse_embedding_df = pd.DataFrame.from_dict(mouse_embedding_dict)

        result = pd.concat([human_embedding_df, mouse_embedding_df], axis=1).corr()
        Var_Corr = result[mouse_embedding_df.columns].loc[human_embedding_df.columns]
        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        color_map = sns.color_palette("vlag", as_cmap=True)

        cmap_used = "vlag" #"YlGnBu_r"
        # color_map = sns.color_palette("rocket_r", as_cmap=
        rcParams["figure.subplot.left"] = 0.2
        rcParams["figure.subplot.right"] = 1
        rcParams["figure.subplot.bottom"] = 0.2

        fig, ax = plt.subplots(figsize=(30, 30))
        sns.heatmap(Var_Corr, xticklabels=mouse_embedding_df.columns, yticklabels=human_embedding_df.columns,
                    annot=False, ax=ax, cmap=cmap_used, linewidth=0.2, linecolor='gray') #
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.savefig(save_path + 'human_mouse_sim_ordered.' + fig_format, format=fig_format)
        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        #with plt.rc_context({"figure.figsize": (32, 32), "figure.dpi": (self.fig_dpi)}):
        #fig, ax = plt.subplots(figsize=(22, 22))


        #------------------------------plot region number distribution------------------------------
        mouse_region_num_dict = Counter(adata_mouse_embedding.obs['region_name'])
        mouse_region_num_dict = {'Region name': mouse_region_num_dict.keys(), 'Sample number':mouse_region_num_dict.values()}
        mouse_region_num_df = pd.DataFrame.from_dict(mouse_region_num_dict)
        print('mouse region number:', mouse_region_num_df)
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.98
        rcParams["figure.subplot.bottom"] = 0.5
        fig, ax = plt.subplots(figsize=(24, 6.5), dpi=self.fig_dpi)
        # sns.set(rc={"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)})
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        sns.barplot(x="Region name", y="Sample number", data=mouse_region_num_df, order=mouse_64_labels_list,
                    palette=self.mouse_64_color_dict, width=0.8, ax=ax)  #
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.xlabel('')
        plt.savefig(save_path + 'mouse_region_num.' + fig_format, format=fig_format)

        rcParams["figure.subplot.bottom"] = 0.45
        human_region_num_dict = Counter(adata_human_embedding.obs['region_name'])
        human_region_num_dict = {'Region name': human_region_num_dict.keys(),
                                 'Sample number': human_region_num_dict.values()}
        human_region_num_df = pd.DataFrame.from_dict(human_region_num_dict)
        print('human region number:', human_region_num_df)
        fig, ax = plt.subplots(figsize=(29, 7), dpi=self.fig_dpi)
        # sns.set(rc={"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)})
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        sns.barplot(x="Region name", y="Sample number", data=human_region_num_df, order=human_88_labels_list,
                    palette=self.human_88_color_dict, width=0.8, ax=ax)  #
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.xlabel('')
        plt.savefig(save_path + 'human_region_num.' + fig_format, format=fig_format)
        #-------------------------------------------------------------------------------------------------------


        # modify region order



        rcParams["figure.subplot.right"] = 0.85
        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.bottom"] = 0.05

        fig, ax = plt.subplots(figsize=(6, 6), dpi=self.fig_dpi)

        hm = sns.clustermap(Var_Corr, cmap=cmap_used,
                       row_colors=human_color_list,
                       col_colors=mouse_color_list,
                       yticklabels=True,
                        row_cluster=False, col_cluster=False, #ax=ax,
                       xticklabels=True, figsize=(30, 30), linewidth=0.2, linecolor='gray', cbar_pos=(0.92, 0.8, 0.04, 0.12))  # cmap=color_map, center=0.6
        ax = hm.ax_heatmap
        lw_v = 6
        ax.add_patch(Rectangle((0, 0), 20, 20, fill=False, edgecolor='blue', lw=lw_v))
        #plt.setp(hm.ax_heatmap.get_xticklabels(), rotation=30)
        for item in hm.ax_heatmap.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.ax_heatmap.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        #plt.subplots_adjust(bottom=0.25, left=0.2)
        plt.savefig(save_path + 'region_Hiercluster_human_mouse.' + fig_format, format=fig_format, dpi=self.fig_dpi)

        c = Colormap()
        color_map = c.cmap_linear('white', 'white', 'red')

        Var_Corr_homo = Var_Corr.iloc[0:20, 0:20]
        rcParams["figure.subplot.right"] = 0.8
        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.bottom"] = 0.04
        rcParams["figure.subplot.top"] = 0.99

        fig, ax = plt.subplots(figsize=(7, 7), dpi=self.fig_dpi)

        hm = sns.clustermap(Var_Corr_homo, cmap=color_map,
                            row_colors=human_homo_region_color_list,
                            col_colors=mouse_homo_region_color_list,
                            yticklabels='auto',
                            row_cluster=False, col_cluster=False,
                            xticklabels='auto', figsize=(7, 7), linewidth=0.12, linecolor='black',
                            cbar_pos=None)  # cmap=color_map, center=0.6 #(0.05, 0.8, 0.04, 0.12)
        #ax = hm.ax_heatmap
        #lw_v = 10
        #ax.add_patch(Rectangle((0, 0), 20, 20, fill=False, edgecolor='cyan', lw=lw_v))
        # plt.setp(hm.ax_heatmap.get_xticklabels(), rotation=30)
        for item in hm.ax_heatmap.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.ax_heatmap.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        #plt.subplots_adjust(bottom=0.26)
        b, t = plt.ylim()  # discover the values for bottom and top
        b += 0.2  # Add 0.5 to the bottom
        # t -= 0.5  # Subtract 0.5 from the top
        plt.ylim(b, t)  # update the ylim(bottom, top) values
        l, r = plt.xlim()  # discover the values for bottom and top
        r += 0.2  # Add 0.5 to the bottom
        # l -= 0.5  # Subtract 0.5 from the top
        plt.xlim(l, r)  # update the ylim(bottom, top) values

        plt.savefig(save_path + 'homo_region_Hiercluster_human_mouse.' + fig_format, format=fig_format, dpi=self.fig_dpi)

        # parent region
        # Step 1
        # sns.set(style='white')
        # TINY_SIZE = 32  # 24  # 39
        # SMALL_SIZE = 36  # 28  # 42
        # MEDIUM_SIZE = 36  # 32  # 46
        # BIGGER_SIZE = 40  # 36  # 46
        #
        # plt.rc('font', size=32)  # 35 controls default text sizes
        # plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        # plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        # plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        # plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        # plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        #
        # rcParams['font.family'] = 'sans-serif'
        # rcParams['font.sans-serif'] = ['Arial']

        human_embedding_dict = OrderedDict()
        # print(list(human_88_labels['region_name']))
        for r_n in self.human_16_labels_list:
            human_embedding_dict[r_n] = None

        mouse_embedding_dict = OrderedDict()
        #mouse_67_labels_list = list(mouse_67_labels['region_name'])
        for r_n in self.mouse_15_labels_list:
            mouse_embedding_dict[r_n] = None

        for region_name in self.human_16_labels_list:
            mean_embedding = np.mean(adata_human_embedding[adata_human_embedding.obs['parent_region_name'] == region_name].X,
                                     axis=0)
            human_embedding_dict[region_name] = mean_embedding

        human_embedding_df = pd.DataFrame.from_dict(human_embedding_dict)

        for region_name in self.mouse_15_labels_list:
            mean_embedding = np.mean(adata_mouse_embedding[adata_mouse_embedding.obs['parent_region_name'] == region_name].X,
                                     axis=0)
            mouse_embedding_dict[region_name] = mean_embedding
            # print(mean_embedding.shape)
        mouse_embedding_df = pd.DataFrame.from_dict(mouse_embedding_dict)

        result = pd.concat([human_embedding_df, mouse_embedding_df], axis=1).corr()
        Var_Corr = result[mouse_embedding_df.columns].loc[human_embedding_df.columns]
        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        color_map = sns.color_palette("vlag", as_cmap=True)
        # color_map = sns.color_palette("rocket_r", as_cmap=
        rcParams["figure.subplot.left"] = 0.2
        rcParams["figure.subplot.bottom"] = 0.2
        fig, ax = plt.subplots(figsize=(10, 12), dpi=self.fig_dpi)
        sns.heatmap(Var_Corr, xticklabels=mouse_embedding_df.columns, yticklabels=human_embedding_df.columns,
                    annot=False, ax=ax, cmap=cmap_used, linewidth=0.2, square=True, linecolor='gray') #
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.savefig(save_path + 'parent_human_mouse_sim_ordered.' + fig_format, format=fig_format)
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.bottom"] = 0.05
        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        #with plt.rc_context({"figure.figsize": (16, 16), "figure.dpi": (self.fig_dpi)}):
        hm = sns.clustermap(Var_Corr, cmap=cmap_used,
                       row_colors=list(self.human_16_labels['color_hex_triplet']),
                       col_colors=list(self.mouse_15_labels['color_hex_triplet']),
                       yticklabels=True,
                       xticklabels=True, figsize=(10, 10), linewidth=0.2, linecolor='gray', cbar_pos=(0.85, 0.85, 0.04, 0.12)
                       )  # cmap=color_map, center=0.6
        for item in hm.ax_heatmap.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.ax_heatmap.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.savefig(save_path + 'parent_region_Hiercluster_human_mouse.' + fig_format, format=fig_format)


    def experiment_7_3_beforealign_cross_species_split(self):
        sns.set(style='white')
        TINY_SIZE = 18  # 39
        SMALL_SIZE = 18  # 42
        MEDIUM_SIZE = 24  # 46
        BIGGER_SIZE = 24  # 46

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
        '''
            Step 1: Compute average embedding of every region in two species, use two dict to store;
            Step 2: Compute similarity matrix, use np array to store;
            Step 3: Heatmap.
            '''
        # Read ordered labels
        fig_format = cfg.BrainAlign.fig_format

        human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
        mouse_64_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)

        adata_mouse_expression = sc.read_h5ad(cfg.CAME.path_rawdata1)
        sc.tl.pca(adata_mouse_expression, svd_solver='arpack', n_comps=30)
        adata_mouse_embedding = ad.AnnData(adata_mouse_expression.obsm['X_pca'])
        adata_mouse_embedding.obs_names = adata_mouse_expression.obs_names
        adata_mouse_embedding.obs['region_name'] = adata_mouse_expression.obs['region_name']

        adata_human_expression = sc.read_h5ad(cfg.CAME.path_rawdata2)
        sc.tl.pca(adata_human_expression, svd_solver='arpack', n_comps=30)
        adata_human_embedding = ad.AnnData(adata_human_expression.obsm['X_pca'])
        adata_human_embedding.obs_names = adata_human_expression.obs_names
        adata_human_embedding.obs['region_name'] = adata_human_expression.obs['region_name']

        adata_mouse_embedding.obs['dataset'] = 'Mouse'
        adata_human_embedding.obs['dataset'] = 'Human'


        human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)
        homo_region_list = list(human_mouse_homo_region['Mouse'].values) + list(human_mouse_homo_region['Human'].values)
        homo_region_color_list = [self.mouse_64_color_dict[m_r] for m_r in
                                  list(human_mouse_homo_region['Mouse'].values)] + \
                                 [self.human_88_color_dict[h_r] for h_r in
                                  list(human_mouse_homo_region['Human'].values)]
        # print(homo_region_list)
        # adata_embedding_homo = adata_embedding[adata_embedding.obs['region_name'].isin(homo_region_list)]
        homo_region_palette = {k: v for k, v in zip(homo_region_list, homo_region_color_list)}

        mouse_homo_region_list = list(human_mouse_homo_region['Mouse'].values)
        human_homo_region_list = list(human_mouse_homo_region['Human'].values)
        mouse_homo_region_color_list = [self.mouse_64_color_dict[m_r] for m_r in
                                        list(human_mouse_homo_region['Mouse'].values)]
        human_homo_region_color_list = [self.human_88_color_dict[h_r] for h_r in
                                        list(human_mouse_homo_region['Human'].values)]


        human_region_list = human_homo_region_list
        mouse_region_list = mouse_homo_region_list

        human_color_list = human_homo_region_color_list
        mouse_color_list = mouse_homo_region_color_list

        for i in range(len(human_88_labels['region_name'])):
            h_r = human_88_labels['region_name'][i]
            if h_r not in set(human_homo_region_list):
                human_region_list.append(h_r)
                human_color_list.append(self.human_88_color_dict[h_r])


        for i in range(len(mouse_64_labels['region_name'])):
            m_r = mouse_64_labels['region_name'][i]
            if m_r not in set(mouse_homo_region_list):
                mouse_region_list.append(m_r)
                mouse_color_list.append(self.mouse_64_color_dict[m_r])


        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/7_3_experiment_beforealignment_align_cross_species/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Step 1
        human_embedding_dict = OrderedDict()
        # print(list(human_88_labels['region_name']))
        human_88_labels_list = human_region_list
        for r_n in human_88_labels_list:
            human_embedding_dict[r_n] = None

        mouse_embedding_dict = OrderedDict()
        mouse_64_labels_list = mouse_region_list
        for r_n in mouse_64_labels_list:
            mouse_embedding_dict[r_n] = None

        for region_name in human_88_labels_list:
            mean_embedding = np.mean(adata_human_embedding[adata_human_embedding.obs['region_name'] == region_name].X,
                                     axis=0)
            human_embedding_dict[region_name] = mean_embedding

        human_embedding_df = pd.DataFrame.from_dict(human_embedding_dict)

        for region_name in mouse_64_labels_list:
            mean_embedding = np.mean(adata_mouse_embedding[adata_mouse_embedding.obs['region_name'] == region_name].X,
                                     axis=0)
            mouse_embedding_dict[region_name] = mean_embedding
            # print(mean_embedding.shape)
        mouse_embedding_df = pd.DataFrame.from_dict(mouse_embedding_dict)

        result = pd.concat([human_embedding_df, mouse_embedding_df], axis=1).corr()
        Var_Corr = result[mouse_embedding_df.columns].loc[human_embedding_df.columns]
        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        color_map = sns.color_palette("vlag", as_cmap=True)

        cmap_used = "vlag" #"YlGnBu_r"
        # color_map = sns.color_palette("rocket_r", as_cmap=
        rcParams["figure.subplot.left"] = 0.2
        rcParams["figure.subplot.right"] = 1
        rcParams["figure.subplot.bottom"] = 0.2

        c = Colormap()
        color_map = c.cmap_linear('white', 'white', 'red')

        Var_Corr_homo = Var_Corr.iloc[0:20, 0:20]
        rcParams["figure.subplot.right"] = 0.98
        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.bottom"] = 0.04
        rcParams["figure.subplot.top"] = 0.99
        fig, ax = plt.subplots(figsize=(7, 7), dpi=self.fig_dpi)
        hm = sns.clustermap(Var_Corr_homo, cmap=color_map,
                            row_colors=human_homo_region_color_list,
                            col_colors=mouse_homo_region_color_list,
                            yticklabels='auto', xticklabels='auto',
                            row_cluster=False, col_cluster=False,
                            figsize=(7, 7), linewidth=0.12, linecolor='black',
                            cbar_pos=(0.02, 0.8, 0.04, 0.12))  # cmap=color_map, center=0.6
        #ax = hm.ax_heatmap
        #lw_v = 10
        #ax.add_patch(Rectangle((0, 0), 20, 20, fill=False, edgecolor='cyan', lw=lw_v))
        # plt.setp(hm.ax_heatmap.get_xticklabels(), rotation=30)
        for item in hm.ax_heatmap.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.ax_heatmap.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        #plt.subplots_adjust(bottom=0.26)
        b, t = plt.ylim()  # discover the values for bottom and top
        b += 0.2  # Add 0.5 to the bottom
        # t -= 0.5  # Subtract 0.5 from the top
        plt.ylim(b, t)  # update the ylim(bottom, top) values
        l, r = plt.xlim()  # discover the values for bottom and top
        r += 0.2  # Add 0.5 to the bottom
        # l -= 0.5  # Subtract 0.5 from the top
        plt.xlim(l, r)  # update the ylim(bottom, top) values

        plt.savefig(save_path + 'homo_region_Hiercluster_human_mouse.' + fig_format, format=fig_format, dpi=self.fig_dpi)



    def experiment_8_cross_evaluation_aligment_cluster(self):

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

        cfg = self.cfg

        fig_format = cfg.BrainAlign.fig_format

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/8_experiment_cross_evaluation_aligment_cluster/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
        adata_mouse_embedding.obs['dataset'] = 'Mouse'
        adata_human_embedding.obs['dataset'] = 'Human'

        mouse_64_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)
        human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)

        # plot human mouse homologous samples
        human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)
        homo_region_list_mouse = list(human_mouse_homo_region['Mouse'].values)
        homo_region_list_human = list(human_mouse_homo_region['Human'].values)

        homo_region_mouse_human_dict = {k: v for k, v in zip(homo_region_list_mouse, homo_region_list_human)}

        adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
        print(adata_embedding)

        sc.pp.neighbors(adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(adata_embedding, n_components=2)

        # PCA and kmeans clustering of the whole dataset
        sc.tl.pca(adata_embedding, svd_solver='arpack', n_comps=30)

        #X_pca = adata_embedding.obsm['X_pca']
        #X_umap = adata_embedding.obsm['X_umap']
        #X = adata_embedding.X

        # Do clustering
        clustering_num = cfg.ANALYSIS.sample_cluster_num
        clustering_name = f'leiden_cluster'

        sc.tl.leiden(adata_embedding, resolution=9, key_added=clustering_name)
        clustering_num = len(Counter(adata_embedding.obs[clustering_name]))

        # Now we know each cluster is actually a pair
        region_labels = adata_embedding.obs['region_name'].values
        sample_names = adata_embedding.obs_names.values
        cluster_labels = adata_embedding.obs[clustering_name].values
        print('cluster_labels.shape', cluster_labels.shape)
        sample_cluter_dict = {k: v for k, v in zip(sample_names, cluster_labels)}
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

        ###########
        # Plot heatmap of cluster proportions on homologous regions
        mouse_region_clusters_dict = {x:[] for x in homo_region_list_mouse}
        human_region_clusters_dict = {x:[] for x in homo_region_list_human}
        ###################################
        # Compute the similarity or distance between the clusters of the two species
        # and identify those highly correlated pairs.
        # here, we only need to list the cluster pairs

        # save sample ratio distribution across clusters to evaluate alignment across species
        mouse_ratio_list = []
        human_ratio_list = []

        for c_label in cluster_labels_unique:
            mouse_adata = adata_embedding[adata_embedding.obs[clustering_name].isin([c_label])]
            mouse_num = mouse_adata[mouse_adata.obs['dataset'].isin(['Mouse'])].n_obs
            mouse_ratio_list.append(mouse_num)
            human_adata = adata_embedding[adata_embedding.obs[clustering_name].isin([c_label])]
            human_num = human_adata[human_adata.obs['dataset'].isin(['Human'])].n_obs
            human_ratio_list.append(human_num)
            print(c_label, f'mouse number = {mouse_num}', f'human number = {human_num}')

        adata_mouse_umap = adata_embedding[adata_embedding.obs['dataset'].isin(['Mouse'])]
        adata_human_umap = adata_embedding[adata_embedding.obs['dataset'].isin(['Human'])]

        mouse_count_all = Counter(adata_mouse_umap.obs['region_name'])
        human_count_all = Counter(adata_human_umap.obs['region_name'])

        palette = sns.color_palette(cc.glasbey, n_colors=clustering_num)

        umap1_x, umap1_y = adata_mouse_umap.obsm['X_umap'].toarray()[:, 0], adata_mouse_umap.obsm['X_umap'].toarray()[:, 1]
        umap1_z_value = 1 / 36 * (
                ((np.max(umap1_x) - np.min(umap1_x)) ** 2 + (np.max(umap1_y) - np.min(umap1_y)) ** 2) ** (1 / 2))

        umap2_x, umap2_y = adata_human_umap.obsm['X_umap'].toarray()[:, 0], adata_human_umap.obsm['X_umap'].toarray()[:, 1]
        umap2_z = np.zeros(umap2_x.shape)

        most_common_num = 3
        rank_region_method = 'count'  # count, count_normalized, count&count_normalized

        # Init a dict to save the homologous regions in each cluster
        cluster_mouse_human_homologs_dict = {}

        identified_homo_pairs = {'Mouse': [], 'Human': []}
        fig = plt.figure(figsize=(4, 4), dpi=self.fig_dpi)
        axes = plt.axes(projection='3d')
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        rcParams["figure.subplot.left"] = 0.05

        marker_s = 5
        line_w = 2

        original_stdout = sys.stdout  # Save a reference to the original standard output

        mouse_r_unique = set()
        human_r_unique = set()

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
                    mouse_count = Counter(
                        {k: (v1 / v2) for k, v1, v2 in zip(dict_c.keys(), dict_c.values(), mouse_count_all.values())})
                elif rank_region_method == 'count&count_normalized':
                    mouse_count_1 = Counter(mouse_adata.obs['region_name'])
                    dict_c = Counter(mouse_adata.obs['region_name'])
                    mouse_count_2 = Counter(
                        {k: (v1 / v2) for k, v1, v2 in zip(dict_c.keys(), dict_c.values(), mouse_count_all.values())})
                    mouse_count = Counter({k: (v1 + 1) * (v2 + 1) for k, v1, v2 in
                                           zip(mouse_count_1.keys(), mouse_count_1.values(), mouse_count_2.values())})
                # print('mouse_count:', mouse_count)
                mouse_region_set = mouse_count.most_common(most_common_num)
                cluster_mouse_human_homologs_dict[c_label]['Mouse'] = {'region name': [],
                                                                       'proportion': []}
                for mouse_region, mouse_region_count in mouse_region_set:
                    # mouse_count = dict(mouse_count)
                    mouse_proportion = mouse_count[mouse_region] / mouse_adata.n_obs
                    cluster_mouse_human_homologs_dict[c_label]['Mouse']['region name'].append(mouse_region)
                    cluster_mouse_human_homologs_dict[c_label]['Mouse']['proportion'].append(mouse_proportion)
                    mouse_adata_region = mouse_adata[mouse_adata.obs['region_name'].isin([mouse_region])]

                human_adata = adata_human_umap[adata_human_umap.obs[clustering_name].isin([c_label])]
                umap2_x, umap2_y = human_adata.obsm['X_umap'].toarray()[:, 0], human_adata.obsm['X_umap'].toarray()[:,
                                                                               1]
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
                    human_count = Counter({k: (v1 + 1) * (v2 + 1) for k, v1, v2 in
                                           zip(human_count_1.keys(), human_count_1.values(), human_count_2.values())})

                human_region_set = human_count.most_common(most_common_num)
                cluster_mouse_human_homologs_dict[c_label]['Human'] = {'region name': [],
                                                                       'proportion': []}
                for human_region, human_region_count in human_region_set:
                    # human_count = dict(human_count)
                    human_proportion = human_count[human_region] / human_adata.n_obs
                    cluster_mouse_human_homologs_dict[c_label]['Human']['region name'].append(human_region)
                    cluster_mouse_human_homologs_dict[c_label]['Human']['proportion'].append(human_proportion)
                    human_adata_region = human_adata[human_adata.obs['region_name'].isin([human_region])]

                for mouse_region in cluster_mouse_human_homologs_dict[c_label]['Mouse']['region name']:
                    for human_region in cluster_mouse_human_homologs_dict[c_label]['Human']['region name']:
                        print(
                            f'Identified homologous regions for cluster {c_label}: Mouse-{mouse_region}, Human-{human_region}')
                        for k, v in homo_region_mouse_human_dict.items():
                            if k == mouse_region and v == human_region:
                                identified_homo_pairs['Mouse'].append(mouse_region)
                                identified_homo_pairs['Human'].append(human_region)

                                mouse_region_clusters_dict[mouse_region].append(c_label)
                                human_region_clusters_dict[human_region].append(c_label)

                                adata_mouse_homo = mouse_adata[
                                    mouse_adata.obs['region_name'].isin([mouse_region])]
                                umap1_x, umap1_y = adata_mouse_homo.obsm['X_umap'].toarray()[:, 0], \
                                                   adata_mouse_homo.obsm['X_umap'].toarray()[:, 1]
                                umap1_z = umap1_z_value * np.ones(umap1_x.shape)

                                adata_human_homo = human_adata[
                                    human_adata.obs['region_name'].isin([human_region])]
                                umap2_x, umap2_y = adata_human_homo.obsm['X_umap'].toarray()[:, 0], \
                                                   adata_human_homo.obsm[
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
                                            color='red', linewidth=line_w+1, alpha=1)

                                #mouse_r_unique.update(mouse_region)



                axes.scatter3D(umap1_x, umap1_y, umap1_z, color=scatter_color, s=marker_s, label=c_label)
                axes.scatter3D(umap2_x, umap2_y, umap2_z, color=scatter_color, s=marker_s)

                # Compute the center point and plot them
                umap1_x_mean = np.mean(umap1_x)
                umap1_y_mean = np.mean(umap1_y)
                umap1_z_mean = umap1_z_value

                umap2_x_mean = np.mean(umap2_x)
                umap2_y_mean = np.mean(umap2_y)
                umap2_z_mean = 0

                axes.plot3D([umap1_x_mean, umap2_x_mean], [umap1_y_mean, umap2_y_mean], [umap1_z_mean, umap2_z_mean],
                            color='gray', linewidth=line_w-1, alpha=0.5)

            print('identified_homo_pairs', identified_homo_pairs)
            identified_homo_pairs_num = len(set(identified_homo_pairs['Mouse']))
            print(f'Identified {identified_homo_pairs_num} homologous pairs.')
            with open(save_path + f'identified_homo_pairs_dict_n{most_common_num}_cluster{clustering_num}.pkl',
                      'wb') as f:
                pickle.dump(identified_homo_pairs, f)

            # plot venn figure
            homo_pair_num = len(homo_region_mouse_human_dict)
            cluster_num = clustering_num
            identified_homo_pairs_set = set([str(x) for x in range(identified_homo_pairs_num)])
            homo_pair_set = identified_homo_pairs_set
            homo_pair_set.update(set([str(x) for x in range(identified_homo_pairs_num, homo_pair_num)] + []))
            cluster_set = identified_homo_pairs_set
            cluster_set.update(set([str(x) for x in range(homo_pair_num, cluster_num - identified_homo_pairs_num)]))

            #---------Rename clusters---------------------------------------------------------------------------
            rename_cluster_list = []

            # cluster_mouse_human_homologs_dict
            for i in range(len(cluster_labels_unique)):
                cluster_id_str = cluster_labels_unique[i]
                mouse_str = self.mouse_64_acronym_dict[cluster_mouse_human_homologs_dict[cluster_labels_unique[i]]['Mouse']['region name'][0]]
                human_str = self.human_88_acronym_dict[cluster_mouse_human_homologs_dict[cluster_labels_unique[i]]['Human']['region name'][0]]
                rename_cluster_list.append(cluster_id_str + '-' + mouse_str + '-' + human_str)
            rename_cluster_dict = {k:v for k,v in zip(cluster_labels_unique, rename_cluster_list)}
            adata_embedding.obs['leiden_cluster_name_max'] = [rename_cluster_dict[x] for x in adata_embedding.obs[clustering_name]]
            adata_embedding.write_h5ad(cfg.BrainAlign.embeddings_file_path + 'adata_embedding.h5ad')
            #---------------------------------------------------------------------------------------------------

            # plot homologous regions pairs in blue lines
            for mouse_region, human_region in homo_region_mouse_human_dict.items():
                adata_mouse_homo = adata_mouse_umap[adata_mouse_umap.obs['region_name'].isin([mouse_region])]
                umap1_x, umap1_y = adata_mouse_homo.obsm['X_umap'].toarray()[:, 0], adata_mouse_homo.obsm[
                                                                                        'X_umap'].toarray()[:, 1]
                umap1_z = umap1_z_value * np.ones(umap1_x.shape)

                adata_human_homo = adata_human_umap[adata_human_umap.obs['region_name'].isin([human_region])]
                umap2_x, umap2_y = adata_human_homo.obsm['X_umap'].toarray()[:, 0], adata_human_homo.obsm[
                                                                                        'X_umap'].toarray()[:, 1]
                umap2_z = np.zeros(umap2_x.shape)

                # Compute the center point and plot them
                umap1_x_mean = np.mean(umap1_x)
                umap1_y_mean = np.mean(umap1_y)
                umap1_z_mean = umap1_z_value

                umap2_x_mean = np.mean(umap2_x)
                umap2_y_mean = np.mean(umap2_y)
                umap2_z_mean = 0

                axes.plot3D([umap1_x_mean, umap2_x_mean], [umap1_y_mean, umap2_y_mean], [umap1_z_mean, umap2_z_mean],
                            color='black', linewidth=line_w, alpha=1)

            # axes.set_zlim(0, umap1_z_value*1.5)
            # Hide grid lines
            axes.grid(False)
            axes.set_xlabel('UMAP1')
            axes.set_ylabel('UMAP2')
            axes.set_zlabel('Species')
            fig.subplots_adjust(left=0, right=0.98, bottom=0, top=1)

            ax = plt.gca()
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

            for line in ax.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.yaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.zaxis.get_ticklines():
                line.set_visible(False)
            #plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), ncol=5, frameon=False)
            # plt.subplots_adjust(right=0.7)
            # axes.view_init(45, 215)
            plt.savefig(save_path + f'Umap3d_human_mouse_n{most_common_num}_clusters{clustering_num}.' + fig_format)

            # plot venn diagram
            sns.set(style='white')
            TINY_SIZE = 20  # 39
            SMALL_SIZE = 20  # 42
            MEDIUM_SIZE = 20  # 46
            BIGGER_SIZE = 20  # 46

            plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
            plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

            rcParams['font.family'] = 'sans-serif'
            rcParams['font.sans-serif'] = ['Arial']

            fig, ax = plt.subplots(figsize=(5, 4), dpi=self.fig_dpi)

            rcParams["figure.subplot.left"] = 0.2
            rcParams["figure.subplot.bottom"] = 0.2
            rcParams["figure.subplot.right"] = 0.95
            #rcParams["figure.subplot.bottom"] = 0.2

            v = venn2(subsets=(homo_pair_num-identified_homo_pairs_num, cluster_num-identified_homo_pairs_num, identified_homo_pairs_num),
                      alpha=0.8, set_labels=(f'{homo_pair_num} homologous\nregion pairs', f'{cluster_num} clusters'), ax=ax,
                      subset_label_fontsize=MEDIUM_SIZE, label_fontsize=MEDIUM_SIZE)
            #v = venn2(subsets=(homo_pair_set, cluster_set), set_labels=('homologous regions', 'clusters'), ax=ax)
            #v.get_label_by_id('010').set_text('identified homologous pairs')
            #plt.rcParams["figure.subplot.left"] = 0.4
            fig.subplots_adjust(left=0.25, right=0.95, bottom=0.1, top=0.95)
            plt.annotate('Identified homologous pairs', xy=v.get_label_by_id('110').get_position()+ np.array([0, 0.1]), #+ np.array([0.1, 0.2])
                         xytext=(30, 80),#xytext=(120, 200),
                         ha='center', textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', color='black',lw=2), fontsize=MEDIUM_SIZE)
            plt.savefig(save_path + 'identified_homo_pairs_venn.' + fig_format)
            rcParams["figure.subplot.left"] = 0.05

            # plot venn diagram
            sns.set(style='white')
            TINY_SIZE = 12  # 39
            SMALL_SIZE = 12  # 42
            MEDIUM_SIZE = 14  # 46
            BIGGER_SIZE = 14  # 46

            plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
            plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

            rcParams['font.family'] = 'sans-serif'
            rcParams['font.sans-serif'] = ['Arial']
            # plot distribution figures
            mouse_sample_num = adata_mouse_embedding.n_obs
            human_sample_num = adata_human_embedding.n_obs
            mouse_percent_list = mouse_ratio_list
            human_percent_list = human_ratio_list
            #mouse_ratio_list = [x/mouse_sample_num for x in mouse_ratio_list]
            #human_ratio_list = [x/human_sample_num for x in human_ratio_list]
            #cluster_labels_unique
            distri_dict_mouse = {'Cluster':[],'Species':[]} # 'Percentage':[],
            distri_dict_human = {'Cluster':[],'Species':[]}
            #distri_dict['Cluster'] = [int(x) for x in list(range(0, clustering_num))] + [int(x) for x in list(range(0, clustering_num))]
            distri_dict_mouse['Cluster'] =  []
            for i in range(clustering_num):
                temp_list = [int(i)]* int(mouse_percent_list[i])
                distri_dict_mouse['Cluster'] = distri_dict_mouse['Cluster'] + temp_list
            for i in range(clustering_num):
                temp_list = [int(i)]* int(human_percent_list[i])
                distri_dict_human['Cluster'] = distri_dict_human['Cluster'] + temp_list
            #distri_dict['Percentage'] = mouse_ratio_list + human_ratio_list
            distri_dict_mouse['Species'] = ['Mouse'] * mouse_sample_num
            distri_dict_human['Species'] = ['Human'] * human_sample_num
            distri_df_mouse = pd.DataFrame.from_dict(distri_dict_mouse)
            distri_df_human = pd.DataFrame.from_dict(distri_dict_human)
            palette = {'Mouse': self.mouse_color, 'Human': self.human_color}
            rcParams["figure.subplot.left"] = 0.25
            rcParams["figure.subplot.bottom"] = 0.2
            fig = plt.figure(figsize=(3.5, 2.5), dpi=self.fig_dpi)
            #sns.kdeplot(data=distri_df_mouse, x="Cluster", hue="Species", multiple="stack", common_norm=False, palette=palette, alpha=0.5)
            sns.kdeplot(data=distri_df_mouse, x="Cluster", common_norm=False, color=self.mouse_color, alpha=0.4, multiple="stack", legend=True)
            sns.kdeplot(data=distri_df_human, x="Cluster", common_norm=False, color=self.human_color, alpha=0.4, multiple="stack", legend=True)
            #handles, labels = ax.get_legend_handles_labels()
            #ax.legend(handles=handles[1:], labels=labels[1:], frameon = False)
            fig.legend(labels = ['Mouse', 'Human'], loc='upper right', frameon = False, bbox_to_anchor=(0.9, 0.9)) #loc='upper right',
            plt.ylabel('Percentage density')
            plt.savefig(save_path + 'cluster_distribution.' + fig_format)
            rcParams["figure.subplot.left"] = 0.05

           #fig = plt.figure(figsize=(2.5, 2.5), dpi=self.fig_dpi)


            sys.stdout = original_stdout  # Reset the standard output to its original value


            sns.set(style='white')
            TINY_SIZE = 16  # 39
            SMALL_SIZE = 16  # 42
            MEDIUM_SIZE = 18  # 46
            BIGGER_SIZE = 18  # 46

            plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
            plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
            plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

            rcParams['font.family'] = 'sans-serif'
            rcParams['font.sans-serif'] = ['Arial']

            rcParams["figure.subplot.right"] = 0.9
            rcParams["figure.subplot.left"] = 0.25
            rcParams["figure.subplot.bottom"] = 0.3
            rcParams["figure.subplot.top"] = 0.9

            ax = plt.figure(figsize=(4, 4), dpi=self.fig_dpi)
            # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
            from scipy import stats
            #print(distri_df_mouse)
            #print(distri_df_mouse.values)
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(human_ratio_list),
                                                                           np.array(mouse_ratio_list))
            g = sns.jointplot(x=human_ratio_list, y=mouse_ratio_list, kind="reg", height=4, ax=ax) #marginal_kws={'color': self.mouse_color},
            plt.text(45, 800, f'R = -0.139, P = 0.232')
            print(f'R = {r_value}, P = {p_value}')
            plt.setp(g.ax_marg_x.patches, color=self.human_color)
            plt.setp(g.ax_marg_y.patches, color=self.mouse_color)
            plt.xlabel('Human spot number')
            plt.ylabel('Mouse spot number')
            plt.subplots_adjust(bottom=0.15, left=0.2, right=0.95, top=0.95)
            #plt.title('Brain region sample number')
            plt.savefig(save_path + 'jointplot_cluster_distribution.' + fig_format, format=fig_format, dpi=self.fig_dpi)

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.9

        # plot heatmap of homologous regions proportions in each clusters
        cluster_order = []
        for region, cluster in mouse_region_clusters_dict.items():
            c_list = mouse_region_clusters_dict[region]
            for c_ in c_list:
                if c_ not in set(cluster_order):
                    cluster_order.append(c_)

        adata_mouse_umap = adata_embedding[adata_embedding.obs['dataset'].isin(['Mouse'])]
        adata_human_umap = adata_embedding[adata_embedding.obs['dataset'].isin(['Human'])]

        mouse_homo_region_cluster_array = np.zeros((len(homo_region_list_mouse), len(cluster_order)))
        for j in range(len(cluster_order)):
            c_ = cluster_order[j]
            mouse_adata = adata_mouse_umap[adata_mouse_umap.obs[clustering_name].isin([c_])]
            mouse_num = mouse_adata.n_obs
            print('mouse num:', mouse_num)
            c_r_count = Counter(mouse_adata.obs['region_name'])
            for i in range(len(homo_region_list_mouse)):
                r_ = homo_region_list_mouse[i]
                if r_ in set(c_r_count.keys()):
                    mouse_homo_region_cluster_array[i, j] = c_r_count[r_] / mouse_num

        human_homo_region_cluster_array = np.zeros((len(homo_region_list_human), len(cluster_order)))
        for j in range(len(cluster_order)):
            c_ = cluster_order[j]
            human_adata = adata_human_umap[adata_human_umap.obs[clustering_name].isin([c_])]
            human_num = human_adata.n_obs
            print('human num:', human_num)
            c_r_count = Counter(human_adata.obs['region_name'])
            for i in range(len(homo_region_list_human)):
                r_ = homo_region_list_human[i]
                if r_ in set(c_r_count.keys()):
                    human_homo_region_cluster_array[i, j] = c_r_count[r_] / human_num

        mouse_homo_region_cluster_df = pd.DataFrame(
            mouse_homo_region_cluster_array,
            index=homo_region_list_mouse,
            columns=cluster_order)

        np.savez(save_path + 'mouse_homo_region_cluster_array.npz',
                     df_index=homo_region_list_mouse,
                     df_columns=cluster_order,
                     mouse_homo_region_cluster_array=mouse_homo_region_cluster_array)

        human_homo_region_cluster_df = pd.DataFrame(
            human_homo_region_cluster_array,
            index=homo_region_list_human,
            columns=cluster_order)

        np.savez(save_path + 'human_homo_region_cluster_array.npz',
                 df_index=homo_region_list_human,
                 df_columns=cluster_order,
                 human_homo_region_cluster_array=human_homo_region_cluster_array)

        sns.set(style='white')
        TINY_SIZE = 16  # 39
        SMALL_SIZE = 16  # 42
        MEDIUM_SIZE = 20  # 46
        BIGGER_SIZE = 20  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']


        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.5


        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.fig_dpi)
        c = Colormap()
        color_map = c.cmap_bicolor('white', self.mouse_color)

        hm = sns.heatmap(mouse_homo_region_cluster_df, square=False, cbar_kws={'location': 'right'},
                         cmap=color_map,  # 'Spectral_r',  # "YlGnBu",
                         ax=ax,
                         xticklabels=True, yticklabels=True, linewidths=0.5, linecolor='lightgrey')
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.legend([],[], frameon=False)
        # plt.xticks(rotation=45)
        #plt.ylabel('Regions')
        #plt.ylabel('Regions')

        plt.savefig(save_path + 'region_clusters_mouse.' + self.fig_format, format=self.fig_format)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1

        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.5
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.fig_dpi)
        c = Colormap()
        color_map = c.cmap_bicolor('white', self.human_color)

        hm = sns.heatmap(human_homo_region_cluster_df, square=False, cbar_kws={'location': 'right'},
                         cmap=color_map,  # 'Spectral_r',  # "YlGnBu",
                         ax=ax,
                         xticklabels=True, yticklabels=True, linewidths=0.5, linecolor='lightgrey')
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.legend([],[], frameon=False)
        # plt.xticks(rotation=45)
        # plt.ylabel('Regions')
        # plt.ylabel('Regions')
        plt.savefig(save_path + 'region_clusters_human.' + self.fig_format, format=self.fig_format)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1



    def experiment_8_1_cross_aligment_cluster_merge_lines(self):

        sns.set(style='white')
        TINY_SIZE = 20  # 39
        SMALL_SIZE = 20  # 42
        MEDIUM_SIZE = 24  # 46
        BIGGER_SIZE = 24  # 46

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

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/8_experiment_cross_evaluation_aligment_cluster/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
        adata_mouse_embedding.obs['dataset'] = 'Mouse'
        adata_human_embedding.obs['dataset'] = 'Human'

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
        sc.tl.umap(adata_embedding, n_components=2)

        # PCA and kmeans clustering of the whole dataset
        sc.tl.pca(adata_embedding, svd_solver='arpack', n_comps=30)

        # X_pca = adata_embedding.obsm['X_pca']
        # X_umap = adata_embedding.obsm['X_umap']
        # X = adata_embedding.X

        # Do clustering
        clustering_num = cfg.ANALYSIS.sample_cluster_num
        # clustering_name = f'kmeans{clustering_num}'
        clustering_name = f'leiden_cluster'
        # kmeans = KMeans(n_clusters=clustering_num, random_state=29).fit(X_pca)
        # kmeans = KMeans(n_clusters=clustering_num, random_state=29).fit(X)

        # adata_embedding.obs[clustering_name] = kmeans.labels_.astype(str)
        sc.tl.leiden(adata_embedding, resolution=9, key_added=clustering_name)
        clustering_num = len(Counter(adata_embedding.obs[clustering_name]))

        # Now we know each cluster is actually a pair
        region_labels = adata_embedding.obs['region_name'].values
        sample_names = adata_embedding.obs_names.values
        cluster_labels = adata_embedding.obs[clustering_name].values
        print('cluster_labels.shape', cluster_labels.shape)
        sample_cluter_dict = {k: v for k, v in zip(sample_names, cluster_labels)}
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
        ###################################
        # Compute the similarity or distance between the clusters of the two species
        # and identify those highly correlated pairs.
        # here, we only need to list the cluster pairs

        # save sample ratio distribution across clusters to evaluate alignment across species
        mouse_ratio_list = []
        human_ratio_list = []

        for c_label in cluster_labels_unique:
            mouse_adata = adata_embedding[adata_embedding.obs[clustering_name].isin([c_label])]
            mouse_num = mouse_adata[mouse_adata.obs['dataset'].isin(['Mouse'])].n_obs
            mouse_ratio_list.append(mouse_num)
            human_adata = adata_embedding[adata_embedding.obs[clustering_name].isin([c_label])]
            human_num = human_adata[human_adata.obs['dataset'].isin(['Human'])].n_obs
            human_ratio_list.append(human_num)
            print(c_label, f'mouse number = {mouse_num}', f'human number = {human_num}')

        adata_mouse_umap = adata_embedding[adata_embedding.obs['dataset'].isin(['Mouse'])]
        adata_human_umap = adata_embedding[adata_embedding.obs['dataset'].isin(['Human'])]

        mouse_count_all = Counter(adata_mouse_umap.obs['region_name'])
        human_count_all = Counter(adata_human_umap.obs['region_name'])

        palette = sns.color_palette(cc.glasbey, n_colors=clustering_num)

        umap1_x, umap1_y = adata_mouse_umap.obsm['X_umap'].toarray()[:, 0], adata_mouse_umap.obsm['X_umap'].toarray()[:,
                                                                            1]
        umap1_z_value = 1 / 36 * (
                ((np.max(umap1_x) - np.min(umap1_x)) ** 2 + (np.max(umap1_y) - np.min(umap1_y)) ** 2) ** (1 / 2))

        umap2_x, umap2_y = adata_human_umap.obsm['X_umap'].toarray()[:, 0], adata_human_umap.obsm['X_umap'].toarray()[:,
                                                                            1]
        umap2_z = np.zeros(umap2_x.shape)

        most_common_num = 3
        rank_region_method = 'count'  # count, count_normalized, count&count_normalized

        # Init a dict to save the homologous regions in each cluster
        cluster_mouse_human_homologs_dict = {}

        identified_homo_pairs = {'Mouse': [], 'Human': [], 'c_labels':[]}
        fig = plt.figure(figsize=(8, 8), dpi=self.fig_dpi)
        axes = plt.axes(projection='3d')
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        rcParams["figure.subplot.left"] = 0.05

        marker_s = 5
        line_w = 2

        original_stdout = sys.stdout  # Save a reference to the original standard output

        mouse_r_unique = set()
        human_r_unique = set()

        with open(save_path + f'cluster{clustering_num}_most_common_num{most_common_num}.txt', 'w') as f:
            sys.stdout = f  # Change the standard output to the file we created.

            legend_k = 0

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
                    mouse_count = Counter(
                        {k: (v1 / v2) for k, v1, v2 in zip(dict_c.keys(), dict_c.values(), mouse_count_all.values())})
                elif rank_region_method == 'count&count_normalized':
                    mouse_count_1 = Counter(mouse_adata.obs['region_name'])
                    dict_c = Counter(mouse_adata.obs['region_name'])
                    mouse_count_2 = Counter(
                        {k: (v1 / v2) for k, v1, v2 in zip(dict_c.keys(), dict_c.values(), mouse_count_all.values())})
                    mouse_count = Counter({k: (v1 + 1) * (v2 + 1) for k, v1, v2 in
                                           zip(mouse_count_1.keys(), mouse_count_1.values(), mouse_count_2.values())})
                # print('mouse_count:', mouse_count)
                mouse_region_set = mouse_count.most_common(most_common_num)
                cluster_mouse_human_homologs_dict[c_label]['Mouse'] = {'region name': [],
                                                                       'proportion': []}
                for mouse_region, mouse_region_count in mouse_region_set:
                    # mouse_count = dict(mouse_count)
                    mouse_proportion = mouse_count[mouse_region] / mouse_adata.n_obs
                    cluster_mouse_human_homologs_dict[c_label]['Mouse']['region name'].append(mouse_region)
                    cluster_mouse_human_homologs_dict[c_label]['Mouse']['proportion'].append(mouse_proportion)
                    mouse_adata_region = mouse_adata[mouse_adata.obs['region_name'].isin([mouse_region])]

                human_adata = adata_human_umap[adata_human_umap.obs[clustering_name].isin([c_label])]
                umap2_x, umap2_y = human_adata.obsm['X_umap'].toarray()[:, 0], human_adata.obsm['X_umap'].toarray()[:,
                                                                               1]
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
                    human_count = Counter({k: (v1 + 1) * (v2 + 1) for k, v1, v2 in
                                           zip(human_count_1.keys(), human_count_1.values(), human_count_2.values())})

                human_region_set = human_count.most_common(most_common_num)
                cluster_mouse_human_homologs_dict[c_label]['Human'] = {'region name': [],
                                                                       'proportion': []}
                for human_region, human_region_count in human_region_set:
                    # human_count = dict(human_count)
                    human_proportion = human_count[human_region] / human_adata.n_obs
                    cluster_mouse_human_homologs_dict[c_label]['Human']['region name'].append(human_region)
                    cluster_mouse_human_homologs_dict[c_label]['Human']['proportion'].append(human_proportion)
                    human_adata_region = human_adata[human_adata.obs['region_name'].isin([human_region])]

                for mouse_region in cluster_mouse_human_homologs_dict[c_label]['Mouse']['region name']:
                    for human_region in cluster_mouse_human_homologs_dict[c_label]['Human']['region name']:
                        print(
                            f'Identified homologous regions for cluster {c_label}: Mouse-{mouse_region}, Human-{human_region}')
                        for k, v in homo_region_mouse_human_dict.items():
                            if k == mouse_region and v == human_region:
                                identified_homo_pairs['Mouse'].append(mouse_region)
                                identified_homo_pairs['Human'].append(human_region)

                                identified_homo_pairs['c_labels'].append(c_label)

                                adata_mouse_homo = mouse_adata[
                                    mouse_adata.obs['region_name'].isin([mouse_region])]
                                umap1_x, umap1_y = adata_mouse_homo.obsm['X_umap'].toarray()[:, 0], \
                                                   adata_mouse_homo.obsm['X_umap'].toarray()[:, 1]
                                umap1_z = umap1_z_value * np.ones(umap1_x.shape)

                                adata_human_homo = human_adata[
                                    human_adata.obs['region_name'].isin([human_region])]
                                umap2_x, umap2_y = adata_human_homo.obsm['X_umap'].toarray()[:, 0], \
                                                   adata_human_homo.obsm[
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

                                # axes.plot3D([umap1_x_mean, umap2_x_mean], [umap1_y_mean, umap2_y_mean],
                                #             [umap1_z_mean, umap2_z_mean],
                                #             color='red', linewidth=line_w + 1, alpha=1)

                                # mouse_r_unique.update(mouse_region)

                axes.scatter3D(umap1_x, umap1_y, umap1_z, color=scatter_color, s=marker_s) #, label=c_label
                axes.scatter3D(umap2_x, umap2_y, umap2_z, color=scatter_color, s=marker_s)

                # Compute the center point and plot them
                umap1_x_mean = np.mean(umap1_x)
                umap1_y_mean = np.mean(umap1_y)
                umap1_z_mean = umap1_z_value

                umap2_x_mean = np.mean(umap2_x)
                umap2_y_mean = np.mean(umap2_y)
                umap2_z_mean = 0
                if legend_k == 0:
                    axes.plot3D([umap1_x_mean, umap2_x_mean], [umap1_y_mean, umap2_y_mean], [umap1_z_mean, umap2_z_mean],
                                color='gray', linewidth=line_w, alpha=0.6, label='Clustering aligned')
                    legend_k += 1
                else:

                    axes.plot3D([umap1_x_mean, umap2_x_mean], [umap1_y_mean, umap2_y_mean],
                                [umap1_z_mean, umap2_z_mean],
                                color='gray', linewidth=line_w, alpha=0.6)

            identified_homo_pairs_dict = {k:v for k,v in zip(identified_homo_pairs['Mouse'], identified_homo_pairs['Human'])}
            unique_mouse_r = list(set(identified_homo_pairs['Mouse']))
            unique_mouse_r_cluster_dict = {k:[] for k in unique_mouse_r}
            for i in range(len(identified_homo_pairs['c_labels'])):
                for m_r in unique_mouse_r:
                    if m_r == identified_homo_pairs['Mouse'][i]:
                        unique_mouse_r_cluster_dict[m_r].append(identified_homo_pairs['c_labels'][i])

            legend_k = 0

            for mouse_r, cluster_list in unique_mouse_r_cluster_dict.items():
                human_r = identified_homo_pairs_dict[mouse_r]

                mouse_adata = adata_mouse_umap[adata_mouse_umap.obs[clustering_name].isin(cluster_list)]
                adata_mouse_homo = mouse_adata[mouse_adata.obs['region_name'].isin([mouse_r])]

                human_adata = adata_human_umap[adata_human_umap.obs[clustering_name].isin(cluster_list)]
                adata_human_homo = human_adata[human_adata.obs['region_name'].isin([human_r])]

                umap1_x, umap1_y = adata_mouse_homo.obsm['X_umap'].toarray()[:, 0], adata_mouse_homo.obsm['X_umap'].toarray()[:, 1]
                umap1_z = umap1_z_value * np.ones(umap1_x.shape)

                umap2_x, umap2_y = adata_human_homo.obsm['X_umap'].toarray()[:, 0], adata_human_homo.obsm['X_umap'].toarray()[:, 1]
                umap2_z = np.zeros(umap2_x.shape)

                # Compute the center point and plot them
                umap1_x_mean = np.mean(umap1_x)
                umap1_y_mean = np.mean(umap1_y)
                umap1_z_mean = umap1_z_value

                umap2_x_mean = np.mean(umap2_x)
                umap2_y_mean = np.mean(umap2_y)
                umap2_z_mean = 0

                if legend_k == 0:
                    axes.plot3D([umap1_x_mean, umap2_x_mean], [umap1_y_mean, umap2_y_mean],
                                [umap1_z_mean, umap2_z_mean],
                                color='red', linewidth=line_w + 1, alpha=1, label='Identified homologous')
                    legend_k += 1
                else:

                    axes.plot3D([umap1_x_mean, umap2_x_mean], [umap1_y_mean, umap2_y_mean],
                                [umap1_z_mean, umap2_z_mean],
                                color='red', linewidth=line_w + 1, alpha=1)


            #for m_r, h_r, c_l in zip(identified_homo_pairs['Mouse'], identified_homo_pairs['Human'], )

            print('identified_homo_pairs', identified_homo_pairs)
            identified_homo_pairs_num = len(set(identified_homo_pairs['Mouse']))
            print(f'Identified {identified_homo_pairs_num} homologous pairs.')
            with open(save_path + f'identified_homo_pairs_dict_n{most_common_num}_cluster{clustering_num}.pkl',
                      'wb') as f:
                pickle.dump(identified_homo_pairs, f)

            # plot venn figure
            homo_pair_num = len(homo_region_mouse_human_dict)
            cluster_num = clustering_num
            identified_homo_pairs_set = set([str(x) for x in range(identified_homo_pairs_num)])
            homo_pair_set = identified_homo_pairs_set
            homo_pair_set.update(set([str(x) for x in range(identified_homo_pairs_num, homo_pair_num)] + []))
            cluster_set = identified_homo_pairs_set
            cluster_set.update(set([str(x) for x in range(homo_pair_num, cluster_num - identified_homo_pairs_num)]))

            # ---------Rename clusters---------------------------------------------------------------------------
            rename_cluster_list = []

            # cluster_mouse_human_homologs_dict
            for i in range(len(cluster_labels_unique)):
                cluster_id_str = cluster_labels_unique[i]
                mouse_str = self.mouse_64_acronym_dict[
                    cluster_mouse_human_homologs_dict[cluster_labels_unique[i]]['Mouse']['region name'][0]]
                human_str = self.human_88_acronym_dict[
                    cluster_mouse_human_homologs_dict[cluster_labels_unique[i]]['Human']['region name'][0]]
                rename_cluster_list.append(cluster_id_str + '-' + mouse_str + '-' + human_str)
            rename_cluster_dict = {k: v for k, v in zip(cluster_labels_unique, rename_cluster_list)}
            adata_embedding.obs['leiden_cluster_name_max'] = [rename_cluster_dict[x] for x in
                                                              adata_embedding.obs[clustering_name]]
            adata_embedding.write_h5ad(cfg.BrainAlign.embeddings_file_path + 'adata_embedding.h5ad')
            # ---------------------------------------------------------------------------------------------------

            legend_k = 0
            # plot homologous regions pairs in blue lines
            for mouse_region, human_region in homo_region_mouse_human_dict.items():
                adata_mouse_homo = adata_mouse_umap[adata_mouse_umap.obs['region_name'].isin([mouse_region])]
                umap1_x, umap1_y = adata_mouse_homo.obsm['X_umap'].toarray()[:, 0], adata_mouse_homo.obsm[
                                                                                        'X_umap'].toarray()[:, 1]
                umap1_z = umap1_z_value * np.ones(umap1_x.shape)

                adata_human_homo = adata_human_umap[adata_human_umap.obs['region_name'].isin([human_region])]
                umap2_x, umap2_y = adata_human_homo.obsm['X_umap'].toarray()[:, 0], adata_human_homo.obsm[
                                                                                        'X_umap'].toarray()[:, 1]
                umap2_z = np.zeros(umap2_x.shape)

                # Compute the center point and plot them
                umap1_x_mean = np.mean(umap1_x)
                umap1_y_mean = np.mean(umap1_y)
                umap1_z_mean = umap1_z_value

                umap2_x_mean = np.mean(umap2_x)
                umap2_y_mean = np.mean(umap2_y)
                umap2_z_mean = 0

                if not mouse_region in identified_homo_pairs['Mouse'] and human_region not in identified_homo_pairs['Human']:
                    if legend_k == 0:
                        axes.plot3D([umap1_x_mean, umap2_x_mean], [umap1_y_mean, umap2_y_mean], [umap1_z_mean, umap2_z_mean],
                                color='black', linewidth=line_w+1, alpha=1, label='Homologous')
                        legend_k += 1
                    else:
                        axes.plot3D([umap1_x_mean, umap2_x_mean], [umap1_y_mean, umap2_y_mean],
                                    [umap1_z_mean, umap2_z_mean],
                                    color='black', linewidth=line_w+1, alpha=1)

            # axes.set_zlim(0, umap1_z_value*1.5)
            # Hide grid lines
            axes.grid(False)
            axes.set_xlabel('UMAP1')
            axes.set_ylabel('UMAP2')
            axes.set_zlabel('Species')
            fig.subplots_adjust(left=0, right=0.95, bottom=0, top=1)

            ax = plt.gca()
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

            for line in ax.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.yaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.zaxis.get_ticklines():
                line.set_visible(False)
            # plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), ncol=5, frameon=False)
            # plt.subplots_adjust(right=0.7)
            # axes.view_init(45, 215)
            plt.legend(loc='lower center', bbox_to_anchor=(0.53, -0.03), frameon=False, ncol=3, columnspacing=0.5, handlelength=0.5)
            plt.savefig(save_path + f'merged_Umap3d_human_mouse_n{most_common_num}_clusters{clustering_num}.' + fig_format)

            rcParams["figure.subplot.bottom"] = 0.1
            rcParams["figure.subplot.left"] = 0.1
            sys.stdout = original_stdout  # Reset the standard output to its original value

    def experiment_9_name_clusters(self):

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
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        # 1. Identify those differentially expressed genes for regions and clusters, and save them.
        # 2. How these genes are correlated to each other - Cluster those genes or get gene modules
        # 3. clustering of samples, do GO analysis to determine enrichment of genes in each region or clusters;
        # 4. dotplot of clusters and homologous regions, go clusters.
        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/9_experiment_rename_clusters/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
        self.adata_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_embedding.h5ad')
        print(self.adata_embedding)

        sc.pp.neighbors(self.adata_embedding, n_neighbors=self.cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(self.adata_embedding, n_components=2)

        # PCA and kmeans clustering of the whole dataset
        sc.tl.pca(self.adata_embedding, svd_solver='arpack', n_comps=30)
        # X_pca = adata_embedding.obsm['X_pca']
        # X_umap = adata_embedding.obsm['X_umap']
        # Do clustering
        clustering_name = f'leiden_cluster'
        # kmeans = KMeans(n_clusters=clustering_num, random_state=29).fit(X_pca)
        # kmeans = KMeans(n_clusters=clustering_num, random_state=29).fit(X)

        # adata_embedding.obs[clustering_name] = kmeans.labels_.astype(str)

        # sc.tl.leiden(adata_embedding, resolution=9, key_added=clustering_name)
        clustering_num = len(Counter(self.adata_embedding.obs[clustering_name]))

        cluster_labels_unique = [str(x) for x in range(clustering_num)]
        # kmeans = KMeans(n_clusters=clustering_num, random_state=29).fit(X_umap)

        # adata_embedding.obs[clustering_name] = kmeans.labels_.astype(str)

        # name those clusters via parent regions
        # name some cluster the region name if the proportion of the samples of the region in this cluster exceeds 50%
        # (50% for parent region name, 30% for region name, the ratio is calculated in each species),
        # k-mouse parent region name-human parent region name
        # or name the cluster via k-mixed(most parent region name of mouse)-mixed(most parent region name of human)
        cluster_names_dict = {k: v for k, v in zip(self.adata_embedding.obs_names, self.adata_embedding.obs[clustering_name])}
        cluster_names_acronym_dict = {k: v for k, v in
                                      zip(self.adata_embedding.obs_names, self.adata_embedding.obs[clustering_name])}

        parent_cluster_names_dict = {k: v for k, v in
                                     zip(self.adata_embedding.obs_names, self.adata_embedding.obs[clustering_name])}
        parent_cluster_names_acronym_dict = {k: v for k, v in
                                             zip(self.adata_embedding.obs_names, self.adata_embedding.obs[clustering_name])}

        parent_region_proportion = 0.5
        region_proportion = 0.3

        for c_label in cluster_labels_unique:
            c_label_adata = self.adata_embedding[self.adata_embedding.obs[clustering_name].isin([c_label])]
            # create parent region names
            mouse_adata = c_label_adata[c_label_adata.obs['dataset'].isin(['Mouse'])]
            mouse_parent_region_Counter = Counter(mouse_adata.obs['parent_region_name'])
            if len(mouse_parent_region_Counter) == 0:
                mouse_parent_region_mc = 'None'
                mouse_parent_region_rate = 0
            else:
                mouse_parent_region_mc = mouse_parent_region_Counter.most_common(1)[0][0]
                mouse_parent_region_rate = mouse_parent_region_Counter[mouse_parent_region_mc] / mouse_adata.n_obs
            # mouse_parent_region_mc = mouse_parent_region_Counter.most_common(1)[0][0]
            # mouse_parent_region_rate = mouse_parent_region_Counter[mouse_parent_region_mc] / mouse_adata.n_obs

            if mouse_parent_region_rate >= parent_region_proportion:
                mouse_parent_region_string = mouse_parent_region_mc
                mouse_parent_region_string_acronym = self.mouse_15_acronym_dict[mouse_parent_region_mc]
            elif mouse_parent_region_rate < parent_region_proportion and mouse_parent_region_rate > 0:
                mouse_parent_region_string = f'mixed({mouse_parent_region_mc})'
                mouse_parent_region_string_acronym = f'mixed({self.mouse_15_acronym_dict[mouse_parent_region_mc]})'
            elif mouse_parent_region_rate == 0:
                mouse_parent_region_string = mouse_parent_region_mc
                mouse_parent_region_string_acronym = mouse_parent_region_mc

            human_adata = c_label_adata[c_label_adata.obs['dataset'].isin(['Human'])]
            human_parent_region_Counter = Counter(human_adata.obs['parent_region_name'])
            if len(human_parent_region_Counter) == 0:
                human_parent_region_mc = 'None'
                human_parent_region_rate = 0
            else:
                human_parent_region_mc = human_parent_region_Counter.most_common(1)[0][0]
                human_parent_region_rate = human_parent_region_Counter[human_parent_region_mc] / human_adata.n_obs

            if human_parent_region_rate >= parent_region_proportion:
                human_parent_region_string = human_parent_region_mc
                human_parent_region_string_acronym = self.human_16_acronym_dict[human_parent_region_mc]
            elif human_parent_region_rate < parent_region_proportion and human_parent_region_rate > 0:
                human_parent_region_string = f'mixed({human_parent_region_mc})'
                human_parent_region_string_acronym = f'mixed({self.human_16_acronym_dict[human_parent_region_mc]})'
            elif human_parent_region_rate == 0:
                human_parent_region_string = human_parent_region_mc
                human_parent_region_string_acronym = human_parent_region_mc

            parent_cluster_name = c_label + '-' + mouse_parent_region_string + '-' + human_parent_region_string
            parent_cluster_name_acronym = c_label + '-' + mouse_parent_region_string_acronym + '-' + human_parent_region_string_acronym

            for sample_index in c_label_adata.obs_names:
                parent_cluster_names_dict[sample_index] = parent_cluster_name
                parent_cluster_names_acronym_dict[sample_index] = parent_cluster_name_acronym

            print(c_label, f'parent cluster name = {parent_cluster_name}',
                  f'parent cluster name acronym = {parent_cluster_name_acronym}')

            # create region names
            mouse_region_Counter = Counter(mouse_adata.obs['region_name'])
            if len(mouse_region_Counter) == 0:
                mouse_region_mc = 'None'
                mouse_region_rate = 0
            else:
                mouse_region_mc = mouse_region_Counter.most_common(1)[0][0]
                mouse_region_rate = mouse_region_Counter[mouse_region_mc] / mouse_adata.n_obs

            if mouse_region_rate >= region_proportion:
                mouse_region_string = mouse_region_mc
                mouse_region_string_acronym = self.mouse_64_acronym_dict[mouse_region_mc]
            elif mouse_region_rate < region_proportion and mouse_region_rate > 0:
                mouse_region_string = f'mixed({mouse_region_mc})'
                mouse_region_string_acronym = f'mixed({self.mouse_64_acronym_dict[mouse_region_mc]})'
            elif mouse_region_rate == 0:
                mouse_region_string = mouse_region_mc
                mouse_region_string_acronym = mouse_region_mc

            human_region_Counter = Counter(human_adata.obs['region_name'])
            if len(human_region_Counter) == 0:
                human_region_mc = 'None'
                human_region_rate = 0
            else:
                human_region_mc = human_region_Counter.most_common(1)[0][0]
                human_region_rate = human_region_Counter[human_region_mc] / human_adata.n_obs

            if human_region_rate >= region_proportion:
                human_region_string = human_region_mc
                human_region_string_acronym = self.human_88_acronym_dict[human_region_mc]
            elif human_region_rate < region_proportion and human_region_rate > 0:
                human_region_string = f'mixed({human_region_mc})'
                human_region_string_acronym = f'mixed({self.human_88_acronym_dict[human_region_mc]})'
            elif human_region_rate == 0:
                human_region_string = human_region_mc
                human_region_string_acronym = human_region_mc

            cluster_name = c_label + '-' + mouse_region_string + '-' + human_region_string
            cluster_name_acronym = c_label + '-' + mouse_region_string_acronym + '-' + human_region_string_acronym

            for sample_index in c_label_adata.obs_names:
                cluster_names_dict[sample_index] = cluster_name
                cluster_names_acronym_dict[sample_index] = cluster_name_acronym

            print(c_label, f'cluster name = {cluster_name}', f'cluster name acronym = {cluster_name_acronym}')

        self.adata_embedding.obs['parent_cluster_name'] = cluster_names_dict.values()
        self.adata_embedding.obs['parent_cluster_name_acronym'] = cluster_names_acronym_dict.values()
        self.adata_embedding.obs['cluster_name'] = cluster_names_dict.values()
        self.adata_embedding.obs['cluster_name_acronym'] = cluster_names_acronym_dict.values()

        self.adata_embedding.write_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_embedding.h5ad')

        adata_mouse_embedding = self.adata_embedding[self.adata_embedding.obs['dataset'] == 'Mouse']
        adata_human_embedding = self.adata_embedding[self.adata_embedding.obs['dataset'] == 'Human']

        adata_mouse_expression = sc.read_h5ad(self.cfg.CAME.path_rawdata1)
        for obs1, obs2 in zip(adata_mouse_embedding.obs_names, adata_mouse_expression.obs_names):
            if obs1 != obs2:
                print('Sample name not aligned!')

        adata_human_expression = sc.read_h5ad(self.cfg.CAME.path_rawdata2)
        for obs1, obs2 in zip(adata_human_embedding.obs_names, adata_human_expression.obs_names):
            if obs1 != obs2:
                print('Sample name not aligned!')

        adata_mouse_embedding.obs['x_grid'] = adata_mouse_expression.obs['x_grid']
        adata_mouse_embedding.obs['y_grid'] = adata_mouse_expression.obs['y_grid']
        adata_mouse_embedding.obs['z_grid'] = adata_mouse_expression.obs['z_grid']

        adata_mouse_expression.obs = adata_mouse_embedding.obs

        adata_human_embedding.obs['mri_voxel_x'] = adata_human_expression.obs['mri_voxel_x']
        adata_human_embedding.obs['mri_voxel_y'] = adata_human_expression.obs['mri_voxel_y']
        adata_human_embedding.obs['mri_voxel_z'] = adata_human_expression.obs['mri_voxel_z']

        adata_human_expression.obs = adata_human_embedding.obs


        adata_mouse_embedding.write_h5ad(self.cfg.BrainAlign.embeddings_file_path + '/s_embeddings.h5ad')
        adata_human_embedding.write_h5ad(self.cfg.BrainAlign.embeddings_file_path + '/v_embeddings.h5ad')

        adata_mouse_expression.write_h5ad(self.cfg.CAME.path_rawdata1)
        adata_human_expression.write_h5ad(self.cfg.CAME.path_rawdata2)


        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/8_experiment_cross_evaluation_aligment_cluster/'

        if os.path.exists(save_path + 'mouse_homo_region_cluster_array.npz'):
            npzfile = np.load(save_path + 'mouse_homo_region_cluster_array.npz')
            mouse_homo_region_cluster_array = npzfile['mouse_homo_region_cluster_array']
            mouse_df_index = npzfile['df_index']
            mouse_df_columns = npzfile['df_columns']

        if os.path.exists(save_path + 'human_homo_region_cluster_array.npz'):
            npzfile = np.load(save_path + 'human_homo_region_cluster_array.npz')
            human_homo_region_cluster_array = npzfile['human_homo_region_cluster_array']
            human_df_index = npzfile['df_index']
            human_df_columns = npzfile['df_columns']

        cluster_labels_unique = [str(x) for x in list(range(0, clustering_num))]

        cluster_name_unique = sorted(Counter(self.adata_embedding.obs['cluster_name_acronym']).keys(),
                                     key=lambda t: int(t.split('-')[0]))

        cluster_label2name_dict = {k:v for k,v in zip(cluster_labels_unique, cluster_name_unique)}

        mouse_df_columns_name = [cluster_label2name_dict[x] for x in mouse_df_columns]
        human_df_columns_name = [cluster_label2name_dict[x] for x in human_df_columns]

        mouse_homo_region_cluster_df = pd.DataFrame(
            mouse_homo_region_cluster_array,
            index=mouse_df_index,
            columns=mouse_df_columns_name)


        human_homo_region_cluster_df = pd.DataFrame(
            human_homo_region_cluster_array,
            index=human_df_index,
            columns=human_df_columns_name)

        sns.set(style='white')
        TINY_SIZE = 16  # 39
        SMALL_SIZE = 16  # 42
        MEDIUM_SIZE = 20  # 46
        BIGGER_SIZE = 20  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.4

        fig, ax = plt.subplots(figsize=(15, 8), dpi=self.fig_dpi)
        c = Colormap()
        #color_map = c.cmap_bicolor('white', self.mouse_color)
        color_map = c.cmap_linear('white', 'white', self.mouse_color)

        hm = sns.heatmap(mouse_homo_region_cluster_df, square=False, cbar_kws={'location': 'right'},
                         cmap=color_map,  # 'Spectral_r',  # "YlGnBu",
                         ax=ax,
                         xticklabels=True, yticklabels=True, linewidths=0.5, linecolor='lightgrey')
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.legend([],[], frameon=False)
        # plt.xticks(rotation=45)
        # plt.ylabel('Regions')
        # plt.ylabel('Regions')
        plt.subplots_adjust(bottom=0.25, left=0.2, top=0.98, right=0.98)

        plt.savefig(save_path + 'region_clusters_mouse.' + self.fig_format, format=self.fig_format)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1

        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.4
        fig, ax = plt.subplots(figsize=(15, 8), dpi=self.fig_dpi)
        c = Colormap()
        #color_map = c.cmap_bicolor('white', self.human_color)
        #color_map = c.cmap_bicolor('white', self.human_color)
        color_map = c.cmap_linear('white', 'white', self.human_color)

        hm = sns.heatmap(human_homo_region_cluster_df, square=False, cbar_kws={'location': 'right'},
                         cmap=color_map,  # 'Spectral_r',  # "YlGnBu",
                         ax=ax,
                         xticklabels=True, yticklabels=True, linewidths=0.5, linecolor='lightgrey')
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.legend([],[], frameon=False)
        # plt.xticks(rotation=45)
        # plt.ylabel('Regions')
        # plt.ylabel('Regions')
        plt.subplots_adjust(bottom=0.25, left=0.2, top=0.98, right=0.98)
        plt.savefig(save_path + 'region_clusters_human.' + self.fig_format, format=self.fig_format)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1

        return None

    def experiment_9_1_region_cluster_heatmap(self):

        self.adata_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_embedding.h5ad')
        print(self.adata_embedding)
        # Do clustering
        clustering_name = f'leiden_cluster'
        # kmeans = KMeans(n_clusters=clustering_num, random_state=29).fit(X_pca)
        # kmeans = KMeans(n_clusters=clustering_num, random_state=29).fit(X)

        # adata_embedding.obs[clustering_name] = kmeans.labels_.astype(str)

        # sc.tl.leiden(adata_embedding, resolution=9, key_added=clustering_name)
        clustering_num = len(Counter(self.adata_embedding.obs[clustering_name]))

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/8_experiment_cross_evaluation_aligment_cluster/'

        if os.path.exists(save_path + 'mouse_homo_region_cluster_array.npz'):
            npzfile = np.load(save_path + 'mouse_homo_region_cluster_array.npz')
            mouse_homo_region_cluster_array = npzfile['mouse_homo_region_cluster_array']
            mouse_df_index = npzfile['df_index']
            mouse_df_columns = npzfile['df_columns']

        if os.path.exists(save_path + 'human_homo_region_cluster_array.npz'):
            npzfile = np.load(save_path + 'human_homo_region_cluster_array.npz')
            human_homo_region_cluster_array = npzfile['human_homo_region_cluster_array']
            human_df_index = npzfile['df_index']
            human_df_columns = npzfile['df_columns']


        cluster_labels_unique = [str(x) for x in list(range(0, clustering_num))]

        cluster_name_unique = sorted(Counter(self.adata_embedding.obs['cluster_name_acronym']).keys(),
                                     key=lambda t: int(t.split('-')[0]))

        cluster_label2name_dict = {k: v for k, v in zip(cluster_labels_unique, cluster_name_unique)}

        mouse_df_columns_name = [cluster_label2name_dict[x] for x in mouse_df_columns]
        human_df_columns_name = [cluster_label2name_dict[x] for x in human_df_columns]

        mouse_homo_region_cluster_df = pd.DataFrame(
            mouse_homo_region_cluster_array,
            index=mouse_df_index,
            columns=mouse_df_columns_name)

        human_homo_region_cluster_df = pd.DataFrame(
            human_homo_region_cluster_array,
            index=human_df_index,
            columns=human_df_columns_name)

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

        # rcParams["figure.subplot.left"] = 0.15
        # rcParams["figure.subplot.right"] = 0.99
        # rcParams["figure.subplot.top"] = 0.9
        # rcParams["figure.subplot.bottom"] = 0.4

        rcParams["figure.subplot.left"] = 0.08
        rcParams["figure.subplot.right"] = 0.8
        rcParams["figure.subplot.top"] = 0.98
        rcParams["figure.subplot.bottom"] = 0.35

        fig, ax = plt.subplots(figsize=(7.5, 3), dpi=self.fig_dpi)
        c = Colormap()
        #color_map = c.cmap_bicolor('white', ''self.mouse_color'')
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap

        viridis = cm.get_cmap('Blues', 256)
        newcolors = viridis(np.linspace(0, 1, 256))
        white = np.array([1, 1, 1, 1])
        newcolors[:5, :] = white
        color_map = ListedColormap(newcolors)
        #color_map = c.cmap_linear('white', 'white', 'darkblue')

        hm = sns.heatmap(mouse_homo_region_cluster_df, square=False, cbar_kws={'location': 'left', 'label': 'Proportion in clusters'},
                         cmap=color_map,#'Blues',  # 'Spectral_r',  # "YlGnBu",
                         ax=ax,
                         xticklabels=False, yticklabels='auto', linewidths=0.1, linecolor='black')
        plt.subplots_adjust(bottom=0.5, left=0.05, right=0.75)
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.legend([],[], frameon=False)
        # plt.xticks(rotation=45)
        # plt.ylabel('Regions')
        # plt.ylabel('Regions')
        hm.yaxis.tick_right()
        # Set y-tick label alignment and rotation
        hm.set_yticklabels(ax.get_yticklabels(), va='center', rotation=0)

        b, t = plt.ylim()  # discover the values for bottom and top
        b += 0.2  # Add 0.5 to the bottom
        #t -= 0.5  # Subtract 0.5 from the top
        plt.ylim(b, t)  # update the ylim(bottom, top) values

        l, r = plt.xlim()  # discover the values for bottom and top
        r += 0.2  # Add 0.5 to the bottom
        #l -= 0.5  # Subtract 0.5 from the top
        plt.xlim(l, r)  # update the ylim(bottom, top) values

        hm1 = hm
        plt.savefig(save_path + 'region_clusters_mouse.' + self.fig_format, format=self.fig_format)

        rcParams["figure.subplot.left"] = 0.08
        rcParams["figure.subplot.right"] = 0.8
        rcParams["figure.subplot.top"] = 0.98
        rcParams["figure.subplot.bottom"] = 0.35

        # rcParams["figure.subplot.left"] = 0.15
        # rcParams["figure.subplot.right"] = 0.99
        # rcParams["figure.subplot.top"] = 0.9
        # rcParams["figure.subplot.bottom"] = 0.4
        fig, ax = plt.subplots(figsize=(7.5, 3), dpi=self.fig_dpi)
        c = Colormap()
        #color_map = c.cmap_bicolor('white', self.human_color)
        #color_map = c.cmap_linear('white', 'white', 'darkred')
        viridis = cm.get_cmap('Reds', 256)
        newcolors = viridis(np.linspace(0, 1, 256))
        white = np.array([1, 1, 1, 1])
        newcolors[:10, :] = white
        color_map = ListedColormap(newcolors)

        hm = sns.heatmap(human_homo_region_cluster_df, square=False, cbar_kws={'location': 'left','label': 'Proportion in clusters'},
                         cmap=color_map,#'Reds',#color_map,  # 'Spectral_r',  # "YlGnBu",
                         ax=ax,
                         xticklabels='auto', yticklabels='auto', linewidths=0.1, linecolor='black')
        plt.subplots_adjust(bottom=0.5, left=0.05, right=0.75)
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.legend([],[], frameon=False)
        # plt.xticks(rotation=45)
        # plt.ylabel('Regions')
        #plt.ylabel('Regions')
        hm.yaxis.tick_right()

        # Set y-tick label alignment and rotation
        hm.set_yticklabels(ax.get_yticklabels(), va='center', rotation=0)
        b, t = plt.ylim()  # discover the values for bottom and top
        b += 0.2  # Add 0.5 to the bottom
        #t -= 0.5  # Subtract 0.5 from the top
        plt.ylim(b, t)  # update the ylim(bottom, top) values
        l, r = plt.xlim()  # discover the values for bottom and top
        r += 0.2  # Add 0.5 to the bottom
        #l -= 0.5  # Subtract 0.5 from the top
        plt.xlim(l, r)  # update the ylim(bottom, top) values

        hm2 = hm
        plt.savefig(save_path + 'region_clusters_human.' + self.fig_format, format=self.fig_format)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1

        f, (ax1, ax2) = plt.subplots(2, 1)
        ax1 = hm1
        ax2 = hm2
        plt.savefig(save_path + 'region_clusters_mouse_human.' + self.fig_format, format=self.fig_format)



    def experiment_10_check_alignment(self):

        """
        Examine whether the alignment clusters across species are biologically meaningful.

        1. Acquire gene modules shared by species;
        2. Compute sample correlations between those cluster pairs, and compared with homologous region pairs;
        3. The enrichment of cluster pair on gene modules is more similar than the other pairs and close to homologous region pairs;
        4. The expression similarity of one-2-one homologous genes is higher for those pairs than random pairs.

        :return:None
        """
        sns.set(style='white')
        TINY_SIZE = 22  # 39
        SMALL_SIZE = 28  # 42
        MEDIUM_SIZE = 32  # 46
        BIGGER_SIZE = 32  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        metric_name = 'correlation'

        # 1. Identify those differentially expressed genes for regions and clusters, and save them.
        # 2. How these genes are correlated to each other - Cluster those genes or get gene modules
        # 3. clustering of samples, do GO analysis to determine enrichment of genes in each region or clusters;
        # 4. dotplot of clusters and homologous regions, go clusters.
        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/10_experiment_check_alignment/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # load sample embedding
        adata_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_embedding.h5ad')

        clustering_name = f'leiden_cluster'

        #sc.tl.leiden(adata_embedding, resolution=1, key_added=clustering_name)
        clustering_num = len(Counter(adata_embedding.obs[clustering_name]))
        # Now we know each cluster is actually a pair
        region_labels = adata_embedding.obs['region_name'].values
        sample_names = adata_embedding.obs_names.values
        cluster_labels = adata_embedding.obs[clustering_name].values
        print('cluster_labels.shape', cluster_labels.shape)
        sample_cluter_dict = {k: v for k, v in zip(sample_names, cluster_labels)}
        # All cluster labels
        cluster_labels_unique = [str(x) for x in list(range(0, clustering_num))]

        cluster_name_unique = sorted(Counter(adata_embedding.obs['cluster_name_acronym']).keys(), key=lambda t:int(t.split('-')[0]))

        # load gene embedding
        adata_mouse_gene_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'm_embeddings.h5ad')
        adata_human_gene_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'h_embeddings.h5ad')
        adata_mouse_gene_embedding.obs['dataset'] = 'Mouse'
        adata_human_gene_embedding.obs['dataset'] = 'Human'

        adata_gene_embedding = ad.concat([adata_mouse_gene_embedding, adata_human_gene_embedding])

        # generate gene modules
        sc.pp.neighbors(adata_gene_embedding, n_neighbors=self.cfg.ANALYSIS.genes_umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.leiden(adata_gene_embedding, resolution=1.5, key_added='module')
        module_name = 'module'
        module_num = len(Counter(adata_gene_embedding.obs[module_name]))

        gene_names = adata_gene_embedding.obs_names.values
        module_labels = adata_gene_embedding.obs[module_name].values
        print('module_labels.shape', module_labels.shape)
        gene_module_dict = {k: v for k, v in zip(gene_names, module_labels)}

        # mouse gene_name to module
        adata_mouse_gene_embedding = adata_gene_embedding[adata_gene_embedding.obs['dataset'].isin(['Mouse'])]
        mouse_gene_names = adata_mouse_gene_embedding.obs_names.values
        mouse_module_labels = adata_mouse_gene_embedding.obs[module_name].values
        mouse_gene_module_dict = {k: v for k, v in zip(mouse_gene_names, mouse_module_labels)}
        # human gene_name to module
        adata_human_gene_embedding = adata_gene_embedding[adata_gene_embedding.obs['dataset'].isin(['Human'])]
        human_gene_names = adata_human_gene_embedding.obs_names.values
        human_module_labels = adata_human_gene_embedding.obs[module_name].values
        human_gene_module_dict = {k: v for k, v in zip(human_gene_names, human_module_labels)}

        # All module labels
        module_labels_unique = [str(x) for x in list(range(0, module_num))]

        # load gene expression data
        adata_mouse_expression = sc.read_h5ad(self.cfg.CAME.path_rawdata1)
        adata_mouse_expression = adata_mouse_expression[:, adata_mouse_gene_embedding.obs_names.tolist()]
        adata_mouse_expression.var['module'] = [mouse_gene_module_dict[g] for g in adata_mouse_expression.var_names]

        adata_human_expression = sc.read_h5ad(self.cfg.CAME.path_rawdata2)
        adata_human_expression = adata_human_expression[:, adata_human_gene_embedding.obs_names.tolist()]
        adata_human_expression.var['module'] = [human_gene_module_dict[g] for g in adata_human_expression.var_names]



        # -----------------------------------------------
        # Compute sample correlations between those cluster pairs, and compared with homologous region pairs
        # compute sample embedding correlations for each cluster pairs, scatters (black points), homologous region pairs (red points)
        # background points (region pairs, shallow grey), black line (average correlation of homologous regions)  x:cluster index  y: correlation
        # -----------------------------------------------

        # -----------------------------------------------
        # Check if the enrichment of cluster pair on gene modules is more similar than the other pairs and close to homologous region pairs;
        # Enrichment: 1. average gene expression level; 2.
        # x: cluster id,  y: correlation of gene module enrichment
        # downside: heatmap of gene module enrichment for two species; upside: correlation of gene module enrichment heatmap,
        # with line as enrichment of homologous region pairs
        # -----------------------------------------------
        '''

        adata_human_expression.obsm['average_module'] = np.zeros((adata_human_expression.n_obs, module_num))
        adata_mouse_expression.obsm['average_module'] = np.zeros((adata_mouse_expression.n_obs, module_num))
        for i in range(module_num):
            # human
            gene_names_of_module_human = adata_human_gene_embedding[adata_human_gene_embedding.obs[module_name].isin([str(int(i+1))])].obs_names
            #print('len(gene_names_of_module_human)', len(gene_names_of_module_human))
            if len(gene_names_of_module_human) != 0:
                adata_human_expression.obsm['average_module'][:, i] = np.mean(adata_human_expression[:, gene_names_of_module_human].X.toarray(), axis=1)

            # mouse
            gene_names_of_module_mouse = adata_mouse_gene_embedding[
                adata_mouse_gene_embedding.obs[module_name].isin([str(int(i+1))])].obs_names
            if len(gene_names_of_module_mouse) != 0:
                adata_mouse_expression.obsm['average_module'][:, i] = np.mean(
                adata_mouse_expression[:, gene_names_of_module_mouse].X.toarray(), axis=1)

        print(adata_human_expression.obsm['average_module'])

        cluster_module_corr_dict = {}
        cluster_module_corr_mean_dict = {}
        for i in range(clustering_num):
            clustering_id = cluster_labels_unique[i]
            cluster_name_id = cluster_name_unique[i]
            human_distri_module_X = \
                adata_human_expression[adata_human_expression.obs[clustering_name].isin([clustering_id])].obsm['average_module']
            mouse_distri_module_X = \
            adata_mouse_expression[adata_mouse_expression.obs[clustering_name].isin([clustering_id])].obsm['average_module']

            corr_list = get_correlation(mouse_distri_module_X, human_distri_module_X, metric_name='correlation')
            cluster_module_corr_dict.update({cluster_name_id:corr_list})
            cluster_module_corr_mean_dict.update({cluster_name_id:np.mean(corr_list)})



        palette = sns.color_palette(cc.glasbey, n_colors=clustering_num)

        palette_dict = {k:v for k,v in zip(cluster_name_unique, palette)}

        #cluster_module_corr_df = pd.DataFrame.from_dict(cluster_module_corr_dict)
        #cluster_module_corr_mean_df = pd.DataFrame.from_dict(cluster_module_corr_mean_dict)
        plot_cluster_module_corr_dict = {'Cluster Name':[], 'Correlation':[]}
        plot_cluster_module_corr_mean_dict = {'Cluster Name': [], 'Mean Correlation': []}
        for k,v in cluster_module_corr_dict.items():
            plot_cluster_module_corr_dict['Cluster Name'] = plot_cluster_module_corr_dict['Cluster Name'] + [k] * len(v)
            plot_cluster_module_corr_dict['Correlation'] = plot_cluster_module_corr_dict['Correlation'] + v

            plot_cluster_module_corr_mean_dict['Cluster Name'] = plot_cluster_module_corr_mean_dict['Cluster Name'] + [k]
            plot_cluster_module_corr_mean_dict['Mean Correlation'] = plot_cluster_module_corr_mean_dict['Mean Correlation'] + np.mean(v)

        #temp_dict = {k:v for k,v in zip(plot_cluster_module_corr_mean_dict['Cluster Name'],  plot_cluster_module_corr_mean_dict['Mean Correlation'])}
        plot_cluster_module_corr_mean_dict_sorted = dict(sorted(cluster_module_corr_mean_dict.items(), key=lambda item: item[1]))
        print(plot_cluster_module_corr_mean_dict_sorted)

        plot_cluster_module_corr_df = pd.DataFrame.from_dict(plot_cluster_module_corr_dict)
        print(plot_cluster_module_corr_df)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.top"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.35

        plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
        plt.figure(figsize=(25, 10), dpi=self.fig_dpi)
        ax = sns.boxplot(x="Cluster Name", y="Correlation", data=plot_cluster_module_corr_df, order=plot_cluster_module_corr_mean_dict_sorted.keys(), palette=palette_dict,
                         width=0.8)
        # add_stat_annotation(ax, data=data_df, x="type", y="Correlation", order=["homologous", "random"],
        #                     box_pairs=[("homologous", "random")],
        #                     test='t-test_ind', text_format='star', loc='inside', verbose=2)  # text_format='star'
        plt.xlabel('')
        for item in ax.get_xticklabels():
            item.set_rotation(60)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        #plt.subplots_adjust(bottom=0.25, left=0.2)


        # Compare with homologous region gene module correlations
        human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        homo_region_dict = OrderedDict()
        for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
            homo_region_dict[x] = y

        mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        human_homo_set = set(human_mouse_homo_region['Human'].values)

        homo_corr_list = []
        homo_corr_mean_dict = dict()
        for i in range(len(homo_region_dict)):
            h_region = list(homo_region_dict.keys())[i]
            m_region = list(homo_region_dict.values())[i]

            human_distri_module_X = \
                adata_human_expression[adata_human_expression.obs['region_name'].isin([h_region])].obsm['average_module']
            mouse_distri_module_X = \
            adata_mouse_expression[adata_mouse_expression.obs['region_name'].isin([m_region])].obsm['average_module']

            corr_list = get_correlation(mouse_distri_module_X, human_distri_module_X, metric_name='correlation')

            homo_corr_mean_dict.update({h_region:np.mean(corr_list)})
            homo_corr_list.append(np.mean(corr_list))


        random_num = 2000
        random_corr_list = []
        mouse_region_num = len(self.mouse_64_labels_list)
        human_region_num = len(self.human_88_labels_list)
        for i in range(random_num):
            m_index = np.random.randint(mouse_region_num)
            h_index = np.random.randint(human_region_num)
            m_region = self.mouse_64_labels_list[m_index]
            h_region = self.human_88_labels_list[h_index]
            if m_region not in mouse_homo_set and h_region not in human_homo_set:
                human_distri_module_X = \
                    adata_human_expression[adata_human_expression.obs['region_name'].isin([h_region])].obsm[
                        'average_module']
                mouse_distri_module_X = \
                    adata_mouse_expression[adata_mouse_expression.obs['region_name'].isin([m_region])].obsm[
                        'average_module']

                corr_list = get_correlation(mouse_distri_module_X, human_distri_module_X, metric_name='correlation')

                random_corr_list.append(np.mean(corr_list))


        random_num = 1000
        random_corr_list_cluster = []
        mouse_cluster_num = len(cluster_name_unique)
        human_cluster_num = len(cluster_name_unique)
        for i in range(random_num):
            m_index = np.random.randint(mouse_cluster_num)
            h_index = np.random.randint(human_cluster_num)
            m_region = cluster_name_unique[m_index]
            h_region = cluster_name_unique[h_index]
            if m_index != h_index:
                human_distri_module_X = \
                    adata_human_expression[adata_human_expression.obs['cluster_name_acronym'].isin([h_region])].obsm[
                        'average_module']
                mouse_distri_module_X = \
                    adata_mouse_expression[adata_mouse_expression.obs['cluster_name_acronym'].isin([m_region])].obsm[
                        'average_module']

                corr_list = get_correlation(mouse_distri_module_X, human_distri_module_X, metric_name='correlation')

                random_corr_list_cluster.append(np.mean(corr_list))

        #for h_region, m_region in home_region_dict.items():

        print(random_corr_list_cluster)

        m = plot_cluster_module_corr_df.groupby('Cluster Name')['Correlation'].median()
        sns.lineplot(y=np.mean(homo_corr_list)*np.ones(m.values.shape), x=m.index, ax=ax, lw=3, color='deepskyblue')
        sns.lineplot(y=np.mean(random_corr_list) * np.ones(m.values.shape), x=m.index, ax=ax, lw=3, color='grey')
        sns.lineplot(y=np.mean(random_corr_list_cluster) * np.ones(m.values.shape), x=m.index, ax=ax, lw=3, color='pink')
        plt.savefig(save_path + 'Correlation_gene_module.' + self.fig_format)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1
        #plt.show()
        '''

        #---------------------------------------------------
        # Try embedding correlation instead of gene module similarity
        #---------------------------------------------------
        adata_human_embedding = adata_embedding[adata_embedding.obs['dataset'].isin(['Human'])]
        adata_mouse_embedding = adata_embedding[adata_embedding.obs['dataset'].isin(['Mouse'])]
        cluster_module_corr_dict = {}
        cluster_module_corr_mean_dict = {}
        for i in range(clustering_num):
            clustering_id = cluster_labels_unique[i]
            cluster_name_id = cluster_name_unique[i]
            human_distri_module_X = \
                adata_human_embedding[adata_human_embedding.obs[clustering_name].isin([clustering_id])].X
            mouse_distri_module_X = \
                adata_mouse_embedding[adata_mouse_embedding.obs[clustering_name].isin([clustering_id])].X
            corr_list = get_correlation(mouse_distri_module_X, human_distri_module_X, metric_name='correlation')
            cluster_module_corr_dict.update({cluster_name_id: corr_list})
            cluster_module_corr_mean_dict.update({cluster_name_id: np.mean(corr_list)})

        palette = sns.color_palette(cc.glasbey, n_colors=clustering_num)

        palette_dict = {k: v for k, v in zip(cluster_name_unique, palette)}

        # cluster_module_corr_df = pd.DataFrame.from_dict(cluster_module_corr_dict)
        # cluster_module_corr_mean_df = pd.DataFrame.from_dict(cluster_module_corr_mean_dict)
        plot_cluster_module_corr_dict = {'Cluster Name': [], 'Correlation': []}
        plot_cluster_module_corr_mean_dict = {'Cluster Name': [], 'Mean Correlation': []}
        for k, v in cluster_module_corr_dict.items():
            plot_cluster_module_corr_dict['Cluster Name'] = plot_cluster_module_corr_dict['Cluster Name'] + [k] * len(v)
            plot_cluster_module_corr_dict['Correlation'] = plot_cluster_module_corr_dict['Correlation'] + v

            plot_cluster_module_corr_mean_dict['Cluster Name'] = plot_cluster_module_corr_mean_dict['Cluster Name'] + [
                k]
            plot_cluster_module_corr_mean_dict['Mean Correlation'] = plot_cluster_module_corr_mean_dict[
                                                                         'Mean Correlation'] + np.mean(v)

        # temp_dict = {k:v for k,v in zip(plot_cluster_module_corr_mean_dict['Cluster Name'],  plot_cluster_module_corr_mean_dict['Mean Correlation'])}
        plot_cluster_module_corr_mean_dict_sorted = dict(
            sorted(cluster_module_corr_mean_dict.items(), key=lambda item: item[1]))
        print(plot_cluster_module_corr_mean_dict_sorted)

        plot_cluster_module_corr_df = pd.DataFrame.from_dict(plot_cluster_module_corr_dict)
        print(plot_cluster_module_corr_df)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.top"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.5

        np.savez(save_path + 'plot_data.npz',
                 plot_cluster_module_corr_df=plot_cluster_module_corr_df,
                 plot_cluster_module_corr_mean_dict_sorted=plot_cluster_module_corr_mean_dict_sorted,
                 palette_dict=palette_dict)

        plt.rc('xtick', labelsize=18)  # fontsize of the tick labels
        plt.figure(figsize=(25, 7), dpi=self.fig_dpi)
        ax = sns.boxplot(x="Cluster Name", y="Correlation", data=plot_cluster_module_corr_df,
                         order=plot_cluster_module_corr_mean_dict_sorted.keys(), palette=palette_dict,
                         width=0.8)
        # add_stat_annotation(ax, data=data_df, x="type", y="Correlation", order=["homologous", "random"],
        #                     box_pairs=[("homologous", "random")],
        #                     test='t-test_ind', text_format='star', loc='inside', verbose=2)  # text_format='star'
        plt.xlabel('')
        for item in ax.get_xticklabels():
            item.set_rotation(60)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.subplots_adjust(bottom=0.25, left=0.2)

        # Compare with homologous region gene module correlations
        human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        homo_region_dict = OrderedDict()
        for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
            homo_region_dict[x] = y

        mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        human_homo_set = set(human_mouse_homo_region['Human'].values)

        homo_corr_list = []
        homo_corr_mean_dict = dict()
        for i in range(len(homo_region_dict)):
            h_region = list(homo_region_dict.keys())[i]
            m_region = list(homo_region_dict.values())[i]

            human_distri_module_X = \
                adata_human_embedding[adata_human_embedding.obs['region_name'].isin([h_region])].X
            mouse_distri_module_X = \
                adata_mouse_embedding[adata_mouse_embedding.obs['region_name'].isin([m_region])].X

            corr_list = get_correlation(mouse_distri_module_X, human_distri_module_X, metric_name='correlation')

            homo_corr_mean_dict.update({h_region: np.mean(corr_list)})
            homo_corr_list.append(np.mean(corr_list))

        random_num = 2000
        random_corr_list = []
        mouse_region_num = len(self.mouse_64_labels_list)
        human_region_num = len(self.human_88_labels_list)
        for i in range(random_num):
            m_index = np.random.randint(mouse_region_num)
            h_index = np.random.randint(human_region_num)
            m_region = self.mouse_64_labels_list[m_index]
            h_region = self.human_88_labels_list[h_index]
            if m_region not in mouse_homo_set and h_region not in human_homo_set:
                human_distri_module_X = \
                    adata_human_embedding[adata_human_embedding.obs['region_name'].isin([h_region])].X
                mouse_distri_module_X = \
                    adata_mouse_embedding[adata_mouse_embedding.obs['region_name'].isin([m_region])].X

                corr_list = get_correlation(mouse_distri_module_X, human_distri_module_X, metric_name='correlation')

                random_corr_list.append(np.mean(corr_list))

        random_num = 1000
        random_corr_list_cluster = []
        mouse_cluster_num = len(cluster_name_unique)
        human_cluster_num = len(cluster_name_unique)
        for i in range(random_num):
            m_index = np.random.randint(mouse_cluster_num)
            h_index = np.random.randint(human_cluster_num)
            m_region = cluster_name_unique[m_index]
            h_region = cluster_name_unique[h_index]
            if m_index != h_index:
                human_distri_module_X = \
                    adata_human_embedding[adata_human_embedding.obs['cluster_name_acronym'].isin([h_region])].X
                mouse_distri_module_X = \
                    adata_mouse_embedding[adata_mouse_embedding.obs['cluster_name_acronym'].isin([m_region])].X

                corr_list = get_correlation(mouse_distri_module_X, human_distri_module_X, metric_name='correlation')

                random_corr_list_cluster.append(np.mean(corr_list))

        # for h_region, m_region in home_region_dict.items():

        print(random_corr_list_cluster)

        m = plot_cluster_module_corr_df.groupby('Cluster Name')['Correlation'].median()
        sns.lineplot(y=np.mean(homo_corr_list) * np.ones(m.values.shape), x=m.index, ax=ax, lw=3, color=self.human_color)
        #ax.text(61, 0.28, "Average correlation between \nhomologous region pairs", fontdict={'color':self.human_color, 'weight':'semibold'})
        ax.annotate('Average correlation between \nhomologous region pairs', xy=(58, 0.45), xytext=(61, 0), arrowprops=dict(width=5, facecolor=self.human_color)) #, fontdict={'color':'deepskyblue'}
        #sns.lineplot(y=np.mean(random_corr_list) * np.ones(m.values.shape), x=m.index, ax=ax, lw=3, color='grey')
        #sns.lineplot(y=np.mean(random_corr_list_cluster) * np.ones(m.values.shape), x=m.index, ax=ax, lw=3,
        #             color='pink')
        plt.subplots_adjust(left=0.06, right=0.99)

        plt.savefig(save_path + 'Correlation_embeddings.' + self.fig_format)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1


        '''
        # -----------------------------------------------
        # The expression similarity of many-2-many homologous genes is higher for those pairs than random pairs.
        # Collect many-2-many homologous genes in cluster pairs; and random pairs (or region pairs);
        # Compare gene expression levels of those clusters and the other pairs.
        # -----------------------------------------------
        # Get one-2-one homologous genes name list
        ######## If one-2-one homologous genes name list in Marker genes of brain regions is more reliable? #################
        path_datapiar = self.cfg.CAME.ROOT + 'datapair_init.pickle'
        path_datapiar_file = open(path_datapiar, 'rb')
        datapair = pickle.load(path_datapiar_file)
        mh_mat = datapair['vv_adj'].toarray()[0:self.cfg.BrainAlign.binary_M, self.cfg.BrainAlign.binary_M:]
        mouse_gene_names = adata_mouse_gene_embedding.obs_names
        human_gene_names = adata_human_gene_embedding.obs_names
        mouse_one2one_gene_list = []
        human_one2one_gene_list = []
        for i in range(self.cfg.BrainAlign.binary_M):
            for j in range(self.cfg.BrainAlign.binary_H):
                if mh_mat[i,j] > 0 and np.sum(mh_mat[i, :]) + np.sum(mh_mat[:, j]) <= 2:
                    mouse_one2one_gene_list.append(mouse_gene_names[i])
                    human_one2one_gene_list.append(human_gene_names[j])


        # # Acquire marker genes for each brain region
        # human_deg_list = []
        # ntop_gene = 30
        # sc.tl.rank_genes_groups(adata_human_expression, groupby='region_name', method='wilcoxon',
        #                         key_added="wilcoxon")
        # sc.pl.rank_genes_groups(adata_human_expression, n_genes=ntop_gene, sharey=False, key="wilcoxon",
        #                         show=False)
        # for region_name in list(set(adata_human_expression.obs['region_name'].tolist())):
        #     #if region_name != 'paraterminal gyrus':
        #     glist_human_cluster = sc.get.rank_genes_groups_df(adata_human_expression,
        #                                                       group=region_name,
        #                                                       key='wilcoxon',
        #                                                       log2fc_min=0.1,
        #                                                       pval_cutoff=0.001)['names'].squeeze().str.strip().tolist()
        #     human_deg_list =  human_deg_list + glist_human_cluster
        #
        # # Acquire marker genes for each brain region
        # mouse_deg_list = []
        # ntop_gene = 30
        # sc.tl.rank_genes_groups(adata_mouse_expression, groupby='region_name', method='wilcoxon',
        #                         key_added="wilcoxon")
        # sc.pl.rank_genes_groups(adata_mouse_expression, n_genes=ntop_gene, sharey=False, key="wilcoxon",
        #                         show=False)
        # for region_name in list(set(adata_mouse_expression.obs['region_name'].tolist())):
        #     glist_mouse_cluster = sc.get.rank_genes_groups_df(adata_mouse_expression,
        #                                                       group=region_name,
        #                                                       key='wilcoxon',
        #                                                       log2fc_min=0.1,
        #                                                       pval_cutoff=0.001)['names'].squeeze().str.strip().tolist()
        #     mouse_deg_list =  mouse_deg_list + glist_mouse_cluster
        #
        # mouse_gene_index_list = []
        # for i in range(len(mouse_one2one_gene_list)):
        #     if mouse_one2one_gene_list[i] in set(mouse_deg_list):
        #         mouse_gene_index_list.append(i)
        # human_gene_index_list = []
        # for i in range(len(human_one2one_gene_list)):
        #     if human_one2one_gene_list[i] in set(human_deg_list):
        #         human_gene_index_list.append(i)
        # gene_index_list = sorted(set(mouse_gene_index_list).intersection(set(human_gene_index_list)))
        #
        # mouse_one2one_gene_list = mouse_one2one_gene_list[gene_index_list]
        # human_one2one_gene_list = human_one2one_gene_list[gene_index_list]

        adata_mouse_expression_one2one = adata_mouse_expression[:, mouse_one2one_gene_list].copy()
        adata_human_expression_one2one = adata_human_expression[:, human_one2one_gene_list].copy()

        print('Number of one-to-one homologous genes: ', len(mouse_one2one_gene_list))


        cluster_module_corr_dict = {}
        cluster_module_corr_mean_dict = {}
        for i in range(clustering_num):
            clustering_id = cluster_labels_unique[i]
            cluster_name_id = cluster_name_unique[i]
            human_distri_module_X = \
                adata_human_expression_one2one[adata_human_expression_one2one.obs[clustering_name].isin([clustering_id])].X.toarray()
            mouse_distri_module_X = \
            adata_mouse_expression_one2one[adata_mouse_expression_one2one.obs[clustering_name].isin([clustering_id])].X.toarray()

            corr_list = get_correlation(mouse_distri_module_X, human_distri_module_X, metric_name='correlation')
            cluster_module_corr_dict.update({cluster_name_id:corr_list})
            cluster_module_corr_mean_dict.update({cluster_name_id:np.mean(corr_list)})



        palette = sns.color_palette(cc.glasbey, n_colors=clustering_num)

        palette_dict = {k:v for k,v in zip(cluster_name_unique, palette)}

        #cluster_module_corr_df = pd.DataFrame.from_dict(cluster_module_corr_dict)
        #cluster_module_corr_mean_df = pd.DataFrame.from_dict(cluster_module_corr_mean_dict)
        plot_cluster_module_corr_dict = {'Cluster Name':[], 'Correlation':[]}
        plot_cluster_module_corr_mean_dict = {'Cluster Name': [], 'Mean Correlation': []}
        for k,v in cluster_module_corr_dict.items():
            plot_cluster_module_corr_dict['Cluster Name'] = plot_cluster_module_corr_dict['Cluster Name'] + [k] * len(v)
            plot_cluster_module_corr_dict['Correlation'] = plot_cluster_module_corr_dict['Correlation'] + v

            plot_cluster_module_corr_mean_dict['Cluster Name'] = plot_cluster_module_corr_mean_dict['Cluster Name'] + [k]
            plot_cluster_module_corr_mean_dict['Mean Correlation'] = plot_cluster_module_corr_mean_dict['Mean Correlation'] + np.mean(v)

        #temp_dict = {k:v for k,v in zip(plot_cluster_module_corr_mean_dict['Cluster Name'],  plot_cluster_module_corr_mean_dict['Mean Correlation'])}
        plot_cluster_module_corr_mean_dict_sorted = dict(sorted(cluster_module_corr_mean_dict.items(), key=lambda item: item[1]))
        print(plot_cluster_module_corr_mean_dict_sorted)

        plot_cluster_module_corr_df = pd.DataFrame.from_dict(plot_cluster_module_corr_dict)
        print(plot_cluster_module_corr_df)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.top"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.35

        plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
        plt.figure(figsize=(25, 10), dpi=self.fig_dpi)
        ax = sns.boxplot(x="Cluster Name", y="Correlation", data=plot_cluster_module_corr_df, order=plot_cluster_module_corr_mean_dict_sorted.keys(), palette=palette_dict,
                         width=0.8)
        # add_stat_annotation(ax, data=data_df, x="type", y="Correlation", order=["homologous", "random"],
        #                     box_pairs=[("homologous", "random")],
        #                     test='t-test_ind', text_format='star', loc='inside', verbose=2)  # text_format='star'
        plt.xlabel('')
        for item in ax.get_xticklabels():
            item.set_rotation(60)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        #plt.subplots_adjust(bottom=0.25, left=0.2)


        # Compare with homologous region gene module correlations
        human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        homo_region_dict = OrderedDict()
        for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
            homo_region_dict[x] = y

        mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        human_homo_set = set(human_mouse_homo_region['Human'].values)

        homo_corr_list = []
        homo_corr_mean_dict = dict()
        for i in range(len(homo_region_dict)):
            h_region = list(homo_region_dict.keys())[i]
            m_region = list(homo_region_dict.values())[i]

            human_distri_module_X = \
                adata_human_expression_one2one[adata_human_expression_one2one.obs['region_name'].isin([h_region])].X.toarray()
            mouse_distri_module_X = \
            adata_mouse_expression_one2one[adata_mouse_expression_one2one.obs['region_name'].isin([m_region])].X.toarray()

            corr_list = get_correlation(mouse_distri_module_X, human_distri_module_X, metric_name='correlation')

            homo_corr_mean_dict.update({h_region:np.mean(corr_list)})
            homo_corr_list.append(np.mean(corr_list))


        random_num = 200
        random_corr_list = []
        mouse_region_num = len(self.mouse_64_labels_list)
        human_region_num = len(self.human_88_labels_list)
        for i in range(random_num):
            m_index = np.random.randint(mouse_region_num)
            h_index = np.random.randint(human_region_num)
            m_region = self.mouse_64_labels_list[m_index]
            h_region = self.human_88_labels_list[h_index]
            if m_region not in mouse_homo_set and h_region not in human_homo_set:
                human_distri_module_X = \
                    adata_human_expression_one2one[adata_human_expression_one2one.obs['region_name'].isin([h_region])].X.toarray()
                mouse_distri_module_X = \
                    adata_mouse_expression_one2one[adata_mouse_expression_one2one.obs['region_name'].isin([m_region])].X.toarray()

                corr_list = get_correlation(mouse_distri_module_X, human_distri_module_X, metric_name='correlation')

                random_corr_list.append(np.mean(corr_list))


        random_num = 200
        random_corr_list_cluster = []
        mouse_cluster_num = len(cluster_name_unique)
        human_cluster_num = len(cluster_name_unique)
        for i in range(random_num):
            m_index = np.random.randint(mouse_cluster_num)
            h_index = np.random.randint(human_cluster_num)
            m_region = cluster_name_unique[m_index]
            h_region = cluster_name_unique[h_index]
            if m_index != h_index:
                human_distri_module_X = \
                    adata_human_expression_one2one[adata_human_expression_one2one.obs['cluster_name_acronym'].isin([h_region])].X.toarray()
                mouse_distri_module_X = \
                    adata_mouse_expression_one2one[adata_mouse_expression_one2one.obs['cluster_name_acronym'].isin([m_region])].X.toarray()

                corr_list = get_correlation(mouse_distri_module_X, human_distri_module_X, metric_name='correlation')

                random_corr_list_cluster.append(np.mean(corr_list))

        #for h_region, m_region in home_region_dict.items():

        print(random_corr_list_cluster)

        m = plot_cluster_module_corr_df.groupby('Cluster Name')['Correlation'].median()
        sns.lineplot(y=np.mean(homo_corr_list)*np.ones(m.values.shape), x=m.index, ax=ax, lw=3, color='deepskyblue')
        sns.lineplot(y=np.mean(random_corr_list) * np.ones(m.values.shape), x=m.index, ax=ax, lw=3, color='grey')
        sns.lineplot(y=np.mean(random_corr_list_cluster) * np.ones(m.values.shape), x=m.index, ax=ax, lw=3, color='pink')
        plt.savefig(save_path + 'Correlation_gene_one2one_homologous_filtered.' + self.fig_format)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1

        adata_gene_embedding.write_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_gene_embedding.h5ad')
        adata_mouse_gene_embedding.write_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_mouse_gene_embedding.h5ad')
        adata_human_gene_embedding.write_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_human_gene_embedding.h5ad')
        
        '''
        return None


    def experiment_10_1_check_alignment_plot(self):

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/10_experiment_check_alignment/'

        """
        Examine whether the alignment clusters across species are biologically meaningful.

        1. Acquire gene modules shared by species;
        2. Compute sample correlations between those cluster pairs, and compared with homologous region pairs;
        3. The enrichment of cluster pair on gene modules is more similar than the other pairs and close to homologous region pairs;
        4. The expression similarity of one-2-one homologous genes is higher for those pairs than random pairs.

        :return:None
        """
        sns.set(style='white')
        TINY_SIZE = 22  # 39
        SMALL_SIZE = 28  # 42
        MEDIUM_SIZE = 32  # 46
        BIGGER_SIZE = 32  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        metric_name = 'correlation'

        # 1. Identify those differentially expressed genes for regions and clusters, and save them.
        # 2. How these genes are correlated to each other - Cluster those genes or get gene modules
        # 3. clustering of samples, do GO analysis to determine enrichment of genes in each region or clusters;
        # 4. dotplot of clusters and homologous regions, go clusters.
        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/10_experiment_check_alignment/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # load sample embedding
        adata_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_embedding.h5ad')

        clustering_name = f'leiden_cluster'

        # sc.tl.leiden(adata_embedding, resolution=1, key_added=clustering_name)
        clustering_num = len(Counter(adata_embedding.obs[clustering_name]))
        # Now we know each cluster is actually a pair
        region_labels = adata_embedding.obs['region_name'].values
        sample_names = adata_embedding.obs_names.values
        cluster_labels = adata_embedding.obs[clustering_name].values
        print('cluster_labels.shape', cluster_labels.shape)
        sample_cluter_dict = {k: v for k, v in zip(sample_names, cluster_labels)}
        # All cluster labels
        cluster_labels_unique = [str(x) for x in list(range(0, clustering_num))]

        cluster_name_unique = sorted(Counter(adata_embedding.obs['cluster_name_acronym']).keys(),
                                     key=lambda t: int(t.split('-')[0]))

        # load gene embedding
        adata_mouse_gene_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'm_embeddings.h5ad')
        adata_human_gene_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'h_embeddings.h5ad')
        adata_mouse_gene_embedding.obs['dataset'] = 'Mouse'
        adata_human_gene_embedding.obs['dataset'] = 'Human'

        adata_gene_embedding = ad.concat([adata_mouse_gene_embedding, adata_human_gene_embedding])

        # generate gene modules
        sc.pp.neighbors(adata_gene_embedding, n_neighbors=self.cfg.ANALYSIS.genes_umap_neighbor, metric='cosine',
                        use_rep='X')
        sc.tl.leiden(adata_gene_embedding, resolution=1.5, key_added='module')
        module_name = 'module'
        module_num = len(Counter(adata_gene_embedding.obs[module_name]))

        gene_names = adata_gene_embedding.obs_names.values
        module_labels = adata_gene_embedding.obs[module_name].values
        print('module_labels.shape', module_labels.shape)
        gene_module_dict = {k: v for k, v in zip(gene_names, module_labels)}

        # mouse gene_name to module
        adata_mouse_gene_embedding = adata_gene_embedding[adata_gene_embedding.obs['dataset'].isin(['Mouse'])]
        mouse_gene_names = adata_mouse_gene_embedding.obs_names.values
        mouse_module_labels = adata_mouse_gene_embedding.obs[module_name].values
        mouse_gene_module_dict = {k: v for k, v in zip(mouse_gene_names, mouse_module_labels)}
        # human gene_name to module
        adata_human_gene_embedding = adata_gene_embedding[adata_gene_embedding.obs['dataset'].isin(['Human'])]
        human_gene_names = adata_human_gene_embedding.obs_names.values
        human_module_labels = adata_human_gene_embedding.obs[module_name].values
        human_gene_module_dict = {k: v for k, v in zip(human_gene_names, human_module_labels)}

        # All module labels
        module_labels_unique = [str(x) for x in list(range(0, module_num))]

        # load gene expression data
        adata_mouse_expression = sc.read_h5ad(self.cfg.CAME.path_rawdata1)
        adata_mouse_expression = adata_mouse_expression[:, adata_mouse_gene_embedding.obs_names.tolist()]
        adata_mouse_expression.var['module'] = [mouse_gene_module_dict[g] for g in adata_mouse_expression.var_names]

        adata_human_expression = sc.read_h5ad(self.cfg.CAME.path_rawdata2)
        adata_human_expression = adata_human_expression[:, adata_human_gene_embedding.obs_names.tolist()]
        adata_human_expression.var['module'] = [human_gene_module_dict[g] for g in adata_human_expression.var_names]


        # ---------------------------------------------------
        # Try embedding correlation instead of gene module similarity
        # ---------------------------------------------------
        adata_human_embedding = adata_embedding[adata_embedding.obs['dataset'].isin(['Human'])]
        adata_mouse_embedding = adata_embedding[adata_embedding.obs['dataset'].isin(['Mouse'])]
        cluster_module_corr_dict = {}
        cluster_module_corr_mean_dict = {}
        for i in range(clustering_num):
            clustering_id = cluster_labels_unique[i]
            cluster_name_id = cluster_name_unique[i]
            human_distri_module_X = \
                adata_human_embedding[adata_human_embedding.obs[clustering_name].isin([clustering_id])].X
            mouse_distri_module_X = \
                adata_mouse_embedding[adata_mouse_embedding.obs[clustering_name].isin([clustering_id])].X
            corr_list = get_correlation(mouse_distri_module_X, human_distri_module_X, metric_name='correlation')
            cluster_module_corr_dict.update({cluster_name_id: corr_list})
            cluster_module_corr_mean_dict.update({cluster_name_id: np.mean(corr_list)})

        palette = sns.color_palette(cc.glasbey, n_colors=clustering_num)

        palette_dict = {k: v for k, v in zip(cluster_name_unique, palette)}

        # cluster_module_corr_df = pd.DataFrame.from_dict(cluster_module_corr_dict)
        # cluster_module_corr_mean_df = pd.DataFrame.from_dict(cluster_module_corr_mean_dict)
        plot_cluster_module_corr_dict = {'Cluster Name': [], 'Correlation': []}
        plot_cluster_module_corr_mean_dict = {'Cluster Name': [], 'Mean Correlation': []}
        for k, v in cluster_module_corr_dict.items():
            plot_cluster_module_corr_dict['Cluster Name'] = plot_cluster_module_corr_dict['Cluster Name'] + [k] * len(v)
            plot_cluster_module_corr_dict['Correlation'] = plot_cluster_module_corr_dict['Correlation'] + v

            plot_cluster_module_corr_mean_dict['Cluster Name'] = plot_cluster_module_corr_mean_dict['Cluster Name'] + [
                k]
            plot_cluster_module_corr_mean_dict['Mean Correlation'] = plot_cluster_module_corr_mean_dict[
                                                                         'Mean Correlation'] + np.mean(v)

        if os.path.exists(save_path + 'plot_data.npz'):
            plot_data = np.load(save_path + 'plot_data.npz')


        # temp_dict = {k:v for k,v in zip(plot_cluster_module_corr_mean_dict['Cluster Name'],  plot_cluster_module_corr_mean_dict['Mean Correlation'])}
        plot_cluster_module_corr_mean_dict_sorted = dict(
            sorted(cluster_module_corr_mean_dict.items(), key=lambda item: item[1]))
        print(plot_cluster_module_corr_mean_dict_sorted)

        plot_cluster_module_corr_df = pd.DataFrame.from_dict(plot_cluster_module_corr_dict)
        print(plot_cluster_module_corr_df)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.top"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.5

        np.savez(save_path + 'plot_data.npz',
                 plot_cluster_module_corr_df=plot_cluster_module_corr_df,
                 plot_cluster_module_corr_mean_dict_sorted=plot_cluster_module_corr_mean_dict_sorted,
                 palette_dict=palette_dict)

        plt.rc('xtick', labelsize=18, weight='normal')  # fontsize of the tick labels
        plt.figure(figsize=(25, 7), dpi=self.fig_dpi)
        ax = sns.boxplot(x="Cluster Name", y="Correlation", data=plot_cluster_module_corr_df,
                         order=plot_cluster_module_corr_mean_dict_sorted.keys(), palette=palette_dict,
                         width=0.8)
        # add_stat_annotation(ax, data=data_df, x="type", y="Correlation", order=["homologous", "random"],
        #                     box_pairs=[("homologous", "random")],
        #                     test='t-test_ind', text_format='star', loc='inside', verbose=2)  # text_format='star'
        plt.xlabel('')
        for item in ax.get_xticklabels():
            item.set_rotation(60)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        # plt.subplots_adjust(bottom=0.25, left=0.2)

        # Compare with homologous region gene module correlations
        human_mouse_homo_region = pd.read_csv(self.cfg.CAME.homo_region_file_path)
        homo_region_dict = OrderedDict()
        for x, y in zip(human_mouse_homo_region['Human'].values, human_mouse_homo_region['Mouse'].values):
            homo_region_dict[x] = y

        mouse_homo_set = set(human_mouse_homo_region['Mouse'].values)
        human_homo_set = set(human_mouse_homo_region['Human'].values)

        homo_corr_list = []
        homo_corr_mean_dict = dict()
        for i in range(len(homo_region_dict)):
            h_region = list(homo_region_dict.keys())[i]
            m_region = list(homo_region_dict.values())[i]

            human_distri_module_X = \
                adata_human_embedding[adata_human_embedding.obs['region_name'].isin([h_region])].X
            mouse_distri_module_X = \
                adata_mouse_embedding[adata_mouse_embedding.obs['region_name'].isin([m_region])].X

            corr_list = get_correlation(mouse_distri_module_X, human_distri_module_X, metric_name='correlation')

            homo_corr_mean_dict.update({h_region: np.mean(corr_list)})
            homo_corr_list.append(np.mean(corr_list))

        random_num = 2000
        random_corr_list = []
        mouse_region_num = len(self.mouse_64_labels_list)
        human_region_num = len(self.human_88_labels_list)
        for i in range(random_num):
            m_index = np.random.randint(mouse_region_num)
            h_index = np.random.randint(human_region_num)
            m_region = self.mouse_64_labels_list[m_index]
            h_region = self.human_88_labels_list[h_index]
            if m_region not in mouse_homo_set and h_region not in human_homo_set:
                human_distri_module_X = \
                    adata_human_embedding[adata_human_embedding.obs['region_name'].isin([h_region])].X
                mouse_distri_module_X = \
                    adata_mouse_embedding[adata_mouse_embedding.obs['region_name'].isin([m_region])].X

                corr_list = get_correlation(mouse_distri_module_X, human_distri_module_X, metric_name='correlation')

                random_corr_list.append(np.mean(corr_list))

        random_num = 1000
        random_corr_list_cluster = []
        mouse_cluster_num = len(cluster_name_unique)
        human_cluster_num = len(cluster_name_unique)
        for i in range(random_num):
            m_index = np.random.randint(mouse_cluster_num)
            h_index = np.random.randint(human_cluster_num)
            m_region = cluster_name_unique[m_index]
            h_region = cluster_name_unique[h_index]
            if m_index != h_index:
                human_distri_module_X = \
                    adata_human_embedding[adata_human_embedding.obs['cluster_name_acronym'].isin([h_region])].X
                mouse_distri_module_X = \
                    adata_mouse_embedding[adata_mouse_embedding.obs['cluster_name_acronym'].isin([m_region])].X

                corr_list = get_correlation(mouse_distri_module_X, human_distri_module_X, metric_name='correlation')

                random_corr_list_cluster.append(np.mean(corr_list))

        # for h_region, m_region in home_region_dict.items():

        print(random_corr_list_cluster)

        m = plot_cluster_module_corr_df.groupby('Cluster Name')['Correlation'].median()
        sns.lineplot(y=np.mean(homo_corr_list) * np.ones(m.values.shape), x=m.index, ax=ax, lw=3,
                     color=self.human_color)
        # ax.text(61, 0.28, "Average correlation between \nhomologous region pairs", fontdict={'color':self.human_color, 'weight':'semibold'})
        ax.annotate('Average correlation between \nhomologous region pairs', xy=(58, 0.45), xytext=(61, 0),
                    arrowprops=dict(width=5, facecolor=self.human_color))  # , fontdict={'color':'deepskyblue'}
        # sns.lineplot(y=np.mean(random_corr_list) * np.ones(m.values.shape), x=m.index, ax=ax, lw=3, color='grey')
        # sns.lineplot(y=np.mean(random_corr_list_cluster) * np.ones(m.values.shape), x=m.index, ax=ax, lw=3,
        #             color='pink')
        plt.subplots_adjust(left=0.06, right=0.99)

        plt.savefig(save_path + 'Correlation_embeddings.' + self.fig_format)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1








        return None



    def experiment_11_paga(self):
        """

        :return: None
        """
        sns.set(style='white')
        TINY_SIZE = 20  # 39
        SMALL_SIZE = 20  # 42
        MEDIUM_SIZE = 24  # 46
        BIGGER_SIZE = 24  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/11_experiment_rename_clusters/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        adata_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_embedding.h5ad')

        adata_human_embedding = adata_embedding[adata_embedding.obs['dataset'].isin(['Human'])]
        adata_mouse_embedding = adata_embedding[adata_embedding.obs['dataset'].isin(['Mouse'])]
        adata_human_embedding.obs['acronym'] = ['H-'+x for x in adata_human_embedding.obs['acronym']]
        adata_mouse_embedding.obs['acronym'] = ['M-' + x for x in adata_mouse_embedding.obs['acronym']]

        adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])

        sc.pp.neighbors(adata_embedding, n_neighbors=self.cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')

        sc.tl.paga(adata_embedding,  groups='parent_acronym')
        #sc.pl.paga(adata_embedding, color='region_name')

        rcParams["figure.subplot.right"] = 0.9
        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.fig_dpi)
        #with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):

        species_color_dict = {'Mouse': self.mouse_color, 'Human': self.human_color}
        species_cmap = colors.ListedColormap(['red', 'blue'])

        parent_acronym_color_dict = {}
        for k,v in self.parent_acronym_color_dict.items():
            parent_acronym_color_dict.update({k:{v:1}})

        graph_fontsize = 16

        sc.pl.paga(adata_embedding,
                   color= parent_acronym_color_dict,
                   node_size_scale=10,
                   min_edge_width = 0.1,
                   ax=ax,
                   max_edge_width=1,
                   fontsize = graph_fontsize,
                   fontoutline=1, colorbar=False, threshold=0.5, frameon=False) # node_size_scale=10, edge_width_scale=2
        plt.title('')
        fig.tight_layout()
        plt.savefig(save_path + 'paga_region_name_parent.' + self.fig_format, format=self.fig_format)

        rcParams["figure.subplot.right"] = 0.9
        fig, ax = plt.subplots(figsize=(10,10), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        sc.pl.paga(adata_embedding,
                   color='dataset',
                   #cmap = species_cmap,
                   #colors = species_color_dict,
                   node_size_scale=10,
                   min_edge_width=0.1,
                   ax=ax,
                   max_edge_width=1,
                   fontsize=graph_fontsize,
                   fontoutline=1, colorbar=False, threshold=0.5, frameon=False)  # node_size_scale=10, edge_width_scale=2
        plt.title('')
        fig.tight_layout()

        plt.savefig(save_path + 'paga_region_name_species_parent.' + self.fig_format, format=self.fig_format)

        #----------------------------------------------------------
        sc.tl.paga(adata_embedding, groups='acronym')
        rcParams["figure.subplot.right"] = 0.9
        fig, ax = plt.subplots(figsize=(35, 13), dpi=self.fig_dpi)
        acronym_color_dict = {}
        for k, v in self.acronym_color_dict.items():
            acronym_color_dict.update({k: {v:1}})
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        sc.pl.paga(adata_embedding,
                   color=acronym_color_dict,
                   node_size_scale=20,
                   min_edge_width= 0.1,
                   ax=ax,
                   max_edge_width=1,
                   layout='fa',
                   fontsize=graph_fontsize,
                   fontoutline=1,colorbar=False,threshold=0.99, frameon=False
                   )  # node_size_scale=10, edge_width_scale=2
        plt.title('')
        fig.tight_layout()

        plt.savefig(save_path + 'paga_region_name.' + self.fig_format, format=self.fig_format)

        rcParams["figure.subplot.right"] = 0.9

        fig, ax = plt.subplots(figsize=(35, 13), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):

        sc.pl.paga(adata_embedding,
                   color='dataset',
                   cmap=species_cmap,
                   node_size_scale=20,
                   min_edge_width=0.1,
                   ax=ax,
                   max_edge_width=1,
                   layout = 'fa',
                   fontsize=graph_fontsize,
                   fontoutline=1,colorbar=False,threshold=0.99, frameon=False
                   )  # node_size_scale=10, edge_width_scale=2
        plt.title('')
        fig.tight_layout()

        plt.savefig(save_path + 'paga_region_name_species.' + self.fig_format, format=self.fig_format)


        # -----------------------------------------------------------------------------------
        # cluster paga
        sc.tl.paga(adata_embedding, groups='cluster_name_acronym')
        rcParams["figure.subplot.right"] = 0.9
        fig, ax = plt.subplots(figsize=(20, 20), dpi=self.fig_dpi)

        cluster_name_unique = sorted(Counter(adata_embedding.obs['cluster_name_acronym']).keys(),
                                     key=lambda t: int(t.split('-')[0]))
        palette_cluster = sns.color_palette(cc.glasbey, n_colors=len(cluster_name_unique))

        cluster_name_color_dict = {}
        for k, v in zip(cluster_name_unique, palette_cluster):
            cluster_name_color_dict.update({k: {v: 1}})
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        sc.pl.paga(adata_embedding,
                   color=cluster_name_color_dict,
                   node_size_scale=20,
                   min_edge_width=0.1,
                   ax=ax,
                   max_edge_width=1, node_size_power=1,single_component=True,
                   fontsize=graph_fontsize,
                   fontoutline=1, colorbar=False, threshold= 1, frameon=False
                   )  # node_size_scale=10, edge_width_scale=2
        plt.title('')
        fig.tight_layout()

        plt.savefig(save_path + 'paga_cluster.' + self.fig_format, format=self.fig_format)

        rcParams["figure.subplot.right"] = 0.9

        fig, ax = plt.subplots(figsize=(20, 20), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):

        sc.pl.paga(adata_embedding,
                   color='dataset',
                   cmap= species_cmap,
                   fontsize=graph_fontsize,
                   node_size_scale=20,
                   min_edge_width=0.1,
                   ax=ax, node_size_power=1,
                   max_edge_width=1, single_component=True,
                   fontoutline=1, colorbar=False, threshold= 1, frameon=False
                   )  # node_size_scale=10, edge_width_scale=2
        plt.title('')
        fig.tight_layout()

        plt.savefig(save_path + 'paga_cluster_species.' + self.fig_format, format=self.fig_format)

        return None


    def experiment_12_corr_clusters(self):

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
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/12_experiment_corr_clusters/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        adata_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_embedding.h5ad')

        adata_human_embedding = adata_embedding[adata_embedding.obs['dataset'].isin(['Human'])]
        adata_mouse_embedding = adata_embedding[adata_embedding.obs['dataset'].isin(['Mouse'])]

        human_mouse_dict_mean = {'Human': [], 'Mouse': []}
        human_mouse_dict_std = {'Human': [], 'Mouse': []}

        # human_mouse_dict_mean['Human'] = human_correlation_dict['mean']
        # human_mouse_dict_mean['Mouse'] = mouse_correlation_dict['mean']
        #
        # human_mouse_dict_std['Human'] = human_correlation_dict['std']
        # human_mouse_dict_std['Mouse'] = mouse_correlation_dict['std']

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.bottom"] = 0.05

        plt.figure(figsize=(8, 8), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        sns.jointplot(data=pd.DataFrame.from_dict(human_mouse_dict_mean), x="Human", y="Mouse", kind="reg", height=8)
        plt.title('')
        plt.savefig(save_path + 'mean_human_mouse.' + self.cfg.fig_format, format=self.cfg.fig_format)


        return None

    def experiment_13_th_clusters_deg_analysis(self):

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
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/1_alignment_STs/13_experiment_th_clusters_deg_analysis/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        adata_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_embedding.h5ad')


        TH_cluster_list = []
        cluster_name_unique = sorted(Counter(adata_embedding.obs['cluster_name_acronym']).keys(),
                                     key=lambda t: int(t.split('-')[0]))
        for cluster_name in cluster_name_unique:
            if cluster_name.split('-')[1] == 'TH' and cluster_name.split('-')[2] == 'TH':
                TH_cluster_list.append(cluster_name)

        adata_embedding_TH = adata_embedding[adata_embedding.obs['cluster_name_acronym'].isin(TH_cluster_list)]

        #adata_embedding_TH_mouse =

        #adata_mouse_expression =


        return None


    def forward(self):

        self.init_data()
        self.experiment_1_cross_species_clustering()
        self.experiment_2_umap_evaluation()
        self.experiment_2_umap_diagram()
        #self.experiment_3_homo_random()
        self.experiment_4_cross_species_genes_analysis()
        self.experiment_5_genes_homo_random()
        self.experiment_6_brain_region_classfier()
        self.experiment_7_align_cross_species()
        self.experiment_8_cross_evaluation_aligment_cluster()
        self.experiment_9_name_clusters()
        self.experiment_10_check_alignment()
        self.experiment_11_paga()


    def forward_single(self):

        #self.init_data()
        #self.experiment_1_cross_species_clustering()
        #self.experiment_2_umap_evaluation()
        aligned_score_came = self.experiment_2_umap_evaluation_single(method='CAME')
        # self.experiment_3_homo_random()
        # self.experiment_4_cross_species_genes_analysis()
        # self.experiment_5_genes_homo_random()
        # self.experiment_6_brain_region_classfier()
        # self.experiment_7_align_cross_species()
        # self.experiment_8_cross_evaluation_aligment_cluster()



def anatomical_STs_analysis_fun(cfg):
    # Gene expression comparative analysis
    fig_format = cfg.BrainAlign.fig_format  # the figure save format
    sns.set(style='white')
    TINY_SIZE = 24  # 39
    SMALL_SIZE = 28  # 42
    MEDIUM_SIZE = 32  # 46
    BIGGER_SIZE = 36  # 46

    plt.rc('font', size=30)  # 35 controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']

    mouse_color = '#ED7D31'
    human_color = '#4472C4'

    # read labels data including acronym, color and parent region name
    mouse_64_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)
    mouse_64_labels_list = list(mouse_64_labels['region_name'])
    mouse_64_acronym_dict = {k: v for k, v in zip(mouse_64_labels['region_name'], mouse_64_labels['acronym'])}
    mouse_64_color_dict = {k: v for k, v in zip(mouse_64_labels['region_name'], mouse_64_labels['color_hex_triplet'])}
    mouse_64_parent_region_dict = {k: v for k, v in
                                   zip(mouse_64_labels['region_name'], mouse_64_labels['parent_region_name'])}
    mouse_15_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.parent_labels_mouse_file)
    mouse_15_acronym_dict = {k: v for k, v in zip(mouse_15_labels['region_name'], mouse_15_labels['acronym'])}
    mouse_15_color_dict = {k: v for k, v in zip(mouse_15_labels['region_name'], mouse_15_labels['color_hex_triplet'])}

    human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
    human_88_labels_list = list(human_88_labels['region_name'])
    human_88_acronym_dict = {k: v for k, v in zip(human_88_labels['region_name'], human_88_labels['acronym'])}
    human_88_color_dict = {k: v for k, v in zip(human_88_labels['region_name'], human_88_labels['color_hex_triplet'])}
    human_88_parent_region_dict = {k: v for k, v in
                                   zip(human_88_labels['region_name'], human_88_labels['parent_region_name'])}
    human_16_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.parent_labels_human_file)
    human_16_acronym_dict = {k: v for k, v in zip(human_16_labels['region_name'], human_16_labels['acronym'])}
    human_16_color_dict = {k: v for k, v in zip(human_16_labels['region_name'], human_16_labels['color_hex_triplet'])}

    mouse_human_64_88_color_dict = dict(mouse_64_color_dict)
    mouse_human_64_88_color_dict.update(human_88_color_dict)

    save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    # load sample embeddings
    adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    adata_mouse_expression = sc.read_h5ad(cfg.CAME.path_rawdata1)
    for obs1, obs2 in zip(adata_mouse_embedding.obs_names, adata_mouse_expression.obs_names):
        if obs1 != obs2:
            print('Sample name not aligned!')

    adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
    adata_human_expression = sc.read_h5ad(cfg.CAME.path_rawdata2)
    for obs1, obs2 in zip(adata_human_embedding.obs_names, adata_human_expression.obs_names):
        if obs1 != obs2:
            print('Sample name not aligned!')

    # load gene embeddings
    adata_mouse_gene_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'm_embeddings.h5ad')
    adata_human_gene_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'h_embeddings.h5ad')

    mouse_gene_num = adata_mouse_gene_embedding.n_obs
    human_gene_num = adata_human_gene_embedding.n_obs
    '''
    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)
    mh_mat = datapair['vv_adj'].toarray()[0:cfg.BrainAlign.binary_M, cfg.BrainAlign.binary_M:]
    print('Homologous gene matrix shape:', mh_mat.shape)
    #
    print('Computing correlation matrix...')
    mouse_df = pd.DataFrame(adata_mouse_gene_embedding.X).T
    mouse_df.columns = ['mouse_' + x for x in adata_mouse_gene_embedding.obs_names]
    # print(mouse_df.shape)
    human_df = pd.DataFrame(adata_human_gene_embedding.X).T
    human_df.columns = ['human_' + x for x in adata_human_gene_embedding.obs_names]
    # print(human_df.shape)

    ## Compute correlation of homologous regions compared to the other pairs
    corr_result = pd.concat([mouse_df, human_df], axis=1).corr()

    Var_Corr = corr_result[human_df.columns].loc[mouse_df.columns]
    mean, std = Var_Corr.mean().mean(), Var_Corr.stack().std()
    # print('mean:', mean, 'std:', std)
    print('Correlation matrix shape:', Var_Corr.shape)
    '''
    def experiment_6_gene_palette_comparison():

        # Generate human palette
        return None


    return None


def gene_comparison(cfg):
    # Gene expression comparative analysis
    fig_format = cfg.BrainAlign.fig_format  # the figure save format
    fig_dpi = cfg.BrainAlign.fig_dpi
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
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']

    mouse_color = '#ED7D31'
    human_color = '#4472C4'

    # read labels data including acronym, color and parent region name
    mouse_64_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)
    mouse_64_labels_list = list(mouse_64_labels['region_name'])
    mouse_64_acronym_dict = {k: v for k, v in zip(mouse_64_labels['region_name'], mouse_64_labels['acronym'])}
    mouse_64_color_dict = {k: v for k, v in zip(mouse_64_labels['region_name'], mouse_64_labels['color_hex_triplet'])}
    mouse_64_parent_region_dict = {k: v for k, v in
                                   zip(mouse_64_labels['region_name'], mouse_64_labels['parent_region_name'])}
    mouse_15_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.parent_labels_mouse_file)
    mouse_15_acronym_dict = {k: v for k, v in zip(mouse_15_labels['region_name'], mouse_15_labels['acronym'])}
    mouse_15_color_dict = {k: v for k, v in zip(mouse_15_labels['region_name'], mouse_15_labels['color_hex_triplet'])}

    human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
    human_88_labels_list = list(human_88_labels['region_name'])
    human_88_acronym_dict = {k: v for k, v in zip(human_88_labels['region_name'], human_88_labels['acronym'])}
    human_88_color_dict = {k: v for k, v in zip(human_88_labels['region_name'], human_88_labels['color_hex_triplet'])}
    human_88_parent_region_dict = {k: v for k, v in
                                   zip(human_88_labels['region_name'], human_88_labels['parent_region_name'])}
    human_16_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.parent_labels_human_file)
    human_16_acronym_dict = {k: v for k, v in zip(human_16_labels['region_name'], human_16_labels['acronym'])}
    human_16_color_dict = {k: v for k, v in zip(human_16_labels['region_name'], human_16_labels['color_hex_triplet'])}

    mouse_human_64_88_color_dict = dict(mouse_64_color_dict)
    mouse_human_64_88_color_dict.update(human_88_color_dict)

    save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load sample embeddings
    adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    adata_mouse_expression = sc.read_h5ad(cfg.CAME.path_rawdata1)
    for obs1, obs2 in zip(adata_mouse_embedding.obs_names, adata_mouse_expression.obs_names):
        if obs1 != obs2:
            print('Sample name not aligned!')


    adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
    adata_human_expression = sc.read_h5ad(cfg.CAME.path_rawdata2)
    for obs1, obs2 in zip(adata_human_embedding.obs_names, adata_human_expression.obs_names):
        if obs1 != obs2:
            print('Sample name not aligned!')

    # load gene embeddings
    adata_mouse_gene_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'm_embeddings.h5ad')
    adata_human_gene_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'h_embeddings.h5ad')

    mouse_gene_num = adata_mouse_gene_embedding.n_obs
    human_gene_num = adata_human_gene_embedding.n_obs

    path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
    path_datapiar_file = open(path_datapiar, 'rb')
    datapair = pickle.load(path_datapiar_file)
    mh_mat = datapair['vv_adj'].toarray()[0:cfg.BrainAlign.binary_M, cfg.BrainAlign.binary_M:]
    print('Homologous gene matrix shape:', mh_mat.shape)
    #
    print('Computing correlation matrix...')
    mouse_df = pd.DataFrame(adata_mouse_gene_embedding.X).T
    mouse_df.columns = ['mouse_' + x for x in adata_mouse_gene_embedding.obs_names]
    #print(mouse_df.shape)
    human_df = pd.DataFrame(adata_human_gene_embedding.X).T
    human_df.columns = ['human_' + x for x in adata_human_gene_embedding.obs_names]
    #print(human_df.shape)

    ## Compute correlation of homologous regions compared to the other pairs
    corr_result = pd.concat([mouse_df, human_df], axis=1).corr()

    Var_Corr = corr_result[human_df.columns].loc[mouse_df.columns]
    Var_Corr.columns = human_df.columns
    Var_Corr.index = mouse_df.columns

    Var_Corr_m = corr_result[mouse_df.columns].loc[mouse_df.columns]
    Var_Corr_m.columns = mouse_df.columns
    Var_Corr_m.index = mouse_df.columns

    Var_Corr_h = corr_result[human_df.columns].loc[human_df.columns]
    Var_Corr_h.columns = human_df.columns
    Var_Corr_h.index = human_df.columns

    #mean, std = Var_Corr.mean().mean(), Var_Corr.stack().std()
    #print('mean:', mean, 'std:', std)
    print('Correlation matrix shape:', Var_Corr.shape)

    # Experiment 1.1
    def experiment_1_calb1():
        # UMAP the known marker of CALB1 in hippocampal subfields
        #print(adata_mouse_expression.var_names)\

        wspace = 0.2
        color_map = 'magma_r'

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/1_experiment_CALB1_whole_umap/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print(adata_mouse_expression[:, 'Calb1'].X.toarray().reshape(-1))
        adata_mouse_embedding.obs['CALB1'] = normalize(adata_mouse_expression[:, 'Calb1'].X.toarray().reshape(-1)[:,np.newaxis], norm='max', axis=0).ravel()

        print(adata_human_expression[:, 'CALB1'].X.toarray().reshape(-1))
        adata_human_embedding.obs['CALB1'] = normalize(adata_human_expression[:, 'CALB1'].X.toarray().reshape(-1)[:,np.newaxis], norm='max', axis=0).ravel()

        adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
        print(adata_embedding)
        sc.pp.neighbors(adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(adata_embedding)


        hippocampal_region_list = ['Field CA1','Field CA2','Field CA3','Dentate gyrus', 'Subiculum',
                                   'CA1 field','CA2 field','CA3 field','dentate gyrus', 'subiculum']
        adata_embedding_hippocampal = adata_embedding[adata_embedding.obs['region_name'].isin(hippocampal_region_list)]

        adata_embedding_hippocampal_mouse = adata_embedding_hippocampal[adata_embedding_hippocampal.obs['dataset'].isin(['Mouse'])]
        adata_embedding_hippocampal_human = adata_embedding_hippocampal[adata_embedding_hippocampal.obs['dataset'].isin(['Human'])]


        print('adata_embedding_hippocampal:',adata_embedding_hippocampal)
        palette = {k:mouse_human_64_88_color_dict[v] for k,v in zip(hippocampal_region_list, hippocampal_region_list)}
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.45
        with plt.rc_context({"figure.figsize": (16, 8), "figure.dpi":(fig_dpi)}):
            fig = sc.pl.umap(adata_embedding_hippocampal, color=['region_name'], return_fig=True, legend_loc='right margin')
            plt.title('')
            fig.savefig(
                save_path + 'umap_regions.' + fig_format, format=fig_format)
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.9

        with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
            fig = sc.pl.umap(adata_embedding_hippocampal_mouse, color=['CALB1', 'region_name'],
                       return_fig=True, legend_loc='on data', color_map=color_map, wspace=wspace, title=['CALB1', ''])
            #plt.title('')
            fig.savefig(
                save_path + 'umap_Calb1_expression_mouse.' + fig_format, format=fig_format)
            #plt.title('')

        with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
            sc.pl.umap(adata_embedding_hippocampal_human, color=['CALB1', 'region_name'], return_fig=True,
                       legend_loc='on data', color_map=color_map, wspace=wspace, title=['CALB1', '']).savefig(
                save_path + 'umap_Calb1_expression_human.' + fig_format, format=fig_format)
            #plt.title('')

        with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
            fig = sc.pl.umap(adata_embedding_hippocampal, color=['CALB1', 'dataset'], return_fig=True, legend_loc='on data', color_map=color_map, wspace=wspace, title=['CALB1', ''])
            #plt.title('')
            fig.savefig(
                save_path + 'umap_CALB1_expression.' + fig_format, format=fig_format)



        # umap independently
        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/1_experiment_CALB1/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print(adata_mouse_expression[:, 'Calb1'].X.toarray().reshape(-1))
        adata_mouse_embedding.obs['CALB1'] = normalize(
            adata_mouse_expression[:, 'Calb1'].X.toarray().reshape(-1)[:, np.newaxis], norm='max', axis=0).ravel()

        print(adata_human_expression[:, 'CALB1'].X.toarray().reshape(-1))
        adata_human_embedding.obs['CALB1'] = normalize(
            adata_human_expression[:, 'CALB1'].X.toarray().reshape(-1)[:, np.newaxis], norm='max', axis=0).ravel()

        adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
        print(adata_embedding)

        hippocampal_region_list = ['Field CA1', 'Field CA2', 'Field CA3', 'Dentate gyrus', 'Subiculum',
                                   'CA1 field', 'CA2 field', 'CA3 field', 'dentate gyrus', 'subiculum']

        adata_embedding_hippocampal = adata_embedding[adata_embedding.obs['region_name'].isin(hippocampal_region_list)]

        sc.pp.neighbors(adata_embedding_hippocampal, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(adata_embedding_hippocampal)

        adata_embedding_hippocampal_mouse = adata_embedding_hippocampal[
            adata_embedding_hippocampal.obs['dataset'].isin(['Mouse'])]
        adata_embedding_hippocampal_human = adata_embedding_hippocampal[
            adata_embedding_hippocampal.obs['dataset'].isin(['Human'])]

        print('adata_embedding_hippocampal:', adata_embedding_hippocampal)
        palette = {k: mouse_human_64_88_color_dict[v] for k, v in zip(hippocampal_region_list, hippocampal_region_list)}
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.45
        with plt.rc_context({"figure.figsize": (16, 8), "figure.dpi":(fig_dpi)}):
            fig = sc.pl.umap(adata_embedding_hippocampal, color=['region_name'], return_fig=True,
                       legend_loc='right margin', title=['CALB1', ''])
            #plt.title('')
            fig.savefig(
                save_path + 'umap_regions.' + fig_format, format=fig_format)

        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.9

        with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
            fig = sc.pl.umap(adata_embedding_hippocampal_mouse, color=['CALB1', 'region_name'],  return_fig=True,
                       legend_loc='on data', color_map=color_map, wspace=wspace, title=['CALB1', ''])
            #plt.title('')
            fig.savefig(
                save_path + 'umap_Calb1_expression_mouse.' + fig_format, format=fig_format)


        with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
            fig = sc.pl.umap(adata_embedding_hippocampal_human, color=['CALB1', 'region_name'], return_fig=True,
                       legend_loc='on data', color_map=color_map, wspace=wspace, title=['CALB1', ''])
            #plt.title('')
            fig.savefig(
                save_path + 'umap_Calb1_expression_human.' + fig_format, format=fig_format)


        with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
            fig = sc.pl.umap(adata_embedding_hippocampal, color=['CALB1', 'dataset'], return_fig=True,
                       legend_loc='on data', color_map=color_map, wspace=wspace, title=['CALB1', ''])
            #plt.title('')
            fig.savefig(
                save_path + 'umap_CALB1_expression.' + fig_format, format=fig_format)



        # plot boxplots with t-test pvalues
        human_dict = {k:None for k in ['CA1 field', 'CA2 field', 'CA3 field', 'dentate gyrus', 'subiculum']}
        mouse_dict = {k:None for k in ['Field CA1', 'Field CA2', 'Field CA3', 'Dentate gyrus', 'Subiculum']}

        for human_region in human_dict.keys():
            adata_embedding_present_region = adata_embedding[adata_embedding.obs['region_name'].isin([human_region])]
            human_dict[human_region] = list(adata_embedding_present_region.obs['CALB1'].values)

        for mouse_region in mouse_dict.keys():
            adata_embedding_present_region = adata_embedding[adata_embedding.obs['region_name'].isin([mouse_region])]
            mouse_dict[mouse_region] = list(adata_embedding_present_region.obs['CALB1'].values)

        human_dict_new = {'Region':[], 'Expression':[]}
        for x,y in human_dict.items():
            human_dict_new['Region'] = human_dict_new['Region'] + [x]*len(y)
            human_dict_new['Expression'] = human_dict_new['Expression'] + y
        human_df = pd.DataFrame.from_dict(human_dict_new)

        mouse_dict_new = {'Region': [], 'Expression': []}
        for x, y in mouse_dict.items():
            mouse_dict_new['Region'] = mouse_dict_new['Region'] + [x] * len(y)
            mouse_dict_new['Expression'] = mouse_dict_new['Expression'] + y
        mouse_df = pd.DataFrame.from_dict(mouse_dict_new)

        img_width = 8
        img_height = 8
        plt.figure(figsize=(img_width, img_height))  # width:20, height:3
        ax = plt.gca()
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        order = ['CA1 field', 'CA2 field', 'CA3 field', 'dentate gyrus', 'subiculum']
        pal_dict = {'CA1 field': (181 / 255, 85 / 255, 250 / 255),
                    'CA2 field': (255 / 255, 66 / 255, 14 / 255),
                    'CA3 field': (137 / 255, 218 / 255, 89 / 255),
                    'dentate gyrus': (255 / 255, 184 / 255, 0 / 255),
                    'subiculum': (76 / 255, 181 / 255, 245 / 255)}


        plt.figure(figsize=(img_width, img_height), dpi=fig_dpi)  # width:20, height:3
        #human_df.index.name = 'Region'
        x = 'Region'
        y = 'Expression'
        ax = sns.boxplot(data=human_df, x=x, y=y, order=order, palette=pal_dict)
        add_stat_annotation(ax, data=human_df, x=x, y=y, order=order,
                            box_pairs=[('CA1 field', 'dentate gyrus'), ('CA2 field', 'dentate gyrus'),
                                       ('CA3 field', 'dentate gyrus'),
                                       ('dentate gyrus', 'subiculum')],
                            test='t-test_ind', text_format='star', loc='inside', verbose=2)

        # plt.legend(loc='best', bbox_to_anchor=(1.03, 1))
        plt.xlabel('')
        plt.ylabel('Expression')
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.subplots_adjust(bottom=0.25, left=0.2)
        plt.savefig(save_path + 'human_hippocampal.'+fig_format, format=fig_format)

        # Mouse boxplots
        order = ['Field CA1', 'Field CA2', 'Field CA3', 'Dentate gyrus', 'Subiculum']
        pal_dict = {'Field CA1': (181 / 255, 85 / 255, 250 / 255),
                    'Field CA2': (255 / 255, 66 / 255, 14 / 255),
                    'Field CA3': (137 / 255, 218 / 255, 89 / 255),
                    'Dentate gyrus': (255 / 255, 184 / 255, 0 / 255),
                    'Subiculum': (76 / 255, 181 / 255, 245 / 255)}
        plt.figure(figsize=(img_width, img_height), dpi=fig_dpi)  # width:20, height:3
        #mouse_df.index.name = 'Region'
        x = 'Region'
        y = 'Expression'
        ax = sns.boxplot(data=mouse_df, x=x, y=y, order=order, palette=pal_dict)
        add_stat_annotation(ax, data=mouse_df, x=x, y=y, order=order,
                            box_pairs=[('Field CA1', 'Field CA2'), ('Field CA2', 'Field CA3'),
                                       ('Field CA3', 'Dentate gyrus'),
                                       ('Dentate gyrus', 'Subiculum')],
                            test='t-test_ind', text_format='star', loc='inside', verbose=2)

        # plt.legend(loc='best', bbox_to_anchor=(1.03, 1))
        plt.xlabel('')
        plt.ylabel('Expression')
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.subplots_adjust(bottom=0.25, left=0.2)
        plt.savefig(save_path + 'mouse_hippocampal.'+fig_format, format=fig_format)


    def experiment_2_marker_genes(adata_mouse_embedding, adata_human_embedding):
        # 1. as correlation increases, how the proportion of homologous genes increase.
        # 2. see how the human genes the most correlated to mouse marker genes distrbute in the human brain
        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/2_experiment_marker_genes/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # mouse color:#ED7D31 , human color:#4472C4
        # 1.
        correlation_vec = np.linspace(0, 0.99, 20)
        proportion_mouse_homo_gene_vec = np.zeros(20)
        proportion_human_homo_gene_vec = np.zeros(20)
        proportion_homo_all_vec = np.zeros(20)
        corr_mat = Var_Corr.values
        homo_corr_mat = np.multiply(corr_mat, mh_mat)
        for i in range(len(correlation_vec)):
            mouse_gene_set = []
            human_gene_set = []
            corr_threshold = correlation_vec[i]
            homo_count_mat = homo_corr_mat > corr_threshold
            homo_mouse_index, homo_human_index = np.nonzero(homo_count_mat)

            corr_count_mat = corr_mat > corr_threshold
            corr_mouse_index, corr_human_index = np.nonzero(corr_count_mat)

            homo_beyond_threshold_num = np.count_nonzero(homo_count_mat)
            corr_beyond_threshold_num = np.count_nonzero(corr_count_mat)
            proportion_homo_all_vec[i] = homo_beyond_threshold_num/corr_beyond_threshold_num
            proportion_mouse_homo_gene_vec[i] = len(set(homo_mouse_index)) / len(set(corr_mouse_index))
            proportion_human_homo_gene_vec[i] = len(set(homo_human_index)) / len(set(corr_human_index))

        plot_dict = {}
        plot_dict['Correlation threshold'] = correlation_vec
        plot_dict['Ratio of mouse homo-genes'] = proportion_mouse_homo_gene_vec
        plot_dict['Ratio of human homo-genes'] = proportion_human_homo_gene_vec

        plot_dict_1 = {}
        plot_dict_1['Correlation threshold'] = correlation_vec
        plot_dict_1['Ratio of homo-pairs'] = proportion_homo_all_vec


        fig, ax = plt.subplots(figsize=(10, 8), dpi=fig_dpi)
        plot_rate_df = pd.DataFrame.from_dict(plot_dict)
        sns.lineplot(x='Correlation threshold', y='value',hue='variable',
                     data=pd.melt(plot_rate_df, ['Correlation threshold']),
                     palette=['#ED7D31', '#4472C4'], marker='o', markersize=10,
                     linewidth=2.5)
        plt.ylabel('Ratio')
        plt.legend(frameon=False)
        plt.subplots_adjust(bottom=0.3, left=0.2)
        plt.savefig(save_path + 'proportion_homo_gene_pairs.' + fig_format, format=fig_format)

        fig, ax = plt.subplots(figsize=(10, 8), dpi=fig_dpi)
        plot_rate_df_1 = pd.DataFrame.from_dict(plot_dict_1)
        sns.lineplot(data = plot_rate_df_1, x='Correlation threshold', y='Ratio of homo-pairs',
                     color='black', marker='o', markersize=10,
                     linewidth=2.5)
        #plt.ylabel('Ratio')
        #plt.legend(frameon=False)
        plt.ticklabel_format(style='sci', scilimits=(0, 0))
        plt.subplots_adjust(bottom=0.3, left=0.2)
        plt.savefig(save_path + 'proportion_homo_pairs.' + fig_format, format=fig_format)

        adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
        print(adata_embedding)
        sc.pp.neighbors(adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(adata_embedding)

        adata_mouse_embedding = adata_embedding[adata_embedding.obs['dataset'] == 'Mouse']
        adata_human_embedding = adata_embedding[adata_embedding.obs['dataset'] == 'Human']

        markersize = 10

        # Take some examples to check the biological meaning of gene correlations
        mouse_marker_gene_dict = {'Agrp': 'Hypothalamus', 'Rora': 'Thalamus'}  # 'Gpr88':'Striatum',
        # cortical: Rasgrf2
        top_gene_num = 3

        for marker_gene, parent_region_name in mouse_marker_gene_dict.items():

            save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/2_experiment_marker_genes/'+f'{marker_gene}/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            marker_gene_index = adata_mouse_gene_embedding.obs_names.tolist().index(marker_gene)
            marker_gene_row = mh_mat[marker_gene_index, :]
            homo_marker_gene_human_index = np.nonzero(marker_gene_row)
            #if len(homo_marker_gene_human_index) == 0:
            corr_mat_row = corr_mat[marker_gene_index, :]
            correlated_set_index = list(np.argsort(corr_mat_row)[::-1][:top_gene_num])
            correlated_set = [adata_human_gene_embedding.obs_names.values[x] for x in correlated_set_index]
            print(marker_gene,'The most correlated gene set')
            print(correlated_set)
            #else:
            print(marker_gene, 'Homologous gene set')
            homologous_set = [adata_human_gene_embedding.obs_names.values[x] for x in homo_marker_gene_human_index]
            print(homologous_set[0])
            homologous_gene = homologous_set[0][0]

            #adata_mouse_embedding.obs[marker_gene] = normalize(
            #    adata_mouse_expression[:, marker_gene].X.toarray().reshape(-1)[:, np.newaxis], norm='max', axis=0).ravel()
            adata_mouse_embedding.obs[marker_gene] = adata_mouse_expression[:, marker_gene].X.toarray().reshape(-1)
            with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
                ax = sc.pl.umap(adata_mouse_embedding, show=False)
                human_parent_region_name = parent_region_name.lower()
                sc.pl.umap(adata_mouse_embedding[
                               adata_mouse_embedding.obs["region_name"].isin([parent_region_name])],
                           color='region_name',
                           palette={parent_region_name: mouse_64_color_dict[parent_region_name]},
                           ax=ax, legend_loc='on data', legend_fontweight='normal', show=False, size=markersize)
                plt.title('')
                plt.savefig(
                    save_path + f'umap_mouse_{parent_region_name}.' + fig_format, format=fig_format)
            with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
                sc.pl.umap(adata_mouse_embedding,color=marker_gene, color_map='viridis_r',legend_loc='on data', legend_fontweight='normal', show=False, size=markersize)
                plt.savefig(
                    save_path + f'umap_mouse_{parent_region_name}_{marker_gene}_expression.' + fig_format,
                    format=fig_format)

            #adata_human_embedding.obs[homologous_gene] = normalize(
            #    adata_human_expression[:, homologous_gene].X.toarray().reshape(-1)[:, np.newaxis], norm='max', axis=0).ravel()
            adata_human_embedding.obs[homologous_gene] = adata_human_expression[:, homologous_gene].X.toarray().reshape(-1)
            with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
                ax = sc.pl.umap(adata_human_embedding, show=False)
                human_parent_region_name = parent_region_name.lower()
                sc.pl.umap(adata_human_embedding[
                               adata_human_embedding.obs["region_name"].isin([human_parent_region_name])],
                           color='region_name',
                           palette={human_parent_region_name: human_88_color_dict[human_parent_region_name]},
                           ax=ax, legend_loc='on data',legend_fontweight='normal', show=False, size=markersize)
                plt.title('')
                plt.savefig(
                    save_path + f'umap_human_{human_parent_region_name}.' + fig_format,
                    format=fig_format)
            with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
                sc.pl.umap(adata_human_embedding,color=homologous_gene, color_map='viridis_r',legend_loc='on data', legend_fontweight='normal',show=False, size=markersize)
                plt.savefig(
                    save_path + f'umap_human_{human_parent_region_name}_{homologous_gene}_homologous_gene.' + fig_format,
                    format=fig_format)

            for correlated_gene in correlated_set:
                adata_human_embedding.obs[correlated_gene] = adata_human_expression[:, correlated_gene].X.toarray().reshape(-1)
                with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
                    sc.pl.umap(adata_human_embedding, color=correlated_gene, color_map='viridis_r', legend_loc='on data', legend_fontweight='normal',
                               show=False, size=markersize)
                    plt.savefig(
                        save_path + f'umap_human_{human_parent_region_name}_{correlated_gene}_correlated_gene.' + fig_format,
                        format=fig_format)


            rcParams["figure.subplot.left"] = 0.05
            rcParams["figure.subplot.right"] = 0.66
            with plt.rc_context({"figure.figsize": (12, 8), "figure.dpi":(fig_dpi)}):
                ax = sc.pl.umap(adata_embedding, show=False)
                human_parent_region_name = parent_region_name.lower()
                ax = sc.pl.umap(adata_embedding[adata_embedding.obs["region_name"].isin([parent_region_name, human_parent_region_name])], color='region_name',
                    palette={parent_region_name:mouse_64_color_dict[parent_region_name], human_parent_region_name:human_88_color_dict[human_parent_region_name]},
                    ax=ax, legend_loc='right margin', legend_fontweight='normal', show=False, title='')
                #plt.title('')
                #ax.legend(['mouse '+parent_region_name, 'human '+human_parent_region_name], loc = 'lower right')
                plt.savefig(save_path + f'umap_{parent_region_name}_{human_parent_region_name}_expression_together.' + fig_format, format=fig_format)
            rcParams["figure.subplot.left"] = 0.05
            rcParams["figure.subplot.right"] = 0.9

        mouse_marker_gene_dict = {'Gpr88':'Striatum'}  # 'Gpr88':'Striatum',
        # cortical: Rasgrf2
        top_gene_num = 3

        for marker_gene, parent_region_name in mouse_marker_gene_dict.items():

            save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/2_experiment_marker_genes/' + f'{marker_gene}/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            marker_gene_index = adata_mouse_gene_embedding.obs_names.tolist().index(marker_gene)
            marker_gene_row = mh_mat[marker_gene_index, :]
            homo_marker_gene_human_index = np.nonzero(marker_gene_row)
            # if len(homo_marker_gene_human_index) == 0:
            corr_mat_row = corr_mat[marker_gene_index, :]
            correlated_set_index = list(np.argsort(corr_mat_row)[::-1][:top_gene_num])
            correlated_set = [adata_human_gene_embedding.obs_names.values[x] for x in correlated_set_index]
            print(marker_gene, 'The most correlated gene set')
            print(correlated_set)
            # else:
            print(marker_gene, 'Homologous gene set')
            homologous_set = [adata_human_gene_embedding.obs_names.values[x] for x in homo_marker_gene_human_index]
            print(homologous_set[0])
            homologous_gene = homologous_set[0][0]

            # adata_mouse_embedding.obs[marker_gene] = normalize(
            #    adata_mouse_expression[:, marker_gene].X.toarray().reshape(-1)[:, np.newaxis], norm='max', axis=0).ravel()
            adata_mouse_embedding.obs[marker_gene] = adata_mouse_expression[:, marker_gene].X.toarray().reshape(-1)
            with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
                ax = sc.pl.umap(adata_mouse_embedding, show=False)
                human_parent_region_name = parent_region_name.lower()
                sc.pl.umap(adata_mouse_embedding[
                               adata_mouse_embedding.obs["parent_region_name"].isin([parent_region_name])],
                           color='parent_region_name',
                           palette={parent_region_name: mouse_15_color_dict[parent_region_name]},
                           ax=ax, legend_loc='on data', legend_fontweight='normal', show=False, size=markersize)
                plt.title('')
                plt.savefig(
                    save_path + f'umap_mouse_{parent_region_name}.' + fig_format,
                    format=fig_format)
            with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
                sc.pl.umap(adata_mouse_embedding, color=marker_gene, color_map='viridis_r', legend_loc='on data',
                           legend_fontweight='normal', show=False, size=markersize)
                plt.savefig(
                    save_path + f'umap_mouse_{parent_region_name}_{marker_gene}_expression.' + fig_format,
                    format=fig_format)

            # adata_human_embedding.obs[homologous_gene] = normalize(
            #    adata_human_expression[:, homologous_gene].X.toarray().reshape(-1)[:, np.newaxis], norm='max', axis=0).ravel()
            adata_human_embedding.obs[homologous_gene] = adata_human_expression[:, homologous_gene].X.toarray().reshape(
                -1)
            rcParams["figure.subplot.left"] = 0.05
            rcParams["figure.subplot.right"] = 0.66
            with plt.rc_context({"figure.figsize": (16, 8), "figure.dpi":(fig_dpi)}):
                sc.pl.umap(adata_human_embedding,
                           color='region_name',
                           palette=human_88_color_dict, legend_loc='right margin', legend_fontweight='normal', show=False, size=markersize)
                plt.title('')
                plt.savefig(save_path + f'umap_human_all_regions.' + fig_format, format=fig_format)
            rcParams["figure.subplot.left"] = 0.05
            rcParams["figure.subplot.right"] = 0.9

            rcParams["figure.subplot.left"] = 0.05
            rcParams["figure.subplot.right"] = 0.45
            with plt.rc_context({"figure.figsize": (16, 8), "figure.dpi":(fig_dpi)}):
                sc.pl.umap(adata_human_embedding,
                           color='parent_region_name',
                           palette=human_16_color_dict, legend_loc='right margin', legend_fontweight='normal',
                           show=False, size=markersize)
                plt.title('')
                plt.savefig(save_path + f'umap_human_all_regions.' + fig_format, format=fig_format)
            rcParams["figure.subplot.left"] = 0.05
            rcParams["figure.subplot.right"] = 0.9

            with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
                sc.pl.umap(adata_human_embedding, color=homologous_gene, color_map='viridis_r', legend_loc='on data',
                           legend_fontweight='normal', show=False, size=markersize)
                plt.savefig(
                    save_path + f'umap_human_{human_parent_region_name}_{homologous_gene}_homologous_gene.' + fig_format,
                    format=fig_format)

            for correlated_gene in correlated_set:
                adata_human_embedding.obs[correlated_gene] = adata_human_expression[:, correlated_gene].X.toarray().reshape(-1)
                with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
                    sc.pl.umap(adata_human_embedding, color=correlated_gene, color_map='viridis_r', legend_loc='on data',
                               legend_fontweight='normal',
                               show=False, size=markersize)
                    plt.savefig(
                        save_path + f'umap_human_{correlated_gene}_correlated_gene.' + fig_format,
                        format=fig_format)

        return None

    def experiment_3_1_cluster_rename(adata_mouse_embedding, adata_human_embedding):
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
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        # 1. Identify those differentially expressed genes for regions and clusters, and save them.
        # 2. How these genes are correlated to each other - Cluster those genes or get gene modules
        # 3. clustering of samples, do GO analysis to determine enrichment of genes in each region or clusters;
        # 4. dotplot of clusters and homologous regions, go clusters.
        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/3_experiment_degs_analysis/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        #adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
        adata_embedding = sc.read_h5ad('')
        print(adata_embedding)

        sc.pp.neighbors(adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(adata_embedding, n_components=5)

        # PCA and kmeans clustering of the whole dataset
        sc.tl.pca(adata_embedding, svd_solver='arpack', n_comps=30)

        X_pca = adata_embedding.obsm['X_pca']
        X_umap = adata_embedding.obsm['X_umap']

        # Do clustering
        clustering_name = f'leiden_cluster'
        # kmeans = KMeans(n_clusters=clustering_num, random_state=29).fit(X_pca)
        # kmeans = KMeans(n_clusters=clustering_num, random_state=29).fit(X)

        # adata_embedding.obs[clustering_name] = kmeans.labels_.astype(str)

        #sc.tl.leiden(adata_embedding, resolution=9, key_added=clustering_name)
        clustering_num = len(Counter(adata_embedding.obs[clustering_name]))

        cluster_labels_unique = [str(x) for x in range(clustering_num)]
        #kmeans = KMeans(n_clusters=clustering_num, random_state=29).fit(X_umap)

        #adata_embedding.obs[clustering_name] = kmeans.labels_.astype(str)

        # name those clusters via parent regions
        # name some cluster the region name if the proportion of the samples of the region in this cluster exceeds 50%
        # (50% for parent region name, 30% for region name, the ratio is calculated in each species),
        # k-mouse parent region name-human parent region name
        # or name the cluster via k-mixed(most parent region name of mouse)-mixed(most parent region name of human)
        cluster_names_dict = {k:v for k,v in zip(adata_embedding.obs_names, adata_embedding.obs[clustering_name])}
        cluster_names_acronym_dict = {k: v for k, v in zip(adata_embedding.obs_names, adata_embedding.obs[clustering_name])}

        parent_cluster_names_dict = {k: v for k, v in zip(adata_embedding.obs_names, adata_embedding.obs[clustering_name])}
        parent_cluster_names_acronym_dict = {k: v for k, v in zip(adata_embedding.obs_names, adata_embedding.obs[clustering_name])}

        parent_region_proportion = 0.5
        region_proportion = 0.3
        for c_label in cluster_labels_unique:
            c_label_adata = adata_embedding[adata_embedding.obs[clustering_name].isin([c_label])]
            # create parent region names
            mouse_adata = c_label_adata[c_label_adata.obs['dataset'].isin(['Mouse'])]
            mouse_parent_region_Counter = Counter(mouse_adata.obs['parent_region_name'])
            if len(mouse_parent_region_Counter) == 0:
                mouse_parent_region_mc = 'None'
                mouse_parent_region_rate = 0
            else:
                mouse_parent_region_mc = mouse_parent_region_Counter.most_common(1)[0][0]
                mouse_parent_region_rate = mouse_parent_region_Counter[mouse_parent_region_mc] / mouse_adata.n_obs
            #mouse_parent_region_mc = mouse_parent_region_Counter.most_common(1)[0][0]
            #mouse_parent_region_rate = mouse_parent_region_Counter[mouse_parent_region_mc] / mouse_adata.n_obs

            if mouse_parent_region_rate >= parent_region_proportion:
                mouse_parent_region_string = mouse_parent_region_mc
                mouse_parent_region_string_acronym = mouse_15_acronym_dict[mouse_parent_region_mc]
            elif mouse_parent_region_rate < parent_region_proportion and mouse_parent_region_rate > 0:
                mouse_parent_region_string = f'mixed({mouse_parent_region_mc})'
                mouse_parent_region_string_acronym = f'mixed({mouse_15_acronym_dict[mouse_parent_region_mc]})'
            elif mouse_parent_region_rate == 0:
                mouse_parent_region_string = mouse_parent_region_mc
                mouse_parent_region_string_acronym = mouse_parent_region_mc

            human_adata = c_label_adata[c_label_adata.obs['dataset'].isin(['Human'])]
            human_parent_region_Counter = Counter(human_adata.obs['parent_region_name'])
            if len(human_parent_region_Counter) == 0:
                human_parent_region_mc = 'None'
                human_parent_region_rate = 0
            else:
                human_parent_region_mc = human_parent_region_Counter.most_common(1)[0][0]
                human_parent_region_rate = human_parent_region_Counter[human_parent_region_mc] / human_adata.n_obs

            if human_parent_region_rate >= parent_region_proportion:
                human_parent_region_string = human_parent_region_mc
                human_parent_region_string_acronym = human_16_acronym_dict[human_parent_region_mc]
            elif human_parent_region_rate < parent_region_proportion and human_parent_region_rate > 0:
                human_parent_region_string = f'mixed({human_parent_region_mc})'
                human_parent_region_string_acronym = f'mixed({human_16_acronym_dict[human_parent_region_mc]})'
            elif human_parent_region_rate == 0:
                human_parent_region_string = human_parent_region_mc
                human_parent_region_string_acronym = human_parent_region_mc

            parent_cluster_name = c_label + '-' + mouse_parent_region_string + '-' + human_parent_region_string
            parent_cluster_name_acronym = c_label + '-' + mouse_parent_region_string_acronym + '-' + human_parent_region_string_acronym

            for sample_index in c_label_adata.obs_names:
                parent_cluster_names_dict[sample_index] = parent_cluster_name
                parent_cluster_names_acronym_dict[sample_index] = parent_cluster_name_acronym

            print(c_label, f'parent cluster name = {parent_cluster_name}', f'parent cluster name acronym = {parent_cluster_name_acronym}')

            # create region names
            mouse_region_Counter = Counter(mouse_adata.obs['region_name'])
            if len(mouse_region_Counter) == 0:
                mouse_region_mc = 'None'
                mouse_region_rate = 0
            else:
                mouse_region_mc = mouse_region_Counter.most_common(1)[0][0]
                mouse_region_rate = mouse_region_Counter[mouse_region_mc] / mouse_adata.n_obs

            if mouse_region_rate >= region_proportion:
                mouse_region_string = mouse_region_mc
                mouse_region_string_acronym = mouse_64_acronym_dict[mouse_region_mc]
            elif mouse_region_rate < region_proportion and mouse_region_rate > 0:
                mouse_region_string = f'mixed({mouse_region_mc})'
                mouse_region_string_acronym = f'mixed({mouse_64_acronym_dict[mouse_region_mc]})'
            elif mouse_region_rate == 0:
                mouse_region_string = mouse_region_mc
                mouse_region_string_acronym = mouse_region_mc

            human_region_Counter = Counter(human_adata.obs['region_name'])
            if len(human_region_Counter) == 0:
                human_region_mc = 'None'
                human_region_rate = 0
            else:
                human_region_mc = human_region_Counter.most_common(1)[0][0]
                human_region_rate = human_region_Counter[human_region_mc] / human_adata.n_obs

            if human_region_rate >= region_proportion:
                human_region_string = human_region_mc
                human_region_string_acronym = human_88_acronym_dict[human_region_mc]
            elif human_region_rate < region_proportion and human_region_rate > 0:
                human_region_string = f'mixed({human_region_mc})'
                human_region_string_acronym = f'mixed({human_88_acronym_dict[human_region_mc]})'
            elif human_region_rate == 0:
                human_region_string = human_region_mc
                human_region_string_acronym = human_region_mc

            cluster_name = c_label + '-' + mouse_region_string + '-' + human_region_string
            cluster_name_acronym = c_label + '-' + mouse_region_string_acronym + '-' + human_region_string_acronym

            for sample_index in c_label_adata.obs_names:
                cluster_names_dict[sample_index] = cluster_name
                cluster_names_acronym_dict[sample_index] = cluster_name_acronym

            print(c_label, f'cluster name = {cluster_name}', f'cluster name acronym = {cluster_name_acronym}')

        adata_embedding.obs['parent_cluster_name'] = cluster_names_dict.values()
        adata_embedding.obs['parent_cluster_name_acronym'] = cluster_names_acronym_dict.values()
        adata_embedding.obs['cluster_name'] = cluster_names_dict.values()
        adata_embedding.obs['cluster_name_acronym'] = cluster_names_acronym_dict.values()

        adata_mouse_embedding = adata_embedding[adata_embedding.obs['dataset'] == 'Mouse']
        adata_human_embedding = adata_embedding[adata_embedding.obs['dataset'] == 'Human']

        adata_mouse_embedding.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/s_embeddings.h5ad')
        adata_human_embedding.write_h5ad(cfg.BrainAlign.embeddings_file_path + '/v_embeddings.h5ad')

        adata_mouse_embedding.obs['x_grid'] = adata_mouse_expression.obs['x_grid']
        adata_mouse_embedding.obs['y_grid'] = adata_mouse_expression.obs['y_grid']
        adata_mouse_embedding.obs['z_grid'] = adata_mouse_expression.obs['z_grid']

        adata_mouse_expression.obs = adata_mouse_embedding.obs

        adata_human_embedding.obs['mri_voxel_x'] = adata_human_expression.obs['mri_voxel_x']
        adata_human_embedding.obs['mri_voxel_y'] = adata_human_expression.obs['mri_voxel_y']
        adata_human_embedding.obs['mri_voxel_z'] = adata_human_expression.obs['mri_voxel_z']

        adata_human_expression.obs = adata_human_embedding.obs

        adata_mouse_expression.write_h5ad(cfg.CAME.path_rawdata1)
        adata_human_expression.write_h5ad(cfg.CAME.path_rawdata2)
        return None

    def deg_plot(adata, ntop_genes, ntop_genes_visual, save_path_parent, groupby, save_pre, ifmouse=True):
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
            sc.tl.filter_rank_genes_groups(adata, key = "wilcoxon", key_added= "wilcoxon",
                                           min_in_group_fraction=0.1,
                                           max_out_group_fraction=0.9,
                                           min_fold_change=5)
        else:
            #pass
            sc.tl.filter_rank_genes_groups(adata, key="wilcoxon", key_added="wilcoxon_filtered",
                                           min_fold_change=2)
        plt.savefig(save_path_parent + save_pre + 'degs.' + fig_format, format=fig_format)

        rcParams["figure.subplot.top"] = 0.8
        rcParams["figure.subplot.bottom"] = 0.2
        rcParams["figure.subplot.left"] = 0.2
        #with plt.rc_context({"figure.figsize": (12, 16)}):
        sc.pl.rank_genes_groups_heatmap(adata, n_genes=ntop_genes_visual, key="wilcoxon",
                                    groupby=groupby, show_gene_labels=True, show=False, dendrogram=True, figsize=(20,24))
        plt.savefig(save_path_parent + save_pre+'_heatmap.' + fig_format, format=fig_format)
        #with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_dotplot(adata, n_genes=ntop_genes_visual, key="wilcoxon",
                                        groupby=groupby, show=False, dendrogram=True, figsize=(20,20))
        plt.savefig(save_path_parent + save_pre + '_dotplot.' + fig_format, format=fig_format)
        #with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_stacked_violin(adata, n_genes=ntop_genes_visual, key="wilcoxon",
                                               groupby=groupby, show=False,dendrogram=True, figsize=(20,20))
        plt.savefig(save_path_parent + save_pre + '_violin.' + fig_format, format=fig_format)

        #with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_matrixplot(adata, n_genes=ntop_genes_visual, key="wilcoxon",
                                           groupby=groupby, show=False, dendrogram=True, figsize=(20,20))
        plt.savefig(save_path_parent + save_pre + '_matrixplot.' + fig_format, format=fig_format)

    def experiment_3_2_deg_analysis():

        save_path_parent = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/3_experiment_degs_analysis/parent_region_cluster/'
        if not os.path.exists(save_path_parent):
            os.makedirs(save_path_parent)

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/3_experiment_degs_analysis/region_cluster/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path_parent_region = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/3_experiment_degs_analysis/parent_region/'
        if not os.path.exists(save_path_parent_region):
            os.makedirs(save_path_parent_region)

        save_path_region = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/3_experiment_degs_analysis/region/'
        if not os.path.exists(save_path_region):
            os.makedirs(save_path_region)

        adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        adata_mouse_expression = sc.read_h5ad(cfg.CAME.path_rawdata1)
        for obs1, obs2 in zip(adata_mouse_embedding.obs_names, adata_mouse_expression.obs_names):
            if obs1 != obs2:
                print('Sample name not aligned!')

        adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
        adata_human_expression = sc.read_h5ad(cfg.CAME.path_rawdata2)
        for obs1, obs2 in zip(adata_human_embedding.obs_names, adata_human_expression.obs_names):
            if obs1 != obs2:
                print('Sample name not aligned!')

        # mouse parent region degs
        ntop_genes = 5

        ntop_genes_visual = 1


        deg_plot(adata_mouse_expression, ntop_genes, ntop_genes_visual, save_path_parent,
                 groupby='parent_cluster_name_acronym', save_pre='mouse_degs_parent')

        deg_plot(adata_mouse_expression, ntop_genes, ntop_genes_visual, save_path_parent_region,
                 groupby='parent_acronym', save_pre='mouse_degs_parent')


        # human parent region degs
        # sc.tl.rank_genes_groups(adata_human_expression, 'parent_cluster_name_acronym', method='wilcoxon', key_added="wilcoxon")
        # sc.pl.rank_genes_groups(adata_human_expression, n_genes=5, sharey=False, key="wilcoxon",
        #                         show=False)
        #plt.savefig(save_path_parent + 'human_degs_parent_region.' + fig_format, format=fig_format)
        deg_plot(adata_human_expression, ntop_genes, ntop_genes_visual, save_path_parent,
                 groupby='parent_cluster_name_acronym', save_pre='human_degs_parent', ifmouse=False)

        deg_plot(adata_human_expression, ntop_genes, ntop_genes_visual, save_path_parent_region,
                 groupby='parent_acronym', save_pre='human_degs_parent', ifmouse=False)


        # mouse region degs
        ntop_genes = 5
        deg_plot(adata_mouse_expression, ntop_genes, ntop_genes_visual, save_path,
                 groupby='cluster_name_acronym', save_pre='mouse_degs')
        deg_plot(adata_mouse_expression, ntop_genes, ntop_genes_visual, save_path_region,
                 groupby='acronym', save_pre='mouse_degs')
        # human region degs
        deg_plot(adata_human_expression, ntop_genes, ntop_genes_visual, save_path,
                 groupby='cluster_name_acronym', save_pre='human_degs', ifmouse=False)
        deg_plot(adata_human_expression, ntop_genes, ntop_genes_visual, save_path_region,
                 groupby='acronym', save_pre='human_degs', ifmouse=False)
        # visualization
        #adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
        #print(adata_embedding)
        return None

    def experiment_3_3_deg_distribution():
        TINY_SIZE = 22  # 39
        SMALL_SIZE = 22  # 42
        MEDIUM_SIZE = 28  # 46
        BIGGER_SIZE = 28  # 46

        plt.rc('font', size=24)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path_parent = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/3_experiment_degs_distribution/parent_region_cluster/'
        if not os.path.exists(save_path_parent):
            os.makedirs(save_path_parent)

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/3_experiment_degs_distribution/region_cluster/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path_parent_region = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/3_experiment_degs_distribution/parent_region/'
        if not os.path.exists(save_path_parent_region):
            os.makedirs(save_path_parent_region)

        save_path_region = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/3_experiment_degs_distribution/region/'
        if not os.path.exists(save_path_region):
            os.makedirs(save_path_region)

        path_datapiar = cfg.CAME.ROOT + 'datapair_init.pickle'
        path_datapiar_file = open(path_datapiar, 'rb')
        datapair = pickle.load(path_datapiar_file)

        mouse_gene_list = datapair['varnames_node'][0]
        human_gene_list = datapair['varnames_node'][1]

        mouse_gene_set = set(mouse_gene_list)
        human_gene_set = set(human_gene_list)

        adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        adata_mouse_expression = sc.read_h5ad(cfg.CAME.path_rawdata1)#[:, mouse_gene_list]
        print(adata_mouse_expression)
        for obs1, obs2 in zip(adata_mouse_embedding.obs_names, adata_mouse_expression.obs_names):
            if obs1 != obs2:
                print('Sample name not aligned!')

        adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
        adata_human_expression = sc.read_h5ad(cfg.CAME.path_rawdata2)#[:, human_gene_list]
        print(adata_human_expression)
        for obs1, obs2 in zip(adata_human_embedding.obs_names, adata_human_expression.obs_names):
            if obs1 != obs2:
                print('Sample name not aligned!')

        # mouse cluster
        ntop_gene = 50
        cluster_name_acronym_list = list(set(adata_mouse_expression.obs['cluster_name_acronym'].tolist()))
        cluster_name_acronym_list.sort(key=lambda x:int(x.split('-')[0]))

        parent_cluster_dict = {k: v for k, v in zip(adata_human_expression.obs['parent_cluster_name'].tolist(),
                                                    [x.replace(' ', '_') for x in adata_human_expression.obs[
                                                        'parent_cluster_name_acronym'].tolist()])}

        #
        sc.tl.rank_genes_groups(adata_mouse_expression, groupby='cluster_name_acronym', method='wilcoxon', n_genes=ntop_gene,
                                key_added="wilcoxon")
        sc.tl.rank_genes_groups(adata_human_expression, groupby='cluster_name_acronym', method='wilcoxon', n_genes=ntop_gene,
                                key_added="wilcoxon")
        gene_name_mouse_list = adata_mouse_expression.var_names.tolist()
        gene_name_human_list = adata_human_expression.var_names.tolist()

        gene_stacked_df = {}
        gene_num_stacked_df = {'Mouse specialized':[], 'Mouse common':[], 'Human common':[], 'Human specialized':[], 'Cluster':[]}

        for cluster_name_acronym in cluster_name_acronym_list:
            gene_stacked_df[cluster_name_acronym] = {}
            mouse_degs_list = sc.get.rank_genes_groups_df(adata_mouse_expression,
                                                          group=cluster_name_acronym,
                                                          key='wilcoxon',
                                                          log2fc_min=0.25,
                                                          pval_cutoff=1e-3)['names'].squeeze().str.strip().tolist()
            mouse_degs_list = list(mouse_gene_set.intersection(set(mouse_degs_list)))
            human_degs_list = sc.get.rank_genes_groups_df(adata_human_expression,
                                                          group=cluster_name_acronym,
                                                          key='wilcoxon',
                                                          log2fc_min=0.25,
                                                          pval_cutoff=1e-3)['names'].squeeze().str.strip().tolist()
            human_degs_list = list(human_gene_set.intersection(set(human_degs_list)))
            gene_list_mouse_special, gene_list_mouse_common, gene_list_human_common, gene_list_human_special = \
                get_common_special_gene_list(mouse_degs_list, human_degs_list,mouse_gene_list, human_gene_list,
                                     mh_mat, Var_Corr.values)
            gene_stacked_df[cluster_name_acronym]['Mouse specialized'] = gene_list_mouse_special
            gene_stacked_df[cluster_name_acronym]['Mouse common'] = gene_list_mouse_common
            gene_stacked_df[cluster_name_acronym]['Human common'] = gene_list_human_common
            gene_stacked_df[cluster_name_acronym]['Human specialized'] = gene_list_human_special

            # update stacked bar data
            gene_num_stacked_df['Mouse specialized'].append(len(gene_list_mouse_special))
            gene_num_stacked_df['Mouse common'].append(len(gene_list_mouse_common))
            gene_num_stacked_df['Human common'].append(len(gene_list_human_common))
            gene_num_stacked_df['Human specialized'].append(len(gene_list_human_special))
            gene_num_stacked_df['Cluster'].append(cluster_name_acronym)
        #print(adata_mouse_expression.uns)
        #sc.pl.rank_genes_groups(adata_mouse_expression, n_genes=ntop_genes, sharey=False, key="wilcoxon", show=False)
        gene_stacked_df = pd.DataFrame.from_dict(gene_stacked_df)
        gene_stacked_df.to_csv(save_path + 'gene_stacked_special_common.csv')
        gene_num_stacked_df = pd.DataFrame.from_dict(gene_num_stacked_df)
        gene_stacked_df.to_csv(save_path + 'gene_stacked_special_common_number.csv')
        colors = [mouse_color, '#8B1C62', '#008B00', human_color]
        rcParams["figure.subplot.bottom"] = 0.35
        rcParams["figure.subplot.right"] = 0.7
        ax=gene_num_stacked_df.plot(x='Cluster', kind='bar', stacked=True, title='DEGs comparison of clusters',
                                 color=colors, figsize=(24,9), rot=60, width=0.9)
        ha_adjust = 'right'
        plt.setp(ax.xaxis.get_majorticklabels(), ha=ha_adjust, rotation_mode="anchor")
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
        plt.ylabel('Gene number')
        plt.savefig(save_path + 'gene_stacked_special_common_number.svg', format=fig_format)


        #plt.savefig(save_path_parent + save_pre + 'degs.' + fig_format, format=fig_format)

    def experiment_4_gene_ontology_analysis():
        TINY_SIZE = 16  # 39
        SMALL_SIZE = 22  # 42
        MEDIUM_SIZE = 24  # 46
        BIGGER_SIZE = 26  # 46

        plt.rc('font', size=18)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path_parent = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/4_experiment_gene_ontology_analysis/parent_region_cluster/'
        if not os.path.exists(save_path_parent):
            os.makedirs(save_path_parent)

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/4_experiment_gene_ontology_analysis/region_cluster/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        adata_mouse_expression = sc.read_h5ad(cfg.CAME.path_rawdata1)
        for obs1, obs2 in zip(adata_mouse_embedding.obs_names, adata_mouse_expression.obs_names):
            if obs1 != obs2:
                print('Sample name not aligned!')

        adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
        adata_human_expression = sc.read_h5ad(cfg.CAME.path_rawdata2)
        for obs1, obs2 in zip(adata_human_embedding.obs_names, adata_human_expression.obs_names):
            if obs1 != obs2:
                print('Sample name not aligned!')

        ntop_gene = 10

        ####################################################
        # human parent cluster
        sc.tl.rank_genes_groups(adata_human_expression, groupby='cluster_name', method='wilcoxon',
                                key_added="wilcoxon")

        cluster_dict = {k: v for k, v in zip(adata_human_expression.obs['cluster_name'].tolist(),
                                             [x.replace(' ', '_') for x in adata_human_expression.obs[
                                                 'cluster_name_acronym'].tolist()])}

        for cluster_name in list(set(adata_human_expression.obs['cluster_name'].tolist())):
            sc.pl.rank_genes_groups(adata_human_expression, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                    show=False)
            print(adata_human_expression)
            glist_human_cluster = sc.get.rank_genes_groups_df(adata_human_expression,
                                                              group=cluster_name,
                                                              key='wilcoxon',
                                                              log2fc_min=0.25,
                                                              pval_cutoff=0.01)['names'].squeeze().str.strip().tolist()

            enr_res = gseapy.enrichr(gene_list=glist_human_cluster,
                                     organism='Human',
                                     gene_sets=['Allen_Brain_Atlas_down', 'Allen_Brain_Atlas_up',
                                                'Azimuth_Cell_Types_2021', 'CellMarker_Augmented_2021',
                                                'GO_Biological_Process_2021', 'GO_Cellular_Component_2021',
                                                'GO_Molecular_Function_2021', 'Human_Gene_Atlas'],
                                     cutoff=0.5)
            gseapy.barplot(enr_res.results, column="Adjusted P-value", group='Gene_set', color=human_color,
                           # set group, so you could do a multi-sample/library comparsion
                           size=5, title=f"GO of human cluster {cluster_name} DEGs", figsize=(12, 15),
                           ofname=save_path + f'human_{cluster_dict[cluster_name]}_barplot.' + fig_format)
            gseapy.dotplot(enr_res.results,
                           column="Adjusted P-value",
                           x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion
                           size=5,
                           top_term=5,  #
                           figsize=(12, 12),
                           title=f"GO of human cluster {cluster_name} DEGs",
                           xticklabels_rot=45,  # rotate xtick labels show_ring=True, set to False to revmove outer ring
                           marker='o',
                           ofname=save_path + f'human_{cluster_dict[cluster_name]}_botplot.' + fig_format,
                           format=fig_format)

        # mouse parent cluster
        sc.tl.rank_genes_groups(adata_mouse_expression, groupby='cluster_name', method='wilcoxon',
                                key_added="wilcoxon")

        cluster_dict = {k: v for k, v in zip(adata_mouse_expression.obs['cluster_name'].tolist(),
                                             [x.replace(' ', '_') for x in adata_mouse_expression.obs[
                                                 'cluster_name_acronym'].tolist()])}

        for cluster_name in list(set(adata_mouse_expression.obs['cluster_name'].tolist())):
            sc.pl.rank_genes_groups(adata_mouse_expression, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                    show=False)
            print(adata_mouse_expression)
            glist_mouse_cluster = sc.get.rank_genes_groups_df(adata_mouse_expression,
                                                              group=cluster_name,
                                                              key='wilcoxon',
                                                              log2fc_min=0.25,
                                                              pval_cutoff=0.01)[
                'names'].squeeze().str.strip().tolist()

            enr_res = gseapy.enrichr(gene_list=glist_mouse_cluster,
                                     organism='Mouse',
                                     gene_sets=['Allen_Brain_Atlas_down',
                                                'Allen_Brain_Atlas_up',
                                                'Azimuth_Cell_Types_2021',
                                                'CellMarker_Augmented_2021',
                                                'GO_Biological_Process_2021',
                                                'GO_Cellular_Component_2021',
                                                'GO_Molecular_Function_2021',
                                                'Mouse_Gene_Atlas'], cutoff=0.5)
            gseapy.barplot(enr_res.results, column="Adjusted P-value", group='Gene_set',
                           # set group, so you could do a multi-sample/library comparsion
                           size=5, title=f"GO of mouse cluster {cluster_name} DEGs", figsize=(12, 15),
                           color=mouse_color,
                           ofname=save_path + f'mouse_{cluster_dict[cluster_name]}_barplot.' + fig_format)
            gseapy.dotplot(enr_res.results,
                           column="Adjusted P-value",
                           x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion
                           size=5,
                           top_term=5,  #
                           figsize=(12, 12),
                           title=f"GO of mouse cluster {cluster_name} DEGs",
                           xticklabels_rot=45,
                           # rotate xtick labels  show_ring=True, set to False to revmove outer ring
                           marker='o',
                           ofname=save_path + f'mouse_{cluster_dict[cluster_name]}_botplot.' + fig_format,
                           format=fig_format)

        ####################################################################
        # human parent cluster
        sc.tl.rank_genes_groups(adata_human_expression, groupby='parent_cluster_name', method='wilcoxon',
                                key_added="wilcoxon")

        parent_cluster_dict = {k: v for k, v in zip(adata_human_expression.obs['parent_cluster_name'].tolist(),
                                                    [x.replace(' ', '_') for x in adata_human_expression.obs[
                                                        'parent_cluster_name_acronym'].tolist()])}

        for parent_cluster_name in list(set(adata_human_expression.obs['parent_cluster_name'].tolist())):
            sc.pl.rank_genes_groups(adata_human_expression, n_genes=ntop_gene, sharey=False, key="wilcoxon", show=False)
            print(adata_human_expression)
            glist_human_parent_cluster = sc.get.rank_genes_groups_df(adata_human_expression,
                                                                     group=parent_cluster_name,
                                                                     key='wilcoxon',
                                                                     log2fc_min=0.25,
                                                                     pval_cutoff=0.01)['names'].squeeze().str.strip().tolist()

            enr_res = gseapy.enrichr(gene_list=glist_human_parent_cluster,
                                     organism='Human',
                                     gene_sets=['Allen_Brain_Atlas_down', 'Allen_Brain_Atlas_up',
                                                'Azimuth_Cell_Types_2021', 'CellMarker_Augmented_2021',
                                                'GO_Biological_Process_2021', 'GO_Cellular_Component_2021',
                                                'GO_Molecular_Function_2021', 'Human_Gene_Atlas'],
                                     cutoff=0.5)
            gseapy.barplot(enr_res.results, column="Adjusted P-value", group='Gene_set', color=human_color, # set group, so you could do a multi-sample/library comparsion
                           size=5, title=f"GO of human cluster {parent_cluster_name} DEGs", figsize=(12, 15),
                           ofname=save_path_parent + f'human_{parent_cluster_dict[parent_cluster_name]}_barplot.' + fig_format)
            gseapy.dotplot(enr_res.results,
                           column="Adjusted P-value",
                           x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion
                           size=5,
                           top_term=5,  #
                           figsize=(12, 12),
                           title=f"GO of human cluster {parent_cluster_name} DEGs",
                           xticklabels_rot=45,  # rotate xtick labels show_ring=True, set to False to revmove outer ring
                           marker='o',
                           ofname=save_path_parent + f'human_{parent_cluster_dict[parent_cluster_name]}_botplot.' + fig_format, format=fig_format)

        # mouse parent cluster
        sc.tl.rank_genes_groups(adata_mouse_expression, groupby='parent_cluster_name', method='wilcoxon',
                                key_added="wilcoxon")

        parent_cluster_dict = {k: v for k, v in zip(adata_mouse_expression.obs['parent_cluster_name'].tolist(),
                                                    [x.replace(' ', '_') for x in adata_mouse_expression.obs[
                                                        'parent_cluster_name_acronym'].tolist()])}

        for parent_cluster_name in list(set(adata_mouse_expression.obs['parent_cluster_name'].tolist())):
            sc.pl.rank_genes_groups(adata_mouse_expression, n_genes=ntop_gene, sharey=False, key="wilcoxon",
                                    show=False)
            print(adata_mouse_expression)
            glist_mouse_parent_cluster = sc.get.rank_genes_groups_df(adata_mouse_expression,
                                                                     group=parent_cluster_name,
                                                                     key='wilcoxon',
                                                                     log2fc_min=0.25,
                                                                     pval_cutoff=0.01)[
                'names'].squeeze().str.strip().tolist()

            enr_res = gseapy.enrichr(gene_list=glist_mouse_parent_cluster,
                                     organism='Mouse',
                                     gene_sets=['Allen_Brain_Atlas_down',
                                                'Allen_Brain_Atlas_up',
                                                'Azimuth_Cell_Types_2021',
                                                'CellMarker_Augmented_2021',
                                                'GO_Biological_Process_2021',
                                                'GO_Cellular_Component_2021',
                                                'GO_Molecular_Function_2021',
                                                'Mouse_Gene_Atlas'],
                                     cutoff=0.5)
            gseapy.barplot(enr_res.results,column="Adjusted P-value", group='Gene_set', # set group, so you could do a multi-sample/library comparsion
                size=5, title=f"GO of mouse cluster {parent_cluster_name} DEGs", figsize=(12, 15), color=mouse_color,
                           ofname=save_path_parent + f'mouse_{parent_cluster_dict[parent_cluster_name]}_barplot.' + fig_format)
            gseapy.dotplot(enr_res.results,
                           column="Adjusted P-value",
                           x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion
                           size=5,
                           top_term=5,  #
                           figsize=(12, 12),
                           title=f"GO of mouse cluster {parent_cluster_name} DEGs",
                           xticklabels_rot=45,  # rotate xtick labels  show_ring=True, set to False to revmove outer ring
                           marker='o',
                           ofname=save_path_parent + f'mouse_{parent_cluster_dict[parent_cluster_name]}_botplot.' + fig_format, format=fig_format)

        return None


    def experiment_5_homologous_genes_analysis():
        TINY_SIZE = 16  # 39
        SMALL_SIZE = 22  # 42
        MEDIUM_SIZE = 24  # 46
        BIGGER_SIZE = 26  # 46

        plt.rc('font', size=24)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path_parent = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/5_experiment_homologous_genes_analysis/parent_region_cluster/'
        if not os.path.exists(save_path_parent):
            os.makedirs(save_path_parent)

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/5_experiment_homologous_genes_analysis/region_cluster/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        adata_mouse_expression = sc.read_h5ad(cfg.CAME.path_rawdata1)
        for obs1, obs2 in zip(adata_mouse_embedding.obs_names, adata_mouse_expression.obs_names):
            if obs1 != obs2:
                print('Sample name not aligned!')

        adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
        adata_human_expression = sc.read_h5ad(cfg.CAME.path_rawdata2)
        for obs1, obs2 in zip(adata_human_embedding.obs_names, adata_human_expression.obs_names):
            if obs1 != obs2:
                print('Sample name not aligned!')

        # Get homologous degs
        # mouse cluster
        ntop_gene = 50

        cluster_name_acronym_list = list(set(adata_mouse_expression.obs['cluster_name'].tolist()))
        cluster_name_acronym_list.sort()
        #
        sc.tl.rank_genes_groups(adata_mouse_expression, groupby='cluster_name', n_genes=ntop_gene, method='wilcoxon', key_added="wilcoxon")
        sc.tl.rank_genes_groups(adata_human_expression, groupby='cluster_name', n_genes=ntop_gene, method='wilcoxon', key_added="wilcoxon")
        # get the large homologous gene sparse mat
        homo_relation_df = pd.read_csv(cfg.CAME.path_varmap)
        homologous_pair_mouse = homo_relation_df['gene.name']
        homologous_pair_human = homo_relation_df['human.gene.name']
        gene_list_mouse = adata_mouse_expression.var_names.tolist()
        gene_list_human = adata_human_expression.var_names.tolist()
        gene_list_mouse_set = gene_list_mouse
        gene_list_human_set = gene_list_human
        gene_list_mouse_homo = []
        gene_list_human_homo = []
        for m_gene, h_gene in zip(homologous_pair_mouse, homologous_pair_human):
            if m_gene in gene_list_mouse_set and h_gene in gene_list_human_set:
                gene_list_mouse_homo.append(m_gene)
                gene_list_human_homo.append(h_gene)

        gene_homo_dict_mouse_human = [(k,v) for k,v in zip(gene_list_mouse_homo, gene_list_human_homo)]
        gene_homo_dict_human_mouse = [(k,v) for k,v in zip(gene_list_human_homo, gene_list_mouse_homo)]

        #MH_mat = get_homologous_mat(gene_list_mouse, gene_list_human, homologous_pair_mouse, homologous_pair_human)
        #gene_list_mouse_homo, gene_list_human_homo = \
        #    get_homologous_gene_list(gene_list_mouse, gene_list_human, gene_name_mouse_list, gene_name_human_list, MH_mat)
        for cluster_name in cluster_name_acronym_list:
            mouse_degs_list = sc.get.rank_genes_groups_df(adata_mouse_expression,
                                                    group=cluster_name,
                                                     key='wilcoxon',
                                                     pval_cutoff=1e-6)['names'].squeeze().str.strip().tolist()
            mouse_degs_list_homo = list(set(gene_list_mouse_homo).intersection(set(mouse_degs_list)))
            mouse_degs_list_homo = [g for g in gene_list_mouse_homo if g in set(mouse_degs_list_homo)]
            #mouse_degs_list_homo_human = [v for (k,v) in gene_homo_dict_mouse_human if k in set(mouse_degs_list_homo)]

            human_degs_list = sc.get.rank_genes_groups_df(adata_human_expression,
                                                          group=cluster_name,
                                                          key='wilcoxon',
                                                          pval_cutoff=1e-6)['names'].squeeze().str.strip().tolist()

            human_degs_list_homo = list(set(gene_list_human_homo).intersection(set(human_degs_list)))
            human_degs_list_homo = [g for g in gene_list_human_homo if g in set(human_degs_list_homo)]
            human_degs_list_homo_mouse = [v for (k,v) in gene_homo_dict_human_mouse if k in set(human_degs_list_homo)]

            mouse_common_homo_degs = list(set(mouse_degs_list_homo).intersection(set(human_degs_list_homo_mouse)))
            mouse_common_homo_degs = [g for g in gene_list_mouse_homo if g in set(mouse_common_homo_degs)]
            human_common_homo_degs = [v for (k,v) in gene_homo_dict_mouse_human if k in set(mouse_common_homo_degs)]

            mouse_degs_list_homo = list(set(mouse_degs_list_homo) - set(mouse_common_homo_degs))
            mouse_degs_list_homo = [g for g in gene_list_mouse_homo if g in set(mouse_degs_list_homo)]
            mouse_degs_list_homo_human = [v for (k,v) in gene_homo_dict_mouse_human if k in set(mouse_degs_list_homo)]

            human_degs_list_homo = list(set(human_degs_list_homo) - set(human_common_homo_degs))
            human_degs_list_homo = [g for g in gene_list_human_homo if g in set(human_degs_list_homo)]
            human_degs_list_homo_mouse = [v for (k,v) in gene_homo_dict_human_mouse if k in set(human_degs_list_homo)]

            mouse_complement = mouse_common_homo_degs + mouse_degs_list_homo + human_degs_list_homo_mouse
            mouse_homo_list_subtracted = [x for x in gene_list_mouse_homo if x not in mouse_complement]
            human_homo_list_subtracted = [v for (k,v) in gene_homo_dict_mouse_human if k in set(mouse_homo_list_subtracted)]

            # compute avarage expression
            adata_mouse_expression_group = adata_mouse_expression[adata_mouse_expression.obs['cluster_name'] == cluster_name]
            adata_human_expression_group = adata_human_expression[adata_human_expression.obs['cluster_name'] == cluster_name]

            #m_X_arr = (adata_mouse_expression_group.X.toarray() - np.mean(adata_mouse_expression_group.X.toarray()))/ np.std(adata_mouse_expression_group.X.toarray())
            #m_X_arr = np.log(adata_mouse_expression_group.X.toarray())
            #adata_mouse_expression_group.X = sp.coo_matrix(m_X_arr).tocsr()
            #adata_mouse_expression_group.X = sp.coo_matrix((m_X_arr- np.min(m_X_arr))/(np.max(m_X_arr)- np.min(m_X_arr))).tocsr()
            #h_X_arr = adata_human_expression_group.X.toarray() - np.mean(adata_human_expression_group.X.toarray())
            #adata_human_expression_group.X = sp.coo_matrix((h_X_arr- np.min(h_X_arr))/(np.max(h_X_arr)- np.min(h_X_arr))).tocsr()
            #h_X_arr = (adata_human_expression_group.X.toarray() - np.mean(
            #    adata_human_expression_group.X.toarray())) / np.std(adata_human_expression_group.X.toarray())
            #m_X_arr = np.log(adata_mouse_expression_group.X.toarray())
            #adata_human_expression_group.X = sp.coo_matrix(h_X_arr).tocsr()


            mouse_common_homo_degs_exp = average_expression(adata_mouse_expression_group, mouse_common_homo_degs)
            human_common_homo_degs_exp = average_expression(adata_human_expression_group, human_common_homo_degs)

            mouse_degs_list_homo_exp = average_expression(adata_mouse_expression_group, mouse_degs_list_homo)
            mouse_degs_list_homo_human_exp = average_expression(adata_human_expression_group, mouse_degs_list_homo_human)

            human_degs_list_homo_exp = average_expression(adata_human_expression_group, human_degs_list_homo)
            human_degs_list_homo_mouse_exp = average_expression(adata_mouse_expression_group, human_degs_list_homo_mouse)

            mouse_homo_list_subtracted_exp = average_expression(adata_mouse_expression_group, mouse_homo_list_subtracted)
            human_homo_list_subtracted_exp = average_expression(adata_human_expression_group, human_homo_list_subtracted)

            plt.figure(figsize=(8, 8))
            sns.scatterplot(x=mouse_homo_list_subtracted_exp, y=human_homo_list_subtracted_exp, color='black', s=5)
            sns.scatterplot(x=mouse_degs_list_homo_exp, y=mouse_degs_list_homo_human_exp, color=mouse_color)
            sns.scatterplot(x=human_degs_list_homo_mouse_exp, y=human_degs_list_homo_exp, color=human_color)
            sns.scatterplot(x=mouse_common_homo_degs_exp, y=human_common_homo_degs_exp, color='red')
            plt.xlabel('Mouse')
            plt.ylabel('Human')
            plt.title(cluster_name)
            cluster_name_save = cluster_name.replace(' ', '_').replace('(','_').replace(')','_')

            plt.savefig(save_path + f'{cluster_name_save}.{fig_format}', format=fig_format, transparent=True)

        return None


    def experiment_5_1_homologous_genes_divergence():
        TINY_SIZE = 16  # 39
        SMALL_SIZE = 22  # 42
        MEDIUM_SIZE = 24  # 46
        BIGGER_SIZE = 26  # 46

        plt.rc('font', size=24)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path_parent = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/5_1_experiment_homologous_genes_divergence/parent_region_cluster/'
        if not os.path.exists(save_path_parent):
            os.makedirs(save_path_parent)

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/5_1_experiment_homologous_genes_divergence/region_cluster/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        adata_mouse_expression = sc.read_h5ad(cfg.CAME.path_rawdata1)
        for obs1, obs2 in zip(adata_mouse_embedding.obs_names, adata_mouse_expression.obs_names):
            if obs1 != obs2:
                print('Sample name not aligned!')

        adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
        adata_human_expression = sc.read_h5ad(cfg.CAME.path_rawdata2)
        for obs1, obs2 in zip(adata_human_embedding.obs_names, adata_human_expression.obs_names):
            if obs1 != obs2:
                print('Sample name not aligned!')

        # Get homologous degs
        # mouse cluster
        ntop_gene = 50

        cluster_name_acronym_list = list(set(adata_mouse_expression.obs['cluster_name'].tolist()))
        cluster_name_acronym_list.sort()
        #
        sc.tl.rank_genes_groups(adata_mouse_expression, groupby='cluster_name', n_genes=ntop_gene, method='wilcoxon',
                                key_added="wilcoxon")
        sc.tl.rank_genes_groups(adata_human_expression, groupby='cluster_name', n_genes=ntop_gene, method='wilcoxon',
                                key_added="wilcoxon")
        # get the large homologous gene sparse mat
        homo_relation_df = pd.read_csv(cfg.CAME.path_varmap)
        homologous_pair_mouse = homo_relation_df['gene.name']
        homologous_pair_human = homo_relation_df['human.gene.name']
        gene_list_mouse = adata_mouse_gene_embedding.obs_names.tolist()
        gene_list_human = adata_human_gene_embedding.obs_names.tolist()
        gene_list_mouse_set = gene_list_mouse
        gene_list_human_set = gene_list_human
        gene_list_mouse_homo = []
        gene_list_human_homo = []

        for m_gene, h_gene in zip(homologous_pair_mouse, homologous_pair_human):
            if m_gene in gene_list_mouse_set and h_gene in gene_list_human_set:
                gene_list_mouse_homo.append(m_gene)
                gene_list_human_homo.append(h_gene)

        gene_homo_dict_mouse_human = [(k,v) for k,v in zip(gene_list_mouse_homo, gene_list_human_homo)]
        gene_homo_dict_human_mouse = [(k,v) for k,v in zip(gene_list_human_homo, gene_list_mouse_homo)]

        corrdistance_divergence_dict = {'Correlation':[], 'Expression divergence':[]}


        for m_gene, h_gene in zip(gene_list_mouse_homo, gene_list_human_homo):

            corrdistance_divergence_dict['Correlation'].append(Var_Corr.loc['mouse_'+m_gene, 'human_'+h_gene])

            mouse_mean_exp = pp.group_mean_adata(adata_mouse_expression, groupby='cluster_name', features=[m_gene], use_raw=False)
            #print(type(mouse_mean_exp))
            #print(mouse_mean_exp)
            human_mean_exp = pp.group_mean_adata(adata_human_expression, groupby='cluster_name', features=[h_gene], use_raw=False)
            corrdistance_divergence_dict['Expression divergence'].append(scipy.spatial.distance.euclidean(mouse_mean_exp, human_mean_exp))
            #print(type(human_mean_exp))
            #print(human_mean_exp)
            # for m_name, h_name in zip(mouse_mean_exp.columns, human_mean_exp.columns):
            #     if m_name != h_name:
            #         print('Not equal!')
        r, p = stats.pearsonr(corrdistance_divergence_dict['Correlation'], corrdistance_divergence_dict['Expression divergence'])

        corrdistance_divergence_df = pd.DataFrame.from_dict(corrdistance_divergence_dict)
        fig, ax = plt.subplots(figsize=(8, 8))
        graph = sns.jointplot(x='Correlation', y='Expression divergence', data=corrdistance_divergence_df, kind='hex')
        phantom, = graph.ax_joint.plot([], [], linestyle="", alpha=0)
        graph.ax_joint.legend([phantom], ['r={:f}, p={:f}'.format(r, p)])
        plt.savefig(save_path + f'corr_divergence.{fig_format}', format=fig_format, transparent=True)

        fig, ax = plt.subplots(figsize=(8, 8))
        graph = sns.jointplot(x='Correlation', y='Expression divergence', data=corrdistance_divergence_df, kind='scatter')
        phantom, = graph.ax_joint.plot([], [], linestyle="", alpha=0)
        graph.ax_joint.legend([phantom], ['r={:f}, p={:f}'.format(r, p)])
        plt.savefig(save_path + f'corr_divergence_scatter.{fig_format}', format=fig_format, transparent=True)

        return None


    def experiment_6_gene_module_enrichment():
        TINY_SIZE = 16  # 39
        SMALL_SIZE = 22  # 42
        MEDIUM_SIZE = 24  # 46
        BIGGER_SIZE = 26  # 46

        plt.rc('font', size=18)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']


        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/6_experiment_gene_module_enrichment/region_cluster/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path_parent = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/6_experiment_gene_module_enrichment/parent_region_cluster/'

        adata_mouse_gene_embedding.obs['dataset'] = 'Mouse'
        adata_human_gene_embedding.obs['dataset'] = 'Human'
        adata_gene_embedding = ad.concat([adata_mouse_gene_embedding, adata_human_gene_embedding])

        key_class = 'cluster_name_acronym'

        gene_module_abstract_graph(cfg, adata_gene_embedding, save_path, fig_format, key_class, resolution=0.5,
                                   umap_min_dist=1, umap_n_neighbors=20)

        gene_module_abstract_graph(cfg, adata_gene_embedding, save_path_parent, fig_format, key_class='parent_cluster_name_acronym', resolution=0.5,
                                   umap_min_dist=1, umap_n_neighbors=20)

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/6_experiment_gene_module_enrichment/parent_region/'
        gene_module_abstract_graph(cfg, adata_gene_embedding, save_path, fig_format,
                                   key_class='parent_acronym', resolution=0.5,
                                   umap_min_dist=1, umap_n_neighbors=20)

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/6_experiment_gene_module_enrichment/region/'
        gene_module_abstract_graph(cfg, adata_gene_embedding, save_path, fig_format,
                                   key_class='acronym', resolution=0.5,
                                   umap_min_dist=1, umap_n_neighbors=20)


        return None


    def experiment_7_region_markergene_enrichment(adata_mouse_embedding, adata_human_embedding):
        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/7_experiment_region_markergene_enrichment/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        gene_topk = 30
        correlated_gene_topk = 20
        markersize = 10

        adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
        print(adata_embedding)
        sc.pp.neighbors(adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(adata_embedding)

        adata_mouse_embedding = adata_embedding[adata_embedding.obs['dataset'] == 'Mouse']
        adata_human_embedding = adata_embedding[adata_embedding.obs['dataset'] == 'Human']

        human_marker_gene_subregion_path = cfg.BrainAlign.used_data_path + 'human_marker_genes_subregion/'
        Marker_gene_dict = {}
        human_88_acronym_list = human_88_acronym_dict.values()

        human_88_acronym_dict_reversed = {v:k for k,v in human_88_acronym_dict.items()}

        human_88_marker_gene_dict = {k:[] for k in human_88_acronym_list}
        pair_88_marker_gene_dict = {k:{'Human':None, 'Mouse':None} for k in human_88_acronym_list}

        #print(mouse_88_marker_gene_dict)

        corr_mat = Var_Corr.values

        print('np.max(np.sum(mh_mat, axis=0))', np.max(np.sum(mh_mat, axis=0)))
        print('np.max(np.sum(mh_mat, axis=1))', np.max(np.sum(mh_mat, axis=1)))

        # get homologous regions human:mouse acronym
        human_mouse_homo_region = pd.read_csv(cfg.CAME.homo_region_file_path)
        home_region_dict = OrderedDict()
        mouse_region_list = human_mouse_homo_region['Mouse'].values
        # random.shuffle(mouse_region_list)
        for x, y in zip(human_mouse_homo_region['Human'].values, mouse_region_list):
            home_region_dict[x] = y


        human_gene_list = adata_human_gene_embedding.obs_names.to_list()
        mouse_gene_list = adata_mouse_gene_embedding.obs_names.to_list()

        for subregion, subregion_acronym in human_88_acronym_dict.items():
            subregion_marker_file = human_marker_gene_subregion_path + f'{subregion_acronym}/Probes.csv'
            marker_gene_df = pd.read_csv(subregion_marker_file)

            top_gene_df = marker_gene_df.loc[0:gene_topk, :]
            human_subregion_gene_list = list(top_gene_df['gene-symbol'].values)
            #print(human_subregion_gene_list)
            human_88_marker_gene_dict[subregion_acronym] = human_subregion_gene_list

            for h_g in human_subregion_gene_list:
                name_m_g_list = []
                if h_g in human_gene_list:
                    index_h_g = human_gene_list.index(h_g)
                    if np.sum(mh_mat[:, index_h_g]) > 0:
                        name_m_g_list = []
                        for i in range(len(mouse_gene_list)):
                            if mh_mat[i, index_h_g] > 0:
                                name_m_g_list.append(mouse_gene_list[i])
                if len(name_m_g_list) > 0:
                    #print([h_g], name_m_g_list)
                    pair_88_marker_gene_dict[subregion_acronym]['Human'] = [h_g]
                    pair_88_marker_gene_dict[subregion_acronym]['Mouse'] = name_m_g_list

        print('mouse_88_marker_gene_dict', pair_88_marker_gene_dict)

        pair_88_marker_correlated_genes_dict = {k: {'Human': None, 'Mouse': None} for k in human_88_acronym_list}
        #save_path_subregion = save_path + 'subregion_human_marker/'
        for subregion_acronym, v in pair_88_marker_gene_dict.items():
            print('human marker: ', v['Human'], 'mouse marker:', v['Mouse'])

            save_path_subregion_acronym = save_path + subregion_acronym + '/'
            if not os.path.exists(save_path_subregion_acronym):
                os.makedirs(save_path_subregion_acronym)

            human_marker_gene = v['Human'][0]
            human_marker_gene_index = adata_human_gene_embedding.obs_names.tolist().index(human_marker_gene)
            human_marker_gene_column = mh_mat[:, human_marker_gene_index]
            corr_mat_column = corr_mat[:, human_marker_gene_index]

            correlated_set_index = list(np.argsort(corr_mat_column)[::-1][:correlated_gene_topk])
            mouse_correlated_set = [adata_mouse_gene_embedding.obs_names.values[x] for x in correlated_set_index]

            mouse_marker_gene = v['Mouse'][0]
            mouse_marker_gene_index = adata_mouse_gene_embedding.obs_names.tolist().index(mouse_marker_gene)
            mouse_marker_gene_column = mh_mat[mouse_marker_gene_index, :]
            corr_mat_column = corr_mat[mouse_marker_gene_index, :]

            correlated_set_index = list(np.argsort(corr_mat_column)[::-1][:correlated_gene_topk])
            human_correlated_set = [adata_human_gene_embedding.obs_names.values[x] for x in correlated_set_index]

            pair_88_marker_correlated_genes_dict[subregion_acronym]['Human'] = human_correlated_set
            pair_88_marker_correlated_genes_dict[subregion_acronym]['Mouse'] = mouse_correlated_set

            human_subregion = human_88_acronym_dict_reversed[subregion_acronym]
            mouse_subregion = None
            if human_subregion in home_region_dict.keys():
                mouse_subregion = home_region_dict[human_subregion]

            # Umap of human region, marker gene and correlated marker genes
            adata_human_embedding.obs[human_marker_gene] = adata_human_expression[:, human_marker_gene].X.toarray().reshape(-1)
            with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
                ax = sc.pl.umap(adata_human_embedding, show=False)
                #human_parent_region_name = human_subregion.lower()
                sc.pl.umap(adata_human_embedding[
                               adata_human_embedding.obs["region_name"].isin([human_subregion])],
                           color='region_name',
                           palette={human_subregion: human_88_color_dict[human_subregion]},
                           ax=ax, legend_loc='on data', legend_fontweight='normal', show=False, size=markersize)
                plt.title('')
                plt.savefig(save_path_subregion_acronym + f'umap_human_{human_subregion}.' + fig_format,
                    format=fig_format)

            with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
                sc.pl.umap(adata_human_embedding, color=human_marker_gene, color_map='viridis_r', legend_loc='on data',
                           legend_fontweight='normal', show=False, size=markersize)
                plt.savefig(
                    save_path_subregion_acronym + f'umap_human_{human_88_acronym_dict[human_subregion]}_{human_marker_gene}_homologous_gene.' + fig_format,
                    format=fig_format)

            for correlated_gene in human_correlated_set:
                adata_human_embedding.obs[correlated_gene] = adata_human_expression[:, correlated_gene].X.toarray().reshape(-1)
                with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
                    sc.pl.umap(adata_human_embedding, color=correlated_gene, color_map='viridis_r', legend_loc='on data', legend_fontweight='normal',
                               show=False, size=markersize)
                    if not os.path.exists(save_path_subregion_acronym + 'human_correlated_genes/'):
                        os.makedirs(save_path_subregion_acronym + 'human_correlated_genes/')
                    plt.savefig(save_path_subregion_acronym + 'human_correlated_genes/' + f'umap_human_{correlated_gene}.' + fig_format,
                        format=fig_format)


            adata_mouse_embedding.obs[mouse_marker_gene] = adata_mouse_expression[:, mouse_marker_gene].X.toarray().reshape(-1)
            #print(mouse_marker_gene)
            # Umap of mouse region, marker gene and correlated marker genes
            if mouse_subregion != None:
                with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
                    ax = sc.pl.umap(adata_mouse_embedding, show=False)
                    # human_parent_region_name = human_subregion.lower()
                    sc.pl.umap(adata_mouse_embedding[
                                   adata_mouse_embedding.obs["region_name"].isin([mouse_subregion])],
                               color='region_name',
                               palette={mouse_subregion: mouse_64_color_dict[mouse_subregion]},
                               ax=ax, legend_loc='on data', legend_fontweight='normal', show=False, size=markersize)
                    plt.title('')
                    plt.savefig(save_path_subregion_acronym + f'umap_mouse_{mouse_subregion}.' + fig_format,
                        format=fig_format)

                # umap of two regions if the region is homologous, else umap only one region
                rcParams["figure.subplot.left"] = 0.1
                rcParams["figure.subplot.right"] = 0.66
                with plt.rc_context({"figure.figsize": (12, 8)}):
                    ax = sc.pl.umap(adata_embedding, show=False)
                    #human_parent_region_name = parent_region_name.lower()
                    sc.pl.umap(adata_embedding[
                                   adata_embedding.obs["region_name"].isin(
                                       [human_subregion, mouse_subregion])],
                               color='region_name',
                               palette={mouse_subregion: mouse_64_color_dict[mouse_subregion],
                                        human_subregion: human_88_color_dict[human_subregion]},
                               ax=ax, legend_loc='right margin', legend_fontweight='normal', show=False, size=markersize)
                    plt.title('')
                    save_path_here = save_path_subregion_acronym +  f'umap_expression_together.' + fig_format
                    save_path_here = save_path_here.replace(' ', '_')
                    plt.savefig(save_path_here, format=fig_format)
                rcParams["figure.subplot.left"] = 0.1
                rcParams["figure.subplot.right"] = 0.9

            with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
                sc.pl.umap(adata_mouse_embedding, color=mouse_marker_gene, color_map='viridis_r', legend_loc='on data',
                           legend_fontweight='normal', show=False, size=markersize)
                if mouse_subregion != None:
                    mouse_subregion_acronym = mouse_64_acronym_dict[mouse_subregion]
                else:
                    mouse_subregion_acronym = 'None'
                plt.savefig(save_path_subregion_acronym + f'umap_mouse_{mouse_subregion_acronym}_{mouse_marker_gene}_homologous_gene.' + fig_format,
                    format=fig_format)

            for correlated_gene in mouse_correlated_set:
                adata_mouse_embedding.obs[correlated_gene] = adata_mouse_expression[:, correlated_gene].X.toarray().reshape(-1)
                with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):
                    sc.pl.umap(adata_mouse_embedding, color=correlated_gene, color_map='viridis_r', legend_loc='on data', legend_fontweight='normal',
                               show=False, size=markersize)
                    if not os.path.exists(save_path_subregion_acronym + 'mouse_correlated_genes/'):
                        os.makedirs(save_path_subregion_acronym + 'mouse_correlated_genes/')
                    plt.savefig(save_path_subregion_acronym + 'mouse_correlated_genes/' + f'umap_mouse_{correlated_gene}.' + fig_format,
                        format=fig_format)


            go_save_path = save_path + f'{subregion_acronym}/gene_ontology/'
            if not os.path.exists(go_save_path):
                os.makedirs(go_save_path)
            # Gene ontology analysis of human
            gene_list_subregion_human = human_88_marker_gene_dict[subregion_acronym]
            print(gene_list_subregion_human)
            print(gene_list_subregion_human[0])
            print(len(gene_list_subregion_human))
            enr_res = gseapy.enrichr(gene_list=gene_list_subregion_human, #[human_marker_gene],
                                     organism='Human',
                                     # gene_sets=['Allen_Brain_Atlas_down', 'Allen_Brain_Atlas_up',
                                     #            'Azimuth_Cell_Types_2021', 'CellMarker_Augmented_2021',
                                     #            'GO_Biological_Process_2021', 'GO_Cellular_Component_2021',
                                     #            'GO_Molecular_Function_2021', 'Human_Gene_Atlas'],
                                     gene_sets=[#'Allen_Brain_Atlas_down', 'Allen_Brain_Atlas_up',
                                                'Azimuth_Cell_Types_2021', 'CellMarker_Augmented_2021',
                                                'GO_Biological_Process_2021',
                                                'GO_Molecular_Function_2021', 'Human_Gene_Atlas'],
                                     cutoff=0.5)
            gseapy.barplot(enr_res.results, column="Adjusted P-value", group='Gene_set', color=human_color,
                           # set group, so you could do a multi-sample/library comparsion
                           size=5, title=f"GO of human {human_subregion} DEGs",
                           figsize=(12, 15),
                           ofname=go_save_path + f'go_human_{human_88_acronym_dict[human_subregion]}_DEGs_barplot.' + fig_format)
            gseapy.dotplot(enr_res.results,
                           column="Adjusted P-value",
                           x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion #
                           size=20,
                           top_term=5,  #
                           figsize=(12, 12),
                           title=f"GO of human {human_subregion} DEGs",
                           xticklabels_rot=45,
                           # rotate xtick labels show_ring=True, set to False to revmove outer ring
                           marker='o',
                           ofname=go_save_path + f'go_human_{human_88_acronym_dict[human_subregion]}_DEGs_botplot.' + fig_format,
                           format=fig_format)


            # Gene ontology analysis of human genes correlated to marker gene
            enr_res = gseapy.enrichr(gene_list=human_correlated_set,
                                     organism='Human',
                                     gene_sets=[#'Allen_Brain_Atlas_down', 'Allen_Brain_Atlas_up',
                                                'Azimuth_Cell_Types_2021', 'CellMarker_Augmented_2021',
                                                'GO_Biological_Process_2021', #'GO_Cellular_Component_2021',
                                                'GO_Molecular_Function_2021', 'Human_Gene_Atlas'],
                                     cutoff=0.5)
            gseapy.barplot(enr_res.results, column="Adjusted P-value", group='Gene_set', color=human_color,
                           # set group, so you could do a multi-sample/library comparsion
                           size=5, title=f"GO of genes correlated to human {human_marker_gene}", figsize=(12, 15),
                           ofname=go_save_path + f'go_human_correlated_{human_marker_gene}_barplot.' + fig_format)
            gseapy.dotplot(enr_res.results,
                           column="Adjusted P-value",
                           x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion #size=5,
                           size=20,
                           top_term=5,  #
                           figsize=(12, 12),
                           title=f"GO of genes correlated to human {human_marker_gene}",
                           xticklabels_rot=45,  # rotate xtick labels show_ring=True, set to False to revmove outer ring
                           marker='o',
                           ofname=go_save_path + f'go_human_correlated_{human_marker_gene}_botplot.' + fig_format,
                           format=fig_format)

            # Gene ontology analysis of mouse
            enr_res = gseapy.enrichr(gene_list=mouse_correlated_set,
                                     organism='Mouse',
                                     gene_sets=[#'Allen_Brain_Atlas_down', 'Allen_Brain_Atlas_up',
                                                'Azimuth_Cell_Types_2021', 'CellMarker_Augmented_2021',
                                                'GO_Biological_Process_2021', #'GO_Cellular_Component_2021',
                                                'GO_Molecular_Function_2021', 'Mouse_Gene_Atlas'],
                                     cutoff=0.5)
            gseapy.barplot(enr_res.results, column="Adjusted P-value", group='Gene_set', color=mouse_color,
                           # set group, so you could do a multi-sample/library comparsion
                           size=5, title=f"GO of genes correlated to mouse {mouse_marker_gene}", figsize=(12, 15),
                           ofname=go_save_path + f'go_mouse_correlated_{mouse_marker_gene}_barplot.' + fig_format)
            gseapy.dotplot(enr_res.results,
                           column="Adjusted P-value",
                           x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion #
                           size=20,
                           top_term=5,  #
                           figsize=(12, 12),
                           title=f"GO of genes correlated to mouse {mouse_marker_gene}",
                           xticklabels_rot=45,  # rotate xtick labels show_ring=True, set to False to revmove outer ring
                           marker='o',
                           ofname=go_save_path + f'go_mouse_correlated_{mouse_marker_gene}_botplot.' + fig_format,
                           format=fig_format)

        with open(save_path + 'human_88_marker_gene_dict.pkl', 'wb') as f:
            pickle.dump(human_88_marker_gene_dict, f)

        with open(save_path + 'pair_88_marker_gene_dict.pkl', 'wb') as f:
            pickle.dump(pair_88_marker_gene_dict, f)

        with open(save_path + 'pair_88_marker_correlated_genes_dict.pkl', 'wb') as f:
            pickle.dump(pair_88_marker_correlated_genes_dict, f)


    def experiment_8_check_unaligned():

        sns.set(style='white')
        TINY_SIZE = 28  # 39
        SMALL_SIZE = 28  # 42
        MEDIUM_SIZE = 36  # 46
        BIGGER_SIZE = 36  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')

        adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])

        cluster_name_acronym_list = list(set(adata_embedding.obs['cluster_name_acronym'].values))
        cluster_name_acronym_list.sort(key=lambda x:int(x.split('-')[0]))

        mouse_human_rate = adata_mouse_embedding.n_obs / adata_human_embedding.n_obs

        mouse_normal_rate = mouse_human_rate/ (mouse_human_rate + 1)
        human_normal_rate = 1 / (mouse_human_rate + 1)

        #mouse_lowest_rate = 0.1
        #human_lowest_rate = 0.1

        mouse_lowest_rate = mouse_normal_rate * 0.5#0.554075#mouse_normal_rate * mouse_lowest_rate
        human_lowest_rate = human_normal_rate * 0.05#human_normal_rate * human_lowest_rate

        human_special_dict = {}
        mouse_special_dict = {}

        human_special_proportion_dict = {}
        mouse_special_proportion_dict = {}

        for cluster_name_acronym in cluster_name_acronym_list:
            adata_human_cluster = adata_human_embedding[adata_human_embedding.obs['cluster_name_acronym'].isin([cluster_name_acronym])]
            adata_mouse_cluster = adata_mouse_embedding[adata_mouse_embedding.obs['cluster_name_acronym'].isin([cluster_name_acronym])]
            adata_cluster = ad.concat([adata_human_cluster, adata_mouse_cluster])

            h_proportion = adata_human_cluster.n_obs / (adata_human_cluster.n_obs + adata_mouse_cluster.n_obs)
            if h_proportion < human_lowest_rate:
                mouse_special_dict[cluster_name_acronym] = {'cluster':cluster_name_acronym, 'human proportion':h_proportion, 'mouse proportion':1-h_proportion, 'specialized':'Mouse'}
                human_category_distri = Counter(list(adata_human_cluster.obs['region_name'].values))
                mouse_category_distri = Counter(list(adata_mouse_cluster.obs['region_name'].values))
                mouse_special_proportion_dict[cluster_name_acronym] = mouse_category_distri#{'human':human_category_distri, 'mouse':mouse_category_distri}
                print('specialized:mouse')
                print('human', human_category_distri)
                print('mouse', mouse_category_distri)

            m_proportion = adata_mouse_cluster.n_obs / (adata_human_cluster.n_obs + adata_mouse_cluster.n_obs)
            if m_proportion < mouse_lowest_rate:
                human_special_dict[cluster_name_acronym] = {'cluster':cluster_name_acronym, 'human proportion': 1 - m_proportion, 'mouse proportion': m_proportion, 'specialized':'Human'}
                human_category_distri = Counter(list(adata_human_cluster.obs['region_name'].values))
                mouse_category_distri = Counter(list(adata_mouse_cluster.obs['region_name'].values))
                human_special_proportion_dict[cluster_name_acronym] = human_category_distri#{'human': human_category_distri,
                                                                      # 'mouse': mouse_category_distri}
                print('specialized:human')
                print('human', human_category_distri)
                print('mouse', mouse_category_distri)

        mouse_special_proportion_df = pd.DataFrame.from_dict(mouse_special_proportion_dict).T
        mouse_special_proportion_df = mouse_special_proportion_df.fillna(0)
        human_special_proportion_df = pd.DataFrame.from_dict(human_special_proportion_dict).T
        human_special_proportion_df = human_special_proportion_df.fillna(0)
        # plot scatters of mouse_special_dict and human_special_dict
        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/8_experiment_check_unaligned/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print('human_special_dict', human_special_dict)
        special_dict = {**human_special_dict, **mouse_special_dict}
        special_df = pd.DataFrame.from_dict(special_dict).T
        print(special_df)
        #with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi":(fig_dpi)}):

        rcParams["figure.subplot.left"] = 0.2
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.2
        fig, ax = plt.subplots(figsize=(8, 8), dpi=fig_dpi)
        sns.scatterplot(data=special_df, x="mouse proportion", y="human proportion",
                        hue='specialized', palette={'Mouse':mouse_color, 'Human':human_color}, sizes=100)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])
        #fig.legend(labels=['Mouse', 'Human'], loc='upper right', frameon=False, bbox_to_anchor=(0.9, 0.9))
        # for label, x, y in zip(special_df['cluster'].values, special_df['mouse proportion'].values, special_df['human proportion'].values):
        #     plt.annotate(label, (x, y))
        plt.savefig(save_path + 'proportion_specialized.' + fig_format, format=fig_format)



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

        #plt.figure(figsize=(12,12))
        #with plt.rc_context({"figure.figsize": (12, 12), "figure.dpi":(fig_dpi)}):
        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.4
        fig, ax = plt.subplots(figsize=(25, 9), dpi=fig_dpi)
        print(mouse_special_proportion_df)
        hm = sns.heatmap(mouse_special_proportion_df, square=False, cbar_kws = {'location':'top'}, cmap="YlGnBu", ax=ax,
                    xticklabels=True, yticklabels=True)
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        #plt.legend([],[], frameon=False)
        #plt.xticks(rotation=45)
        plt.savefig(save_path + 'mouse_specialized_heatmap.' + fig_format, format=fig_format)

        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.right"] = 0.98
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.6
        fig, ax = plt.subplots(figsize=(25, 5), dpi=fig_dpi)
        print(human_special_proportion_df)
        hm = sns.heatmap(human_special_proportion_df, square=False, cbar_kws = {'location':'top'}, cmap="YlGnBu", ax=ax,
                    xticklabels=True, yticklabels=True)
        for item in hm.get_xticklabels():
            item.set_rotation(45)
        plt.setp(hm.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        #plt.legend([],[], frameon=False)
        #plt.xticks(rotation=45)
        plt.savefig(save_path + 'human_specialized_heatmap.' + fig_format, format=fig_format)

        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.05
        return None


    #experiment_1_calb1()
    #experiment_2_marker_genes(adata_mouse_embedding, adata_human_embedding)
    #experiment_3_1_cluster_rename(adata_mouse_embedding, adata_human_embedding)
    #experiment_3_2_deg_analysis()
    #experiment_3_3_deg_distribution()
    experiment_4_gene_ontology_analysis()
    experiment_5_homologous_genes_analysis()
    experiment_5_1_homologous_genes_divergence()
    experiment_6_gene_module_enrichment()
    experiment_7_region_markergene_enrichment(adata_mouse_embedding, adata_human_embedding)
    experiment_8_check_unaligned()

    return None


class local_region_analysis:
    def __init__(self, cfg):
        self.cfg = cfg

    def experiment_1_neocortex_analysis(self):

        return None

    def forward(self):

        self.experiment_1_neocortex_analysis()
        return None





if __name__ == '__main__':
    cfg = heco_config._C
    print('Analysis of srrsc embeddings.')
