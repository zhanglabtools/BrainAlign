# -- coding: utf-8 --
# @Time : 2023/4/3 18:26
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : analysis_anatomical.py
# @Description: This file is used to ...

import pickle

import matplotlib.patches
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
#from BrainAlign.code.utils import set_params
from BrainAlign.brain_analysis.configs import heco_config
from BrainAlign.brain_analysis.data_utils import plot_marker_selection_umap, plot_marker_selection_umap_embedding, get_submin
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
from scGeneFit.functions import *

# color map
import colorcet as cc

from collections import Counter

# get gene ontology
import gseapy

from imblearn.over_sampling import RandomOverSampler
from statannot import add_stat_annotation

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

from webcolors import hex_to_rgb

import plotly.graph_objects as go  # Import the graphical object
from plotly.offline import plot

from colormap import Colormap
from matplotlib.patches import Patch


def reorder(X_1, X_ordered, Y_1):
    Y_ordered = []
    for i in range(len(X_1)):
        X_ele_index = X_1.index(X_ordered[i])
        Y_ordered.append(Y_1[X_ele_index])
    return Y_ordered

class spatial_analysis():
    def __init__(self, cfg):
        # Gene expression comparative analysis
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

        self.adata_mouse_exp = sc.read_h5ad(cfg.CAME.path_rawdata1)#[:, mouse_gene_list]
        self.adata_human_exp = sc.read_h5ad(cfg.CAME.path_rawdata2)#[:, human_gene_list]

        self.adata_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_embedding.h5ad')

        self.adata_gene_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'adata_gene_embedding.h5ad')
        self.adata_mouse_gene_embedding = self.adata_gene_embedding[
            self.adata_gene_embedding.obs['dataset'].isin(['Mouse'])]
        self.adata_human_gene_embedding = self.adata_gene_embedding[
            self.adata_gene_embedding.obs['dataset'].isin(['Human'])]

        self.adata_mouse_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + '/s_embeddings.h5ad')
        self.adata_human_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + '/v_embeddings.h5ad')

        self.mouse_gene_num = self.adata_mouse_gene_embedding.n_obs
        self.human_gene_num = self.adata_human_gene_embedding.n_obs

        self.cluster_name_unique = sorted(Counter(self.adata_embedding.obs['cluster_name_acronym']).keys(),
                                          key=lambda t: int(t.split('-')[0]))

        palette = sns.color_palette(cc.glasbey, n_colors=len(self.cluster_name_unique))

        self.cluster_color_dict = {k: v for k, v in zip(self.cluster_name_unique, palette)}

    def experiment_1_1_paga_test(self):
        """
        Adjust paga points to acquire two groups of points

        :return: None
        """

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
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/4_spatial_analysis/1_experiment_paga_test/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Visualize the anatomical regions of regions;
        hippocampal_region_mouse_list = ['Field CA1', 'Field CA2', 'Field CA3', 'Dentate gyrus', 'Subiculum']
        hippocampal_acronym_mouse_list = ['CA1', 'CA2', 'CA3', 'DG', 'SUB']
        hippocampal_region_human_list = ['CA1 field', 'CA2 field', 'CA3 field', 'dentate gyrus', 'subiculum']
        hippocampal_acronym_human_list = ['CA1', 'CA2', 'CA3', 'DG', 'S']

        hippocampal_region_list = hippocampal_region_mouse_list + hippocampal_region_human_list
        hippocampal_acronym_list = hippocampal_acronym_mouse_list + hippocampal_acronym_human_list

        adata_human_embedding_hip = self.adata_human_embedding[
            self.adata_human_embedding.obs['region_name'].isin(hippocampal_region_human_list)]
        adata_mouse_embedding_hip = self.adata_mouse_embedding[
            self.adata_mouse_embedding.obs['region_name'].isin(hippocampal_region_mouse_list)]
        adata_human_embedding_hip.obs['acronym'] = ['H-' + x for x in adata_human_embedding_hip.obs['acronym']]
        adata_mouse_embedding_hip.obs['acronym'] = ['M-' + x for x in adata_mouse_embedding_hip.obs['acronym']]

        adata_human_embedding_hip.obs['region_name'] = ['H-' + x for x in adata_human_embedding_hip.obs['acronym']]
        adata_mouse_embedding_hip.obs['region_name'] = ['M-' + x for x in adata_mouse_embedding_hip.obs['acronym']]

        adata_embedding_hip = self.adata_embedding[
            self.adata_embedding.obs['region_name'].isin(hippocampal_region_list)]

        region_name_map_list = ['M-' + x for x in hippocampal_region_mouse_list] + ['H-' + x for x in
                                                                                    hippocampal_region_human_list]
        acronym_name_map_list = ['M-' + x for x in hippocampal_acronym_mouse_list] + ['H-' + x for x in
                                                                                      hippocampal_acronym_human_list]

        region_name_map_dict = {k: v for k, v in zip(hippocampal_region_list, region_name_map_list)}
        acronym_name_map_dict = {k: v for k, v in zip(hippocampal_acronym_list, acronym_name_map_list)}

        adata_embedding_hip.obs['region_name'] = [region_name_map_dict[x] for x in
                                                  adata_embedding_hip.obs['region_name']]
        adata_embedding_hip.obs['acronym'] = [acronym_name_map_dict[x] for x in
                                              adata_embedding_hip.obs['acronym']]

        mouse_human_64_88_color_dict = dict({'M-' + k: v for k, v in self.mouse_64_color_dict.items()})
        mouse_human_64_88_color_dict.update({'H-' + k: v for k, v in self.human_88_color_dict.items()})

        # UMAP ------------------------------------------------------
        color_list = sns.color_palette(cc.glasbey, n_colors=len(region_name_map_list)//2)
        color_list = color_list + color_list
        palette = {k: v for k, v in zip(region_name_map_list, color_list)}
        palette_acronym = {k: v for k, v in zip(acronym_name_map_list, color_list)}

        sc.pp.neighbors(adata_embedding_hip, n_neighbors=self.cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(adata_embedding_hip, n_components=2, min_dist=0.3)

        with plt.rc_context({"figure.figsize": (4, 2), "figure.dpi": (self.fig_dpi)}):
            rcParams["figure.subplot.left"] = 0.1
            rcParams["figure.subplot.right"] = 0.6
            rcParams["figure.subplot.bottom"] = 0.1
            rcParams["figure.subplot.top"] = 0.9

            fig = sc.pl.umap(adata_embedding_hip, color=['region_name'],
                             return_fig=True,
                             legend_loc='right margin',
                             palette=palette,  # wspace=wspace,
                             # add_outline=True,
                             # legend_fontsize=12, legend_fontoutline=2,
                             size=80,
                             title=['']
                             )
            # plt.title('')
            fig.savefig(save_path + 'umap_hippcampal.' + self.fig_format, format=self.fig_format)


        dataset_palette = {'Mouse':self.mouse_color, 'Human':self.human_color}

        with plt.rc_context({"figure.figsize": (4, 2), "figure.dpi": (self.fig_dpi)}):
            rcParams["figure.subplot.left"] = 0.1
            rcParams["figure.subplot.right"] = 0.6
            rcParams["figure.subplot.bottom"] = 0.1
            rcParams["figure.subplot.top"] = 0.9

            fig = sc.pl.umap(adata_embedding_hip, color=['dataset'],
                             return_fig=True,
                             legend_loc='right margin',
                             palette=dataset_palette,  # wspace=wspace,
                             # add_outline=True,
                             # legend_fontsize=12, legend_fontoutline=2,
                             size=80,
                             title=['']
                             )
            # plt.title('')
            fig.savefig(save_path + 'umap_hippcampal_dataset.' + self.fig_format, format=self.fig_format)

        # PAGA of hippocampal
        sc.tl.paga(adata_embedding_hip, groups='region_name')
        rcParams["figure.subplot.right"] = 0.9
        fig, ax = plt.subplots(figsize=(8, 8), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        graph_fontsize = 10

        palette_paga = {}
        for k, v in palette.items():
            palette_paga.update({k: {v: 1}})

        pos_arr = np.zeros((10, 2))
        pos_arr[0:5, 0] = 0
        pos_arr[0:5, 0] = np.array([0, 0.5, 1, 0.5,0])
        pos_arr[0:5, 1] = np.array(range(0, 5))
        pos_arr[5:10, 0] = 3
        pos_arr[0:5, 0] = np.array([3, 2.5, 2, 2.5, 3])
        pos_arr[5:10, 1] = np.array(range(0, 5))


        pos_arr[5:10, 1] = np.array([3, 0,1,2, 4])



        sc.pl.paga(adata_embedding_hip,
                   color=palette_paga,
                   node_size_scale=5,
                   # min_edge_width=0.1,
                   pos = pos_arr,
                   ax=ax,
                   # max_edge_width=1,
                   #layout='f',
                   fontsize=graph_fontsize,
                   fontoutline=1, colorbar=False, threshold=0.8,
                   frameon=False)  # node_size_scale=10, edge_width_scale=2
        plt.title('')
        fig.tight_layout()
        plt.savefig(save_path + 'paga_region_name_test.' + self.fig_format, format=self.fig_format)

        # paga species
        palette_species = {'M-' + k: self.mouse_color for k in hippocampal_region_mouse_list}
        palette_species.update({'H-' + k: self.human_color for k in hippocampal_region_human_list})
        palette_paga_species = {}
        for k, v in palette_species.items():
            palette_paga_species.update({k: {v: 1}})

        fig, ax = plt.subplots(figsize=(3, 3), dpi=self.fig_dpi)
        sc.pl.paga(adata_embedding_hip,
                   color=palette_paga_species,
                   node_size_scale=5,
                   # min_edge_width=0.1,
                   pos = pos_arr,
                   ax=ax,
                   # max_edge_width=1,
                   #layout='eq_tree',
                   fontsize=graph_fontsize,
                   fontoutline=1, colorbar=False, threshold=0.8,
                   frameon=False)  # node_size_scale=10, edge_width_scale=2
        plt.title('')
        fig.tight_layout()
        plt.savefig(save_path + 'paga_species_test.' + self.fig_format, format=self.fig_format)

        return None


    def experiment_1_spatial_hippocampal(self):
        """
        Spatial integration and alignment of hippocampal regions, to display that BrainAlign keeps the relative spatial positions of
        samples and help to detect STs difference across species.

        - 1. Visualize the anatomical regions of two regions;
        - 2. Spatial scatter of regions separately;
        - 3. UMAP of two regions together;
        - 4. PAGA abstracted graph of regions of two species;
        - 4. For each Marker genes, plot each expression of marker genes on UMAP of two species (with different thresholds)

        :return: None
        """

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

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/4_spatial_analysis/1_experiment_spatial_hippocampal/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Visualize the anatomical regions of regions;
        hippocampal_region_mouse_list = ['Field CA1', 'Field CA2', 'Field CA3', 'Dentate gyrus', 'Subiculum']
        hippocampal_acronym_mouse_list = ['CA1', 'CA2', 'CA3', 'DG', 'SUB']
        hippocampal_region_human_list = ['CA1 field', 'CA2 field', 'CA3 field', 'dentate gyrus', 'subiculum']
        hippocampal_acronym_human_list = ['CA1', 'CA2', 'CA3', 'DG', 'S']

        hippocampal_region_list = hippocampal_region_mouse_list + hippocampal_region_human_list
        hippocampal_acronym_list = hippocampal_acronym_mouse_list + hippocampal_acronym_human_list

        adata_human_embedding_hip = self.adata_human_embedding[self.adata_human_embedding.obs['region_name'].isin(hippocampal_region_human_list)]
        adata_mouse_embedding_hip = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['region_name'].isin(hippocampal_region_mouse_list)]
        adata_human_embedding_hip.obs['acronym'] = ['H-' + x for x in adata_human_embedding_hip.obs['acronym']]
        adata_mouse_embedding_hip.obs['acronym'] = ['M-' + x for x in adata_mouse_embedding_hip.obs['acronym']]

        adata_human_embedding_hip.obs['region_name'] = ['H-' + x for x in adata_human_embedding_hip.obs['acronym']]
        adata_mouse_embedding_hip.obs['region_name'] = ['M-' + x for x in adata_mouse_embedding_hip.obs['acronym']]

        adata_embedding_hip = self.adata_embedding[self.adata_embedding.obs['region_name'].isin(hippocampal_region_list)]

        region_name_map_list = ['M-'+x for x in hippocampal_region_mouse_list] + ['H-'+x for x in hippocampal_region_human_list]
        acronym_name_map_list = ['M-'+x for x in hippocampal_acronym_mouse_list] + ['H-'+x for x in hippocampal_acronym_human_list]

        region_name_map_dict = {k:v for k,v in zip(hippocampal_region_list, region_name_map_list)}
        acronym_name_map_dict = {k: v for k, v in zip(hippocampal_acronym_list, acronym_name_map_list)}

        adata_embedding_hip.obs['region_name'] = [region_name_map_dict[x] for x in adata_embedding_hip.obs['region_name']]
        adata_embedding_hip.obs['acronym'] = [acronym_name_map_dict[x] for x in
                                                  adata_embedding_hip.obs['acronym']]

        mouse_human_64_88_color_dict = dict({'M-'+k:v for k,v in self.mouse_64_color_dict.items()})
        mouse_human_64_88_color_dict.update({'H-'+k:v for k,v in self.human_88_color_dict.items()})


        # UMAP ------------------------------------------------------
        color_list = sns.color_palette(cc.glasbey, n_colors=len(region_name_map_list)//2)
        color_list = color_list + color_list
        palette = {k: v for k, v in zip(region_name_map_list, color_list)}
        palette_acronym = {k:v for k,v in zip(acronym_name_map_list, color_list)}

        sc.pp.neighbors(adata_embedding_hip, n_neighbors=self.cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(adata_embedding_hip, n_components=2, min_dist=0.3)

        with plt.rc_context({"figure.figsize": (7, 2.5), "figure.dpi": (self.fig_dpi)}):
            rcParams["figure.subplot.left"] = 0.1
            rcParams["figure.subplot.right"] = 0.4
            rcParams["figure.subplot.bottom"] = 0.2
            rcParams["figure.subplot.top"] = 0.98
            adata_embedding_hip.obs['region_name'] = adata_embedding_hip.obs['region_name'].astype('category')
            adata_embedding_hip.obs['region_name'].cat.\
                reorder_categories(['M-Field CA1', 'M-Field CA2', 'M-Field CA3', 'M-Dentate gyrus', 'M-Subiculum',
                                    'H-CA1 field', 'H-CA2 field', 'H-CA3 field', 'H-dentate gyrus', 'H-subiculum'], inplace = True)

            fig = sc.pl.umap(adata_embedding_hip, color=['region_name'],
                             return_fig=True,
                             legend_loc='right margin',
                             palette=palette, #wspace=wspace,
                             #add_outline=True,
                             #legend_fontsize=12, legend_fontoutline=2,
                             size=20,
                             title=['']
                             )
            # plt.title('')
            fig.savefig(save_path + 'umap_hippcampal.' + self.fig_format, format=self.fig_format)

        dataset_palette = {'Mouse': self.mouse_color, 'Human': self.human_color}

        with plt.rc_context({"figure.figsize": (7, 2.5), "figure.dpi": (self.fig_dpi)}):
            rcParams["figure.subplot.left"] = 0.1
            rcParams["figure.subplot.right"] = 0.4
            rcParams["figure.subplot.bottom"] = 0.2
            rcParams["figure.subplot.top"] = 0.98
            adata_embedding_hip.obs['region_name'] = adata_embedding_hip.obs['region_name'].astype('category')
            adata_embedding_hip.obs['region_name'].cat.\
                reorder_categories(['M-Field CA1', 'M-Field CA2', 'M-Field CA3', 'M-Dentate gyrus', 'M-Subiculum',
                                    'H-CA1 field', 'H-CA2 field', 'H-CA3 field', 'H-dentate gyrus', 'H-subiculum'], inplace = True)

            fig = sc.pl.umap(adata_embedding_hip, color=['dataset'],
                             return_fig=True,
                             legend_loc='right margin',
                             palette=dataset_palette, #wspace=wspace,
                             #add_outline=True,
                             #legend_fontsize=12, legend_fontoutline=2,
                             size=20,
                             title=['']
                             )
            # plt.title('')
            fig.savefig(save_path + 'umap_hippcampal_dataset.' + self.fig_format, format=self.fig_format)

        # PAGA of hippocampal
        sc.tl.paga(adata_embedding_hip, groups='region_name')
        rcParams["figure.subplot.right"] = 0.95
        fig, ax = plt.subplots(figsize=(4, 3.7), dpi=self.fig_dpi)

        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.98
        rcParams["figure.subplot.bottom"] = 0.02
        rcParams["figure.subplot.top"] = 0.98

        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        graph_fontsize = 12

        palette_paga = {}
        for k, v in palette.items():
            palette_paga.update({k: {v: 1}})

        pos_arr = np.zeros((10, 2))
        pos_arr[0:5, 0] = 0
        pos_arr[0:5, 1] = np.array(range(0, 5))
        pos_arr[5:10, 0] = 3
        pos_arr[5:10, 1] = np.array(range(0, 5))
        pos_arr[5:10, 1] = np.array([0, 1, 2, 3, 4])
        pos_arr[0:5, 0] = np.array([1, 0.5, 0, 0.5, 1])
        pos_arr[5:10, 0] = np.array([2, 2.5, 3, 2.5, 2])

        sc.pl.paga(adata_embedding_hip,
                   color=palette_paga,
                   node_size_scale=3,
                   min_edge_width=0.1,
                   pos=pos_arr,
                   ax=ax,
                   max_edge_width=10,
                   fontweight='normal',
                   layout='eq_tree',
                   fontsize=graph_fontsize,
                   fontoutline=0, colorbar=False, threshold=0.8,
                   frameon=False)  # node_size_scale=10, edge_width_scale=2
        plt.title('')
        fig.tight_layout()
        plt.savefig(save_path + 'paga_region_name.' + self.fig_format, format=self.fig_format)

        # paga species
        palette_species = {'M-'+k: self.mouse_color for k in hippocampal_region_mouse_list}
        palette_species.update({'H-'+k: self.human_color for k in hippocampal_region_human_list})
        palette_paga_species = {}
        for k, v in palette_species.items():
            palette_paga_species.update({k: {v: 1}})

        fig, ax = plt.subplots(figsize=(4.2, 3.7), dpi=self.fig_dpi)

        sc.pl.paga(adata_embedding_hip,
                   color=palette_paga_species,
                   node_size_scale=3,
                   min_edge_width=0.1,
                   pos=pos_arr,
                   ax=ax,
                   max_edge_width=10,
                   layout='eq_tree',
                   fontweight='normal',
                   fontsize=graph_fontsize,
                   fontoutline=0, colorbar=False, threshold=0.8,
                   frameon=False)  # node_size_scale=10, edge_width_scale=2
        plt.title('')
        fig.tight_layout()
        plt.savefig(save_path + 'paga_species.' + self.fig_format, format=self.fig_format)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.9

        #print(self.adata_mouse_embedding)

        # 3D plot parameters
        marker_s = 5


        # --------------------------------------------------------------------------------------
        # 3D coordinates of regions of mouse
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.6
        fig = plt.figure(figsize=(3, 2), dpi=self.fig_dpi)
        axes = plt.axes(projection='3d')
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for region in ['M-' + x for x in hippocampal_region_mouse_list]:
            region = region.strip('M-')
            print(region)
            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['M-' + region]
            x_vec = adata_m.obs['x_grid']
            y_vec = adata_m.obs['y_grid']
            z_vec = adata_m.obs['z_grid']
            axes.scatter3D(x_vec, y_vec, z_vec, color=scatter_color, s=marker_s, label=region)
        axes.grid(False)
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')
        axes.set(frame_on=False)
        fig.subplots_adjust(left=0, right=0.6, bottom=0, top=1)
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
        plt.legend(loc='right', bbox_to_anchor=(1.5, 0.5), ncol=1, frameon=False) #bbox_to_anchor=(1.1, 0.5),
        plt.savefig(save_path + f'Spatial_mouse.' + self.fig_format)


        # multiview
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.8
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2))
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        #rcParams["figure.subplot.left"] = 0.05
        for region in ['M-'+x for x in hippocampal_region_mouse_list]:
            region = region.strip('M-')
            #print(region)
            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['region_name'].isin([region])]
            #print(adata_m)
            #print(adata_m)
            scatter_color = palette['M-'+region]
            x_vec = adata_m.obs['x_grid']
            y_vec = adata_m.obs['y_grid']
            z_vec = adata_m.obs['z_grid']
            ax1.scatter(x_vec, y_vec, color=scatter_color, s=marker_s, label=region)
            ax2.scatter(x_vec, z_vec, color=scatter_color, s=marker_s, label=region)
            ax3.scatter(y_vec, z_vec, color=scatter_color, s=marker_s, label=region)

        ax1.grid(False)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        ax2.grid(False)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')

        ax3.grid(False)
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')

        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticklabels([])

        ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])

        ax3.xaxis.set_ticklabels([])
        ax3.yaxis.set_ticklabels([])


        for line in ax1.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax1.yaxis.get_ticklines():
            line.set_visible(False)

        for line in ax2.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax2.yaxis.get_ticklines():
            line.set_visible(False)

        for line in ax3.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax3.yaxis.get_ticklines():
            line.set_visible(False)

        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), ncol=1, frameon=False) #bbox_to_anchor=(1.1, 0.5),
        # plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        plt.savefig(save_path + f'Spatial_mouse_multiview.' + self.fig_format)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9


        # --------------------------------------------------------------------------
        # Scatter according regions
        # multiview
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.19
        rcParams["figure.subplot.top"] = 0.99

        fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi)
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for i, region in zip(range(len(hippocampal_region_mouse_list)), ['M-' + x for x in hippocampal_region_mouse_list]):
            region = region.strip('M-')
            print(region)
            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['M-' + region]
            x_vec = adata_m.obs['x_grid']
            y_vec = adata_m.obs['y_grid']
            z_vec = adata_m.obs['z_grid']
            for j in range(len(hippocampal_region_mouse_list)):
                if i != j:
                    axes[j].scatter(x_vec, y_vec, color='lightgrey', s=marker_s)


        for i, region in zip(range(len(hippocampal_region_mouse_list)), ['M-' + x for x in hippocampal_region_mouse_list]):
            region = region.strip('M-')
            print(region)
            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['M-' + region]
            x_vec = adata_m.obs['x_grid']
            y_vec = adata_m.obs['y_grid']
            z_vec = adata_m.obs['z_grid']
            axes[i].scatter(x_vec, y_vec, color=scatter_color, s=marker_s, label=region)


        for region, ax1 in zip(hippocampal_region_mouse_list, axes):
            ax1.grid(False)
            ax1.set_xlabel(region)
            ax1.set_ylabel('')
            ax1.set(frame_on=False)
            ax1.xaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])

            for line in ax1.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax1.yaxis.get_ticklines():
                line.set_visible(False)


        #plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), ncol=1, frameon=False)  # bbox_to_anchor=(1.1, 0.5),
        # plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        plt.savefig(save_path + f'Spatial_mouse_eachregion.' + self.fig_format)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9



        #############################################################################################
        #--------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # 3D Scatter according regions
        # multiview
        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.99

        #fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi, projection='3d')
        fig = plt.figure(figsize=(10, 2), dpi=self.fig_dpi)
        axes = []
        for i in range(5):
            ax = fig.add_subplot(1, 5, i+1, projection='3d')
            axes.append(ax)
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for i, region in zip(range(len(hippocampal_region_mouse_list)), ['M-' + x for x in hippocampal_region_mouse_list]):
            region = region.strip('M-')
            print(region)
            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['M-' + region]
            x_vec = adata_m.obs['x_grid']
            y_vec = adata_m.obs['y_grid']
            z_vec = adata_m.obs['z_grid']
            for j in range(len(hippocampal_region_mouse_list)):
                if i != j:
                    axes[j].scatter3D(x_vec, y_vec, z_vec, color='lightgrey', s=marker_s-4, alpha=0.5)

        for i, region in zip(range(len(hippocampal_region_mouse_list)),
                             ['M-' + x for x in hippocampal_region_mouse_list]):
            region = region.strip('M-')
            print(region)
            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['M-' + region]
            x_vec = adata_m.obs['x_grid']
            y_vec = adata_m.obs['y_grid']
            z_vec = adata_m.obs['z_grid']
            axes[i].scatter3D(x_vec, y_vec, z_vec, color=scatter_color, s=marker_s, label=region)

        for region, ax1 in zip(hippocampal_region_mouse_list, axes):
            ax1.grid(True)
            #ax1.set_xlabel(region, labelpad=0.05)
            # ax1.set_ylabel('')
            # ax1.set_zlabel('')

            ax1.set_xlabel(region)
            # ax1.set_ylabel('', linespacing=0)
            # ax1.set_zlabel('', linespacing=0)

            ax1.xaxis.labelpad = -12
            ax1.yaxis.labelpad = 0
            ax1.zaxis.labelpad = 0

            #ax1.set(frame_on=False)
            ax1.xaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])
            ax1.zaxis.set_ticklabels([])

            # No ticks
            # ax1.set_xticks([])
            # ax1.set_yticks([])
            # ax1.set_zticks([])
            #
            # # Transparent spines
            # # ax1.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # # ax1.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # # ax1.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            #
            # for line in ax1.xaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax1.yaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax1.zaxis.get_ticklines():
            #     line.set_visible(False)

            #ax1.view_init(60, -80)

        # plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), ncol=1, frameon=False)  # bbox_to_anchor=(1.1, 0.5),
        # plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        fig.subplots_adjust(wspace=0, hspace=0)

        plt.savefig(save_path + f'Spatial_mouse_eachregion_3D.' + self.fig_format, dpi=self.fig_dpi)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9


        #==============================================================================================
        # Spatial visualization of human
        # 3D coordinates of regions of mouse
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.7
        fig = plt.figure(figsize=(12, 8), dpi=self.fig_dpi)
        axes = plt.axes(projection='3d')
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for region in ['H-' + x for x in hippocampal_region_human_list]:
            region = region.strip('H-')
            print(region)
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['H-' + region]
            x_vec = adata_h.obs['mri_voxel_x']
            y_vec = adata_h.obs['mri_voxel_y']
            z_vec = adata_h.obs['mri_voxel_z']
            axes.scatter3D(x_vec, y_vec, z_vec, color=scatter_color, s=marker_s, label=region)
        axes.grid(False)
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')
        axes.set(frame_on=False)
        fig.subplots_adjust(left=0, right=0.7, bottom=0, top=1)
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
        plt.legend(loc='right', bbox_to_anchor=(1.5, 0.5), ncol=1, frameon=False)  # bbox_to_anchor=(1.1, 0.5),
        plt.savefig(save_path + f'Spatial_human.' + self.fig_format)

        # multiview
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.8
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for region in ['H-' + x for x in hippocampal_region_human_list]:
            region = region.strip('H-')
            print(region)
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['H-' + region]
            x_vec = adata_h.obs['mri_voxel_x']
            y_vec = adata_h.obs['mri_voxel_y']
            z_vec = adata_h.obs['mri_voxel_z']
            ax1.scatter(x_vec, y_vec, color=scatter_color, s=marker_s, label=region)
            ax2.scatter(x_vec, z_vec, color=scatter_color, s=marker_s, label=region)
            ax3.scatter(y_vec, z_vec, color=scatter_color, s=marker_s, label=region)

        ax1.grid(False)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        ax2.grid(False)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')

        ax3.grid(False)
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')

        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticklabels([])

        ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])

        ax3.xaxis.set_ticklabels([])
        ax3.yaxis.set_ticklabels([])

        for line in ax1.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax1.yaxis.get_ticklines():
            line.set_visible(False)

        for line in ax2.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax2.yaxis.get_ticklines():
            line.set_visible(False)

        for line in ax3.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax3.yaxis.get_ticklines():
            line.set_visible(False)

        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), ncol=1, frameon=False)  # bbox_to_anchor=(1.1, 0.5),
        # plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        plt.savefig(save_path + f'Spatial_human_multiview.' + self.fig_format)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9

        # --------------------------------------------------------------------------
        # Scatter according regions
        # multiview
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.18
        rcParams["figure.subplot.top"] = 0.98

        fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi)
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for i, region in zip(range(len(hippocampal_region_human_list)),
                             ['H-' + x for x in hippocampal_region_human_list]):
            region = region.strip('H-')
            print(region)
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['H-' + region]
            x_vec = adata_h.obs['mri_voxel_x']
            y_vec = adata_h.obs['mri_voxel_y']
            z_vec = adata_h.obs['mri_voxel_z']
            for j in range(len(hippocampal_region_human_list)):
                if i != j:
                    axes[j].scatter(x_vec, z_vec, color='lightgrey', s=marker_s*2)
            #axes[i].scatter(x_vec, z_vec, color=scatter_color, s=marker_s*3, label=region)


        for i, region in zip(range(len(hippocampal_region_human_list)),
                             ['H-' + x for x in hippocampal_region_human_list]):
            region = region.strip('H-')
            print(region)
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['H-' + region]
            x_vec = adata_h.obs['mri_voxel_x']
            y_vec = adata_h.obs['mri_voxel_y']
            z_vec = adata_h.obs['mri_voxel_z']
            axes[i].scatter(x_vec, z_vec, color=scatter_color, s=marker_s*3, label=region)


        for region, ax1 in zip(hippocampal_region_human_list, axes):
            ax1.grid(False)
            ax1.set_xlabel(region)
            ax1.set_ylabel('')
            ax1.set(frame_on=False)
            ax1.xaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])

            for line in ax1.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax1.yaxis.get_ticklines():
                line.set_visible(False)

        # plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), ncol=1, frameon=False)  # bbox_to_anchor=(1.1, 0.5),
        # plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        plt.savefig(save_path + f'Spatial_human_eachregion.' + self.fig_format)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.9



        #------------------------------------------------------------------------------
        # 3D plot of each human brain region
        #---------------------------------------------------------------------
        #############################################################################################
        # --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # 3D Scatter according regions
        # multiview
        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.99

        #marker_s =

        # fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi, projection='3d')
        fig = plt.figure(figsize=(10, 2), dpi=self.fig_dpi)
        axes = []
        for i in range(5):
            ax = fig.add_subplot(1, 5, i + 1, projection='3d')
            axes.append(ax)
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for i, region in zip(range(len(hippocampal_region_human_list)),
                             ['H-' + x for x in hippocampal_region_human_list]):
            region = region.strip('H-')
            print(region)
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['H-' + region]
            x_vec = adata_h.obs['mri_voxel_x']
            y_vec = adata_h.obs['mri_voxel_y']
            z_vec = adata_h.obs['mri_voxel_z']
            for j in range(len(hippocampal_region_human_list)):
                if i != j:
                    axes[j].scatter3D(x_vec, y_vec, z_vec, color='lightgrey', s=marker_s - 1, alpha=0.75)

        for i, region in zip(range(len(hippocampal_region_human_list)),
                             ['H-' + x for x in hippocampal_region_human_list]):
            region = region.strip('H-')
            print(region)
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['H-' + region]
            x_vec = adata_h.obs['mri_voxel_x']
            y_vec = adata_h.obs['mri_voxel_y']
            z_vec = adata_h.obs['mri_voxel_z']
            axes[i].scatter3D(x_vec, y_vec, z_vec, color=scatter_color, s=marker_s, label=region)

        for region, ax1 in zip(hippocampal_region_human_list, axes):
            ax1.grid(True)
            ax1.set_xlabel(region)
            # ax1.set_ylabel('', linespacing=0)
            # ax1.set_zlabel('', linespacing=0)

            ax1.xaxis.labelpad = -12
            ax1.yaxis.labelpad = 0
            ax1.zaxis.labelpad = 0

            #ax1.set(frame_on=False)
            ax1.xaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])
            ax1.zaxis.set_ticklabels([])

            # No ticks
            # ax1.set_xticks([])
            # ax1.set_yticks([])
            # ax1.set_zticks([])
            #
            # # Transparent spines
            # # ax1.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # # ax1.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # # ax1.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            #
            # for line in ax1.xaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax1.yaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax1.zaxis.get_ticklines():
            #     line.set_visible(False)

            #ax1.view_init(60, -80)
            # ax1.tick_params(axis='x', pad=2)  # Adjust as needed
            # ax1.tick_params(axis='y', pad=2)  # Adjust as needed
            # ax1.tick_params(axis='z', pad=2)  # Adjust as needed

        # plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), ncol=1, frameon=False)  # bbox_to_anchor=(1.1, 0.5),
        # plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        fig.subplots_adjust(wspace=0, hspace=0)

        plt.savefig(save_path + f'Spatial_human_eachregion_3D.' + self.fig_format, dpi=self.fig_dpi)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9


        # ------------------------------------------------------------
        # umap plots of region pairs
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.top"] = 0.995
        rcParams["figure.subplot.bottom"] = 0.195

        fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi)

        markersize = 60

        for i in range(len(hippocampal_region_mouse_list)):
            region_m = 'M-'+hippocampal_region_mouse_list[i]
            region_h = 'H-'+hippocampal_region_human_list[i]

            adata_mh = adata_embedding_hip[adata_embedding_hip.obs['region_name'].isin([region_m, region_h])]
            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i])

            sc.pl.umap(adata_mh,
                       color='region_name',
                       palette={region_m: palette[region_m], region_h:palette[region_h]},
                       ax=ax, legend_loc='lower center', show=False, size=markersize) #legend_fontweight='normal',
            #ax.legend()
            #c_p = 0,0
            # handles = [matplotlib.patches.Circle([0, 0], facecolor=palette[name], radius=5) for name in [region_m, region_h]]
            # ax.legend(handles, palette, labels=[region_m, region_h], # title='',
            #            bbox_to_anchor=(0.3, 0),
            #            #bbox_transform=plt.gcf().transFigure,
            #            loc='lower center', frameon=False)

        for ax1 in axes:
            ax1.grid(False)
            ax1.set_xlabel('')
            ax1.set_title('')
            ax1.set_ylabel('')
            ax1.set(frame_on=False)
            ax1.xaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])

            for line in ax1.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax1.yaxis.get_ticklines():
                line.set_visible(False)

        plt.savefig(save_path + f'Spatial_eachregion_pair.' + self.fig_format)


        # ----------------------------------------------------------------------
        # Statistics of the fact: homologous regions such as Mouse CA1 and Human CA1 are more close to each other
        # 1. Distance of pair region samples are significantly higher than those are not homologous
        # boxplot (mark t-test p-values)
        Name_list = ['Field CA1', 'Field CA2', 'Field CA3', 'Dentate gyrus', 'Subiculum', 'Non-homologous']
        Dist_dict = {k:[] for k in Name_list}

        #sc.pp.neighbors(adata_embedding_hip, n_neighbors=self.cfg.ANALYSIS.umap_neighbor, metric='euclidean', use_rep='X')
        #sc.tl.umap(adata_embedding_hip, n_components=2, min_dist=0.3)

        # homologous
        for i in range(len(hippocampal_region_mouse_list)):
            region_m = 'M-' + hippocampal_region_mouse_list[i]
            region_h = 'H-' + hippocampal_region_human_list[i]

            adata_m = adata_embedding_hip[adata_embedding_hip.obs['region_name'].isin([region_m])]
            adata_h = adata_embedding_hip[adata_embedding_hip.obs['region_name'].isin([region_h])]

            dist_m = scipy.spatial.distance_matrix(adata_m.obsm['X_umap'], adata_h.obsm['X_umap'], p=2)
            Dist_dict[hippocampal_region_mouse_list[i]] = list(np.min(dist_m, axis=0).flatten()) + list(np.min(dist_m, axis=1).flatten())

        # Non-homologous
        for i in range(len(hippocampal_region_mouse_list)):
            for j in range(len(hippocampal_region_human_list)):
                if i != j: # and max(i, j) >= 3
                    region_m = 'M-' + hippocampal_region_mouse_list[i]
                    region_h = 'H-' + hippocampal_region_human_list[i]

                    adata_m = adata_embedding_hip[adata_embedding_hip.obs['region_name'].isin([region_m])]
                    adata_h = adata_embedding_hip[adata_embedding_hip.obs['region_name'].isin([region_h])]

                    dist_m = scipy.spatial.distance_matrix(adata_m.obsm['X_umap'], adata_h.obsm['X_umap'], p=2)
                    Dist_dict['Non-homologous'] = Dist_dict['Non-homologous'] + list(np.min(dist_m, axis=0).flatten()) + list(np.min(dist_m, axis=1).flatten())

        Boxplot_dict = {'Region pairs':[], 'Distance':[]}
        for k,v in Dist_dict.items():
            Boxplot_dict['Region pairs'] = Boxplot_dict['Region pairs'] + [k] * len(v)
            Boxplot_dict['Distance'] = Boxplot_dict['Distance'] + v

        Boxplot_df = pd.DataFrame.from_dict(Boxplot_dict)

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.22
        rcParams["figure.subplot.bottom"] = 0.3
        rcParams["figure.subplot.top"] = 0.9

        color_pal = {k:v for k,v in zip(Name_list, color_list[0:6])}
        color_pal['Non-homologous'] = 'grey'

        plt.figure(figsize=(5, 8), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        # sns.set(rc={"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)})
        ax = sns.boxplot(x="Region pairs", y='Distance', data=Boxplot_df, order=Name_list, palette=color_pal,
                         width=0.75)  #
        add_stat_annotation(ax, data=Boxplot_df, x="Region pairs", y='Distance', order=Name_list,
                            box_pairs=[('Field CA1', 'Non-homologous'),
                                       ('Field CA2', 'Non-homologous'),
                                       ('Field CA3', 'Non-homologous'),
                                       ('Dentate gyrus', 'Non-homologous'),
                                       ('Subiculum', 'Non-homologous')],
                            test='t-test_ind', text_format='star', loc='inside', verbose=2)  # text_format='star'
        plt.title('')
        plt.ylabel('Distance')
        plt.xlabel('')
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.savefig(save_path + 'pair_region_distance.' + self.fig_format, format=self.fig_format)
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.9

        #-----------------------------------------------------------------------------------------------------------
        # Statistics of the fact: homologous regions such as Mouse CA1 and Human CA1 are more close to each other
        # 1. Alignment score of pair region samples are significantly higher than those are not homologous
        # boxplot (mark t-test p-values)
        Name_list = ['Field CA1', 'Field CA2', 'Field CA3', 'Dentate gyrus', 'Subiculum', 'Non-homologous']
        Dist_dict = {k: 0 for k in Name_list}

        # sc.pp.neighbors(adata_embedding_hip, n_neighbors=self.cfg.ANALYSIS.umap_neighbor, metric='euclidean', use_rep='X')
        # sc.tl.umap(adata_embedding_hip, n_components=2, min_dist=0.3)

        # homologous
        for i in range(len(hippocampal_region_mouse_list)):
            region_m = hippocampal_region_mouse_list[i] #'M-' +
            region_h = hippocampal_region_human_list[i] #'H-' +

            adata_m = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_m])]
            adata_h = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_h])]

            X = np.concatenate((adata_m.obsm['X_umap'], adata_h.obsm['X_umap']), axis=0)
            Y = np.concatenate([np.zeros((adata_m.n_obs, 1)), np.ones((adata_h.n_obs, 1))], axis=0)
            Dist_dict[hippocampal_region_mouse_list[i]] = seurat_alignment_score(X, Y)

        # Non-homologous
        seurat_as_list = []
        for i in range(len(self.mouse_64_labels_list)):
            for j in range(len(self.human_88_labels_list)):
                if i != j: #and max(i, j) >= 3
                    region_m = self.mouse_64_labels_list[i] #'M-' +
                    region_h = self.human_88_labels_list[i] #'H-' +

                    adata_m = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_m])]
                    adata_h = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_h])]

                    X = np.concatenate((adata_m.obsm['X_umap'], adata_h.obsm['X_umap']), axis=0)
                    Y = np.concatenate([np.zeros((adata_m.n_obs, 1)), np.ones((adata_h.n_obs, 1))], axis=0)

                    seurat_as_list.append(seurat_alignment_score(X, Y))

        Dist_dict['Non-homologous'] = np.mean(seurat_as_list)

        Boxplot_dict = {'Region pairs': [], 'Alignment score': []}
        for k, v in Dist_dict.items():
            Boxplot_dict['Region pairs'] = Boxplot_dict['Region pairs'] + [k]
            Boxplot_dict['Alignment score'] = Boxplot_dict['Alignment score'] + [v]

        Boxplot_df = pd.DataFrame.from_dict(Boxplot_dict)

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.22
        rcParams["figure.subplot.bottom"] = 0.3
        rcParams["figure.subplot.top"] = 0.9

        color_pal = {k: v for k, v in zip(Name_list, color_list[0:6])}
        color_pal['Non-homologous'] = 'grey'

        plt.figure(figsize=(5, 8), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        # sns.set(rc={"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)})
        ax = sns.barplot(x="Region pairs", y='Alignment score', data=Boxplot_df, order=Name_list, palette=color_pal,
                         width=0.75)  #
        plt.title('')
        plt.ylabel('Alignment score')
        plt.xlabel('')
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.savefig(save_path + 'pair_region_distance_alignment_score.' + self.fig_format, format=self.fig_format)
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.9


        #---------------------------------------------------------------------------------------
        # Fold change of alignement score
        Name_list = ['Field CA1', 'Field CA2', 'Field CA3', 'Dentate gyrus', 'Subiculum']
        Dist_dict = {k: 0 for k in Name_list}

        # sc.pp.neighbors(adata_embedding_hip, n_neighbors=self.cfg.ANALYSIS.umap_neighbor, metric='euclidean', use_rep='X')
        # sc.tl.umap(adata_embedding_hip, n_components=2, min_dist=0.3)

        # homologous
        for i in range(len(hippocampal_region_mouse_list)):
            region_m = hippocampal_region_mouse_list[i]  # 'M-' +
            region_h = hippocampal_region_human_list[i]  # 'H-' +

            adata_m = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_m])]
            adata_h = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_h])]

            X = np.concatenate((adata_m.obsm['X_umap'], adata_h.obsm['X_umap']), axis=0)
            Y = np.concatenate([np.zeros((adata_m.n_obs, 1)), np.ones((adata_h.n_obs, 1))], axis=0)
            Dist_dict[hippocampal_region_mouse_list[i]] = seurat_alignment_score(X, Y)

        # Non-homologous
        seurat_as_list = []
        for i in range(len(self.mouse_64_labels_list)):
            for j in range(len(self.human_88_labels_list)):
                if i != j:  # and max(i, j) >= 3
                    region_m = self.mouse_64_labels_list[i]  # 'M-' +
                    region_h = self.human_88_labels_list[i]  # 'H-' +

                    adata_m = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_m])]
                    adata_h = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_h])]

                    X = np.concatenate((adata_m.obsm['X_umap'], adata_h.obsm['X_umap']), axis=0)
                    Y = np.concatenate([np.zeros((adata_m.n_obs, 1)), np.ones((adata_h.n_obs, 1))], axis=0)

                    seurat_as_list.append(seurat_alignment_score(X, Y))

        Non_homo_score = np.mean(seurat_as_list)

        Boxplot_dict = {'Region pairs': [], 'Alignment score fold change': []}
        for k, v in Dist_dict.items():
            Boxplot_dict['Region pairs'] = Boxplot_dict['Region pairs'] + [k]
            Boxplot_dict['Alignment score fold change'] = Boxplot_dict['Alignment score fold change'] + [v/Non_homo_score]

        Boxplot_df = pd.DataFrame.from_dict(Boxplot_dict)

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.24
        rcParams["figure.subplot.bottom"] = 0.3
        rcParams["figure.subplot.top"] = 0.92

        color_pal = {k: v for k, v in zip(Name_list, color_list[0:6])}
        color_pal['Non-homologous'] = 'grey'

        plt.figure(figsize=(1.8, 3), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        # sns.set(rc={"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)})
        ax = sns.barplot(x="Region pairs", y='Alignment score fold change', data=Boxplot_df, order=Name_list, palette=color_pal,
                         width=0.62)  #
        ax.axhline(1, color='grey', linestyle='--')
        plt.title('')
        plt.ylabel('Alignment score fold change')
        plt.xlabel('')
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.savefig(save_path + 'pair_region_distance_fold_change.' + self.fig_format, format=self.fig_format)
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.9



        # Marker genes of CA1,CA2,CA2,Dentate gyrus
        # CA1:
        adata_mouse_embedding_hip = adata_embedding_hip[adata_embedding_hip.obs['dataset'].isin(['Mouse'])]
        adata_human_embedding_hip = adata_embedding_hip[adata_embedding_hip.obs['dataset'].isin(['Human'])]

        Mouse_markergene_list = ['Neurod6', 'Scgn', 'Tspan18', 'Pdyn', 'Gfra1']#['Itpka', 'Amigo2', 'Hs3st4', 'Lrrtm4', 'Nts'] #['Fndc1', 'Scgn', 'Tspan18', 'Chrna1', 'Nts']# 'Gfra1'
        Human_markergene_list = ['NEUROD6', 'SCGN', 'TSPAN18', 'PDYN', 'GFRA1']#['ITPKA', 'AMIGO2', 'HS3ST4', 'LRRTM4', 'NTSR2']

        adata_mouse_exp_hip = self.adata_mouse_exp[self.adata_mouse_exp.obs['region_name'].isin(hippocampal_region_mouse_list)]
        adata_human_exp_hip = self.adata_human_exp[
            self.adata_human_exp.obs['region_name'].isin(hippocampal_region_human_list)]

        print(self.adata_human_exp.var_names.tolist())
        print(self.adata_mouse_exp.var_names.tolist())

        #color_map = 'viridis'#'magma_r'
        c = Colormap()
        #color_map = c.cmap_linear('#3288BD', 'white', '#D53E4F')
        color_map = c.cmap_linear('white', 'white', '#D53E4F')

        markersize = 40
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.18
        rcParams["figure.subplot.top"] = 0.98
        # mouse marker genes
        fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi)
        for i in range(len(Mouse_markergene_list)):
            m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_mouse_embedding_hip.obs[m_gene] = adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)
            diff = False
            for k, v in zip(adata_mouse_embedding_hip.obs_names, adata_mouse_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i])

            sc.pl.umap(adata_mouse_embedding_hip,
                       color= m_gene,
                       #palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize, color_map=color_map, colorbar_loc=None)

            ax.grid(False)
            ax.set_xlabel(m_gene, fontdict=dict(style='italic'))
            ax.set_title('')
            ax.set_ylabel('')
            ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

            for line in ax.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.yaxis.get_ticklines():
                line.set_visible(False)

        plt.savefig(save_path + f'Mouse_marker_gene_human.' + self.fig_format)

        # human marker genes
        markersize = 60

        fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi)
        for i in range(len(Human_markergene_list)):
            #m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_human_embedding_hip.obs[h_gene] = adata_human_exp_hip[:, h_gene].X.toarray().reshape(-1) # normalize(adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)[:, np.newaxis], norm='max', axis=0).ravel()

            diff = False
            for k, v in zip(adata_human_embedding_hip.obs_names, adata_human_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i])

            sc.pl.umap(adata_human_embedding_hip,
                       color=h_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None)

            ax.grid(False)
            ax.set_xlabel(h_gene, fontdict=dict(style='italic'))
            ax.set_title('')
            ax.set_ylabel('')
            ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

            for line in ax.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.yaxis.get_ticklines():
                line.set_visible(False)

        plt.savefig(save_path + f'Human_marker_gene_human.' + self.fig_format)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9

        # ----------------------------------------------------------------------
        # Another pair marker gene
        Mouse_markergene_list = ['Itpka', 'Amigo2', 'Hs3st4', 'Lrrtm4', 'Nts'] #['Fndc1', 'Scgn', 'Tspan18', 'Chrna1', 'Nts']# 'Gfra1'
        Human_markergene_list = ['ITPKA', 'AMIGO2', 'HS3ST4', 'LRRTM4', 'NTSR2']

        adata_mouse_exp_hip = self.adata_mouse_exp[
            self.adata_mouse_exp.obs['region_name'].isin(hippocampal_region_mouse_list)]
        adata_human_exp_hip = self.adata_human_exp[
            self.adata_human_exp.obs['region_name'].isin(hippocampal_region_human_list)]

        #print(self.adata_mouse_exp.var_names.tolist())

        #color_map = 'viridis'  # 'magma_r'
        c = Colormap()
        color_map = c.cmap_linear('#3288BD', 'white', '#D53E4F')

        markersize = 40
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.18
        rcParams["figure.subplot.top"] = 0.98
        # mouse marker genes
        fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi)
        for i in range(len(Mouse_markergene_list)):
            m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_mouse_embedding_hip.obs[m_gene] = adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)
            diff = False
            for k, v in zip(adata_mouse_embedding_hip.obs_names, adata_mouse_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i])

            sc.pl.umap(adata_mouse_embedding_hip,
                       color=m_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None)

            ax.grid(False)
            ax.set_xlabel(m_gene, fontdict=dict(style='italic'))
            ax.set_title('')
            ax.set_ylabel('')
            ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

            for line in ax.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.yaxis.get_ticklines():
                line.set_visible(False)

        plt.savefig(save_path + f'Mouse_marker_gene_mouse.' + self.fig_format)

        # human marker genes
        markersize = 60

        fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi)
        for i in range(len(Human_markergene_list)):
            # m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_human_embedding_hip.obs[h_gene] = adata_human_exp_hip[:, h_gene].X.toarray().reshape(
                -1)  # normalize(adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)[:, np.newaxis], norm='max', axis=0).ravel()

            diff = False
            for k, v in zip(adata_human_embedding_hip.obs_names, adata_human_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i])

            sc.pl.umap(adata_human_embedding_hip,
                       color=h_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None)

            ax.grid(False)
            ax.set_xlabel(h_gene, fontdict=dict(style='italic'))
            ax.set_title('')
            ax.set_ylabel('')
            ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

            for line in ax.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.yaxis.get_ticklines():
                line.set_visible(False)

        plt.savefig(save_path + f'Human_marker_gene_mouse.' + self.fig_format)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9




        #---------------------------------------------------------------------------------
        # UMAP 3 components
        ###################################################################################

        sc.tl.umap(adata_embedding_hip, n_components=3, min_dist=0.3)

        # ------------------------------------------------------------
        # umap plots of region pairs
        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.top"] = 0.99
        rcParams["figure.subplot.bottom"] = 0.2

        fig = plt.figure(figsize=(10, 2.2), dpi=self.fig_dpi)
        axes = []
        for i in range(5):
            ax = fig.add_subplot(1, 5, i + 1, projection='3d')
            ax.grid(True)
            axes.append(ax)

        markersize = 60

        for i in range(len(hippocampal_region_mouse_list)):
            region_m = 'M-' + hippocampal_region_mouse_list[i]
            region_h = 'H-' + hippocampal_region_human_list[i]

            adata_mh = adata_embedding_hip[adata_embedding_hip.obs['region_name'].isin([region_m, region_h])]
            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i], projection='3d', size=markersize-55, alpha=0.8) #, wspace=0.25 , s=1, alpha=0.2,

            # umap1_x, umap1_y, umap1_z = adata_embedding_hip.obsm['X_umap'].toarray()[:, 0], \
            #                    adata_embedding_hip.obsm['X_umap'].toarray()[:, 1], \
            #                             adata_embedding_hip.obsm['X_umap'].toarray()[:, 2]
            ax.grid(True)
            sc.pl.umap(adata_mh,
                       color='region_name',
                       palette={region_m: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc='lower center', show=False, size=markersize,  projection='3d', alpha=1)  # legend_fontweight='normal',
            # ax.legend()
            # c_p = 0,0
            # handles = [matplotlib.patches.Circle([0, 0], facecolor=palette[name], radius=5) for name in [region_m, region_h]]
            # ax.legend(handles, palette, labels=[region_m, region_h], # title='',
            #            bbox_to_anchor=(0.3, 0),
            #            #bbox_transform=plt.gcf().transFigure,
            #            loc='lower center', frameon=False)

        for ax1 in axes:
            ax1.grid(True)
            ax1.set_xlabel('')
            ax1.set_ylabel('')
            ax1.set_zlabel('')

            ax1.xaxis.labelpad = -12
            ax1.yaxis.labelpad = 0
            ax1.zaxis.labelpad = 0

            #ax1.set(frame_on=False)
            ax1.xaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])
            ax1.zaxis.set_ticklabels([])

            ax.set_title('')

            # No ticks
            # ax1.set_xticks([])
            # ax1.set_yticks([])
            # ax1.set_zticks([])
            #
            # # Transparent spines
            # # ax1.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # # ax1.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # # ax1.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            #
            # for line in ax1.xaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax1.yaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax1.zaxis.get_ticklines():
            #     line.set_visible(False)

            #ax1.view_init(60, -80)
        plt.subplots_adjust(wspace=0, hspace=0)

        plt.savefig(save_path + f'Spatial_eachregion_pair_3D.' + self.fig_format, dpi=self.fig_dpi)



        ####################################################################################
        # 3D plot of
        ######################
        # Marker genes of CA1,CA2,CA2,Dentate gyrus
        # CA1:
        adata_mouse_embedding_hip = adata_embedding_hip[adata_embedding_hip.obs['dataset'].isin(['Mouse'])]
        adata_human_embedding_hip = adata_embedding_hip[adata_embedding_hip.obs['dataset'].isin(['Human'])]

        Mouse_markergene_list = ['Neurod6', 'Scgn', 'Tspan18', 'Pdyn',
                                 'Gfra1']  # ['Itpka', 'Amigo2', 'Hs3st4', 'Lrrtm4', 'Nts'] #['Fndc1', 'Scgn', 'Tspan18', 'Chrna1', 'Nts']# 'Gfra1'
        Human_markergene_list = ['NEUROD6', 'SCGN', 'TSPAN18', 'PDYN',
                                 'GFRA1']  # ['ITPKA', 'AMIGO2', 'HS3ST4', 'LRRTM4', 'NTSR2']

        adata_mouse_exp_hip = self.adata_mouse_exp[
            self.adata_mouse_exp.obs['region_name'].isin(hippocampal_region_mouse_list)]
        adata_human_exp_hip = self.adata_human_exp[
            self.adata_human_exp.obs['region_name'].isin(hippocampal_region_human_list)]

        print(self.adata_human_exp.var_names.tolist())
        print(self.adata_mouse_exp.var_names.tolist())

        # color_map = 'viridis'#'magma_r'
        c = Colormap()
        #color_map = c.cmap_linear('#3288BD', 'white', '#D53E4F')
        color_map = c.cmap_linear('#3288BD', 'white', '#D53E4F')

        markersize = 20
        # mouse marker genes
        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.top"] = 0.99
        rcParams["figure.subplot.bottom"] = 0.1

        fig = plt.figure(figsize=(10, 2), dpi=self.fig_dpi)
        axes = []
        for i in range(5):
            ax = fig.add_subplot(1, 5, i + 1, projection='3d')
            ax.grid(True)
            axes.append(ax)

        for i in range(len(Mouse_markergene_list)):
            m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_mouse_embedding_hip.obs[m_gene] = adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)
            diff = False
            for k, v in zip(adata_mouse_embedding_hip.obs_names, adata_mouse_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i], projection='3d', size=markersize-17, alpha=0.2) #, alpha=0.15
            ax.grid(True)
            # if i == len(Mouse_markergene_list):
            #     colorbar_loc = 'right'
            # else:
            #     colorbar_loc = None



            sc.pl.umap(adata_mouse_embedding_hip,
                       color=m_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None, projection='3d', alpha=1)

            ax.grid(True)

            if len(m_gene) <= 4:
                m_gene = m_gene + ' '
            ax.set_xlabel(m_gene, fontdict=dict(style='italic')) #, linespacing=-12
            ax.set_ylabel('', linespacing=0)
            ax.set_zlabel('', linespacing=0)

            ax.set_title('')

            ax.xaxis.labelpad = -12
            ax.yaxis.labelpad = 0
            ax.zaxis.labelpad = 0

            #ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

            # No ticks
            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.set_zticks([])

            # Transparent spines
            # ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

            # for line in ax.xaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax.yaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax.zaxis.get_ticklines():
            #     line.set_visible(False)

            #ax.view_init(60, -80)


        plt.subplots_adjust(wspace=0, hspace=0)

        plt.savefig(save_path + f'Mouse_marker_gene_human_3D.' + self.fig_format, dpi=self.fig_dpi)

        # human marker genes
        markersize = 80

        fig = plt.figure(figsize=(10, 2), dpi=self.fig_dpi)
        axes = []
        for i in range(5):
            ax = fig.add_subplot(1, 5, i + 1, projection='3d')
            axes.append(ax)

        for i in range(len(Human_markergene_list)):
            # m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_human_embedding_hip.obs[h_gene] = adata_human_exp_hip[:, h_gene].X.toarray().reshape(
                -1)  # normalize(adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)[:, np.newaxis], norm='max', axis=0).ravel()

            diff = False
            for k, v in zip(adata_human_embedding_hip.obs_names, adata_human_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i], size=markersize-75, projection='3d', alpha=0.8) #, alpha=0.15
            ax.grid(True)
            # if i == len(Human_markergene_list):
            #     colorbar_loc = 'right'
            # else:
            #     colorbar_loc = None

            sc.pl.umap(adata_human_embedding_hip,
                       color=h_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None, projection='3d', alpha=1)

            ax.grid(True)
            if len(h_gene) <= 4:
                h_gene = h_gene + ' '
            ax.set_xlabel(h_gene, fontdict=dict(style='italic')) #, linespacing=-12
            ax.set_ylabel('', linespacing=0)
            ax.set_zlabel('', linespacing=0)

            ax.set_title('')

            ax.xaxis.labelpad = -12
            ax.yaxis.labelpad = 0
            ax.zaxis.labelpad = 0

            #ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

            # No ticks
            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.set_zticks([])
            #
            # # Transparent spines
            # # ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # # ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # # ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            #
            # for line in ax.xaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax.yaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax.zaxis.get_ticklines():
            #     line.set_visible(False)

            #ax.view_init(60, -80)

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(save_path + f'Human_marker_gene_human_3D.' + self.fig_format, dpi=self.fig_dpi)


        # ----------------------------------------------------------------------
        # Another pair marker gene
        Mouse_markergene_list = ['Itpka', 'Amigo2', 'Hs3st4', 'Lrrtm4',
                                 'Nts']  # ['Fndc1', 'Scgn', 'Tspan18', 'Chrna1', 'Nts']# 'Gfra1'
        Human_markergene_list = ['ITPKA', 'AMIGO2', 'HS3ST4', 'LRRTM4', 'NTSR2']

        adata_mouse_exp_hip = self.adata_mouse_exp[
            self.adata_mouse_exp.obs['region_name'].isin(hippocampal_region_mouse_list)]
        adata_human_exp_hip = self.adata_human_exp[
            self.adata_human_exp.obs['region_name'].isin(hippocampal_region_human_list)]

        # print(self.adata_mouse_exp.var_names.tolist())

        # color_map = 'viridis'  # 'magma_r'
        c = Colormap()
        color_map = c.cmap_linear('#3288BD', 'white', '#D53E4F')

        markersize = 20

        fig = plt.figure(figsize=(10, 2), dpi=self.fig_dpi)
        axes = []
        for i in range(5):
            ax = fig.add_subplot(1, 5, i + 1, projection='3d')
            ax.grid(True)
            axes.append(ax)

        for i in range(len(Mouse_markergene_list)):
            m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_mouse_embedding_hip.obs[m_gene] = adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)
            diff = False
            for k, v in zip(adata_mouse_embedding_hip.obs_names, adata_mouse_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i], size=markersize-17, projection='3d', alpha=0.2) #, alpha=0.15
            ax.grid(True)
            # if i == len(Mouse_markergene_list):
            #     colorbar_loc = 'right'
            # else:
            #     colorbar_loc = None
            sc.pl.umap(adata_mouse_embedding_hip,
                       color=m_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None, projection='3d', alpha=1)

            ax.grid(True)
            if len(m_gene) <= 4:
                m_gene = m_gene + ' '
            ax.set_xlabel(m_gene, fontdict=dict(style='italic')) #, linespacing=-12
            ax.set_ylabel('', linespacing=0)
            ax.set_zlabel('', linespacing=0)

            ax.set_title('')

            ax.xaxis.labelpad = -12
            ax.yaxis.labelpad = 0
            ax.zaxis.labelpad = 0

            #ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

            # No ticks
            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.set_zticks([])
            #
            # # Transparent spines
            # # ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # # ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # # ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            #
            # for line in ax.xaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax.yaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax.zaxis.get_ticklines():
            #     line.set_visible(False)

            #ax.view_init(60, -80)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(save_path + f'Mouse_marker_gene_mouse_3D.' + self.fig_format, dpi=self.fig_dpi)

        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.right"] = 0.7
        rcParams["figure.subplot.top"] = 0.99
        rcParams["figure.subplot.bottom"] = 0.1

        from numpy.random import randn

        fig, ax = plt.subplots(figsize=(3, 2), dpi=500)

        data = np.clip(randn(250, 250), 0, 4)

        cax = ax.imshow(data, cmap=color_map)
        ax.set_title('')

        cbar = fig.colorbar(cax, pad=0.1, fraction=0.10, aspect=5, shrink=0.4, location='right', ticks=[0, 4])
        cbar.ax.set_yticklabels(['Low', 'High'])  #
        cbar.outline.set_visible(False)

        plt.savefig(save_path + f'colorbar.' + self.fig_format)
        #plt.show()


        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.top"] = 0.99
        rcParams["figure.subplot.bottom"] = 0.1

        # human marker genes
        markersize = 80

        fig = plt.figure(figsize=(10, 2), dpi=self.fig_dpi)
        axes = []
        for i in range(5):
            ax = fig.add_subplot(1, 5, i + 1, projection='3d')
            ax.grid(True)
            axes.append(ax)


        for i in range(len(Human_markergene_list)):
            # m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_human_embedding_hip.obs[h_gene] = adata_human_exp_hip[:, h_gene].X.toarray().reshape(
                -1)  # normalize(adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)[:, np.newaxis], norm='max', axis=0).ravel()

            diff = False
            for k, v in zip(adata_human_embedding_hip.obs_names, adata_human_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i], size=markersize-75, projection='3d', alpha=0.8) #, wspace=0.25 , alpha=0.15

            # if i == len(Mouse_markergene_list):
            #     colorbar_loc = 'right'
            # else:
            #     colorbar_loc = None

            sc.pl.umap(adata_human_embedding_hip,
                       color=h_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None, projection='3d', alpha=1)

            ax.grid(True)
            if len(h_gene) <= 4:
                h_gene = h_gene + ' '
            ax.set_xlabel(h_gene, fontdict=dict(style='italic')) #, linespacing=-12
            ax.set_ylabel('', linespacing=0)
            ax.set_zlabel('', linespacing=0)
            ax.set_title('')

            ax.xaxis.labelpad = -12
            ax.yaxis.labelpad = 0
            ax.zaxis.labelpad = 0

            #ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

            # No ticks
            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.set_zticks([])
            #
            # # Transparent spines
            # # ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # # ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # # ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            #
            # for line in ax.xaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax.yaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax.zaxis.get_ticklines():
            #     line.set_visible(False)

            #ax.view_init(60, -80)

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(save_path + f'Human_marker_gene_mouse_3D.' + self.fig_format, dpi=self.fig_dpi)




        #plt.scatter(x, y, c=colors)

        #plt.colorbar()


        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.top"] = 0.9
        rcParams["figure.subplot.bottom"] = 0.1


        return None




    def experiment_2_spatial_isocortex(self):
        """
        Spatial integration and alignment of isocortex regions, to display that BrainAlign keeps the relative spatial positions of
        samples and help to detect STs difference across species.

        - 1. Visualize the anatomical regions of two regions;
        - 2. Spatial scatter of regions separately;
        - 3. UMAP of two regions together;
        - 4. PAGA abstracted graph of regions of two species;
        - 4. For each Marker genes, plot each expression of marker genes on UMAP of two species (with different thresholds)

        :return: None
        """

        sns.set(style='white')
        TINY_SIZE = 7  # 39
        SMALL_SIZE = 7  # 42
        MEDIUM_SIZE = 8  # 46
        BIGGER_SIZE = 8  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/4_spatial_analysis/2_experiment_spatial_isocortex/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Visualize the anatomical regions of regions;
        isocortex_region_human_list = ['cingulate gyrus',
                                       'limbic lobe',
                                       'parahippocampal gyrus',
                                       'piriform cortex',
                                       'anterior orbital gyrus',
                                       'frontal operculum',
                                       'frontal pole',
                                       'gyrus rectus',
                                       'inferior frontal gyrus',
                                       'inferior rostral gyrus',
                                       'lateral orbital gyrus',
                                       'medial orbital gyrus',
                                       'middle frontal gyrus',
                                       '"paracentral lobule, anterior part"',
                                       'paraterminal gyrus',
                                       'parolfactory gyri',
                                       'posterior orbital gyrus',
                                       'precentral gyrus',
                                       'superior frontal gyrus',
                                       'insula',
                                       'cuneus',
                                       'inferior occipital gyrus',
                                       'lingual gyrus',
                                       'occipital pole',
                                       'occipito-temporal gyrus',
                                       'superior occipital gyrus',
                                       'inferior parietal lobule',
                                       '"paracentral lobule, posterior part"',
                                       'postcentral gyrus',
                                       'precuneus',
                                       'superior parietal lobule',
                                       'fusiform gyrus',
                                       'Heschl\'s gyrus',
                                       'inferior temporal gyrus',
                                       'middle temporal gyrus',
                                       'planum polare',
                                       'planum temporale',
                                       'superior temporal gyrus', 'temporal pole', 'transverse gyri']

        #isocortex_acronym_mouse_list = ['CA1', 'CA2', 'CA3', 'DG', 'SUB']
        isocortex_region_mouse_list = ['Agranular insular area',
                                        'Anterior cingulate area',
                                        'Auditory areas',
                                        'Ectorhinal area',
                                        'Gustatory areas',
                                        'Infralimbic area',
                                        'Orbital area',
                                        'Perirhinal area',
                                        'Posterior parietal association areas',
                                        'Prelimbic area',
                                        'Retrosplenial area',
                                        'Primary motor area',
                                        'Secondary motor area',
                                        'Primary somatosensory area',
                                        'Supplemental somatosensory area',
                                        'Temporal association areas',
                                        'Visceral area',
                                        'Visual areas']

        #isocortex_acronym_human_list = ['CA1', 'CA2', 'CA3', 'DG', 'S']

        #------Compute alignment score of all the pairs and select the top k pairs
        # k = 5
        minimum_sample_num = 7
        alignment_score_dict = {}
        for region_m in isocortex_region_mouse_list:
            for region_h in isocortex_region_human_list:

                adata_m = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_m])]
                adata_h = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_h])]

                X = np.concatenate((adata_m.X, adata_h.X), axis=0)
                Y = np.concatenate([np.zeros((adata_m.n_obs, 1)), np.ones((adata_h.n_obs, 1))], axis=0)
                score = seurat_alignment_score(X, Y)
                if not np.isnan(score) and adata_m.n_obs >=minimum_sample_num and adata_h.n_obs >= minimum_sample_num:
                    alignment_score_dict[f'{region_m}-{region_h}'] = seurat_alignment_score(X, Y)

        alignment_score_dict_sorted = dict(sorted(alignment_score_dict.items(), key=lambda item:item[1], reverse=True))
        #print(alignment_score_dict_sorted)

        topk = 5
        isocortex_region_human_list = []
        isocortex_region_mouse_list = []
        for k, v in alignment_score_dict_sorted.items():
            m_region = k.split('-')[0]
            h_region = k.split('-')[1]
            if len(isocortex_region_mouse_list) < topk:
                if m_region not in set(isocortex_region_mouse_list) and h_region not in set(isocortex_region_human_list):
                    isocortex_region_mouse_list.append(m_region)
                    isocortex_region_human_list.append(h_region)
        print(isocortex_region_mouse_list)
        print(isocortex_region_human_list)



        isocortex_region_list = isocortex_region_mouse_list + isocortex_region_human_list
        #isocortex_acronym_list = isocortex_acronym_mouse_list + isocortex_acronym_human_list

        adata_human_embedding_hip = self.adata_human_embedding[self.adata_human_embedding.obs['region_name'].isin(isocortex_region_human_list)]
        adata_mouse_embedding_hip = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['region_name'].isin(isocortex_region_mouse_list)]
        adata_human_embedding_hip.obs['acronym'] = ['H-' + x for x in adata_human_embedding_hip.obs['acronym']]
        adata_mouse_embedding_hip.obs['acronym'] = ['M-' + x for x in adata_mouse_embedding_hip.obs['acronym']]

        adata_human_embedding_hip.obs['region_name'] = ['H-' + x for x in adata_human_embedding_hip.obs['acronym']]
        adata_mouse_embedding_hip.obs['region_name'] = ['M-' + x for x in adata_mouse_embedding_hip.obs['acronym']]

        adata_embedding_hip = self.adata_embedding[self.adata_embedding.obs['region_name'].isin(isocortex_region_list)]

        region_name_map_list = ['M-'+x for x in isocortex_region_mouse_list] + ['H-'+x for x in isocortex_region_human_list]
        #acronym_name_map_list = ['M-'+x for x in isocortex_acronym_mouse_list] + ['H-'+x for x in isocortex_acronym_human_list]

        region_name_map_dict = {k:v for k,v in zip(isocortex_region_list, region_name_map_list)}
        #acronym_name_map_dict = {k: v for k, v in zip(isocortex_acronym_list, acronym_name_map_list)}

        adata_embedding_hip.obs['region_name'] = [region_name_map_dict[x] for x in adata_embedding_hip.obs['region_name']]
        #adata_embedding_hip.obs['acronym'] = [acronym_name_map_dict[x] for x in
                                                  #adata_embedding_hip.obs['acronym']]

        mouse_human_64_88_color_dict = dict({'M-'+k:v for k,v in self.mouse_64_color_dict.items()})
        mouse_human_64_88_color_dict.update({'H-'+k:v for k,v in self.human_88_color_dict.items()})


        # UMAP ------------------------------------------------------
        color_list = sns.color_palette(cc.glasbey, n_colors=len(region_name_map_list))
        palette = {k:v for k,v in zip(region_name_map_list, color_list)}
        #palette_acronym = {k:v for k,v in zip(acronym_name_map_list, color_list)}

        sc.pp.neighbors(adata_embedding_hip, n_neighbors=self.cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(adata_embedding_hip, n_components=2, min_dist=0.3)

        marker_size = 100

        with plt.rc_context({"figure.figsize": (7, 2), "figure.dpi": (self.fig_dpi)}):
            rcParams["figure.subplot.left"] = 0.1
            rcParams["figure.subplot.right"] = 0.35
            rcParams["figure.subplot.bottom"] = 0.25
            rcParams["figure.subplot.top"] = 0.98
            adata_embedding_hip.obs['region_name'] = adata_embedding_hip.obs['region_name'].astype('category')
            adata_embedding_hip.obs['region_name'].cat.reorder_categories(['M-Primary motor area', 'M-Primary somatosensory area', 'M-Supplemental somatosensory area', 'M-Visceral area',
             'M-Posterior parietal association areas', 'H-posterior orbital gyrus', 'H-superior parietal lobule', "H-Heschl\'s gyrus", 'H-parolfactory gyri',
             'H-lateral orbital gyrus'], inplace=True)

            fig = sc.pl.umap(adata_embedding_hip, color=['region_name'],
                             return_fig=True,
                             legend_loc='right margin',
                             palette=palette,  # wspace=wspace,
                             # add_outline=True,
                             # legend_fontsize=12, legend_fontoutline=2,
                             size=20,
                             title=['']
                             )
            # plt.title('')
            fig.savefig(save_path + 'umap_isocortex.' + self.fig_format, format=self.fig_format)

        # PAGA of isocortex
        sc.tl.paga(adata_embedding_hip, groups='region_name')
        rcParams["figure.subplot.right"] = 0.95

        fig, ax = plt.subplots(figsize=(5, 4), dpi=self.fig_dpi)

        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.98
        rcParams["figure.subplot.bottom"] = 0.02
        rcParams["figure.subplot.top"] = 0.98

        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        graph_fontsize = 8

        palette_paga = {}
        for k, v in palette.items():
            palette_paga.update({k: {v: 1}})

        pos_arr = np.zeros((10, 2))
        pos_arr[0:5, 0] = 0
        pos_arr[0:5, 1] = np.array(range(0, 5))
        pos_arr[5:10, 0] = 3
        pos_arr[5:10, 1] = np.array(range(0, 5))
        pos_arr[5:10, 1] = np.array([0, 1, 2, 3, 4])
        pos_arr[0:5, 0] = np.array([0, 0.5, 0, 0.5, 0])
        pos_arr[5:10, 0] = np.array([3, 2.5, 3, 2.5, 3])

        sc.pl.paga(adata_embedding_hip,
                   color=palette_paga,
                   node_size_scale=3,
                   min_edge_width=0.1,
                   pos=pos_arr,
                   ax=ax,
                   max_edge_width=10,
                   fontweight='normal',
                   layout='eq_tree',
                   fontsize=graph_fontsize,
                   fontoutline=0, colorbar=False, threshold=0.8,
                   frameon=False)  # node_size_scale=10, edge_width_scale=2
        plt.title('')
        fig.tight_layout()
        plt.savefig(save_path + 'paga_region_name.' + self.fig_format, format=self.fig_format)

        # paga species
        palette_species = {'M-' + k: self.mouse_color for k in isocortex_region_mouse_list}
        palette_species.update({'H-' + k: self.human_color for k in isocortex_region_human_list})
        palette_paga_species = {}
        for k, v in palette_species.items():
            palette_paga_species.update({k: {v: 1}})

        fig, ax = plt.subplots(figsize=(4.2, 3.7), dpi=self.fig_dpi)

        sc.pl.paga(adata_embedding_hip,
                   color=palette_paga_species,
                   node_size_scale=3,
                   min_edge_width=0.1,
                   pos=pos_arr,
                   ax=ax,
                   max_edge_width=10,
                   layout='eq_tree',
                   fontweight='normal',
                   fontsize=graph_fontsize,
                   fontoutline=0, colorbar=False, threshold=0.8,
                   frameon=False)  # node_size_scale=10, edge_width_scale=2
        plt.title('')
        fig.tight_layout()
        plt.savefig(save_path + 'paga_species.' + self.fig_format, format=self.fig_format)

        #print(self.adata_mouse_embedding)

        # 3D plot parameters
        marker_s = 5

        # --------------------------------------------------------------------------------------
        # 3D coordinates of regions of mouse
        rcParams["figure.subplot.left"] = 0.03
        rcParams["figure.subplot.right"] = 0.55
        fig = plt.figure(figsize=(16, 8), dpi=self.fig_dpi)
        axes = plt.axes(projection='3d')
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for region in ['M-' + x for x in isocortex_region_mouse_list]:
            region = region.strip('M-')
            #print(region)
            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['M-' + region]
            x_vec = adata_m.obs['x_grid']
            y_vec = adata_m.obs['y_grid']
            z_vec = adata_m.obs['z_grid']
            axes.scatter3D(x_vec, y_vec, z_vec, color=scatter_color, s=marker_s, label=region)
        axes.grid(False)
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')
        axes.set(frame_on=False)
        fig.subplots_adjust(left=0, right=0.7, bottom=0, top=1)
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
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), ncol=1, frameon=False) #bbox_to_anchor=(1.1, 0.5),
        plt.savefig(save_path + f'Spatial_mouse.' + self.fig_format)


        # multiview
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.7
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 6))
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        #rcParams["figure.subplot.left"] = 0.05
        for region in ['M-'+x for x in isocortex_region_mouse_list]:
            region = region.strip('M-')
            print(region)
            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['region_name'].isin([region])]
            #print(adata_m)
            #print(adata_m)
            scatter_color = palette['M-'+region]
            x_vec = adata_m.obs['x_grid']
            y_vec = adata_m.obs['y_grid']
            z_vec = adata_m.obs['z_grid']
            ax1.scatter(x_vec, y_vec, color=scatter_color, s=marker_s, label=region)
            ax2.scatter(x_vec, z_vec, color=scatter_color, s=marker_s, label=region)
            ax3.scatter(y_vec, z_vec, color=scatter_color, s=marker_s, label=region)

        ax1.grid(False)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        ax2.grid(False)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')

        ax3.grid(False)
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')

        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticklabels([])

        ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])

        ax3.xaxis.set_ticklabels([])
        ax3.yaxis.set_ticklabels([])


        for line in ax1.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax1.yaxis.get_ticklines():
            line.set_visible(False)

        for line in ax2.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax2.yaxis.get_ticklines():
            line.set_visible(False)

        for line in ax3.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax3.yaxis.get_ticklines():
            line.set_visible(False)

        plt.legend(loc='right', bbox_to_anchor=(2.5, 0.5), ncol=1, frameon=False) #bbox_to_anchor=(1.1, 0.5),
        # plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        plt.savefig(save_path + f'Spatial_mouse_multiview.' + self.fig_format)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9

        #############################################################################################
        # --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # 3D Scatter according regions
        # multiview

        sc.tl.umap(adata_embedding_hip, n_components=3, min_dist=0.3)

        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.99

        # fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi, projection='3d')
        fig = plt.figure(figsize=(10, 2), dpi=self.fig_dpi)
        axes = []
        for i in range(5):
            ax = fig.add_subplot(1, 5, i + 1, projection='3d')
            axes.append(ax)
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for i, region in zip(range(len(isocortex_region_mouse_list)),
                             ['M-' + x for x in isocortex_region_mouse_list]):
            region = region.strip('M-')
            print(region)
            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['M-' + region]
            x_vec = adata_m.obs['x_grid']
            y_vec = adata_m.obs['y_grid']
            z_vec = adata_m.obs['z_grid']
            for j in range(len(isocortex_region_mouse_list)):
                if i != j:
                    axes[j].scatter3D(x_vec, y_vec, z_vec, color='lightgrey', s=marker_s - 4, alpha=0.5)

        for i, region in zip(range(len(isocortex_region_mouse_list)),
                             ['M-' + x for x in isocortex_region_mouse_list]):
            region = region.strip('M-')
            print(region)
            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['M-' + region]
            x_vec = adata_m.obs['x_grid']
            y_vec = adata_m.obs['y_grid']
            z_vec = adata_m.obs['z_grid']
            axes[i].scatter3D(x_vec, y_vec, z_vec, color=scatter_color, s=marker_s, label=region)

        for region, ax1 in zip(isocortex_region_mouse_list, axes):
            ax1.grid(True)
            # ax1.set_xlabel(region, labelpad=0.05)
            # ax1.set_ylabel('')
            # ax1.set_zlabel('')

            ax1.set_xlabel(region)
            # ax1.set_ylabel('', linespacing=0)
            # ax1.set_zlabel('', linespacing=0)

            ax1.xaxis.labelpad = -12
            ax1.yaxis.labelpad = 0
            ax1.zaxis.labelpad = 0

            # ax1.set(frame_on=False)
            ax1.xaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])
            ax1.zaxis.set_ticklabels([])
            #Transparent spines
            # ax1.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # ax1.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # ax1.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            #
            # for line in ax1.xaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax1.yaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax1.zaxis.get_ticklines():
            #     line.set_visible(False)

            # ax1.view_init(60, -80)

        # plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), ncol=1, frameon=False)  # bbox_to_anchor=(1.1, 0.5),
        # plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        fig.subplots_adjust(wspace=0, hspace=0)

        plt.savefig(save_path + f'Spatial_mouse_eachregion_3D.' + self.fig_format, dpi=self.fig_dpi)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9




        # --------------------------------------------------------------------------
        # Scatter according regions
        # multiview
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.19
        rcParams["figure.subplot.top"] = 0.99

        plt.rc('legend', fontsize=22)
        plt.rc('axes', labelsize=22)

        fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi)
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for i, region in zip(range(len(isocortex_region_mouse_list)), ['M-' + x for x in isocortex_region_mouse_list]):
            region = region.strip('M-')
            print(region)
            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['M-' + region]
            x_vec = adata_m.obs['x_grid']
            y_vec = adata_m.obs['y_grid']
            z_vec = adata_m.obs['z_grid']
            for j in range(len(isocortex_region_mouse_list)):
                if i != j:
                    axes[j].scatter(x_vec, y_vec, color='lightgrey', s=marker_s)
            #axes[i].scatter(x_vec, y_vec, color=scatter_color, s=marker_s, label=region)

        for i, region in zip(range(len(isocortex_region_mouse_list)), ['M-' + x for x in isocortex_region_mouse_list]):
            region = region.strip('M-')
            print(region)
            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['M-' + region]
            x_vec = adata_m.obs['x_grid']
            y_vec = adata_m.obs['y_grid']
            z_vec = adata_m.obs['z_grid']
            axes[i].scatter(x_vec, y_vec, color=scatter_color, s=marker_s, label=region)


        for region, ax1 in zip(isocortex_region_mouse_list, axes):
            ax1.grid(False)
            ax1.set_xlabel(region, fontdict={'fontsize':22})
            ax1.set_ylabel('')
            ax1.set(frame_on=False)
            ax1.xaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])

            for line in ax1.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax1.yaxis.get_ticklines():
                line.set_visible(False)


        #plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), ncol=1, frameon=False)  # bbox_to_anchor=(1.1, 0.5),
        # plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        plt.savefig(save_path + f'Spatial_mouse_eachregion.' + self.fig_format)

        plt.rc('legend', fontsize=MEDIUM_SIZE)
        plt.rc('axes', labelsize=MEDIUM_SIZE)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9


        #==============================================================================================
        # Spatial visualization of human
        # 3D coordinates of regions of mouse
        rcParams["figure.subplot.left"] = 0.03
        rcParams["figure.subplot.right"] = 0.55
        fig = plt.figure(figsize=(16, 8), dpi=self.fig_dpi)
        axes = plt.axes(projection='3d')
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for region in isocortex_region_human_list:
            #region = region.strip('H-')
            #print(region)
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['H-' + region]
            x_vec = adata_h.obs['mri_voxel_x']
            y_vec = adata_h.obs['mri_voxel_y']
            z_vec = adata_h.obs['mri_voxel_z']
            axes.scatter3D(x_vec, y_vec, z_vec, color=scatter_color, s=marker_s, label=region)
        axes.grid(False)
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')
        axes.set(frame_on=False)
        fig.subplots_adjust(left=0, right=0.7, bottom=0, top=1)
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
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), ncol=1, frameon=False)  # bbox_to_anchor=(1.1, 0.5),
        plt.savefig(save_path + f'Spatial_human.' + self.fig_format)

        # multiview
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.7
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 6))
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for region in isocortex_region_human_list:
            #region = region.strip('H-')
            #print(region)
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['H-' + region]
            x_vec = adata_h.obs['mri_voxel_x']
            y_vec = adata_h.obs['mri_voxel_y']
            z_vec = adata_h.obs['mri_voxel_z']
            ax1.scatter(x_vec, y_vec, color=scatter_color, s=marker_s, label=region)
            ax2.scatter(x_vec, z_vec, color=scatter_color, s=marker_s, label=region)
            ax3.scatter(y_vec, z_vec, color=scatter_color, s=marker_s, label=region)

        ax1.grid(False)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        ax2.grid(False)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')

        ax3.grid(False)
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')

        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticklabels([])

        ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])

        ax3.xaxis.set_ticklabels([])
        ax3.yaxis.set_ticklabels([])

        for line in ax1.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax1.yaxis.get_ticklines():
            line.set_visible(False)

        for line in ax2.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax2.yaxis.get_ticklines():
            line.set_visible(False)

        for line in ax3.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax3.yaxis.get_ticklines():
            line.set_visible(False)

        plt.legend(loc='right', bbox_to_anchor=(2.5, 0.5), ncol=1, frameon=False)  # bbox_to_anchor=(1.1, 0.5),
        # plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        plt.savefig(save_path + f'Spatial_human_multiview.' + self.fig_format)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9

        # --------------------------------------------------------------------------
        # Scatter according regions
        # multiview
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.18
        rcParams["figure.subplot.top"] = 0.98

        fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi)
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for i, region in zip(range(len(isocortex_region_human_list)),
                             isocortex_region_human_list):
            #region = region.strip('H-')
            #print(region)
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['H-' + region]
            x_vec = adata_h.obs['mri_voxel_x']
            y_vec = adata_h.obs['mri_voxel_y']
            z_vec = adata_h.obs['mri_voxel_z']
            for j in range(len(isocortex_region_human_list)):
                if i != j:
                    axes[j].scatter(x_vec, z_vec, color='lightgrey', s=marker_s * 2)
            #axes[i].scatter(x_vec, z_vec, color=scatter_color, s=marker_s*3, label=region)

        fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi)
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for i, region in zip(range(len(isocortex_region_human_list)),
                             isocortex_region_human_list):
            # region = region.strip('H-')
            # print(region)
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['H-' + region]
            x_vec = adata_h.obs['mri_voxel_x']
            y_vec = adata_h.obs['mri_voxel_y']
            z_vec = adata_h.obs['mri_voxel_z']
            axes[i].scatter(x_vec, z_vec, color=scatter_color, s=marker_s * 3, label=region)


        for region, ax1 in zip(isocortex_region_human_list, axes):
            ax1.grid(False)
            ax1.set_xlabel(region, fontdict={'fontsize':22})
            ax1.set_ylabel('')
            ax1.set(frame_on=False)
            ax1.xaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])

            for line in ax1.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax1.yaxis.get_ticklines():
                line.set_visible(False)

        # plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), ncol=1, frameon=False)  # bbox_to_anchor=(1.1, 0.5),
        # plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        plt.savefig(save_path + f'Spatial_human_eachregion.' + self.fig_format)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9

        # ------------------------------------------------------------------------------
        # 3D plot of each human brain region
        # ---------------------------------------------------------------------
        #############################################################################################
        # --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # 3D Scatter according regions
        # multiview
        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.99

        # marker_s =

        # fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi, projection='3d')
        fig = plt.figure(figsize=(10, 2), dpi=self.fig_dpi)
        axes = []
        for i in range(5):
            ax = fig.add_subplot(1, 5, i + 1, projection='3d')
            axes.append(ax)
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for i, region in zip(range(len(isocortex_region_human_list)),
                             isocortex_region_human_list):
            #region = region.strip('H-')
            print(region)
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['H-' + region]
            x_vec = adata_h.obs['mri_voxel_x']
            y_vec = adata_h.obs['mri_voxel_y']
            z_vec = adata_h.obs['mri_voxel_z']
            for j in range(len(isocortex_region_human_list)):
                if i != j:
                    axes[j].scatter3D(x_vec, y_vec, z_vec, color='lightgrey', s=marker_s - 1)

        for i, region in zip(range(len(isocortex_region_human_list)),
                             isocortex_region_human_list):
            #region = region.strip('H-')
            print(region)
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['region_name'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['H-'+region]
            x_vec = adata_h.obs['mri_voxel_x']
            y_vec = adata_h.obs['mri_voxel_y']
            z_vec = adata_h.obs['mri_voxel_z']
            axes[i].scatter3D(x_vec, y_vec, z_vec, color=scatter_color, s=marker_s, label=region)

        for region, ax1 in zip(isocortex_region_human_list, axes):
            ax1.grid(True)
            ax1.set_xlabel(region)
            # ax1.set_ylabel('', linespacing=0)
            # ax1.set_zlabel('', linespacing=0)

            ax1.xaxis.labelpad = -12
            ax1.yaxis.labelpad = 0
            ax1.zaxis.labelpad = 0

            # ax1.set(frame_on=False)
            ax1.xaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])
            ax1.zaxis.set_ticklabels([])

            # No ticks
            # ax1.set_xticks([])
            # ax1.set_yticks([])
            # ax1.set_zticks([])
            #
            # # Transparent spines
            # # ax1.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # # ax1.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # # ax1.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            #
            # for line in ax1.xaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax1.yaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax1.zaxis.get_ticklines():
            #     line.set_visible(False)

            # ax1.view_init(60, -80)
            # ax1.tick_params(axis='x', pad=2)  # Adjust as needed
            # ax1.tick_params(axis='y', pad=2)  # Adjust as needed
            # ax1.tick_params(axis='z', pad=2)  # Adjust as needed

        # plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), ncol=1, frameon=False)  # bbox_to_anchor=(1.1, 0.5),
        # plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        fig.subplots_adjust(wspace=0, hspace=0)

        plt.savefig(save_path + f'Spatial_human_eachregion_3D.' + self.fig_format, dpi=self.fig_dpi)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9

        # ------------------------------------------------------------
        # umap plots of region pairs
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.top"] = 0.995
        rcParams["figure.subplot.bottom"] = 0.195

        plt.rc('legend', fontsize=22)

        fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi)

        markersize = 60

        for i in range(len(isocortex_region_mouse_list)):
            region_m = 'M-'+isocortex_region_mouse_list[i]
            region_h = 'H-'+isocortex_region_human_list[i]

            adata_mh = adata_embedding_hip[adata_embedding_hip.obs['region_name'].isin([region_m, region_h])]
            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i])

            sc.pl.umap(adata_mh,
                       color='region_name',
                       palette={region_m: palette[region_m], region_h:palette[region_h]},
                       ax=ax, legend_loc='lower center', legend_fontweight='normal', show=False, size=markersize)

        for ax1 in axes:
            ax1.grid(False)
            ax1.set_xlabel('')
            ax1.set_title('')
            ax1.set_ylabel('')
            ax1.set(frame_on=False)
            ax1.xaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])

            for line in ax1.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax1.yaxis.get_ticklines():
                line.set_visible(False)

        plt.savefig(save_path + f'Spatial_eachregion_pair.' + self.fig_format)
        plt.rc('legend', fontsize=MEDIUM_SIZE)

        # ----------------------------------------------------------------------
        # Statistics of the fact: homologous regions such as Mouse CA1 and Human CA1 are more close to each other
        # 1. Distance of pair region samples are significantly higher than those are not homologous
        # boxplot (mark t-test p-values)
        Name_list = isocortex_region_mouse_list + ['Non-homologous']
        Dist_dict = {k:[] for k in Name_list}

        #sc.pp.neighbors(adata_embedding_hip, n_neighbors=self.cfg.ANALYSIS.umap_neighbor, metric='euclidean', use_rep='X')
        #sc.tl.umap(adata_embedding_hip, n_components=2, min_dist=0.1)

        # homologous
        for i in range(len(isocortex_region_mouse_list)):
            region_m = 'M-' + isocortex_region_mouse_list[i]
            region_h = 'H-' + isocortex_region_human_list[i]

            adata_m = adata_embedding_hip[adata_embedding_hip.obs['region_name'].isin([region_m])]
            adata_h = adata_embedding_hip[adata_embedding_hip.obs['region_name'].isin([region_h])]

            dist_m = scipy.spatial.distance_matrix(adata_m.obsm['X_umap'], adata_h.obsm['X_umap'], p=2)
            Dist_dict[isocortex_region_mouse_list[i]] = list(np.min(dist_m, axis=0).flatten()) + list(np.min(dist_m, axis=1).flatten())

        # Non-homologous
        for i in range(len(isocortex_region_mouse_list)):
            for j in range(len(isocortex_region_human_list)):
                if i != j: # and max(i, j) >= 3
                    region_m = 'M-' + isocortex_region_mouse_list[i]
                    region_h = 'H-' + isocortex_region_human_list[i]

                    adata_m = adata_embedding_hip[adata_embedding_hip.obs['region_name'].isin([region_m])]
                    adata_h = adata_embedding_hip[adata_embedding_hip.obs['region_name'].isin([region_h])]

                    dist_m = scipy.spatial.distance_matrix(adata_m.obsm['X_umap'], adata_h.obsm['X_umap'], p=2)
                    Dist_dict['Non-homologous'] = Dist_dict['Non-homologous'] + list(np.min(dist_m, axis=0).flatten()) + list(np.min(dist_m, axis=1).flatten())

        Boxplot_dict = {'Region pairs':[], 'Distance':[]}
        for k,v in Dist_dict.items():
            Boxplot_dict['Region pairs'] = Boxplot_dict['Region pairs'] + [k] * len(v)
            Boxplot_dict['Distance'] = Boxplot_dict['Distance'] + v

        Boxplot_df = pd.DataFrame.from_dict(Boxplot_dict)

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.22
        rcParams["figure.subplot.bottom"] = 0.3
        rcParams["figure.subplot.top"] = 0.9

        color_pal = {k:v for k,v in zip(Name_list, color_list[0:6])}
        color_pal['Non-homologous'] = 'grey'

        plt.figure(figsize=(5, 8), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        # sns.set(rc={"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)})
        ax = sns.boxplot(x="Region pairs", y='Distance', data=Boxplot_df, order=Name_list, palette=color_pal,
                         width=0.75)  #
        add_stat_annotation(ax, data=Boxplot_df, x="Region pairs", y='Distance', order=Name_list,
                            box_pairs=[(isocortex_region_mouse_list[0], 'Non-homologous'),
                                       (isocortex_region_mouse_list[1], 'Non-homologous'),
                                       (isocortex_region_mouse_list[2], 'Non-homologous'),
                                       (isocortex_region_mouse_list[3], 'Non-homologous'),
                                       (isocortex_region_mouse_list[4], 'Non-homologous')],
                            test='t-test_ind', text_format='star', loc='inside', verbose=2)  # text_format='star'
        plt.title('')
        plt.ylabel('Distance')
        plt.xlabel('')
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.savefig(save_path + 'pair_region_distance.' + self.fig_format, format=self.fig_format)
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.9

        #-----------------------------------------------------------------------------------------------------------
        # Statistics of the fact: homologous regions such as Mouse CA1 and Human CA1 are more close to each other
        # 1. Alignment score of pair region samples are significantly higher than those are not homologous
        # boxplot (mark t-test p-values)
        Name_list = isocortex_region_mouse_list + ['Non-homologous']
        Dist_dict = {k: 0 for k in Name_list}

        # sc.pp.neighbors(adata_embedding_hip, n_neighbors=self.cfg.ANALYSIS.umap_neighbor, metric='euclidean', use_rep='X')
        # sc.tl.umap(adata_embedding_hip, n_components=2, min_dist=0.3)

        # homologous
        for i in range(len(isocortex_region_mouse_list)):
            region_m = isocortex_region_mouse_list[i] #'M-' +
            region_h = isocortex_region_human_list[i] #'H-' +

            adata_m = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_m])]
            adata_h = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_h])]

            X = np.concatenate((adata_m.obsm['X_umap'], adata_h.obsm['X_umap']), axis=0)
            Y = np.concatenate([np.zeros((adata_m.n_obs, 1)), np.ones((adata_h.n_obs, 1))], axis=0)
            Dist_dict[isocortex_region_mouse_list[i]] = seurat_alignment_score(X, Y)

        # Non-homologous
        seurat_as_list = []
        for i in range(len(self.mouse_64_labels_list)):
            for j in range(len(self.human_88_labels_list)):
                if i != j: #and max(i, j) >= 3
                    region_m = self.mouse_64_labels_list[i] #'M-' +
                    region_h = self.human_88_labels_list[i] #'H-' +

                    adata_m = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_m])]
                    adata_h = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_h])]

                    X = np.concatenate((adata_m.obsm['X_umap'], adata_h.obsm['X_umap']), axis=0)
                    Y = np.concatenate([np.zeros((adata_m.n_obs, 1)), np.ones((adata_h.n_obs, 1))], axis=0)

                    seurat_as_list.append(seurat_alignment_score(X, Y))

        Dist_dict['Non-homologous'] = np.mean(seurat_as_list)

        Boxplot_dict = {'Region pairs': [], 'Alignment score': []}
        for k, v in Dist_dict.items():
            Boxplot_dict['Region pairs'] = Boxplot_dict['Region pairs'] + [k]
            Boxplot_dict['Alignment score'] = Boxplot_dict['Alignment score'] + [v]

        Boxplot_df = pd.DataFrame.from_dict(Boxplot_dict)

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.22
        rcParams["figure.subplot.bottom"] = 0.3
        rcParams["figure.subplot.top"] = 0.9

        color_pal = {k: v for k, v in zip(Name_list, color_list[0:6])}
        color_pal['Non-homologous'] = 'grey'

        plt.figure(figsize=(5, 8), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        # sns.set(rc={"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)})
        ax = sns.barplot(x="Region pairs", y='Alignment score', data=Boxplot_df, order=Name_list, palette=color_pal,
                         width=0.75)  #
        # add_stat_annotation(ax, data=Boxplot_df, x="Region pairs", y='Distance', order=Name_list,
        #                     box_pairs=[('Field CA1', 'Non-homologous'),
        #                                ('Field CA2', 'Non-homologous'),
        #                                ('Field CA3', 'Non-homologous'),
        #                                ('Dentate gyrus', 'Non-homologous'),
        #                                ('Subiculum', 'Non-homologous')],
        #                     test='t-test_ind', text_format='star', loc='inside', verbose=2)  # text_format='star'
        plt.title('')
        plt.ylabel('Alignment score')
        plt.xlabel('')
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.savefig(save_path + 'pair_region_distance_alignment_score.' + self.fig_format, format=self.fig_format)
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.9

        #--------------------------------------------------------
        # ---------------------------------------------------------------------------------------
        # Fold change of alignement score
        Name_list = isocortex_region_mouse_list
        Dist_dict = {k: 0 for k in Name_list}

        # sc.pp.neighbors(adata_embedding_hip, n_neighbors=self.cfg.ANALYSIS.umap_neighbor, metric='euclidean', use_rep='X')
        # sc.tl.umap(adata_embedding_hip, n_components=2, min_dist=0.3)

        # homologous
        for i in range(len(isocortex_region_mouse_list)):
            region_m = isocortex_region_mouse_list[i]  # 'M-' +
            region_h = isocortex_region_human_list[i]  # 'H-' +

            adata_m = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_m])]
            adata_h = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_h])]

            X = np.concatenate((adata_m.obsm['X_umap'], adata_h.obsm['X_umap']), axis=0)
            Y = np.concatenate([np.zeros((adata_m.n_obs, 1)), np.ones((adata_h.n_obs, 1))], axis=0)
            Dist_dict[isocortex_region_mouse_list[i]] = seurat_alignment_score(X, Y)

        # Non-homologous
        seurat_as_list = []
        for i in range(len(self.mouse_64_labels_list)):
            for j in range(len(self.human_88_labels_list)):
                if i != j:  # and max(i, j) >= 3
                    region_m = self.mouse_64_labels_list[i]  # 'M-' +
                    region_h = self.human_88_labels_list[i]  # 'H-' +

                    adata_m = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_m])]
                    adata_h = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_h])]

                    X = np.concatenate((adata_m.obsm['X_umap'], adata_h.obsm['X_umap']), axis=0)
                    Y = np.concatenate([np.zeros((adata_m.n_obs, 1)), np.ones((adata_h.n_obs, 1))], axis=0)

                    seurat_as_list.append(seurat_alignment_score(X, Y))

        Non_homo_score = np.mean(seurat_as_list)

        Boxplot_dict = {'Region pairs': [], 'Alignment score fold change': []}
        for k, v in Dist_dict.items():
            Boxplot_dict['Region pairs'] = Boxplot_dict['Region pairs'] + [k]
            Boxplot_dict['Alignment score fold change'] = Boxplot_dict['Alignment score fold change'] + [
                v / Non_homo_score]

        Boxplot_df = pd.DataFrame.from_dict(Boxplot_dict)

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.24
        rcParams["figure.subplot.bottom"] = 0.45
        rcParams["figure.subplot.top"] = 0.92

        color_pal = {k: v for k, v in zip(Name_list, color_list[0:6])}
        color_pal['Non-homologous'] = 'grey'

        plt.figure(figsize=(2.2, 3), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        # sns.set(rc={"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)})
        ax = sns.barplot(x="Region pairs", y='Alignment score fold change', data=Boxplot_df, order=Name_list,
                         palette=color_pal,
                         width=0.62)  #
        ax.axhline(1, color='grey', linestyle='--')
        plt.title('')
        plt.ylabel('Alignment score fold change')
        plt.xlabel('')
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.savefig(save_path + 'pair_region_distance_fold_change.' + self.fig_format, format=self.fig_format)
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.9




        # Marker genes of CA1,CA2,CA2,Dentate gyrus
        # CA1:
        adata_mouse_embedding_hip = adata_embedding_hip[adata_embedding_hip.obs['dataset'].isin(['Mouse'])]
        adata_human_embedding_hip = adata_embedding_hip[adata_embedding_hip.obs['dataset'].isin(['Human'])]

        # minimum_num = 10
        #Mouse_markergene_list = ['Cyp39a1', 'Edc3', 'B3gat2', 'Gprin3', 'Phactr2']#['Itpka', 'Amigo2', 'Hs3st4', 'Lrrtm4', 'Nts'] #['Fndc1', 'Scgn', 'Tspan18', 'Chrna1', 'Nts']# 'Gfra1'
        #Human_markergene_list = ['PVALB', 'SLN', 'DDX3Y', 'TRIM54', 'PTGER3']#['KCNS1', 'TMEM155', 'TAC3', 'LHX6', 'CCK']#['ITPKA', 'AMIGO2', 'HS3ST4', 'LRRTM4', 'NTSR2']

        # minimum_num = 5
        #['Primary somatosensory area', 'Supplemental somatosensory area', 'Ectorhinal area', 'Auditory areas',
        # 'Perirhinal area']
        #['cuneus', 'lingual gyrus', 'temporal pole', 'superior parietal lobule', 'parahippocampal gyrus']

        # ['Primary motor area', 'Primary somatosensory area', 'Supplemental somatosensory area', 'Visceral area',
        #  'Posterior parietal association areas']
        # mouse marker gene: Wnt6, Slc16a3, Gprin3, Ankfn1 (Tgfb1), Trim16

        # ['posterior orbital gyrus', 'superior parietal lobule', "Heschl's gyrus", 'parolfactory gyri',
        #  'lateral orbital gyrus']
        # fold change human marker gene: TFAP2D, A_32_P16323, TTR, SLN, TFAP2D, CARTPT
        # pvalues: MBLAC2, A_24_P788227, CDH7, ZCCHC18, TFAP2D
        # -, SHD, -, -, -
        # MBLAC2, SHD,  CDH7, ZCCHC18, TFAP2D
        #Mouse_markergene_list = ['Wnt6', 'Slc16a3', 'Gprin3', 'Tgfb1', 'Trim16']#['Ddx3y', 'Zcchc18', 'Tmem196', 'Eepd1', 'Rbp4']
        #Human_markergene_list = ['MBLAC2', 'SHD',  'CDH7', 'ZCCHC18', 'CARTPT']#['PVALB', 'TRPC3', 'PTGER3', 'SHD', 'TTR']

        # picked by Seurat
        Mouse_markergene_list = ['Dkk3', 'Vamp1', 'Penk', 'Gpr88', 'Spink8']
        Human_markergene_list = ['LRRC36', 'ALDH1A3', 'BHLHE22', 'PPM1M', 'RTP1']

        # 'RTP1', SLN, COL22A1, PVALB, CARTPT
        adata_mouse_exp_hip = self.adata_mouse_exp[self.adata_mouse_exp.obs['region_name'].isin(isocortex_region_mouse_list)]
        adata_human_exp_hip = self.adata_human_exp[
            self.adata_human_exp.obs['region_name'].isin(isocortex_region_human_list)]

        adata_mouse_exp_hip.write_h5ad(save_path + 'adata_mouse_exp_isocortex.h5ad')
        adata_human_exp_hip.write_h5ad(save_path + 'adata_human_exp_isocortex.h5ad')

        #print(self.adata_mouse_exp.var_names.tolist())
        #print(self.adata_human_exp.var_names.tolist())

        #color_map = 'viridis'#'magma_r'
        c = Colormap()
        color_map = c.cmap_linear('#3288BD', 'white', '#D53E4F')

        markersize = 40
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.18
        rcParams["figure.subplot.top"] = 0.98
        # mouse marker genes
        fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi)
        for i in range(len(Mouse_markergene_list)):
            m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_mouse_embedding_hip.obs[m_gene] = adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)
            diff = False
            for k, v in zip(adata_mouse_embedding_hip.obs_names, adata_mouse_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i])

            sc.pl.umap(adata_mouse_embedding_hip,
                       color= m_gene,
                       #palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize, color_map=color_map, colorbar_loc=None)

            ax.grid(False)
            ax.set_xlabel(m_gene)
            ax.set_title('')
            ax.set_ylabel('')
            ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

            for line in ax.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.yaxis.get_ticklines():
                line.set_visible(False)

        plt.savefig(save_path + f'Mouse_marker_gene.' + self.fig_format)

        # human marker genes
        markersize = 60

        fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi)
        for i in range(len(Human_markergene_list)):
            #m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_human_embedding_hip.obs[h_gene] = adata_human_exp_hip[:, h_gene].X.toarray().reshape(-1) # normalize(adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)[:, np.newaxis], norm='max', axis=0).ravel()

            diff = False
            for k, v in zip(adata_human_embedding_hip.obs_names, adata_human_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i])

            sc.pl.umap(adata_human_embedding_hip,
                       color=h_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None)

            ax.grid(False)
            ax.set_xlabel(h_gene)
            ax.set_title('')
            ax.set_ylabel('')
            ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

            for line in ax.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.yaxis.get_ticklines():
                line.set_visible(False)

        plt.savefig(save_path + f'Human_marker_gene.' + self.fig_format)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9

        # ----------------------------------------------------------------------
        # Another pair marker gene
        Mouse_markergene_list = ['Lrrc36', 'Aldh1a3', 'Bhlhe22', 'Ppm1m', 'Rtp1']
        Human_markergene_list = ['DKK3', 'VAMP1', 'PENK', 'GPR88',
                                 'SPINK7']  # ['Fndc1', 'Scgn', 'Tspan18', 'Chrna1', 'Nts']# 'Gfra1'
        # Human_markergene_list = ['ITPKA', 'AMIGO2', 'HS3ST4', 'LRRTM4', 'NTSR2']

        adata_mouse_exp_hip = self.adata_mouse_exp[
            self.adata_mouse_exp.obs['region_name'].isin(isocortex_region_mouse_list)]
        adata_human_exp_hip = self.adata_human_exp[
            self.adata_human_exp.obs['region_name'].isin(isocortex_region_human_list)]

        print(self.adata_human_exp.var_names.tolist())
        print(self.adata_mouse_exp.var_names.tolist())

        # color_map = 'viridis'  # 'magma_r'
        c = Colormap()
        color_map = c.cmap_linear('#3288BD', 'white', '#D53E4F')

        markersize = 150
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.95
        # mouse marker genes
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        for i in range(len(Mouse_markergene_list)):
            m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_mouse_embedding_hip.obs[m_gene] = adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)
            diff = False
            for k, v in zip(adata_mouse_embedding_hip.obs_names, adata_mouse_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i])

            sc.pl.umap(adata_mouse_embedding_hip,
                       color=m_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None)

            ax.grid(False)
            ax.set_xlabel(m_gene)
            ax.set_title('')
            ax.set_ylabel('')
            ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

            for line in ax.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.yaxis.get_ticklines():
                line.set_visible(False)

        plt.savefig(save_path + f'Mouse_marker_gene_human.' + self.fig_format)

        # human marker genes
        markersize = 300

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        for i in range(len(Human_markergene_list)):
            # m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_human_embedding_hip.obs[h_gene] = adata_human_exp_hip[:, h_gene].X.toarray().reshape(
                -1)  # normalize(adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)[:, np.newaxis], norm='max', axis=0).ravel()

            diff = False
            for k, v in zip(adata_human_embedding_hip.obs_names, adata_human_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i])

            sc.pl.umap(adata_human_embedding_hip,
                       color=h_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None)

            ax.grid(False)
            ax.set_xlabel(h_gene)
            ax.set_title('')
            ax.set_ylabel('')
            ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

            for line in ax.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.yaxis.get_ticklines():
                line.set_visible(False)

        plt.savefig(save_path + f'Human_marker_gene_mouse.' + self.fig_format)



        #-----------------------------------------------------------------------------------
        ##################################################################################
        # 3D plot of mouse and human markers
        # picked by Seurat
        Mouse_markergene_list = ['Dkk3', 'Vamp1', 'Penk', 'Gpr88', 'Spink8']
        Human_markergene_list = ['LRRC36', 'ALDH1A3', 'BHLHE22', 'PPM1M', 'RTP1']

        adata_mouse_exp_hip = self.adata_mouse_exp[
            self.adata_mouse_exp.obs['region_name'].isin(isocortex_region_mouse_list)]
        adata_human_exp_hip = self.adata_human_exp[
            self.adata_human_exp.obs['region_name'].isin(isocortex_region_human_list)]

        adata_mouse_exp_hip.write_h5ad(save_path + 'adata_mouse_exp_isocortex.h5ad')
        adata_human_exp_hip.write_h5ad(save_path + 'adata_human_exp_isocortex.h5ad')

        # print(self.adata_mouse_exp.var_names.tolist())
        # print(self.adata_human_exp.var_names.tolist())

        # color_map = 'viridis'#'magma_r'
        c = Colormap()
        color_map = c.cmap_linear('#3288BD', 'white', '#D53E4F')

        markersize = 20
        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.top"] = 0.99
        rcParams["figure.subplot.bottom"] = 0.1

        fig = plt.figure(figsize=(10, 2), dpi=self.fig_dpi)
        axes = []
        for i in range(5):
            ax = fig.add_subplot(1, 5, i + 1, projection='3d')
            ax.grid(True)
            axes.append(ax)

        # mouse marker genes
        for i in range(len(Mouse_markergene_list)):
            m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_mouse_embedding_hip.obs[m_gene] = adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)
            diff = False
            for k, v in zip(adata_mouse_embedding_hip.obs_names, adata_mouse_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i], projection='3d',size=markersize-17,  alpha=0.2)

            sc.pl.umap(adata_mouse_embedding_hip,
                       color=m_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None, projection='3d',  alpha=1)

            ax.grid(True)
            if len(m_gene) <= 4:
                m_gene = m_gene + ' '
            ax.set_xlabel(m_gene, fontdict=dict(style='italic'))  # , linespacing=-12
            ax.set_ylabel('', linespacing=0)
            ax.set_zlabel('', linespacing=0)

            ax.set_title('')

            ax.xaxis.labelpad = -12
            ax.yaxis.labelpad = 0
            ax.zaxis.labelpad = 0

            # ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

        plt.savefig(save_path + f'Mouse_marker_gene_3D.' + self.fig_format)

        # human marker genes
        markersize = 80

        fig = plt.figure(figsize=(10, 2), dpi=self.fig_dpi)
        axes = []
        for i in range(5):
            ax = fig.add_subplot(1, 5, i + 1, projection='3d')
            axes.append(ax)

        for i in range(len(Human_markergene_list)):
            # m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_human_embedding_hip.obs[h_gene] = adata_human_exp_hip[:, h_gene].X.toarray().reshape(
                -1)  # normalize(adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)[:, np.newaxis], norm='max', axis=0).ravel()

            diff = False
            for k, v in zip(adata_human_embedding_hip.obs_names, adata_human_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i], projection='3d', size=markersize-75, alpha=0.2)

            sc.pl.umap(adata_human_embedding_hip,
                       color=h_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None, projection='3d', alpha=1)

            ax.grid(True)
            if len(h_gene) <= 4:
                h_gene = h_gene + ' '
            ax.set_xlabel(h_gene, fontdict=dict(style='italic'))  # , linespacing=-12
            ax.set_ylabel('', linespacing=0)
            ax.set_zlabel('', linespacing=0)

            ax.set_title('')

            ax.xaxis.labelpad = -12
            ax.yaxis.labelpad = 0
            ax.zaxis.labelpad = 0

            # ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

        plt.savefig(save_path + f'Human_marker_gene_3D.' + self.fig_format)


        # ---------------------------------------------------------------------------------
        # UMAP 3 components
        ###################################################################################

        #sc.tl.umap(adata_embedding_hip, n_components=3, min_dist=0.3)

        # ------------------------------------------------------------
        # umap plots of region pairs

        # Another pair marker gene
        Mouse_markergene_list = ['Lrrc36', 'Aldh1a3', 'Bhlhe22', 'Ppm1m', 'Rtp1']
        Human_markergene_list = ['DKK3', 'VAMP1', 'PENK', 'GPR88',
                                 'SPINK7']  # ['Fndc1', 'Scgn', 'Tspan18', 'Chrna1', 'Nts']#

        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.top"] = 0.99
        rcParams["figure.subplot.bottom"] = 0.2

        fig = plt.figure(figsize=(10, 2.2), dpi=self.fig_dpi)
        axes = []
        for i in range(5):
            ax = fig.add_subplot(1, 5, i + 1, projection='3d')
            ax.grid(True)
            axes.append(ax)

        markersize = 40

        for i in range(len(isocortex_region_mouse_list)):
            region_m = 'M-' + isocortex_region_mouse_list[i]
            region_h = 'H-' + isocortex_region_human_list[i]

            adata_mh = adata_embedding_hip[adata_embedding_hip.obs['region_name'].isin([region_m, region_h])]
            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i], projection='3d', size=markersize - 35,
                            alpha=0.2)  # , wspace=0.25 , s=1, alpha=0.2,

            # umap1_x, umap1_y, umap1_z = adata_embedding_hip.obsm['X_umap'].toarray()[:, 0], \
            #                    adata_embedding_hip.obsm['X_umap'].toarray()[:, 1], \
            #                             adata_embedding_hip.obsm['X_umap'].toarray()[:, 2]
            ax.grid(True)
            sc.pl.umap(adata_mh,
                       color='region_name',
                       palette={region_m: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc='lower center', show=False, size=markersize, projection='3d',
                       alpha=1)  # legend_fontweight='normal',
            # ax.legend()
            # c_p = 0,0
            # handles = [matplotlib.patches.Circle([0, 0], facecolor=palette[name], radius=5) for name in [region_m, region_h]]
            # ax.legend(handles, palette, labels=[region_m, region_h], # title='',
            #            bbox_to_anchor=(0.3, 0),
            #            #bbox_transform=plt.gcf().transFigure,
            #            loc='lower center', frameon=False)

        for ax1 in axes:
            ax1.grid(True)
            ax1.set_xlabel('')
            ax1.set_ylabel('')
            ax1.set_zlabel('')

            ax1.xaxis.labelpad = -12
            ax1.yaxis.labelpad = 0
            ax1.zaxis.labelpad = 0

            # ax1.set(frame_on=False)
            ax1.xaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])
            ax1.zaxis.set_ticklabels([])

            ax.set_title('')

            # ax1.view_init(60, -80)
        plt.subplots_adjust(wspace=0, hspace=0)

        plt.savefig(save_path + f'Spatial_eachregion_pair_3D.' + self.fig_format, dpi=self.fig_dpi)



        ##########################################################################
        # 3D plot of human or mouse marker gene homologies
        #---------------------------------------------------------------------------

        adata_mouse_exp_hip = self.adata_mouse_exp[
            self.adata_mouse_exp.obs['region_name'].isin(isocortex_region_mouse_list)]
        adata_human_exp_hip = self.adata_human_exp[
            self.adata_human_exp.obs['region_name'].isin(isocortex_region_human_list)]

        print(self.adata_human_exp.var_names.tolist())
        print(self.adata_mouse_exp.var_names.tolist())

        # color_map = 'viridis'  # 'magma_r'
        c = Colormap()
        color_map = c.cmap_linear('#3288BD', 'white', '#D53E4F')

        markersize = 20
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.95
        # mouse marker genes
        fig = plt.figure(figsize=(10, 2), dpi=self.fig_dpi)
        axes = []
        for i in range(5):
            ax = fig.add_subplot(1, 5, i + 1, projection='3d')
            axes.append(ax)

        for i in range(len(Mouse_markergene_list)):
            m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_mouse_embedding_hip.obs[m_gene] = adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)
            diff = False
            for k, v in zip(adata_mouse_embedding_hip.obs_names, adata_mouse_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i], projection='3d', size=markersize-17, alpha=0.2)

            sc.pl.umap(adata_mouse_embedding_hip,
                       color=m_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize-10,
                       color_map=color_map, colorbar_loc=None, projection='3d', alpha=1)

            ax.grid(True)
            if len(m_gene) <= 4:
                m_gene = m_gene + ' '
            ax.set_xlabel(m_gene, fontdict=dict(style='italic'))  # , linespacing=-12
            ax.set_ylabel('', linespacing=0)
            ax.set_zlabel('', linespacing=0)

            ax.set_title('')

            ax.xaxis.labelpad = -12
            ax.yaxis.labelpad = 0
            ax.zaxis.labelpad = 0

            # ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

        plt.savefig(save_path + f'Mouse_marker_gene_human_3D.' + self.fig_format)

        # human marker genes
        markersize = 80

        fig = plt.figure(figsize=(10, 2), dpi=self.fig_dpi)
        axes = []
        for i in range(5):
            ax = fig.add_subplot(1, 5, i + 1, projection='3d')
            axes.append(ax)

        for i in range(len(Human_markergene_list)):
            # m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_human_embedding_hip.obs[h_gene] = adata_human_exp_hip[:, h_gene].X.toarray().reshape(
                -1)  # normalize(adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)[:, np.newaxis], norm='max', axis=0).ravel()

            diff = False
            for k, v in zip(adata_human_embedding_hip.obs_names, adata_human_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i], projection='3d', size=markersize-75, alpha=0.2)

            sc.pl.umap(adata_human_embedding_hip,
                       color=h_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None, projection='3d', alpha=1)

            ax.grid(True)
            if len(h_gene) <= 4:
                h_gene = h_gene + ' '
            ax.set_xlabel(h_gene, fontdict=dict(style='italic'))  # , linespacing=-12
            ax.set_ylabel('', linespacing=0)
            ax.set_zlabel('', linespacing=0)

            ax.set_title('')

            ax.xaxis.labelpad = -12
            ax.yaxis.labelpad = 0
            ax.zaxis.labelpad = 0

            # ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

        plt.savefig(save_path + f'Human_marker_gene_mouse_3D.' + self.fig_format)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9

        return None


    def experiment_3_spatial_clusters(self):
        """
        Spatial integration and alignment of hypothalamus clusters, to display that BrainAlign keeps the relative spatial positions of
        samples and help to detect STs difference across species.

        - 1. Visualize the anatomical regions of two regions;
        - 2. Spatial scatter of regions separately;
        - 3. UMAP of two regions together;
        - 4. PAGA abstracted graph of regions of two species;
        - 4. For each Marker genes, plot each expression of marker genes on UMAP of two species (with different thresholds)

        :return: None
        """

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
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        save_path = self.cfg.BrainAlign.embeddings_file_path + 'figs/4_spatial_analysis/3_experiment_spatial_clusters/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        TH_cluster_list = []
        cluster_name_unique = sorted(Counter(self.adata_embedding.obs['cluster_name_acronym']).keys(),
                                     key=lambda t: int(t.split('-')[0]))
        for cluster_name in cluster_name_unique:
            if cluster_name.split('-')[1] == 'TH' and cluster_name.split('-')[2] == 'TH':
                TH_cluster_list.append(cluster_name)

        TH_mouse_list = ['M-'+x for x in TH_cluster_list]
        TH_human_list = ['H-'+x for x in TH_cluster_list]
        TH_mouse_human_list = TH_mouse_list + TH_human_list
        # Visualize the anatomical regions of regions;
        #th_cluster_list = self.human_88_labels_list

        adata_human_embedding_hip = self.adata_human_embedding[self.adata_human_embedding.obs['cluster_name_acronym'].isin(TH_cluster_list)]
        adata_mouse_embedding_hip = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['cluster_name_acronym'].isin(TH_cluster_list)]


        adata_human_embedding_hip.obs['cluster_name_acronym'] = ['H-' + x for x in adata_human_embedding_hip.obs['cluster_name_acronym']]
        adata_mouse_embedding_hip.obs['cluster_name_acronym'] = ['M-' + x for x in adata_mouse_embedding_hip.obs['cluster_name_acronym']]

        adata_human_embedding_hip.obs['dataset'] = 'Human'
        adata_mouse_embedding_hip.obs['dataset'] = 'Mouse'

        #adata_embedding_hip = self.adata_embedding[self.adata_embedding.obs['cluster_name_acronym'].isin(TH_cluster_list)]
        adata_embedding_hip = ad.concat([adata_mouse_embedding_hip, adata_human_embedding_hip])

        # UMAP ------------------------------------------------------
        color_list = [self.cluster_color_dict[x] for x in TH_cluster_list]
        palette = {k:v for k,v in zip(TH_cluster_list, color_list)}

        #color_list = sns.color_palette(cc.glasbey, n_colors=len(TH_mouse_human_list))
        color_list = sns.color_palette(cc.glasbey, n_colors=len(TH_mouse_human_list))
        palette = {k: v for k, v in zip(TH_mouse_human_list, color_list)}


        #palette_acronym = {k:v for k,v in zip(acronym_name_map_list, color_list)}

        sc.pp.neighbors(adata_embedding_hip, n_neighbors=self.cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(adata_embedding_hip, n_components=2, min_dist=0.3)

        adata_mouse_embedding_hip = adata_embedding_hip[adata_embedding_hip.obs['dataset'].isin(['Mouse'])]
        adata_human_embedding_hip = adata_embedding_hip[adata_embedding_hip.obs['dataset'].isin(['Human'])]

        marker_size = 15

        with plt.rc_context({"figure.figsize": (7.5, 2.5), "figure.dpi": (self.fig_dpi)}):
            rcParams["figure.subplot.left"] = 0.15
            rcParams["figure.subplot.right"] = 0.4
            rcParams["figure.subplot.top"] = 0.95
            rcParams["figure.subplot.bottom"] = 0.25

            fig = sc.pl.umap(adata_embedding_hip, color=['cluster_name_acronym'],
                             return_fig=True,
                             legend_loc='right margin',
                             palette=palette, #wspace=wspace,
                             #add_outline=True,
                             #legend_fontsize=12, legend_fontoutline=2,
                             size=marker_size,
                             title=['']
                             )
            # plt.title('')
            fig.savefig(save_path + 'umap_isocortex.' + self.fig_format, format=self.fig_format)

            rcParams["figure.subplot.left"] = 0.1
            rcParams["figure.subplot.right"] = 0.9

        # PAGA of isocortex
        sc.tl.paga(adata_embedding_hip, groups='cluster_name_acronym')
        rcParams["figure.subplot.right"] = 0.95
        fig, ax = plt.subplots(figsize=(4.5, 5), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        graph_fontsize = 14

        palette_paga = {}
        for k, v in palette.items():
            palette_paga.update({k: {v: 1}})

        pos_arr = np.zeros((10, 2))
        pos_arr[0:5, 0] = 0
        pos_arr[0:5, 1] = np.array(range(0, 5))
        pos_arr[5:10, 0] = 3
        pos_arr[5:10, 1] = np.array(range(0, 5))
        pos_arr[5:10, 1] = np.array([0, 1, 2, 3, 4])
        pos_arr[0:5, 0] = np.array([1, 0.5, 0, 0.5, 1])
        pos_arr[5:10, 0] = np.array([2, 2.5, 3, 2.5, 2])

        sc.pl.paga(adata_embedding_hip,
                   color=palette_paga,
                   node_size_scale=5,
                   min_edge_width=0.1,
                   ax=ax,
                   pos=pos_arr,
                   max_edge_width=20,
                   layout='eq_tree',
                   fontsize=graph_fontsize,
                   fontoutline=1, colorbar=False, threshold=0.4,
                   frameon=False)  # node_size_scale=10, edge_width_scale=2
        plt.title('')
        fig.tight_layout()
        plt.savefig(save_path + 'paga_region_name.' + self.fig_format, format=self.fig_format)


        # paga species
        palette_species = {k: self.mouse_color for k in TH_mouse_list}
        palette_species.update({k: self.human_color for k in TH_human_list})
        palette_paga_species = {}
        for k, v in palette_species.items():
            palette_paga_species.update({k: {v: 1}})

        fig, ax = plt.subplots(figsize=(4.5, 5), dpi=self.fig_dpi)
        sc.pl.paga(adata_embedding_hip,
                   color=palette_paga_species,
                   node_size_scale=10,
                   min_edge_width=0.1,
                   ax=ax,
                   pos=pos_arr,
                   max_edge_width=20,
                   layout='eq_tree',
                   fontsize=graph_fontsize,
                   fontoutline=1, colorbar=False, threshold=0.4,
                   frameon=False)  # node_size_scale=10, edge_width_scale=2
        plt.title('')
        fig.tight_layout()
        plt.savefig(save_path + 'paga_species.' + self.fig_format, format=self.fig_format)

        #print(self.adata_mouse_embedding)

        # 3D plot parameters
        marker_s = 15

        # --------------------------------------------------------------------------------------
        # 3D coordinates of regions of mouse
        rcParams["figure.subplot.left"] = 0.03
        rcParams["figure.subplot.right"] = 0.55
        fig = plt.figure(figsize=(16, 8), dpi=self.fig_dpi)
        axes = plt.axes(projection='3d')
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for region in ['M-' + x for x in TH_mouse_list]:
            region = region.strip('M-')
            print(region)
            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['cluster_name_acronym'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['M-' + region]
            x_vec = adata_m.obs['x_grid']
            y_vec = adata_m.obs['y_grid']
            z_vec = adata_m.obs['z_grid']
            axes.scatter3D(x_vec, y_vec, z_vec, color=scatter_color, s=marker_s, label=region)
        axes.grid(False)
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')
        axes.set(frame_on=False)
        fig.subplots_adjust(left=0, right=0.7, bottom=0, top=1)
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
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), ncol=1, frameon=False) #bbox_to_anchor=(1.1, 0.5),
        plt.savefig(save_path + f'Spatial_mouse.' + self.fig_format)


        # multiview
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.7
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 6))
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        #rcParams["figure.subplot.left"] = 0.05
        for region in ['M-'+x for x in TH_mouse_list]:
            region = region.strip('M-')
            print(region)
            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['cluster_name_acronym'].isin([region])]
            #print(adata_m)
            #print(adata_m)
            scatter_color = palette['M-'+region]
            x_vec = adata_m.obs['x_grid']
            y_vec = adata_m.obs['y_grid']
            z_vec = adata_m.obs['z_grid']
            ax1.scatter(x_vec, y_vec, color=scatter_color, s=marker_s, label=region)
            ax2.scatter(x_vec, z_vec, color=scatter_color, s=marker_s, label=region)
            ax3.scatter(y_vec, z_vec, color=scatter_color, s=marker_s, label=region)

        ax1.grid(False)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        ax2.grid(False)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')

        ax3.grid(False)
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')

        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticklabels([])

        ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])

        ax3.xaxis.set_ticklabels([])
        ax3.yaxis.set_ticklabels([])


        for line in ax1.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax1.yaxis.get_ticklines():
            line.set_visible(False)

        for line in ax2.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax2.yaxis.get_ticklines():
            line.set_visible(False)

        for line in ax3.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax3.yaxis.get_ticklines():
            line.set_visible(False)

        plt.legend(loc='right', bbox_to_anchor=(2.5, 0.5), ncol=1, frameon=False) #bbox_to_anchor=(1.1, 0.5),
        # plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        plt.savefig(save_path + f'Spatial_mouse_multiview.' + self.fig_format)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9


        # --------------------------------------------------------------------------
        # Scatter according regions
        # multiview
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.19
        rcParams["figure.subplot.top"] = 0.99

        fig, axes = plt.subplots(1, len(TH_cluster_list), figsize=(4*len(TH_cluster_list), 5))
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for i, region in zip(range(len(TH_mouse_list)), ['M-' + x for x in TH_mouse_list]):
            region = region.strip('M-')
            print(region)
            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['cluster_name_acronym'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['M-' + region]
            x_vec = adata_m.obs['x_grid']
            y_vec = adata_m.obs['y_grid']
            z_vec = adata_m.obs['z_grid']
            for j in range(len(TH_mouse_list)):
                if i != j:
                    axes[j].scatter(x_vec, y_vec, color='lightgrey', s=marker_s)

            #axes[i].scatter(x_vec, y_vec, color=scatter_color, s=marker_s, label=region)

        for i, region in zip(range(len(TH_mouse_list)), ['M-' + x for x in TH_mouse_list]):
            region = region.strip('M-')
            print(region)
            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['cluster_name_acronym'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['M-' + region]
            x_vec = adata_m.obs['x_grid']
            y_vec = adata_m.obs['y_grid']
            z_vec = adata_m.obs['z_grid']
            axes[i].scatter(x_vec, y_vec, color=scatter_color, s=marker_s, label=region)

        for region, ax1 in zip(TH_mouse_list, axes):
            ax1.grid(False)
            ax1.set_xlabel(region)
            ax1.set_ylabel('')
            ax1.set(frame_on=False)
            ax1.xaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])

            for line in ax1.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax1.yaxis.get_ticklines():
                line.set_visible(False)
        #plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), ncol=1, frameon=False)  # bbox_to_anchor=(1.1, 0.5),
        # plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        plt.savefig(save_path + f'Spatial_mouse_eachregion.' + self.fig_format)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9


        #---------------------------------------------------------------------------------------
        # 3D plot of mouse regions
        #---------------------------------------------------------------------------
        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.99

        marker_s = 5

        # fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi, projection='3d')
        fig = plt.figure(figsize=(2 * len(TH_cluster_list), 2), dpi=self.fig_dpi)
        axes = []
        for i in range(len(TH_cluster_list)):
            ax = fig.add_subplot(1, len(TH_cluster_list), i + 1, projection='3d')
            axes.append(ax)

        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for i, region in zip(range(len(TH_mouse_list)), ['M-' + x for x in TH_mouse_list]):
            region = region.strip('M-')
            print(region)
            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['cluster_name_acronym'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['M-' + region]
            x_vec = adata_m.obs['x_grid']
            y_vec = adata_m.obs['y_grid']
            z_vec = adata_m.obs['z_grid']
            for j in range(len(TH_mouse_list)):
                if i != j:
                    axes[j].scatter3D(x_vec, y_vec, z_vec, color='lightgrey', s=marker_s-4)

            # axes[i].scatter(x_vec, y_vec, color=scatter_color, s=marker_s, label=region)

        for i, region in zip(range(len(TH_mouse_list)), ['M-' + x for x in TH_mouse_list]):
            region = region.strip('M-')
            print(region)
            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['cluster_name_acronym'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['M-' + region]
            x_vec = adata_m.obs['x_grid']
            y_vec = adata_m.obs['y_grid']
            z_vec = adata_m.obs['z_grid']
            axes[i].scatter3D(x_vec, y_vec, z_vec, color=scatter_color, s=marker_s, label=region)

        for region, ax1 in zip(TH_mouse_list, axes):
            ax1.grid(True)
            # ax1.set_xlabel(region, labelpad=0.05)
            # ax1.set_ylabel('')
            # ax1.set_zlabel('')

            ax1.set_xlabel(region)
            # ax1.set_ylabel('', linespacing=0)
            # ax1.set_zlabel('', linespacing=0)

            ax1.xaxis.labelpad = -12
            ax1.yaxis.labelpad = 0
            ax1.zaxis.labelpad = 0

            # ax1.set(frame_on=False)
            ax1.xaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])
            ax1.zaxis.set_ticklabels([])
        # plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), ncol=1, frameon=False)  # bbox_to_anchor=(1.1, 0.5),
        # plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        plt.savefig(save_path + f'Spatial_mouse_eachregion_3D.' + self.fig_format)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9



        #==============================================================================================
        # Spatial visualization of human
        # 3D coordinates of regions of mouse
        rcParams["figure.subplot.left"] = 0.03
        rcParams["figure.subplot.right"] = 0.55
        fig = plt.figure(figsize=(16, 8), dpi=self.fig_dpi)
        axes = plt.axes(projection='3d')
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for region in TH_cluster_list:
            #region = region.strip('H-')
            print(region)
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['cluster_name_acronym'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['H-'+region]
            x_vec = adata_h.obs['mri_voxel_x']
            y_vec = adata_h.obs['mri_voxel_y']
            z_vec = adata_h.obs['mri_voxel_z']
            axes.scatter3D(x_vec, y_vec, z_vec, color=scatter_color, s=marker_s, label=region)
        axes.grid(False)
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')
        axes.set(frame_on=False)
        fig.subplots_adjust(left=0, right=0.7, bottom=0, top=1)
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
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), ncol=1, frameon=False)  # bbox_to_anchor=(1.1, 0.5),
        plt.savefig(save_path + f'Spatial_human.' + self.fig_format)

        # multiview
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.7
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 6))
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for region in TH_cluster_list:
            #region = region.strip('H-')
            print(region)
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['cluster_name_acronym'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['H-'+region]
            x_vec = adata_h.obs['mri_voxel_x']
            y_vec = adata_h.obs['mri_voxel_y']
            z_vec = adata_h.obs['mri_voxel_z']
            ax1.scatter(x_vec, y_vec, color=scatter_color, s=marker_s, label=region)
            ax2.scatter(x_vec, z_vec, color=scatter_color, s=marker_s, label=region)
            ax3.scatter(y_vec, z_vec, color=scatter_color, s=marker_s, label=region)

        ax1.grid(False)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        ax2.grid(False)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')

        ax3.grid(False)
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')

        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticklabels([])

        ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])

        ax3.xaxis.set_ticklabels([])
        ax3.yaxis.set_ticklabels([])

        for line in ax1.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax1.yaxis.get_ticklines():
            line.set_visible(False)

        for line in ax2.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax2.yaxis.get_ticklines():
            line.set_visible(False)

        for line in ax3.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax3.yaxis.get_ticklines():
            line.set_visible(False)

        plt.legend(loc='right', bbox_to_anchor=(2.5, 0.5), ncol=1, frameon=False)  # bbox_to_anchor=(1.1, 0.5),
        # plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        plt.savefig(save_path + f'Spatial_human_multiview.' + self.fig_format)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9

        # --------------------------------------------------------------------------
        # Scatter according regions
        # multiview
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.18
        rcParams["figure.subplot.top"] = 0.98

        fig, axes = plt.subplots(1, len(TH_cluster_list), figsize=(4*len(TH_cluster_list), 5), dpi=self.fig_dpi)
        # plt.subplots_adjust(bottom=0.2, left=0.6)
        # rcParams["figure.subplot.left"] = 0.05
        for i, region in zip(range(len(TH_cluster_list)),
                             TH_cluster_list):
            #region = region.strip('H-')
            #print(region)
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['cluster_name_acronym'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['H-'+region]
            x_vec = adata_h.obs['mri_voxel_x']
            y_vec = adata_h.obs['mri_voxel_y']
            z_vec = adata_h.obs['mri_voxel_z']
            for j in range(len(TH_human_list)):
                if i != j:
                    axes[j].scatter(x_vec, z_vec, color='grey', s=marker_s*2)
            #axes[i].scatter(x_vec, z_vec, color=scatter_color, s=marker_s*3, label=region)

        for i, region in zip(range(len(TH_cluster_list)),
                             TH_cluster_list):
            # region = region.strip('H-')
            # print(region)
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['cluster_name_acronym'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['H-' + region]
            x_vec = adata_h.obs['mri_voxel_x']
            y_vec = adata_h.obs['mri_voxel_y']
            z_vec = adata_h.obs['mri_voxel_z']
            axes[i].scatter(x_vec, z_vec, color=scatter_color, s=marker_s * 3, label=region)


        for region, ax1 in zip(TH_human_list, axes):
            ax1.grid(False)
            ax1.set_xlabel(region)
            ax1.set_ylabel('')
            ax1.set(frame_on=False)
            ax1.xaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])

            for line in ax1.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax1.yaxis.get_ticklines():
                line.set_visible(False)

        # plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), ncol=1, frameon=False)  # bbox_to_anchor=(1.1, 0.5),
        # plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        plt.savefig(save_path + f'Spatial_human_eachregion.' + self.fig_format)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9


        #--------------------------------------------------------------
        # human 3D each region
        #-----------------------------------------------------------------------
        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.99

        # marker_s =

        # fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=self.fig_dpi, projection='3d')
        fig = plt.figure(figsize=(2*len(TH_cluster_list), 2), dpi=self.fig_dpi)
        axes = []
        for i in range(len(TH_cluster_list)):
            ax = fig.add_subplot(1, len(TH_cluster_list), i + 1, projection='3d')
            axes.append(ax)


        for i, region in zip(range(len(TH_cluster_list)),
                             TH_cluster_list):
            # region = region.strip('H-')
            # print(region)
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['cluster_name_acronym'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['H-' + region]
            x_vec = adata_h.obs['mri_voxel_x']
            y_vec = adata_h.obs['mri_voxel_y']
            z_vec = adata_h.obs['mri_voxel_z']
            for j in range(len(TH_human_list)):
                if i != j:
                    axes[j].scatter3D(x_vec, y_vec, z_vec, color='lightgrey', s=marker_s)
            # axes[i].scatter(x_vec, z_vec, color=scatter_color, s=marker_s*3, label=region)

        for i, region in zip(range(len(TH_cluster_list)),
                             TH_cluster_list):
            # region = region.strip('H-')
            # print(region)
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['cluster_name_acronym'].isin([region])]
            # print(adata_m)
            # print(adata_m)
            scatter_color = palette['H-' + region]
            x_vec = adata_h.obs['mri_voxel_x']
            y_vec = adata_h.obs['mri_voxel_y']
            z_vec = adata_h.obs['mri_voxel_z']
            axes[i].scatter3D(x_vec, y_vec, z_vec, color=scatter_color, s=marker_s * 3, label=region)

        for region, ax1 in zip(TH_human_list, axes):
            #for region, ax1 in zip(isocortex_region_human_list, axes):
            ax1.grid(True)
            ax1.set_xlabel(region)
            # ax1.set_ylabel('', linespacing=0)
            # ax1.set_zlabel('', linespacing=0)

            ax1.xaxis.labelpad = -12
            ax1.yaxis.labelpad = 0
            ax1.zaxis.labelpad = 0

            # ax1.set(frame_on=False)
            ax1.xaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])
            ax1.zaxis.set_ticklabels([])

            # No ticks
            # ax1.set_xticks([])
            # ax1.set_yticks([])
            # ax1.set_zticks([])
            #
            # # Transparent spines
            # # ax1.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # # ax1.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # # ax1.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            #
            # for line in ax1.xaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax1.yaxis.get_ticklines():
            #     line.set_visible(False)
            # for line in ax1.zaxis.get_ticklines():
            #     line.set_visible(False)

            # ax1.view_init(60, -80)
            # ax1.tick_params(axis='x', pad=2)  # Adjust as needed
            # ax1.tick_params(axis='y', pad=2)  # Adjust as needed
            # ax1.tick_params(axis='z', pad=2)  # Adjust as needed

        # plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), ncol=1, frameon=False)  # bbox_to_anchor=(1.1, 0.5),
        # plt.subplots_adjust(right=0.7)
        # axes.view_init(45, 215)
        plt.savefig(save_path + f'Spatial_human_eachregion_3D.' + self.fig_format)
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9



        # ------------------------------------------------------------
        # umap plots of region pairs
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.top"] = 0.995
        rcParams["figure.subplot.bottom"] = 0.195
        fig, axes = plt.subplots(1, len(TH_cluster_list), figsize=(6*len(TH_cluster_list), 5), dpi=self.fig_dpi)

        markersize = 60

        for i in range(len(TH_mouse_list)):
            region_m = TH_mouse_list[i]
            region_h = TH_human_list[i]

            adata_mh = adata_embedding_hip[adata_embedding_hip.obs['cluster_name_acronym'].isin([region_m, region_h])]
            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i])

            sc.pl.umap(adata_mh,
                       color='cluster_name_acronym',
                       palette={region_m: palette[region_m], region_h:palette[region_h]},
                       ax=ax, legend_loc='lower center', legend_fontweight='normal', show=False, size=markersize)

        for ax1 in axes:
            ax1.grid(False)
            ax1.set_xlabel('')
            ax1.set_title('')
            ax1.set_ylabel('')
            ax1.set(frame_on=False)
            ax1.xaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])

            for line in ax1.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax1.yaxis.get_ticklines():
                line.set_visible(False)

        plt.savefig(save_path + f'Spatial_eachregion_pair.' + self.fig_format)


        # ----------------------------------------------------------------------
        # Statistics of the fact: homologous regions such as Mouse CA1 and Human CA1 are more close to each other
        # 1. Distance of pair region samples are significantly higher than those are not homologous
        # boxplot (mark t-test p-values)
        Name_list = TH_cluster_list + ['Non-homologous']
        Dist_dict = {k:[] for k in Name_list}

        #sc.pp.neighbors(adata_embedding_hip, n_neighbors=self.cfg.ANALYSIS.umap_neighbor, metric='euclidean', use_rep='X')
        #sc.tl.umap(adata_embedding_hip, n_components=2, min_dist=0.1)

        # homologous
        for i in range(len(TH_cluster_list)):
            region_m = TH_mouse_list[i]
            region_h = TH_human_list[i]

            adata_m = adata_embedding_hip[adata_embedding_hip.obs['cluster_name_acronym'].isin([region_m])]
            adata_h = adata_embedding_hip[adata_embedding_hip.obs['cluster_name_acronym'].isin([region_h])]

            dist_m = scipy.spatial.distance_matrix(adata_m.obsm['X_umap'], adata_h.obsm['X_umap'], p=2)
            Dist_dict[TH_cluster_list[i]] = list(np.min(dist_m, axis=0).flatten()) + list(np.min(dist_m, axis=1).flatten())

        # Non-homologous
        for i in range(len(TH_mouse_list)):
            for j in range(len(TH_human_list)):
                if i != j: # and max(i, j) >= 3
                    region_m = TH_mouse_list[i]
                    region_h = TH_human_list[i]

                    adata_m = adata_embedding_hip[adata_embedding_hip.obs['cluster_name_acronym'].isin([region_m])]
                    adata_h = adata_embedding_hip[adata_embedding_hip.obs['cluster_name_acronym'].isin([region_h])]

                    dist_m = scipy.spatial.distance_matrix(adata_m.obsm['X_umap'], adata_h.obsm['X_umap'], p=2)
                    Dist_dict['Non-homologous'] = Dist_dict['Non-homologous'] + list(np.min(dist_m, axis=0).flatten()) + list(np.min(dist_m, axis=1).flatten())

        Boxplot_dict = {'Cluster pairs':[], 'Distance':[]}
        for k,v in Dist_dict.items():
            Boxplot_dict['Cluster pairs'] = Boxplot_dict['Cluster pairs'] + [k] * len(v)
            Boxplot_dict['Distance'] = Boxplot_dict['Distance'] + v

        Boxplot_df = pd.DataFrame.from_dict(Boxplot_dict)

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.22
        rcParams["figure.subplot.bottom"] = 0.3
        rcParams["figure.subplot.top"] = 0.9

        color_pal = {k:v for k,v in zip(Name_list, color_list[0:len(TH_cluster_list)])}
        color_pal['Non-homologous'] = 'grey'

        plt.figure(figsize=(5, 8), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        # sns.set(rc={"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)})
        ax = sns.boxplot(x="Cluster pairs", y='Distance', data=Boxplot_df, order=Name_list, palette=color_pal,
                         width=0.75)  #
        add_stat_annotation(ax, data=Boxplot_df, x="Cluster pairs", y='Distance', order=Name_list,
                            box_pairs=[(TH_cluster_list[0], 'Non-homologous'),
                                       (TH_cluster_list[1], 'Non-homologous'),
                                       (TH_cluster_list[2], 'Non-homologous'),
                                       (TH_cluster_list[3], 'Non-homologous'),
                                       (TH_cluster_list[4], 'Non-homologous')],
                            test='t-test_ind', text_format='star', loc='inside', verbose=2)  # text_format='star'
        plt.title('')
        plt.ylabel('Distance')
        plt.xlabel('')
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.savefig(save_path + 'pair_region_distance.' + self.fig_format, format=self.fig_format)
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.9

        #-----------------------------------------------------------------------------------------------------------
        # Statistics of the fact: homologous regions such as Mouse CA1 and Human CA1 are more close to each other
        # 1. Alignment score of pair region samples are significantly higher than those are not homologous
        # boxplot (mark t-test p-values)
        Name_list = TH_cluster_list + ['Non-homologous']
        Dist_dict = {k: 0 for k in Name_list}

        # sc.pp.neighbors(adata_embedding_hip, n_neighbors=self.cfg.ANALYSIS.umap_neighbor, metric='euclidean', use_rep='X')
        # sc.tl.umap(adata_embedding_hip, n_components=2, min_dist=0.3)

        # homologous
        for i in range(len(TH_cluster_list)):
            region_m = TH_cluster_list[i] #'M-' +
            region_h =TH_cluster_list[i] #'H-' +

            adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['cluster_name_acronym'].isin([region_m])]
            adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['cluster_name_acronym'].isin([region_h])]

            X = np.concatenate((adata_m.obsm['X_umap'], adata_h.obsm['X_umap']), axis=0)
            Y = np.concatenate([np.zeros((adata_m.n_obs, 1)), np.ones((adata_h.n_obs, 1))], axis=0)
            Dist_dict[TH_cluster_list[i]] = seurat_alignment_score(X, Y)

        # Non-homologous
        seurat_as_list = []
        for i in range(len(self.mouse_64_labels_list)):
            for j in range(len(self.human_88_labels_list)):
                if i != j: #and max(i, j) >= 3
                    region_m = self.mouse_64_labels_list[i] #'M-' +
                    region_h = self.human_88_labels_list[j] #'H-' +

                    adata_m = self.adata_mouse_embedding[self.adata_mouse_embedding.obs['region_name'].isin([region_m])]
                    adata_h = self.adata_human_embedding[self.adata_human_embedding.obs['region_name'].isin([region_h])]

                    X = np.concatenate((adata_m.obsm['X_umap'], adata_h.obsm['X_umap']), axis=0)
                    Y = np.concatenate([np.zeros((adata_m.n_obs, 1)), np.ones((adata_h.n_obs, 1))], axis=0)

                    seurat_as_list.append(seurat_alignment_score(X, Y))

        Dist_dict['Non-homologous'] = np.mean(seurat_as_list)

        Boxplot_dict = {'Cluster pairs': [], 'Alignment score': []}
        for k, v in Dist_dict.items():
            Boxplot_dict['Cluster pairs'] = Boxplot_dict['Cluster pairs'] + [k]
            Boxplot_dict['Alignment score'] = Boxplot_dict['Alignment score'] + [v]

        Boxplot_df = pd.DataFrame.from_dict(Boxplot_dict)

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.22
        rcParams["figure.subplot.bottom"] = 0.3
        rcParams["figure.subplot.top"] = 0.9

        color_pal = {k: v for k, v in zip(Name_list, color_list[0:len(TH_cluster_list)])}
        color_pal['Non-homologous'] = 'grey'

        plt.figure(figsize=(5, 8), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        # sns.set(rc={"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)})
        ax = sns.barplot(x="Cluster pairs", y='Alignment score', data=Boxplot_df, order=Name_list, palette=color_pal,
                         width=0.75)  #
        # add_stat_annotation(ax, data=Boxplot_df, x="Region pairs", y='Distance', order=Name_list,
        #                     box_pairs=[('Field CA1', 'Non-homologous'),
        #                                ('Field CA2', 'Non-homologous'),
        #                                ('Field CA3', 'Non-homologous'),
        #                                ('Dentate gyrus', 'Non-homologous'),
        #                                ('Subiculum', 'Non-homologous')],
        #                     test='t-test_ind', text_format='star', loc='inside', verbose=2)  # text_format='star'
        plt.title('')
        plt.ylabel('Alignment score')
        plt.xlabel('')
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.savefig(save_path + 'pair_region_distance_alignment_score.' + self.fig_format, format=self.fig_format)
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.9



        # ---------------------------------------------------------------------------------------
        # Fold change of alignement score
        Name_list = TH_cluster_list
        Dist_dict = {k: 0 for k in Name_list}

        # sc.pp.neighbors(adata_embedding_hip, n_neighbors=self.cfg.ANALYSIS.umap_neighbor, metric='euclidean', use_rep='X')
        # sc.tl.umap(adata_embedding_hip, n_components=2, min_dist=0.3)

        # homologous
        for i in range(len(TH_cluster_list)):
            region_m = TH_cluster_list[i]  # 'M-' +
            region_h = TH_cluster_list[i]  # 'H-' +

            adata_m = self.adata_embedding[self.adata_embedding.obs['cluster_name_acronym'].isin([region_m])]
            adata_h = self.adata_embedding[self.adata_embedding.obs['cluster_name_acronym'].isin([region_h])]

            X = np.concatenate((adata_m.obsm['X_umap'], adata_h.obsm['X_umap']), axis=0)
            Y = np.concatenate([np.zeros((adata_m.n_obs, 1)), np.ones((adata_h.n_obs, 1))], axis=0)
            Dist_dict[TH_cluster_list[i]] = seurat_alignment_score(X, Y)

        # Non-homologous
        seurat_as_list = []
        for i in range(len(self.mouse_64_labels_list)):
            for j in range(len(self.human_88_labels_list)):
                if i != j:  # and max(i, j) >= 3
                    region_m = self.mouse_64_labels_list[i]  # 'M-' +
                    region_h = self.human_88_labels_list[i]  # 'H-' +

                    adata_m = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_m])]
                    adata_h = self.adata_embedding[self.adata_embedding.obs['region_name'].isin([region_h])]

                    X = np.concatenate((adata_m.obsm['X_umap'], adata_h.obsm['X_umap']), axis=0)
                    Y = np.concatenate([np.zeros((adata_m.n_obs, 1)), np.ones((adata_h.n_obs, 1))], axis=0)

                    seurat_as_list.append(seurat_alignment_score(X, Y))

        Non_homo_score = np.mean(seurat_as_list)

        Boxplot_dict = {'Cluster': [], 'Alignment score fold change': []}
        for k, v in Dist_dict.items():
            Boxplot_dict['Cluster'] = Boxplot_dict['Cluster'] + [k]
            Boxplot_dict['Alignment score fold change'] = Boxplot_dict['Alignment score fold change'] + [
                v / Non_homo_score]

        Boxplot_df = pd.DataFrame.from_dict(Boxplot_dict)

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.22
        rcParams["figure.subplot.bottom"] = 0.3
        rcParams["figure.subplot.top"] = 0.9

        color_pal = {k: v for k, v in zip(Name_list, color_list[0:5])}
        #color_pal['Non-homologous'] = 'grey'

        plt.figure(figsize=(3, 3), dpi=self.fig_dpi)
        # with plt.rc_context({"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)}):
        # sns.set(rc={"figure.figsize": (8, 8), "figure.dpi": (self.fig_dpi)})
        ax = sns.barplot(x="Cluster", y='Alignment score fold change', data=Boxplot_df, order=Name_list,
                         palette=color_pal,
                         width=0.62)  #
        ax.axhline(1, color='grey', linestyle='--')
        plt.title('')
        plt.ylabel('Alignment score fold change')
        plt.xlabel('')
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")
        plt.savefig(save_path + 'cluster_distance_fold_change.' + self.fig_format, format=self.fig_format)
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.bottom"] = 0.1
        rcParams["figure.subplot.top"] = 0.9





        # Marker genes of CA1,CA2,CA2,Dentate gyrus
        # CA1:
        #adata_mouse_embedding_hip = adata_embedding_hip[adata_embedding_hip.obs['dataset'].isin(['Mouse'])]
        #adata_human_embedding_hip = adata_embedding_hip[adata_embedding_hip.obs['dataset'].isin(['Human'])]

        # minimum_num = 10
        #Mouse_markergene_list = ['Cyp39a1', 'Edc3', 'B3gat2', 'Gprin3', 'Phactr2']#['Itpka', 'Amigo2', 'Hs3st4', 'Lrrtm4', 'Nts'] #['Fndc1', 'Scgn', 'Tspan18', 'Chrna1', 'Nts']# 'Gfra1'
        #Human_markergene_list = ['PVALB', 'SLN', 'DDX3Y', 'TRIM54', 'PTGER3']#['KCNS1', 'TMEM155', 'TAC3', 'LHX6', 'CCK']#['ITPKA', 'AMIGO2', 'HS3ST4', 'LRRTM4', 'NTSR2']

        # minimum_num = 5
        #['Primary somatosensory area', 'Supplemental somatosensory area', 'Ectorhinal area', 'Auditory areas',
        # 'Perirhinal area']
        #['cuneus', 'lingual gyrus', 'temporal pole', 'superior parietal lobule', 'parahippocampal gyrus']

        # 'Anterior cingulate area', 'Orbital area', 'Auditory areas', 'Primary somatosensory area', 'Secondary motor area'
        # 'inferior rostral gyrus', 'parolfactory gyri', "Heschl's gyrus", 'cuneus', 'posterior orbital gyrus'
        #Mouse_markergene_list = ['Ddx3y', 'Cbln3', 'Rad9b', 'Epsti1', 'Gtf3a']#['Cyp39a1', 'Wfdc18', 'Il16', 'Rad9b', 'Phactr2']
        # '', '', 'Tmigd1', 'Epsti1', '	Gtf3a'
        #Mouse_markergene_list = ['Ddx3y', 'Cbln3', 'Rad9b', 'Epsti1', 'Gtf3a']

        #####################################################################################################

        adata_exp_TH_mouse = self.adata_mouse_exp[adata_mouse_embedding_hip.obs_names, :]
        adata_exp_TH_human = self.adata_human_exp[adata_human_embedding_hip.obs_names, :]

        adata_exp_TH_mouse.write_h5ad(save_path+'adata_exp_TH_mouse.h5ad')
        adata_exp_TH_human.write_h5ad(save_path + 'adata_exp_TH_human.h5ad')

        # mouse degs
        groupby = 'cluster_name_acronym'
        # sc.pp.filter_cells(adata, min_counts=2)
        ntop_genes = 100
        ntop_genes_visual = 1
        save_pre = 'mouse'
        key =  'wilcoxon' # 't-test_overestim_var'#'wilcoxon'#'t-test' 'wilcoxon'
        key_filtered = 'wilcoxon'  # 't-test_overestim_var'# 'wilcoxon-filtered'#'t-test-filtered' 'wilcoxon'

        cluster_counts = adata_exp_TH_mouse.obs[groupby].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        adata = adata_exp_TH_mouse[adata_exp_TH_mouse.obs[groupby].isin(keep)].copy()

        sc.tl.rank_genes_groups(adata, groupby, method='t-test', key_added="t-test")
        sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added="wilcoxon")
        sc.tl.rank_genes_groups(adata, groupby, method='t-test_overestim_var', key_added="t-test_overestim_var")
        sc.tl.rank_genes_groups(adata, groupby, method='logreg', key_added="logreg")

        sc.pl.rank_genes_groups(adata, n_genes=ntop_genes, sharey=False, key=key, show=False)

        rcParams["figure.subplot.top"] = 0.7
        rcParams["figure.subplot.bottom"] = 0.25
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        # with plt.rc_context({"figure.figsize": (12, 16)}):

        var_group_positions = []
        for i in range(len(TH_cluster_list)):
            var_group_positions.append((i * ntop_genes_visual + 1, (i + 1) * ntop_genes_visual + 1))

        dp = sc.pl.rank_genes_groups_dotplot(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                        groupby=groupby, show=False,
                                        dendrogram=True,
                                        figsize=(12, 4),
                                        smallest_dot=20,
                                        return_fig=True,
                                        # var_group_positions=var_group_positions,
                                        values_to_plot="logfoldchanges",
                                        colorbar_title='log fold change',
                                        standard_scale='var')  # , values_to_plot="logfoldchanges" , cmap='bwr'
        categories_order_mouse = dp.categories_order
        #var_names_mouse = dp.var_names
        X_1 = TH_cluster_list
        print('X_1', X_1)
        X_ordered = categories_order_mouse
        print('X_ordered', X_ordered)
        Y_1 = dp.var_names
        print('Y_1', Y_1)
        var_names_ordered = reorder(X_1, X_ordered, Y_1)
        dict_gene_mouse = {k:v for k,v in zip(categories_order_mouse, var_names_ordered)}
        Mouse_markergene_list = [dict_gene_mouse[k] for k in TH_cluster_list]
        dp.savefig(save_path + 'mouse_deg_dotplot.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)

        # Human marker gene list
        cluster_counts = adata_exp_TH_human.obs[groupby].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        adata = adata_exp_TH_human[adata_exp_TH_human.obs[groupby].isin(keep)].copy()

        sc.tl.rank_genes_groups(adata, groupby, method='t-test', key_added="t-test")
        sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added="wilcoxon")
        sc.tl.rank_genes_groups(adata, groupby, method='t-test_overestim_var', key_added="t-test_overestim_var")
        sc.tl.rank_genes_groups(adata, groupby, method='logreg', key_added="logreg")

        sc.pl.rank_genes_groups(adata, n_genes=ntop_genes, sharey=False, key=key, show=False)

        rcParams["figure.subplot.top"] = 0.7
        rcParams["figure.subplot.bottom"] = 0.25
        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        # with plt.rc_context({"figure.figsize": (12, 16)}):

        var_group_positions = []
        for i in range(len(TH_cluster_list)):
            var_group_positions.append((i * ntop_genes_visual + 1, (i + 1) * ntop_genes_visual + 1))

        dp = sc.pl.rank_genes_groups_dotplot(adata, n_genes=ntop_genes_visual, key=key_filtered,
                                             groupby=groupby, show=False,
                                             dendrogram=True,
                                             figsize=(12, 4),
                                             smallest_dot=20,
                                             return_fig=True,
                                             # var_group_positions=var_group_positions,
                                             values_to_plot="logfoldchanges",
                                             colorbar_title='log fold change',
                                             standard_scale='var')  # , values_to_plot="logfoldchanges" , cmap='bwr'
        categories_order_human = dp.categories_order
        # var_names_mouse = dp.var_names
        X_1 = TH_cluster_list
        print('X_1', X_1)
        X_ordered = categories_order_human
        print('X_ordered', X_ordered)
        Y_1 = dp.var_names
        print('Y_1', Y_1)
        var_names_ordered = reorder(X_1, X_ordered, Y_1)
        dict_gene_human = {k: v for k, v in zip(categories_order_human, var_names_ordered)}
        Human_markergene_list = [dict_gene_human[k] for k in TH_cluster_list]
        dp.savefig(save_path + 'human_deg_dotplot.' + self.fig_format, format=self.fig_format,
                    dpi=self.fig_dpi)

        #Mouse_markergene_list = ['Ndst4', 'Cck', 'Mef2c', 'Snap25', 'Cck']#['Ddx3y', 'Zcchc18', 'Tmem196', 'Eepd1', 'Rbp4']
        #Human_markergene_list = ['SYT10', 'ZCCHC18', 'TMEM196', 'EEPD1', 'RBP4']#['PVALB', 'TRPC3', 'PTGER3', 'SHD', 'TTR']
        # 'RTP1', SLN, COL22A1, PVALB, CARTPT
        adata_mouse_exp_hip = self.adata_mouse_exp[self.adata_mouse_exp.obs['cluster_name_acronym'].isin(TH_cluster_list)]
        adata_human_exp_hip = self.adata_human_exp[
            self.adata_human_exp.obs['cluster_name_acronym'].isin(TH_cluster_list)]

        print(self.adata_mouse_exp.var_names.tolist())
        print(self.adata_human_exp.var_names.tolist())

        #color_map = 'viridis'#'magma_r'
        c = Colormap()
        color_map = c.cmap_linear('#3288BD', 'white', '#D53E4F')

        markersize = 40
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.18
        rcParams["figure.subplot.top"] = 0.98
        # mouse marker genes
        fig, axes = plt.subplots(1, len(TH_cluster_list), figsize=(4*len(TH_cluster_list), 5.7))
        for i in range(len(Mouse_markergene_list)):
            m_gene = Mouse_markergene_list[i]
            #h_gene = Human_markergene_list[i]

            adata_mouse_embedding_hip.obs[m_gene] = adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)
            diff = False
            for k, v in zip(adata_mouse_embedding_hip.obs_names, adata_mouse_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i])

            sc.pl.umap(adata_mouse_embedding_hip,
                       color= m_gene,
                       #palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize, color_map=color_map, colorbar_loc=None)

            ax.grid(False)
            ax.set_xlabel(m_gene)
            ax.set_title('')
            ax.set_ylabel('')
            ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

            for line in ax.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.yaxis.get_ticklines():
                line.set_visible(False)

        plt.savefig(save_path + f'Mouse_marker_gene.' + self.fig_format)

        # human marker genes
        markersize = 60

        fig, axes = plt.subplots(1, len(TH_cluster_list), figsize=(4*len(TH_cluster_list), 5.7), dpi=self.fig_dpi)
        for i in range(len(Human_markergene_list)):
            #m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_human_embedding_hip.obs[h_gene] = adata_human_exp_hip[:, h_gene].X.toarray().reshape(-1) # normalize(adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)[:, np.newaxis], norm='max', axis=0).ravel()

            diff = False
            for k, v in zip(adata_human_embedding_hip.obs_names, adata_human_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i])

            sc.pl.umap(adata_human_embedding_hip,
                       color=h_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None)

            ax.grid(False)
            ax.set_xlabel(h_gene)
            ax.set_title('')
            ax.set_ylabel('')
            ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

            for line in ax.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.yaxis.get_ticklines():
                line.set_visible(False)

        plt.savefig(save_path + f'Human_marker_gene.' + self.fig_format)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9

        # # ----------------------------------------------------------------------
        # # Another pair marker gene
        # 0, 24, 26, 43, 46
        Mouse_markergene_list = ['Dlgap3', 'Fam104a', 'Cygb', 'Cdk13', 'Dip2a'] #['Fndc1', 'Scgn', 'Tspan18', 'Chrna1', 'Nts']# 'Gfra1'
        Human_markergene_list = ['JPH1', 'WNT4', 'CD6', 'PRR16', 'STK17B']

        adata_mouse_exp_hip = self.adata_mouse_exp[
            self.adata_mouse_exp.obs['cluster_name_acronym'].isin(TH_cluster_list)]
        adata_human_exp_hip = self.adata_human_exp[
            self.adata_human_exp.obs['cluster_name_acronym'].isin(TH_cluster_list)]

        print(self.adata_human_exp.var_names.tolist())
        print(self.adata_mouse_exp.var_names.tolist())

        #color_map = 'viridis'  # 'magma_r'
        c = Colormap()
        color_map = c.cmap_linear('#3288BD', 'white', '#D53E4F')

        markersize = 40
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.95
        rcParams["figure.subplot.bottom"] = 0.18
        rcParams["figure.subplot.top"] = 0.98
        # mouse marker genes
        fig, axes = plt.subplots(1, len(TH_cluster_list), figsize=(4*len(TH_cluster_list), 5.7))
        for i in range(len(Mouse_markergene_list)):
            m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_mouse_embedding_hip.obs[m_gene] = adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)
            diff = False
            for k, v in zip(adata_mouse_embedding_hip.obs_names, adata_mouse_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i])

            sc.pl.umap(adata_mouse_embedding_hip,
                       color=m_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None)

            ax.grid(False)
            ax.set_xlabel(m_gene)
            ax.set_title('')
            ax.set_ylabel('')
            ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

            for line in ax.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.yaxis.get_ticklines():
                line.set_visible(False)

        plt.savefig(save_path + f'Mouse_marker_gene_seurat.' + self.fig_format)

        # human marker genes
        markersize = 60

        fig, axes = plt.subplots(1, len(TH_cluster_list), figsize=(4*len(TH_cluster_list), 5.7), dpi=self.fig_dpi)
        for i in range(len(Human_markergene_list)):
            # m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_human_embedding_hip.obs[h_gene] = adata_human_exp_hip[:, h_gene].X.toarray().reshape(
                -1)  # normalize(adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)[:, np.newaxis], norm='max', axis=0).ravel()

            diff = False
            for k, v in zip(adata_human_embedding_hip.obs_names, adata_human_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i])

            sc.pl.umap(adata_human_embedding_hip,
                       color=h_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None)

            ax.grid(False)
            ax.set_xlabel(h_gene)
            ax.set_title('')
            ax.set_ylabel('')
            ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

            for line in ax.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.yaxis.get_ticklines():
                line.set_visible(False)

        plt.savefig(save_path + f'Human_marker_gene_seurat.' + self.fig_format)

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9


        #--------------------------------------------------------------------------------------
        # 3D region pair plot
        #-----------------------------------------------------------------------------
        #########################################################
        sc.tl.umap(adata_embedding_hip, n_components=3, min_dist=0.3)

        adata_mouse_embedding_hip = adata_embedding_hip[adata_embedding_hip.obs['dataset'].isin(['Mouse'])]
        adata_human_embedding_hip = adata_embedding_hip[adata_embedding_hip.obs['dataset'].isin(['Human'])]

        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.top"] = 0.99
        rcParams["figure.subplot.bottom"] = 0.2

        fig = plt.figure(figsize=(2*len(TH_cluster_list), 2.2), dpi=self.fig_dpi)
        axes = []
        for i in range(len(TH_cluster_list)):
            ax = fig.add_subplot(1, len(TH_cluster_list), i + 1, projection='3d')
            ax.grid(True)
            axes.append(ax)

        markersize = 40

        for i in range(len(TH_mouse_list)):
            region_m = TH_mouse_list[i]
            region_h = TH_human_list[i]

            adata_mh = adata_embedding_hip[adata_embedding_hip.obs['cluster_name_acronym'].isin([region_m, region_h])]
            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i], projection='3d', size=markersize - 25,
                            alpha=0.3)

            sc.pl.umap(adata_mh,
                       color='cluster_name_acronym',
                       palette={region_m: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc='lower center', legend_fontweight='normal', show=False, size=markersize, projection='3d', alpha=1)

        for ax1 in axes:
            ax1.grid(True)
            ax1.set_xlabel('')
            ax1.set_ylabel('')
            ax1.set_zlabel('')

            ax1.xaxis.labelpad = -12
            ax1.yaxis.labelpad = 0
            ax1.zaxis.labelpad = 0

            # ax1.set(frame_on=False)
            ax1.xaxis.set_ticklabels([])
            ax1.yaxis.set_ticklabels([])
            ax1.zaxis.set_ticklabels([])

            ax.set_title('')

        plt.savefig(save_path + f'Spatial_eachregion_pair_3D.' + self.fig_format)



        # -----------------------------------------------------------------------------
        # 3D marker gene
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        Mouse_markergene_list = ['Dlgap3', 'Fam104a', 'Cygb', 'Cdk13',
                                 'Dip2a']  # ['Fndc1', 'Scgn', 'Tspan18', 'Chrna1', 'Nts']# 'Gfra1'
        Human_markergene_list = ['JPH1', 'WNT4', 'CD6', 'PRR16', 'STK17B']

        adata_mouse_exp_hip = self.adata_mouse_exp[
            self.adata_mouse_exp.obs['cluster_name_acronym'].isin(TH_cluster_list)]
        adata_human_exp_hip = self.adata_human_exp[
            self.adata_human_exp.obs['cluster_name_acronym'].isin(TH_cluster_list)]

        print(self.adata_human_exp.var_names.tolist())
        print(self.adata_mouse_exp.var_names.tolist())

        # color_map = 'viridis'  # 'magma_r'
        c = Colormap()
        color_map = c.cmap_linear('#3288BD', 'white', '#D53E4F')

        markersize = 20
        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.top"] = 0.99
        rcParams["figure.subplot.bottom"] = 0.1

        fig = plt.figure(figsize=(2*len(TH_cluster_list), 2), dpi=self.fig_dpi)
        axes = []
        for i in range(len(TH_cluster_list)):
            ax = fig.add_subplot(1, len(TH_cluster_list), i + 1, projection='3d')
            ax.grid(True)
            axes.append(ax)

        # mouse marker genes

        for i in range(len(Mouse_markergene_list)):
            m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_mouse_embedding_hip.obs[m_gene] = adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)
            diff = False
            for k, v in zip(adata_mouse_embedding_hip.obs_names, adata_mouse_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i], projection='3d',size=markersize-17,  alpha=0.2)

            sc.pl.umap(adata_mouse_embedding_hip,
                       color=m_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None, projection='3d',  alpha=1)

            ax.grid(True)
            if len(m_gene) <= 4:
                m_gene = m_gene + ' '
            ax.set_xlabel(m_gene, fontdict=dict(style='italic'))  # , linespacing=-12
            ax.set_ylabel('', linespacing=0)
            ax.set_zlabel('', linespacing=0)

            ax.set_title('')

            ax.xaxis.labelpad = -12
            ax.yaxis.labelpad = 0
            ax.zaxis.labelpad = 0

            # ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

        plt.savefig(save_path + f'Mouse_marker_gene_3D.' + self.fig_format)

        # human marker genes
        markersize = 80

        fig = plt.figure(figsize=(2 * len(TH_cluster_list), 2), dpi=self.fig_dpi)
        axes = []
        for i in range(len(TH_cluster_list)):
            ax = fig.add_subplot(1, len(TH_cluster_list), i + 1, projection='3d')
            ax.grid(True)
            axes.append(ax)

        for i in range(len(Human_markergene_list)):
            # m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_human_embedding_hip.obs[h_gene] = adata_human_exp_hip[:, h_gene].X.toarray().reshape(
                -1)  # normalize(adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)[:, np.newaxis], norm='max', axis=0).ravel()

            diff = False
            for k, v in zip(adata_human_embedding_hip.obs_names, adata_human_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i], projection='3d', size=markersize-75, alpha=0.2)

            sc.pl.umap(adata_human_embedding_hip,
                       color=h_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None, projection='3d', alpha=1)

            ax.grid(True)
            if len(h_gene) <= 4:
                h_gene = h_gene + ' '
            ax.set_xlabel(h_gene, fontdict=dict(style='italic'))  # , linespacing=-12
            ax.set_ylabel('', linespacing=0)
            ax.set_zlabel('', linespacing=0)

            ax.set_title('')

            ax.xaxis.labelpad = -12
            ax.yaxis.labelpad = 0
            ax.zaxis.labelpad = 0

            # ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

        plt.savefig(save_path + f'Human_marker_gene_3D.' + self.fig_format)


        Mouse_markergene_list = ['Jph1', 'Wnt4', 'Cd6', 'Prr16', 'Stk17b']   # ['Fndc1', 'Scgn', 'Tspan18', 'Chrna1', 'Nts']# 'Gfra1'
        Human_markergene_list = ['DLGAP3', 'FAM104A', 'PRR16', 'CDK10',
                                 'DIP2A']

        adata_mouse_exp_hip = self.adata_mouse_exp[
            self.adata_mouse_exp.obs['cluster_name_acronym'].isin(TH_cluster_list)]
        adata_human_exp_hip = self.adata_human_exp[
            self.adata_human_exp.obs['cluster_name_acronym'].isin(TH_cluster_list)]

        print(self.adata_human_exp.var_names.tolist())
        print(self.adata_mouse_exp.var_names.tolist())

        # color_map = 'viridis'  # 'magma_r'
        c = Colormap()
        color_map = c.cmap_linear('#3288BD', 'white', '#D53E4F')

        markersize = 20
        rcParams["figure.subplot.left"] = 0.01
        rcParams["figure.subplot.right"] = 0.99
        rcParams["figure.subplot.top"] = 0.99
        rcParams["figure.subplot.bottom"] = 0.1

        fig = plt.figure(figsize=(2 * len(TH_cluster_list), 2), dpi=self.fig_dpi)
        axes = []
        for i in range(len(TH_cluster_list)):
            ax = fig.add_subplot(1, len(TH_cluster_list), i + 1, projection='3d')
            ax.grid(True)
            axes.append(ax)

        # mouse marker genes

        for i in range(len(Mouse_markergene_list)):
            m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_mouse_embedding_hip.obs[m_gene] = adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)
            diff = False
            for k, v in zip(adata_mouse_embedding_hip.obs_names, adata_mouse_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i], projection='3d', size=markersize - 17,
                            alpha=0.2)

            sc.pl.umap(adata_mouse_embedding_hip,
                       color=m_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None, projection='3d', alpha=1)

            ax.grid(True)
            if len(m_gene) <= 4:
                m_gene = m_gene + ' '
            ax.set_xlabel(m_gene, fontdict=dict(style='italic'))  # , linespacing=-12
            ax.set_ylabel('', linespacing=0)
            ax.set_zlabel('', linespacing=0)

            ax.set_title('')

            ax.xaxis.labelpad = -12
            ax.yaxis.labelpad = 0
            ax.zaxis.labelpad = 0

            # ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

        plt.savefig(save_path + f'Mouse_marker_gene_human_3D.' + self.fig_format)

        # human marker genes
        markersize = 80

        fig = plt.figure(figsize=(2 * len(TH_cluster_list), 2), dpi=self.fig_dpi)
        axes = []
        for i in range(len(TH_cluster_list)):
            ax = fig.add_subplot(1, len(TH_cluster_list), i + 1, projection='3d')
            ax.grid(True)
            axes.append(ax)

        for i in range(len(Human_markergene_list)):
            # m_gene = Mouse_markergene_list[i]
            h_gene = Human_markergene_list[i]

            adata_human_embedding_hip.obs[h_gene] = adata_human_exp_hip[:, h_gene].X.toarray().reshape(
                -1)  # normalize(adata_mouse_exp_hip[:, m_gene].X.toarray().reshape(-1)[:, np.newaxis], norm='max', axis=0).ravel()

            diff = False
            for k, v in zip(adata_human_embedding_hip.obs_names, adata_human_exp_hip.obs_names):
                if k != v:
                    print("The is difference!")
                    diff = True
            if diff == False:
                print('They are identical!')

            ax = sc.pl.umap(adata_embedding_hip, show=False, ax=axes[i], projection='3d', size=markersize - 75,
                            alpha=0.2)

            sc.pl.umap(adata_human_embedding_hip,
                       color=h_gene,
                       # palette={m_gene: palette[region_m], region_h: palette[region_h]},
                       ax=ax, legend_loc=None, legend_fontweight='normal', show=False, size=markersize,
                       color_map=color_map, colorbar_loc=None, projection='3d', alpha=1)

            ax.grid(True)
            if len(h_gene) <= 4:
                h_gene = h_gene + ' '
            ax.set_xlabel(h_gene, fontdict=dict(style='italic'))  # , linespacing=-12
            ax.set_ylabel('', linespacing=0)
            ax.set_zlabel('', linespacing=0)

            ax.set_title('')

            ax.xaxis.labelpad = -12
            ax.yaxis.labelpad = 0
            ax.zaxis.labelpad = 0

            # ax.set(frame_on=False)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

        plt.savefig(save_path + f'Human_marker_gene_mouse_3D.' + self.fig_format)


        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9



        return None




    def experiment_1_spatial_alignment_specie2(self):
        '''
               Step 1: Compute average embedding and position of every region in two species, use two dict to store;
               Step 2: Compute distance between regions, select one-nearest neighbor of each region, compute correlations;
               Step 3: plot Boxplot.

            '''

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

        cfg = self.cfg
        fig_format = cfg.BrainAlign.fig_format

        human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
        mouse_67_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)

        adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')

        ## Human experiment
        # Read ordered labels
        expression_human_path = cfg.CAME.path_rawdata2
        adata_human = sc.read_h5ad(expression_human_path)
        human_embedding_dict = OrderedDict()
        # print(list(human_88_labels['region_name']))
        human_88_labels_list = list(human_88_labels['region_name'])
        for r_n in human_88_labels_list:
            human_embedding_dict[r_n] = None

        for region_name in human_88_labels_list:
            mean_embedding = np.mean(adata_human_embedding[adata_human_embedding.obs['region_name'] == region_name].X,
                                     axis=0)
            human_embedding_dict[region_name] = mean_embedding

        human_embedding_df = pd.DataFrame.from_dict(human_embedding_dict)

        # compute mean mri coordinates
        human_mri_xyz_dict = OrderedDict()
        for r_n in human_88_labels_list:
            human_mri_xyz_dict[r_n] = {'mri_voxel_x': 0, 'mri_voxel_y': 0, 'mri_voxel_z': 0}
        # print(np.mean(adata_human.obs['mri_voxel_x'].values))
        for region_name in human_88_labels_list:
            mean_mri_voxel_x = np.mean(
                adata_human[adata_human.obs['region_name'] == region_name].obs['mri_voxel_x'].values)
            human_mri_xyz_dict[region_name]['mri_voxel_x'] = mean_mri_voxel_x
            mean_mri_voxel_y = np.mean(
                adata_human[adata_human.obs['region_name'] == region_name].obs['mri_voxel_y'].values)
            human_mri_xyz_dict[region_name]['mri_voxel_y'] = mean_mri_voxel_y
            mean_mri_voxel_z = np.mean(
                adata_human[adata_human.obs['region_name'] == region_name].obs['mri_voxel_z'].values)
            human_mri_xyz_dict[region_name]['mri_voxel_z'] = mean_mri_voxel_z

        human_mri_xyz_df = pd.DataFrame.from_dict(human_mri_xyz_dict).T
        ## Compute distance matrix
        human_dist_df = pd.DataFrame(squareform(pdist(human_mri_xyz_df.values)), columns=human_mri_xyz_df.index,
                                     index=human_mri_xyz_df.index)
        for i in range(human_dist_df.shape[0]):
            human_dist_df.iloc[i, i] = 1e5

        human_mri_xyz_nearest = OrderedDict()
        for r_n in human_88_labels_list:
            human_mri_xyz_nearest[r_n] = {'region': None, 'distance': None}

        for region_name in human_88_labels_list:
            values_list = human_dist_df.loc[region_name].values
            index_v = np.argmin(values_list)
            human_mri_xyz_nearest[region_name]['region'] = human_dist_df.index[index_v]
            value_min = np.min(values_list)
            human_mri_xyz_nearest[region_name]['distance'] = value_min
        # print(human_mri_xyz_nearest)

        # pearson correlation matrix of embedding
        pearson_corr_df = human_embedding_df.corr()
        # print(pearson_corr_df)

        # Get nearst neighbor correlation and not neighbor correlation lists
        neighbor_pearson_list = []
        not_neighbor_pearson_list = []
        for region_n, r_dict in human_mri_xyz_nearest.items():
            neighbor_pearson_list.append(pearson_corr_df.loc[region_n, r_dict['region']])
        # print(neighbor_pearson_list)

        for region_x in human_88_labels_list:
            for region_y in human_88_labels_list:
                if region_x != region_y and human_mri_xyz_nearest[region_x]['region'] != region_y and \
                        human_mri_xyz_nearest[region_y]['region'] != region_x:
                    not_neighbor_pearson_list.append(pearson_corr_df.loc[region_x, region_y])

        # print(not_neighbor_pearson_list)

        neighbor_pearson_list_type = ["1-nearest neighbor" for i in range(len(neighbor_pearson_list))]
        not_neighbor_pearson_list_type = ["Not near" for i in range(len(not_neighbor_pearson_list))]

        data_dict = {'Pearson Correlation': neighbor_pearson_list + not_neighbor_pearson_list,
                     'Spatial Relation': neighbor_pearson_list_type + not_neighbor_pearson_list_type}
        data_df = pd.DataFrame.from_dict(data_dict)

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/2_anatomical_STs_analysis/2_experiment_spatial_alignment/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

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
        my_pal = {"1-nearest neighbor": (64 / 255, 125 / 255, 82 / 255), "Not near": (142 / 255, 41 / 255, 97 / 255)}
        # sns.set_theme(style="whitegrid")
        # tips = sns.load_dataset("tips")
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.bottom"] = 0.15
        plt.figure(figsize=(8, 8))
        ax = sns.boxplot(x="Spatial Relation", y="Pearson Correlation", data=data_df,
                         order=["1-nearest neighbor", "Not near"], palette=my_pal, width=0.65)
        add_stat_annotation(ax, data=data_df, x="Spatial Relation", y="Pearson Correlation",
                            order=["1-nearest neighbor", "Not near"],
                            box_pairs=[("1-nearest neighbor", "Not near")],
                            test='t-test_ind', text_format='star', loc='inside', verbose=2)  # text_format='star'
        plt.savefig(save_path + 'human_mri_xyz_pearson.' + fig_format, format=fig_format)
        plt.show()

        ## plot joint
        dist_pearson_dict = {'Euclidean Distance': [], 'Pearson Correlation': []}
        human_dist_list = []
        for i in range(len(human_dist_df.index)):
            for j in range(i + 1, len(human_dist_df.index)):
                human_dist_list.append(human_dist_df.iloc[i, j])

        pearson_corr_list = []
        for i in range(len(pearson_corr_df.index)):
            for j in range(i + 1, len(pearson_corr_df.index)):
                pearson_corr_list.append(pearson_corr_df.iloc[i, j])

        dist_pearson_dict['Euclidean Distance'] = human_dist_list
        dist_pearson_dict['Pearson Correlation'] = pearson_corr_list

        plt.figure(figsize=(8, 8))
        sns.jointplot(data=pd.DataFrame.from_dict(dist_pearson_dict), x="Euclidean Distance", y="Pearson Correlation",
                      kind="reg", scatter_kws={"s": 3}, height=8)
        plt.savefig(save_path + 'human_dist_corr_joint.' + fig_format, format=fig_format, dpi=500)
        plt.show()

    def experiment_1_spatial_alignment_specie1(self):

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

        cfg = self.cfg
        '''
               Step 1: Compute average embedding and position of every region in two species, use two dict to store;
               Step 2: Compute distance between regions, select one-nearest neighbor of each region, compute correlations;
               Step 3: plot Boxplot.

        '''
        fig_format = cfg.BrainAlign.fig_format

        # human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
        mouse_67_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)

        adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        # adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
        ## Human experiment
        # Read ordered labels
        expression_mouse_path = cfg.CAME.path_rawdata1
        adata_mouse = sc.read_h5ad(expression_mouse_path)

        mouse_embedding_dict = OrderedDict()
        mouse_67_labels_list = list(mouse_67_labels['region_name'])
        for r_n in mouse_67_labels_list:
            mouse_embedding_dict[r_n] = None

        for region_name in mouse_67_labels_list:
            mean_embedding = np.mean(adata_mouse_embedding[adata_mouse_embedding.obs['region_name'] == region_name].X,
                                     axis=0)
            mouse_embedding_dict[region_name] = mean_embedding

        mouse_embedding_df = pd.DataFrame.from_dict(mouse_embedding_dict)

        # compute mean mri coordinates
        mouse_xyz_dict = OrderedDict()
        for r_n in mouse_67_labels_list:
            mouse_xyz_dict[r_n] = {'x_grid': 0, 'y_grid': 0, 'z_grid': 0}

        for region_name in mouse_67_labels_list:
            mean_voxel_x = np.mean(adata_mouse[adata_mouse.obs['region_name'] == region_name].obs['x_grid'].values)
            mouse_xyz_dict[region_name]['x_grid'] = mean_voxel_x
            mean_voxel_y = np.mean(adata_mouse[adata_mouse.obs['region_name'] == region_name].obs['y_grid'].values)
            mouse_xyz_dict[region_name]['y_grid'] = mean_voxel_y
            mean_voxel_z = np.mean(adata_mouse[adata_mouse.obs['region_name'] == region_name].obs['z_grid'].values)
            mouse_xyz_dict[region_name]['z_grid'] = mean_voxel_z

        xyz_df = pd.DataFrame.from_dict(mouse_xyz_dict).T

        ## Compute distance matrix
        dist_df = pd.DataFrame(squareform(pdist(xyz_df.values)), columns=xyz_df.index, index=xyz_df.index)

        # print(human_dist_df)
        # values_array = human_dist_df.values
        for i in range(dist_df.shape[0]):
            dist_df.iloc[i, i] = 1e7

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
        # print(human_mri_xyz_nearest)

        # pearson correlation matrix of embedding
        pearson_corr_df = mouse_embedding_df.corr()
        # print(pearson_corr_df)

        # Get nearst neighbor correlation and not neighbor correlation lists
        neighbor_pearson_list = []
        not_neighbor_pearson_list = []
        for region_n, r_dict in xyz_nearest.items():
            neighbor_pearson_list.append(pearson_corr_df.loc[region_n, r_dict['region']])
        # print(neighbor_pearson_list)

        for region_x in mouse_67_labels_list:
            for region_y in mouse_67_labels_list:
                if region_x != region_y and xyz_nearest[region_x]['region'] != region_y and xyz_nearest[region_y][
                    'region'] != region_x:
                    not_neighbor_pearson_list.append(pearson_corr_df.loc[region_x, region_y])

        # print(not_neighbor_pearson_list)
        print('neighbor_pearson_list', neighbor_pearson_list)

        # print('not_neighbor_pearson_list', not_neighbor_pearson_list)

        neighbor_pearson_list_type = ["1-nearest neighbor" for i in range(len(neighbor_pearson_list))]
        not_neighbor_pearson_list_type = ["Not near" for i in range(len(not_neighbor_pearson_list))]

        data_dict = {'Pearson Correlation': neighbor_pearson_list + not_neighbor_pearson_list,
                     'Spatial Relation': neighbor_pearson_list_type + not_neighbor_pearson_list_type}
        data_df = pd.DataFrame.from_dict(data_dict)

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/2_anatomical_STs_analysis/2_experiment_spatial_alignment/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print('len of neighbor_pearson_list', len(neighbor_pearson_list))
        neighbor_pearson_list = [x for x in neighbor_pearson_list if
                                 str(x) != 'nan']  # np.nan_to_num(neighbor_pearson_list, nan=0)
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
        my_pal = {"1-nearest neighbor": (64 / 255, 125 / 255, 82 / 255), "Not near": (142 / 255, 41 / 255, 97 / 255)}
        # sns.set_theme(style="whitegrid")
        # tips = sns.load_dataset("tips")
        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.bottom"] = 0.15
        plt.figure(figsize=(8, 8))
        ax = sns.boxplot(x="Spatial Relation", y="Pearson Correlation", data=data_df,
                         order=["1-nearest neighbor", "Not near"], palette=my_pal, width=0.65)
        add_stat_annotation(ax, data=data_df, x="Spatial Relation", y="Pearson Correlation", order=["1-nearest neighbor", "Not near"],
                            box_pairs=[("1-nearest neighbor", "Not near")],
                            test='t-test_ind', text_format='star', loc='inside', verbose=2)  # text_format='star'
        plt.savefig(save_path + 'mouse_xyz_pearson.' + fig_format, format=fig_format)
        plt.show()

        ## plot joint
        dist_pearson_dict = {'Euclidean Distance': [], 'Pearson Correlation': []}
        mouse_dist_list = []
        for i in range(len(dist_df.index)):
            for j in range(i + 1, len(dist_df.index)):
                mouse_dist_list.append(dist_df.iloc[i, j])

        pearson_corr_list = []
        for i in range(len(pearson_corr_df.index)):
            for j in range(i + 1, len(pearson_corr_df.index)):
                pearson_corr_list.append(pearson_corr_df.iloc[i, j])

        dist_pearson_dict['Euclidean Distance'] = mouse_dist_list
        dist_pearson_dict['Pearson Correlation'] = pearson_corr_list

        plt.figure(figsize=(8, 8))
        sns.jointplot(data=pd.DataFrame.from_dict(dist_pearson_dict), x="Euclidean Distance", y="Pearson Correlation",
                      kind="reg", scatter_kws={"s": 3}, height=8)
        plt.savefig(save_path + 'mouse_dist_corr_joint.' + fig_format, format=fig_format, dpi=500)
        plt.show()

    def experiment_2_spatial_hippocampal(self):

        cfg = self.cfg
        fig_format = self.fig_format

        save_path = self.save_path + '2_experiment_spatial_hippocampal/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        adata_mouse_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        adata_human_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')

        adata_mouse_embedding.obs['dataset'] = 'mouse'
        adata_human_embedding.obs['dataset'] = 'human'

        mouse_human_64_88_color_dict = dict(self.mouse_64_color_dict)
        mouse_human_64_88_color_dict.update(self.human_88_color_dict)

        adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
        print(adata_embedding)
        sc.pp.neighbors(adata_embedding, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine', use_rep='X')
        sc.tl.umap(adata_embedding)

        hippocampal_region_list = ['Field CA1', 'Field CA2', 'Field CA3', 'Dentate gyrus', 'Subiculum',
                                   'CA1 field', 'CA2 field', 'CA3 field', 'dentate gyrus', 'subiculum']
        adata_embedding_hippocampal = adata_embedding[
            adata_embedding.obs['region_name'].isin(hippocampal_region_list)]

        adata_embedding_hippocampal_mouse = adata_embedding_hippocampal[
            adata_embedding_hippocampal.obs['dataset'].isin(['mouse'])]
        adata_embedding_hippocampal_human = adata_embedding_hippocampal[
            adata_embedding_hippocampal.obs['dataset'].isin(['human'])]

        print('adata_embedding_hippocampal:', adata_embedding_hippocampal)
        palette = {k: mouse_human_64_88_color_dict[v] for k, v in
                   zip(hippocampal_region_list, hippocampal_region_list)}

        # umap independently
        #save_path = cfg.BrainAlign.embeddings_file_path + 'figs/3_gene_comparison/1_experiment_CALB1/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])
        print(adata_embedding)

        hippocampal_region_list = ['Field CA1', 'Field CA2', 'Field CA3', 'Dentate gyrus', 'Subiculum',
                                   'CA1 field', 'CA2 field', 'CA3 field', 'dentate gyrus', 'subiculum']

        adata_embedding_hippocampal = adata_embedding[
            adata_embedding.obs['region_name'].isin(hippocampal_region_list)]

        sc.pp.neighbors(adata_embedding_hippocampal, n_neighbors=cfg.ANALYSIS.umap_neighbor, metric='cosine',
                        use_rep='X')
        sc.tl.umap(adata_embedding_hippocampal)

        adata_embedding_hippocampal_mouse = adata_embedding_hippocampal[
            adata_embedding_hippocampal.obs['dataset'].isin(['mouse'])]
        adata_embedding_hippocampal_human = adata_embedding_hippocampal[
            adata_embedding_hippocampal.obs['dataset'].isin(['human'])]

        print('adata_embedding_hippocampal:', adata_embedding_hippocampal)
        palette = {k: mouse_human_64_88_color_dict[v] for k, v in
                   zip(hippocampal_region_list, hippocampal_region_list)}

        adata_embedding_hippocampal_mouse.obs['dataset_region_name'] = ['mouse' + ' ' + x for x in
                                                           adata_embedding_hippocampal_mouse.obs['region_name']]
        adata_embedding_hippocampal_human.obs['dataset_region_name'] = ['human' + ' ' + x for x in
                                                                        adata_embedding_hippocampal_human.obs[
                                                                            'region_name']]

        adata_embedding_hippocampal = ad.concat([adata_embedding_hippocampal_mouse, adata_embedding_hippocampal_human])

        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.45
        with plt.rc_context({"figure.figsize": (16, 8)}):
            sc.pl.umap(adata_embedding_hippocampal, color=['region_name'], return_fig=True,
                       legend_loc='right margin', title='').savefig(
                save_path + 'umap_regions.' + fig_format, format=fig_format)
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.9

        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.45
        with plt.rc_context({"figure.figsize": (16, 8)}):
            sc.pl.umap(adata_embedding_hippocampal, color=['dataset_region_name'], return_fig=True,
                       legend_loc='right margin', title='').savefig(
                save_path + 'umap_dataset_region_name.' + fig_format, format=fig_format)
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.9

        sns.set(style='white')
        TINY_SIZE = 18  # 39
        SMALL_SIZE = 20  # 42
        MEDIUM_SIZE = 32  # 46
        BIGGER_SIZE = 36  # 46

        plt.rc('font', size=18)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        rcParams["figure.subplot.left"] = 0.1
        rcParams["figure.subplot.right"] = 0.9
        with plt.rc_context({"figure.figsize": (8, 8)}):
            sc.pl.umap(adata_embedding_hippocampal, color=['dataset_region_name'], return_fig=True,
                       legend_loc='on data', title='').savefig(
                save_path + 'umap_regions_dataset.' + fig_format, format=fig_format)
        rcParams["figure.subplot.left"] = 0.05
        rcParams["figure.subplot.right"] = 0.9




        return None


    def experiment_3_spatial_umap_3d(self):

        return None


    def forward(self):
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

        self.experiment_1_spatial_hippocampal()
        self.experiment_2_spatial_isocortex()
        self.experiment_3_spatial_clusters()
        #self.experiment_1_1_paga_test()
        #self.experiment_2_spatial_hippocampal()