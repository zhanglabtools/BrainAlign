# -- coding: utf-8 --
# @Time : 2023/4/3 18:26
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : analysis_anatomical.py
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

class anatomical_STs_analysis():
    def __init__(self, cfg):
        # Gene expression comparative analysis
        self.cfg = cfg
        self.fig_format = cfg.BrainAlign.fig_format  # the figure save format
        self.mouse_color = '#ED7D31'
        self.human_color = '#4472C4'

        # read labels data including acronym, color and parent region name
        self.mouse_64_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_mouse_file)
        self.mouse_64_labels_list = list(self.mouse_64_labels['region_name'])
        self.mouse_64_acronym_dict = {k: v for k, v in
                                      zip(self.mouse_64_labels['region_name'], self.mouse_64_labels['acronym'])}
        self.mouse_64_color_dict = {k: v for k, v in
                                    zip(self.mouse_64_labels['region_name'], self.mouse_64_labels['color_hex_triplet'])}
        self.mouse_64_parent_region_dict = {k: v for k, v in
                                            zip(self.mouse_64_labels['region_name'],
                                                self.mouse_64_labels['parent_region_name'])}
        self.mouse_15_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.parent_labels_mouse_file)
        self.mouse_15_labels_list = list(self.mouse_15_labels['region_name'])
        self.mouse_15_acronym_dict = {k: v for k, v in
                                      zip(self.mouse_15_labels['region_name'], self.mouse_15_labels['acronym'])}
        self.mouse_15_color_dict = {k: v for k, v in
                                    zip(self.mouse_15_labels['region_name'], self.mouse_15_labels['color_hex_triplet'])}

        self.human_88_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.labels_human_file)
        self.human_88_labels_list = list(self.human_88_labels['region_name'])
        self.human_88_acronym_dict = {k: v for k, v in
                                      zip(self.human_88_labels['region_name'], self.human_88_labels['acronym'])}
        self.human_88_color_dict = {k: v for k, v in
                                    zip(self.human_88_labels['region_name'], self.human_88_labels['color_hex_triplet'])}
        self.human_88_parent_region_dict = {k: v for k, v in
                                            zip(self.human_88_labels['region_name'],
                                                self.human_88_labels['parent_region_name'])}

        self.human_16_labels = pd.read_csv(cfg.CAME.labels_dir + cfg.CAME.parent_labels_human_file)
        self.human_16_labels_list = list(self.human_16_labels ['region_name'])
        self.human_16_acronym_dict = {k: v for k, v in
                                      zip(self.human_16_labels['region_name'], self.human_16_labels['acronym'])}
        self.human_16_color_dict = {k: v for k, v in
                                    zip(self.human_16_labels['region_name'], self.human_16_labels['color_hex_triplet'])}

        self.mouse_human_64_88_color_dict = dict(self.mouse_64_color_dict)
        self.mouse_human_64_88_color_dict.update(self.human_88_color_dict)

        self.save_path = cfg.BrainAlign.embeddings_file_path + 'figs/2_anatomical_STs_analysis/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # load sample embeddings
        self.adata_mouse_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        self.adata_mouse_expression = sc.read_h5ad(cfg.CAME.path_rawdata1)
        for obs1, obs2 in zip(self.adata_mouse_embedding.obs_names, self.adata_mouse_expression.obs_names):
            if obs1 != obs2:
                print('Sample name not aligned!')

        self.adata_human_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
        self.adata_human_expression = sc.read_h5ad(cfg.CAME.path_rawdata2)
        for obs1, obs2 in zip(self.adata_human_embedding.obs_names, self.adata_human_expression.obs_names):
            if obs1 != obs2:
                print('Sample name not aligned!')

        # load gene embeddings
        self.adata_mouse_gene_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'm_embeddings.h5ad')
        self.adata_human_gene_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'h_embeddings.h5ad')

        self.mouse_gene_num = self.adata_mouse_gene_embedding.n_obs
        self.human_gene_num = self.adata_human_gene_embedding.n_obs

    def experiment_1_palette_comparison(self):
        cfg = self.cfg
        save_path = self.save_path + '1_palette_comparison/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        mouse_palette_df = pd.read_csv(cfg.BrainAlign.palette_mouse_file)
        mouse_palette_gene_list = mouse_palette_df['Gene']
        human_palette_df = pd.read_csv(cfg.BrainAlign.palette_human_file)
        human_palette_gene_unfiltered = human_palette_df['Gene']

        human_gene_set_appear = self.adata_human_expression.var_names.tolist()
        human_palette_gene_unfiltered = [g for g in human_palette_gene_unfiltered if g in human_gene_set_appear]

        # filtering human palette gene
        ntop_genes = 2
        groupby = 'region_name'
        sc.pp.filter_cells(self.adata_human_expression, min_counts=2)

        cluster_counts = self.adata_human_expression.obs[groupby].value_counts()
        keep = cluster_counts.index[cluster_counts >= 2]
        human_88_labels_list_keep = [r for r in self.human_88_labels_list if r in keep]
        self.adata_human_expression = self.adata_human_expression[
            self.adata_human_expression.obs[groupby].isin(keep)].copy()
        adata_human_expression_filtered = self.adata_human_expression[:, human_palette_gene_unfiltered]
        print(adata_human_expression_filtered)
        sc.tl.rank_genes_groups(adata_human_expression_filtered, groupby='region_name', n_genes=ntop_genes,
                                method='wilcoxon', key_added='wilcoxon_filtered')

        # n_genes=ntop_genes,
        # sc.tl.filter_rank_genes_groups(adata_human_expression_filtered, min_in_group_fraction=0,
        #                                max_out_group_fraction=1, key='rank_genes_groups', key_added='wilcoxon_filtered',
        #                                min_fold_change=1)
        sc.pl.rank_genes_groups(adata_human_expression_filtered, sharey=False, key='wilcoxon_filtered', show=False)

        print(adata_human_expression_filtered)

        glist_human_region_df = sc.get.rank_genes_groups_df(adata_human_expression_filtered,
                                                            group=human_88_labels_list_keep,
                                                            key='wilcoxon_filtered',
                                                            pval_cutoff=1e-9)

        glist_human_cluster = glist_human_region_df['names'].squeeze().str.strip().tolist()
        print(glist_human_cluster)
        glist_human_region_df.to_csv(save_path + f'gene_palette_human_{len(set(glist_human_cluster))}.csv')
        print('Number of human palette genes after filtering:', len(set(glist_human_cluster)))

        # with plt.rc_context({"figure.figsize": (12, 12)}):
        sc.pl.rank_genes_groups_tracksplot(adata_human_expression_filtered, key='wilcoxon_filtered', groupby=groupby,
                                           var_names=glist_human_cluster, figsize=(20, 20))  # , return_fig=True
        plt.savefig(save_path + 'human_filtered_gene_map_tracksplot.' + self.fig_format, format=self.fig_format)

        # 3D umap of palette genes across species plot line if gene are homologous

        # rank the correlations of genes, plot the distribution of correlatons

    def experiment_2_spatial_alignment_specie2(self):
        '''
               Step 1: Compute average embedding and position of every region in two species, use two dict to store;
               Step 2: Compute distance between regions, select one-nearest neighbor of each region, compute correlations;
               Step 3: plot Boxplot.

            '''
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

        plt.figure(figsize=(8, 6))
        ax = sns.boxplot(x="Spatial Relation", y="Pearson Correlation", data=data_df,
                         order=["1-nearest neighbor", "Not near"], palette=my_pal)
        plt.savefig(save_path + 'mri_xyz_pearson.' + fig_format, format=fig_format)
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
                      kind="reg", scatter_kws={"s": 3})
        plt.savefig(save_path + 'human_dist_corr_joint.' + fig_format, format=fig_format)
        plt.show()

    def experiment_2_spatial_alignment_specie1(self):
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

        save_path = cfg.BrainAlign.embeddings_file_path + 'figs/mouse_67/neighbor_random/'
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

        plt.figure(figsize=(8, 6))
        ax = sns.boxplot(x="Spatial Relation", y="Pearson Correlation", data=data_df,
                         order=["1-nearest neighbor", "Not near"], palette=my_pal)
        plt.savefig(save_path + 'xyz_pearson.' + fig_format, format=fig_format)
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
                      kind="reg", scatter_kws={"s": 3})
        plt.savefig(save_path + 'mouse_dist_corr_joint.' + fig_format, format=fig_format)
        plt.show()

    def experiment_3_alignment_sankey_plot(self):

        save_path = self.save_path + '3_experiment_alignment_sankey/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        adata_mouse_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
        adata_human_embedding = sc.read_h5ad(self.cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')

        adata_mouse_embedding.obs['dataset'] = 'Mouse'
        adata_human_embedding.obs['dataset'] = 'Human'

        adata_embedding = ad.concat([adata_mouse_embedding, adata_human_embedding])

        sc.pp.neighbors(adata_embedding, n_neighbors=self.cfg.ANALYSIS.umap_neighbor, metric='cosine',
                        use_rep='X')
        sc.tl.umap(adata_embedding, n_components=2)

        adata_mouse_embedding = adata_embedding[adata_embedding.obs['dataset'].isin(['Mouse'])]
        adata_human_embedding = adata_embedding[adata_embedding.obs['dataset'].isin(['Human'])]

        alignment_score_mat = np.zeros((len(self.human_88_labels_list), len(self.mouse_64_labels_list)))
        for i in range(len(self.human_88_labels_list)):
            for j in range(len(self.mouse_64_labels_list)):
                human_region = self.human_88_labels_list[i]
                mouse_region = self.mouse_64_labels_list[j]

                adata_mouse_embedding_region = adata_mouse_embedding[
                    adata_mouse_embedding.obs['region_name'].isin([mouse_region])]
                adata_human_embedding_region = adata_human_embedding[
                    adata_human_embedding.obs['region_name'].isin([human_region])]
                adata_embedding_region = ad.concat([adata_mouse_embedding_region, adata_human_embedding_region])

                # compute alignment score for aligned data
                #X = adata_embedding_region.obsm['X_pca']
                X = adata_embedding_region.obsm['X_umap']
                #X = adata_embedding_region.X
                Y = np.concatenate(
                    [np.zeros((adata_mouse_embedding_region.n_obs, 1)), np.ones((adata_human_embedding_region.n_obs, 1))],
                    axis=0)

                aligned_score = seurat_alignment_score(X, Y)
                if human_region == 'paraterminal gyrus':
                    alignment_score_mat[i, j] = 0
                else:
                    alignment_score_mat[i, j] = aligned_score

        quantile = 0.98
        threshold_score = np.quantile(alignment_score_mat, quantile)
        print('threshold score = :', threshold_score)
        node_label = self.human_88_labels_list + self.mouse_64_labels_list
        node_dict = {y: x for x, y in enumerate(node_label)}

        source = []
        target = []
        values = []

        for i in range(len(self.human_88_labels_list)):
            for j in range(len(self.mouse_64_labels_list)):
                if alignment_score_mat[i, j] >= threshold_score:
                    mouse_region = self.mouse_64_labels_list[j]
                    human_region = self.human_88_labels_list[i]
                    source.append(mouse_region)
                    target.append(human_region)
                    values.append(alignment_score_mat[i, j])

        source_node = [node_dict[x] for x in source]
        target_node = [node_dict[x] for x in target]

        node_color = self.human_88_labels['color_hex_triplet'].tolist() + self.mouse_64_labels['color_hex_triplet'].tolist()
        print('len(node_color)', len(node_color))
        print(node_dict)

        node_label_color = {x: y for x, y in zip(node_label, node_color)}
        link_color = [node_label_color[x] for x in target]

        link_color = ['rgba({},{},{}, 0.8)'.format(
            hex_to_rgb(x)[0],
            hex_to_rgb(x)[1],
            hex_to_rgb(x)[2]) for x in link_color]

        fig = go.Figure(
            data=[go.Sankey(
                node=dict(
                    #pad=15,
                    #thickness=15,
                    line=dict(color="black", width=0.5),
                    label=node_label,
                    color=node_color
                ),
                link=dict(
                    source=source_node,
                    target=target_node,
                    value=values,
                    color=link_color,
                ))])
        plot(fig,
             image_filename='sankey_plot.'+self.fig_format,
             image=self.fig_format,
             image_width=1000,
             image_height=1300,
             auto_open=False
             )
        fig.update_layout(
            font_family="Arial",
            font_size=22
        )
        fig.write_image(save_path + "sankey_subregion."+self.fig_format, width=1000, height=1300, scale=5)

        # parent region
        alignment_score_mat = np.zeros((len(self.human_16_labels_list), len(self.mouse_15_labels_list)))
        for i in range(len(self.human_16_labels_list)):
            for j in range(len(self.mouse_15_labels_list)):
                mouse_region = self.mouse_15_labels_list[j]
                human_region = self.human_16_labels_list[i]

                adata_mouse_embedding_region = adata_mouse_embedding[
                    adata_mouse_embedding.obs['parent_region_name'].isin([mouse_region])]
                adata_human_embedding_region = adata_human_embedding[
                    adata_human_embedding.obs['parent_region_name'].isin([human_region])]
                adata_embedding_region = ad.concat([adata_mouse_embedding_region, adata_human_embedding_region])

                # compute alignment score for aligned data
                #X = adata_embedding_region.obsm['X_pca']
                X = adata_embedding_region.obsm['X_umap']
                #X = adata_embedding_region.X
                Y = np.concatenate(
                    [np.zeros((adata_mouse_embedding_region.n_obs, 1)),
                     np.ones((adata_human_embedding_region.n_obs, 1))],
                    axis=0)
                aligned_score = seurat_alignment_score(X, Y)
                alignment_score_mat[i, j] = aligned_score

        quantile = 0.95#0.85
        threshold_score = np.quantile(alignment_score_mat, quantile)
        print('threshold score = :', threshold_score)
        node_label =  self.human_16_labels_list + self.mouse_15_labels_list
        node_dict = {y: x for x, y in enumerate(node_label)}

        source = []
        target = []
        values = []

        for i in range(len(self.human_16_labels_list)):
            for j in range(len(self.mouse_15_labels_list)):
                if alignment_score_mat[i, j] >= threshold_score:
                    mouse_region = self.mouse_15_labels_list[j]
                    human_region = self.human_16_labels_list[i]
                    source.append(mouse_region)
                    target.append(human_region)
                    values.append(alignment_score_mat[i, j])

        source_node = [node_dict[x] for x in source]
        target_node = [node_dict[x] for x in target]

        node_color = self.human_16_labels['color_hex_triplet'].tolist() + self.mouse_15_labels['color_hex_triplet'].tolist()
        print('len(node_color)', len(node_color))
        print(node_dict)

        node_label_color = {x: y for x, y in zip(node_label, node_color)}
        link_color = [node_label_color[x] for x in target]

        link_color = ['rgba({},{},{}, 0.5)'.format(
            hex_to_rgb(x)[0],
            hex_to_rgb(x)[1],
            hex_to_rgb(x)[2]) for x in link_color]

        fig = go.Figure(
            data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=15,
                    line=dict(color="black", width=0.5),
                    label=node_label,
                    color=node_color
                ),
                link=dict(
                    source=source_node,
                    target=target_node,
                    value=values,
                    color=link_color,
                ))])
        plot(fig,
             image_filename='sankey_plot.'+self.fig_format,
             image=self.fig_format,
             image_width=600,
             image_height=800,
             auto_open=False
             )
        fig.update_layout(
            font_family="Arial",
            font_size=28
        )
        fig.write_image(save_path + f"sankey_parent_region_{quantile}."+self.fig_format, width=600, height=800, scale=5)

        return None



    def experiment_4_alignment_neocortex(self):

        # Neocortex is comprised of

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

        #self.experiment_1_palette_comparison()
        #self.experiment_2_spatial_alignment_specie1()
        #self.experiment_2_spatial_alignment_specie2()
        self.experiment_3_alignment_sankey_plot()