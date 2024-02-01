# -- coding: utf-8 --
# @Time : 2023/3/12 21:34
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : analysis_utils.py
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
from scGeneFit.functions import *

# color map
import colorcet as cc

from collections import Counter

# get gene ontology
import gseapy

try:
    import matplotlib as mpl
    mpl.use('agg')
except Exception as e:
    print(f"An error occurred when setting matplotlib backend ({e})")


def get_common_special_gene_list(gene_list_mouse, gene_list_human, gene_name_mouse_list, gene_name_human_list, mh_matrix, corr_matrix):
    # pick homologous genes from two gene lists
    gene_list_mouse_homo = []
    gene_list_human_homo = []
    gene_index_mouse_homo = []
    gene_index_human_homo = []
    #print(type())
    gene_index_mouse_list = [gene_name_mouse_list.index(g) for g in gene_list_mouse]
    gene_index_human_list = [gene_name_human_list.index(g) for g in gene_list_human]
    for gene_mouse_index in gene_index_mouse_list:
        for gene_human_index in gene_index_human_list:
            if mh_matrix[gene_mouse_index, gene_human_index]:
                gene_index_mouse_homo.append(gene_mouse_index)
                gene_index_human_homo.append(gene_human_index)
                gene_list_mouse_homo.append(gene_name_mouse_list[gene_mouse_index])
                gene_list_human_homo.append(gene_name_human_list[gene_human_index])
    gene_list_mouse_homo = list(set(gene_list_mouse_homo))
    gene_list_human_homo = list(set(gene_list_human_homo))

    # pick those genes has correlation higher than homologous genes top 95% correlations
    corr_homo_list = []
    for gene_index_mouse, gene_index_human in zip(gene_index_mouse_homo, gene_index_human_homo):
        corr_homo_list.append(corr_matrix[gene_index_mouse, gene_index_human])
    if len(corr_homo_list) == 0:
        corr_threshold = 1
    else:
        corr_threshold = np.quantile(corr_homo_list, 0.8) # quantile of threshold
    high_corr_mouse_gene = []
    high_corr_human_gene = []
    for gene_index_mouse, gene_index_human in zip(gene_index_mouse_list, gene_index_human_list):
        if corr_matrix[gene_index_mouse, gene_index_human] >= corr_threshold:
            high_corr_mouse_gene.append(gene_index_mouse)
            high_corr_human_gene.append(gene_index_human)

    gene_list_mouse_common = [gene_name_mouse_list[g_i] for g_i in list(set(gene_index_mouse_homo+high_corr_mouse_gene))]
    gene_list_human_common = [gene_name_human_list[g_i] for g_i in list(set(gene_index_human_homo+high_corr_human_gene))]

    gene_list_mouse_special = [item for item in gene_list_mouse if item not in gene_list_mouse_common]
    gene_list_human_special = [item for item in gene_list_human if item not in gene_list_human_common]

    return gene_list_mouse_special, gene_list_mouse_common, gene_list_human_common, gene_list_human_special



def get_common_special_gene_list_homologous(gene_list_mouse, gene_list_human, gene_name_mouse_list, gene_name_human_list, mh_matrix):
    # pick homologous genes from two gene lists
    gene_list_mouse_homo = []
    gene_list_human_homo = []
    gene_index_mouse_homo = []
    gene_index_human_homo = []
    #print(type())
    gene_index_mouse_list = [gene_name_mouse_list.index(g) for g in gene_list_mouse]
    gene_index_human_list = [gene_name_human_list.index(g) for g in gene_list_human]
    for gene_mouse_index in gene_index_mouse_list:
        for gene_human_index in gene_index_human_list:
            if mh_matrix[gene_mouse_index, gene_human_index]:
                gene_index_mouse_homo.append(gene_mouse_index)
                gene_index_human_homo.append(gene_human_index)
                gene_list_mouse_homo.append(gene_name_mouse_list[gene_mouse_index])
                gene_list_human_homo.append(gene_name_human_list[gene_human_index])
    gene_list_mouse_homo = list(set(gene_list_mouse_homo))
    gene_list_human_homo = list(set(gene_list_human_homo))

    # pick those genes has correlation higher than homologous genes top 95% correlations
    # corr_homo_list = []
    # for gene_index_mouse, gene_index_human in zip(gene_index_mouse_homo, gene_index_human_homo):
    #     corr_homo_list.append(corr_matrix[gene_index_mouse, gene_index_human])
    # if len(corr_homo_list) == 0:
    #     corr_threshold = 1
    # else:
    #     corr_threshold = np.quantile(corr_homo_list, 0.8) # quantile of threshold
    # high_corr_mouse_gene = []
    # high_corr_human_gene = []
    # for gene_index_mouse, gene_index_human in zip(gene_index_mouse_list, gene_index_human_list):
    #     if corr_matrix[gene_index_mouse, gene_index_human] >= corr_threshold:
    #         high_corr_mouse_gene.append(gene_index_mouse)
    #         high_corr_human_gene.append(gene_index_human)

    gene_list_mouse_common = [gene_name_mouse_list[g_i] for g_i in list(set(gene_index_mouse_homo))]
    gene_list_human_common = [gene_name_human_list[g_i] for g_i in list(set(gene_index_human_homo))]

    gene_list_mouse_special = [item for item in gene_list_mouse if item not in gene_list_mouse_common]
    gene_list_human_special = [item for item in gene_list_human if item not in gene_list_human_common]

    #gene_list_common = gene_list_mouse_common + gene_list_human_common

    return gene_list_mouse_special, gene_list_mouse_common,gene_list_human_common, gene_list_human_special


def get_homologous_mat(gene_list_mouse, gene_list_human, homologous_pair_mouse, homologous_pair_human):
    M = []
    H = []
    V = []
    gene_set_mouse = set(gene_list_mouse)
    gene_set_human = set(gene_list_human)
    for m_gene, h_gene in zip(homologous_pair_mouse, homologous_pair_human):
        if m_gene in gene_set_mouse and h_gene in gene_set_human:
            m_index = gene_list_mouse.index(m_gene)
            h_index = gene_list_human.index(h_gene)
            M.append(m_index)
            H.append(h_index)
            V.append(1)
    MH_mat = sp.coo_matrix((np.array(V), (np.array(M), np.array(H))), shape=(len(gene_list_mouse), len(gene_list_human))).tocsr()
    return MH_mat


def get_homologous_gene_list(gene_list_mouse, gene_list_human, gene_name_mouse_list, gene_name_human_list, MH_mat):
    gene_list_mouse_homo = []
    gene_list_human_homo = []
    gene_index_mouse_homo = []
    gene_index_human_homo = []
    # print(type())
    gene_index_mouse_list = [gene_name_mouse_list.index(g) for g in gene_list_mouse]
    gene_index_human_list = [gene_name_human_list.index(g) for g in gene_list_human]
    for gene_mouse_index in gene_index_mouse_list:
        for gene_human_index in gene_index_human_list:
            if MH_mat[gene_mouse_index, gene_human_index]:
                gene_index_mouse_homo.append(gene_mouse_index)
                gene_index_human_homo.append(gene_human_index)
                gene_list_mouse_homo.append(gene_name_mouse_list[gene_mouse_index])
                gene_list_human_homo.append(gene_name_human_list[gene_human_index])
    gene_list_mouse_homo = list(set(gene_list_mouse_homo))
    gene_list_human_homo = list(set(gene_list_human_homo))
    return gene_list_mouse_homo, gene_list_human_homo


def average_expression(adata: ad.AnnData, genes: list) -> list:
    gene_exp_list = []
    for g in genes:
        if g in set(adata.var_names):
            g_exp = adata[:, g].X.toarray().mean().mean()
            gene_exp_list.append(g_exp)
        else:
            print(f'Gene {g} is not in the gene list!')
            break

    return gene_exp_list


def get_expression_group(adata, groupby, gene_name):

    return None

def gene_module_abstract_graph(cfg, adata_gene_embedding, save_path, fig_format, key_class, resolution=0.5, umap_min_dist=0.85, umap_n_neighbors = 30):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sc.pp.neighbors(adata_gene_embedding, n_neighbors=umap_n_neighbors, metric='cosine',
                        use_rep='X')
    sc.tl.umap(adata_gene_embedding, min_dist=umap_min_dist)

    # plot gene modules
    with plt.rc_context({"figure.figsize": (8, 8)}):
        sc.tl.leiden(adata_gene_embedding, resolution=resolution, key_added='module')
        sc.pl.umap(adata_gene_embedding, color='module', ncols=1, palette='tab20b', return_fig=True).savefig(
            save_path + 'Genes_module_concat.' + fig_format, format=fig_format)

    # adata_gene_embedding.obs_names = adata_gene_embedding.obs_names.astype(str)
    gadt1, gadt2 = pp.bisplit_adata(adata_gene_embedding, 'dataset',
                                    cfg.BrainAlign.dsnames[0])  # weight_linked_vars

    color_by = 'module'
    palette = 'tab20b'
    sc.pl.umap(gadt1, color=color_by, s=10,  # edges=True, edges_width=0.05,
               palette=palette,
               save=f'_{color_by}-{cfg.BrainAlign.dsnames[0]}', return_fig=True).savefig(
        save_path + '{}_genes_module_concat.'.format(cfg.BrainAlign.dsnames[0]) + fig_format, format=fig_format)
    sc.pl.umap(gadt2, color=color_by, s=10,  # edges=True, edges_width=0.05,
               palette=palette,
               save=f'_{color_by}-{cfg.BrainAlign.dsnames[1]}', return_fig=True).savefig(
        save_path + '{}_genes_module_concat.'.format(cfg.BrainAlign.dsnames[1]) + fig_format, format=fig_format)

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
    key_class1 = key_class  # set by user
    key_class2 = key_class  # set by user
    # averaged expressions

    # adata_mouse_sample_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 's_embeddings.h5ad')
    # adata_human_sample_embedding = sc.read_h5ad(cfg.BrainAlign.embeddings_file_path + 'v_embeddings.h5ad')
    # adatas = [adata_mouse_sample_embedding, adata_human_sample_embedding]
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
    obs_labels1, obs_labels2 = adata_mouse_sample_embedding.obs[key_class], \
                               adata_human_sample_embedding.obs[key_class]  # adt.obs['celltype'][dpair.obs_ids1], adt.obs['celltype'][dpair.obs_ids2]
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
        figsize=(8, 20), alpha=0.5, fp=fp_abs, colors = ('pink', 'lightblue', 'lightblue', 'pink'))  # nodelist=nodelist,

    # ax.figure
    came.save_pickle(g, save_path + 'abs_graph.pickle')
    return None


from adjustText import adjust_text

def gen_mpl_labels(adata, groupby, exclude=(), ax=None, adjust_kwargs=None, text_kwargs=None):
    if adjust_kwargs is None:
        adjust_kwargs = {"text_from_points": False}
    if text_kwargs is None:
        text_kwargs = {}

    medians = {}

    for g, g_idx in adata.obs.groupby(groupby).groups.items():
        if g in exclude:
            continue
        medians[g] = np.median(adata[g_idx].obsm["X_umap"], axis=0)

    if ax is None:
        texts = [
            plt.text(x=x, y=y, s=k, **text_kwargs) for k, (x, y) in medians.items()
        ]
    else:
        texts = [ax.text(x=x, y=y, s=k, **text_kwargs) for k, (x, y) in medians.items()]

    adjust_text(texts, **adjust_kwargs)


if __name__ == '__main__':
    print('Analysis utils.')
