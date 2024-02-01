# -*- coding: utf-8 -*-
"""
Created on Wed May  5 20:08:52 2021

@author: Xingyan Liu
"""
import came
from came import pipeline, pp, pl

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt  # optional
import seaborn as sns  # optional

import scanpy as sc
from scipy import sparse

import networkx as nx
import torch


try:
    import matplotlib as mpl
    mpl.use('agg')
except Exception as e:
    print(f"An error occurred when setting matplotlib backend ({e})")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s-%(lineno)d-%(funcName)s(): '
               '%(levelname)s\n %(message)s')

    #came.__test1__(10, batch_size=None, reverse=False)
    #came.__test2__(10, batch_size=2048)
    dsnames = ('Baron_human', 'Baron_mouse')  # the dataset names, set by user
    dsn1, dsn2 = dsnames

    path_rawdata1 = './brain_human_mouse/human_brain_region_88_imputed.h5ad'
    path_rawdata2 = './brain_human_mouse/mouse_brain_region_67.h5ad'
    
    path_varmap = './came/sample_data/gene_matches_human2mouse.csv'
    path_varmap_1v1 = './came/sample_data/gene_matches_1v1_human2mouse.csv'
    
    # ========= load data =========
    df_varmap = pd.read_csv(path_varmap)
    df_varmap_1v1 = pd.read_csv(path_varmap_1v1) if path_varmap_1v1 else came.pp.take_1v1_matches(df_varmap)

    adata_raw1 = sc.read_h5ad(path_rawdata1)
    adata_raw2 = sc.read_h5ad(path_rawdata2)
    adatas = [adata_raw1, adata_raw2]
    
    key_class1 = 'region_name'  # set by user
    key_class2 = 'region_name'  # set by user
    
    ROOT = '/home1/zhangbiao/CAME/brain_human_mouse/'
    time_tag = came.make_nowtime_tag()
    resdir = ROOT +  f'{dsnames}-{time_tag}'  # set by user
    figdir = resdir + '/figs'
    came.check_dirs(figdir)  # check and make the directory
    
    sc.pp.filter_genes(adata_raw1, min_cells=3)
    sc.pp.filter_genes(adata_raw2, min_cells=3)
    
    # Inspect classes
    if key_class2 is not None:
        group_counts_ori = pd.concat([
            pd.value_counts(adata_raw1.obs[key_class1]),
            pd.value_counts(adata_raw2.obs[key_class2]),
        ], axis=1, keys=dsnames)
    else:
        group_counts_ori = pd.value_counts(adata_raw1.obs[key_class1])
        
    print(group_counts_ori)
        
        
    # The numer of training epochs
    # (a recommended setting is 200-400 for whole-graph training, and 80-200 for sub-graph training)
    n_epochs = 300

    # The training batch size
    # When the GPU memory is limited, set 4096 or more if possible.
    batch_size = 4096

    # The number of epochs to skip for checkpoint backup
    n_pass = 100

    # Whether to use the single-cell networks
    use_scnets = True

    # The number of top DEGs to take as the node-features of each cells.
    # You set it 70-100 for distant species pairs.
    ntop_deg = 50

    # The number of top DEGs to take as the graph nodes, which can be directly displayed on the UMAP plot.
    ntop_deg_nodes = 50
    # The source of the node genes; use both DEGs and HVGs by default
    node_source = 'deg,hvg'

    # Whether to take into account the non-1v1 variables as the node features.
    keep_non1v1_feats = True
    
    came_inputs, (adata1, adata2) = pipeline.preprocess_unaligned(
        adatas,
        key_class=key_class1,
        use_scnets=use_scnets,
        ntop_deg=ntop_deg,
        ntop_deg_nodes=ntop_deg_nodes,
        node_source=node_source,
        )

    outputs = pipeline.main_for_unaligned(
        **came_inputs,
        df_varmap=df_varmap,
        df_varmap_1v1=df_varmap_1v1,
        dataset_names=dsnames,
        key_class1=key_class1,
        key_class2=key_class2,
        do_normalize=True,
        keep_non1v1_feats=keep_non1v1_feats,
        n_epochs=n_epochs,
        resdir=resdir,
        n_pass=n_pass,
        batch_size=batch_size,
        plot_results=True,
        )
        
    dpair = outputs['dpair']
    trainer = outputs['trainer']
    h_dict = outputs['h_dict']
    out_cell = outputs['out_cell']
    predictor = outputs['predictor']


    obs_ids1, obs_ids2 = dpair.obs_ids1, dpair.obs_ids2
    obs = dpair.obs
    classes = predictor.classes
    
    # plot figures
    y_true = obs['celltype'][obs_ids2].values
    y_pred = obs['predicted'][obs_ids2].values

    ax, contmat = pl.plot_contingency_mat(y_true, y_pred, norm_axis=1, order_rows=False, order_cols=False,)
    pl._save_with_adjust(ax.figure, figdir / 'contingency_mat.png')
    ax.figure
    
    name_label = 'celltype'
    cols_anno = ['celltype', 'predicted'][:]
    df_probs = obs[list(classes)]


    gs = pl.wrapper_heatmap_scores(df_probs.iloc[obs_ids2], obs.iloc[obs_ids2], ignore_index=True,
    col_label='celltype', col_pred='predicted',
    n_subsample=50,  # sampling 50 cells for each group
    cmap_heat='magma_r',
    fp=figdir / f'heatmap_probas.pdf')

    gs.figure
    
    
    
    # further analysis
    hidden_list = came.load_hidden_states(resdir / 'hidden_list.h5')
    #hidden_list  # a list of dicts
    h_dict = hidden_list[-1]
    # the last layer of hidden states
    
    adt = pp.make_adata(h_dict['cell'], obs=dpair.obs, assparse=False, ignore_index=True)
    gadt = pp.make_adata(h_dict['gene'], obs=dpair.var.iloc[:, :2], assparse=False, ignore_index=True)
    
    # UMAP of cell embeddings
    sc.set_figure_params(dpi_save=200)

    sc.pp.neighbors(adt, n_neighbors=15, metric='cosine', use_rep='X')
    sc.tl.umap(adt)
    # sc.pl.umap(adt, color=['dataset', 'celltype'], ncols=1)

    ftype = ['.svg', ''][1]
    sc.pl.umap(adt, color='dataset', save=f'-dataset{ftype}')
    sc.pl.umap(adt, color='celltype', save=f'-ctype{ftype}')
    
    obs_umap = adt.obsm['X_umap']
    obs['UMAP1'] = obs_umap[:, 0]
    obs['UMAP2'] = obs_umap[:, 1]
    obs.to_csv(resdir / 'obs.csv')
    adt.write(resdir / 'adt_hidden_cell.h5ad')
    
    adata1.obsm['X_umap'] = obs_umap[obs_ids1]
    adata2.obsm['X_umap'] = obs_umap[obs_ids2]
    
    # Umap of genes
    sc.pp.neighbors(gadt, n_neighbors=15, metric='cosine', use_rep='X')

    # gadt = pp.make_adata(h_dict['gene'], obs=dpair.var.iloc[:, :2], assparse=False, ignore_index=True)
    sc.tl.umap(gadt)
    sc.pl.umap(gadt, color='dataset', )
    
    # joint gene module extraction
    sc.tl.leiden(gadt, resolution=.8, key_added='module')
    sc.pl.umap(gadt, color='module', ncols=1, palette='tab20b')
    
    # gadt.obs_names = gadt.obs_names.astype(str)
    gadt1, gadt2 = pp.bisplit_adata(gadt, 'dataset', dsnames[0], reset_index_by='name')

    color_by = 'module'
    palette = 'tab20b'
    sc.pl.umap(gadt1, color=color_by, s=10, edges=True, edges_width=0.05,
           palette=palette,
           save=f'_{color_by}-{dsnames[0]}')
    sc.pl.umap(gadt2, color=color_by, s=10, edges=True, edges_width=0.05,
           palette=palette,
           save=f'_{color_by}-{dsnames[0]}')
           
    f_var_links = came.weight_linked_vars(
    gadt.X, dpair._vv_adj, names=dpair.get_vnode_names(),
    matric='cosine', index_names=dsnames,)

    gadt1.write(resdir / 'adt_hidden_gene1.h5ad')
    gadt2.write(resdir / 'adt_hidden_gene2.h5ad')
    
    # Gene-expression-profiles (for each cell type) on gene UMAP
    # averaged expressions
    avg_expr1 = pp.group_mean_adata(adatas[0], groupby=key_class1,features=dpair.vnode_names1, use_raw=True)
    avg_expr2 = pp.group_mean_adata(adatas[1], groupby=key_class2 if key_class2 else 'predicted',features=dpair.vnode_names2, use_raw=True)
    
    plkwds = dict(cmap='RdYlBu_r', vmax=2.5, vmin=-1.5, do_zscore=True,
              axscale=3, ncols=5, with_cbar=True)

    fig1, axs1 = pl.adata_embed_with_values(gadt1, avg_expr1, fp=figdir / f'umap_exprAvgs-{dsnames[0]}-all.png', **plkwds)
    fig2, axs2 = pl.adata_embed_with_values(gadt2, avg_expr2, fp=figdir / f'umap_exprAvgs-{dsnames[1]}-all.png', **plkwds)
    
    # Abstracted graph
    norm_ov = ['max', 'zs', None][1]
    cut_ov = 0.

    groupby_var = 'module'
    obs_labels1, obs_labels2 = adt.obs['celltype'][dpair.obs_ids1], \
                           adt.obs['celltype'][dpair.obs_ids2]
    var_labels1, var_labels2 = gadt1.obs[groupby_var], gadt2.obs[groupby_var]

    sp1, sp2 = 'human', 'mouse'
    g = came.make_abstracted_graph(
        obs_labels1, obs_labels2,
        var_labels1, var_labels2,
        avg_expr1, avg_expr2,
        df_var_links,
        tags_obs=(f'{sp1} ', f'{sp2} '),
        tags_var=(f'{sp1} module ', f'{sp2} module '),
        cut_ov=cut_ov,
        norm_mtd_ov=norm_ov,)
        
    #visualization
    fp_abs = figdir / f'abstracted_graph-{groupby_var}-cut{cut_ov}-{norm_ov}.pdf'
    ax = pl.plot_multipartite_graph(g, edge_scale=10, figsize=(9, 7.5), alpha=0.5, fp=fp_abs)  # nodelist=nodelist,

    ax.figure
    
    came.save_pickle(g, resdir / 'abs_graph.pickle')
    
    
    
    

