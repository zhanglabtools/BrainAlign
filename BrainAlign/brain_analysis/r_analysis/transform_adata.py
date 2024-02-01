# -- coding: utf-8 --
# @Time : 2023/7/23 17:57
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : transform_adata.py
# @Description: This file is used to ...

import anndata
import scanpy as sc
from pathlib import Path
from scipy import io
import os

if __name__ == '__main__':

    output_for_R_path = "R_isocortex_mouse"

    adata_mouse_path_isocortex = "../../data/srrsc_mouse_human_binary/results_20_1000genes_all_came_selfloop/2023-06-23_20-31-14/embeds/figs/4_spatial_analysis/2_experiment_spatial_isocortex/adata_mouse_exp_isocortex.h5ad"
    adata = sc.read_h5ad(adata_mouse_path_isocortex)
    ### Set the directory for saving files
    save_dir = '../../data/srrsc_mouse_human_binary/results_20_1000genes_all_came_selfloop/2023-06-23_20-31-14/embeds/figs/4_spatial_analysis/2_experiment_spatial_isocortex/'

    Path(save_dir + output_for_R_path).mkdir(parents=True, exist_ok=True)
    print(save_dir + output_for_R_path)

    io.mmwrite(save_dir + output_for_R_path+'/counts.mtx', adata.X)
    cell_meta = adata.obs.copy()
    cell_meta['Barcode'] = cell_meta.index
    #cell_meta['UMAP1'] = adata.obsm['X_umap'][:, 0]
    #cell_meta['UMAP2'] = adata.obsm['X_umap'][:, 1]
    cell_meta['region_name'] = adata.obs['region_name']
    cell_meta['cluster_name_acronym'] = adata.obs['cluster_name_acronym']

    gene_meta = adata.var.copy()
    gene_meta['GeneName'] = gene_meta.index

    if not os.path.exists(save_dir + output_for_R_path):
        os.makedirs(save_dir + output_for_R_path)
    cell_meta.to_csv(save_dir + output_for_R_path + '/counts_cellMeta.csv', index=None)
    gene_meta.to_csv(save_dir + output_for_R_path + '/counts_geneMeta.csv', index=None)


    #------------------------------------------------------------------------------
    # human------------------------------------
    output_for_R_path = "R_isocortex_human"

    adata_human_path_isocortex = "../../data/srrsc_mouse_human_binary/results_20_1000genes_all_came_selfloop/2023-06-23_20-31-14/embeds/figs/4_spatial_analysis/2_experiment_spatial_isocortex/adata_human_exp_isocortex.h5ad"
    adata = sc.read_h5ad(adata_human_path_isocortex)
    ### Set the directory for saving files
    save_dir = '../../data/srrsc_mouse_human_binary/results_20_1000genes_all_came_selfloop/2023-06-23_20-31-14/embeds/figs/4_spatial_analysis/2_experiment_spatial_isocortex/'

    Path(save_dir + output_for_R_path).mkdir(parents=True, exist_ok=True)
    print(save_dir + output_for_R_path)

    io.mmwrite(save_dir + output_for_R_path + '/counts.mtx', adata.X)
    cell_meta = adata.obs.copy()
    cell_meta['Barcode'] = cell_meta.index
    # cell_meta['UMAP1'] = adata.obsm['X_umap'][:, 0]
    # cell_meta['UMAP2'] = adata.obsm['X_umap'][:, 1]
    cell_meta['region_name'] = adata.obs['region_name']
    cell_meta['cluster_name_acronym'] = adata.obs['cluster_name_acronym']

    gene_meta = adata.var.copy()
    gene_meta['GeneName'] = gene_meta.index

    if not os.path.exists(save_dir + output_for_R_path):
        os.makedirs(save_dir + output_for_R_path)
    cell_meta.to_csv(save_dir + output_for_R_path + '/counts_cellMeta.csv', index=None)
    gene_meta.to_csv(save_dir + output_for_R_path + '/counts_geneMeta.csv', index=None)