

import scanpy as sc, anndata as ad
import numpy as np
import pandas as pd


if __name__ == "__main__":

    
    path_rawdata_human = "./brain_human_mouse/human_brain_region_88_sparse.h5ad"
    adata = sc.read_h5ad(path_rawdata_human)
    
    sc.settings.verbosity = 3
    #sc.settings.set_figure_params(dpi=80)

    
    #sc.tl.leiden(adata, key_added = "leiden_1.0") # default resolution in 1.0
    #sc.tl.leiden(adata, resolution = 0.6, key_added = "leiden_0.6")
    #sc.tl.leiden(adata, resolution = 0.4, key_added = "leiden_0.4")
    #sc.tl.leiden(adata, resolution = 1.4, key_added = "leiden_1.4")
    
    #sc.pl.umap(adata, color=['leiden_0.4', 'leiden_0.6', 'leiden_1.0','leiden_1.4'])
    
    #sc.tl.dendrogram(adata, groupby = "leiden_0.6")
    #sc.pl.dendrogram(adata, groupby = "leiden_0.6")
    
    
    #sc.pp.log1p(adata)
    # store normalized counts in the raw slot, 
    # we will subset adata.X for variable genes, but want to keep all genes matrix as well.
    #adata.raw = adata
    #adata.X =  pd.df(adata.X).fillna(0)
    #sc.pp.highly_variable_genes(adata, min_mean=0.001, max_mean=3, min_disp=0.1)
    #print("Highly variable genes: %d"%sum(adata.var.highly_variable))

    #plot variable genes
    #sc.pl.highly_variable_genes(adata)

    # subset for variable genes in the dataset
    #adata = adata[:, adata.var['highly_variable']]
    
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pl.pca(adata, color='region_name', components = ['1,2','3,4','5,6','7,8'], ncols=2, return_fig=True).savefig('./figs/human_88/pca.png')
    
    

    
    
    
    path_rawdata_human = "./brain_human_mouse_16_11/human_brain_region_16_sparse.h5ad"
    adata = sc.read_h5ad(path_rawdata_human)
    
    sc.settings.verbosity = 3
    #sc.settings.set_figure_params(dpi=80)

    
    #sc.tl.leiden(adata, key_added = "leiden_1.0") # default resolution in 1.0
    #sc.tl.leiden(adata, resolution = 0.6, key_added = "leiden_0.6")
    #sc.tl.leiden(adata, resolution = 0.4, key_added = "leiden_0.4")
    #sc.tl.leiden(adata, resolution = 1.4, key_added = "leiden_1.4")
    
    #sc.pl.umap(adata, color=['leiden_0.4', 'leiden_0.6', 'leiden_1.0','leiden_1.4'])
    
    #sc.tl.dendrogram(adata, groupby = "leiden_0.6")
    #sc.pl.dendrogram(adata, groupby = "leiden_0.6")
    
    
    #sc.pp.log1p(adata)
    # store normalized counts in the raw slot, 
    # we will subset adata.X for variable genes, but want to keep all genes matrix as well.
    #adata.raw = adata
    #adata.X =  pd.df(adata.X).fillna(0)
    #sc.pp.highly_variable_genes(adata, min_mean=0.001, max_mean=3, min_disp=0.1)
    #print("Highly variable genes: %d"%sum(adata.var.highly_variable))

    #plot variable genes
    #sc.pl.highly_variable_genes(adata)

    # subset for variable genes in the dataset
    #adata = adata[:, adata.var['highly_variable']]

    sc.tl.pca(adata, svd_solver='arpack')
    sc.pl.pca(adata, color='region_name', components = ['1,2','3,4','5,6','7,8'], ncols=2, return_fig=True).savefig('./figs/human_16/pca.png')
    
