

import scanpy as sc, anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

import re, seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap



if __name__ == "__main__":

    
    path_rawdata_mouse = "./brain_human_mouse/human_brain_region_88_sparse_with3d.h5ad"
    adata = sc.read_h5ad(path_rawdata_mouse)
    
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
    
    '''
    sc.tl.leiden(adata, key_added = "leiden_1.0") # default resolution in 1.0
    sc.tl.leiden(adata, resolution = 0.6, key_added = "leiden_0.6")
    sc.tl.leiden(adata, resolution = 0.4, key_added = "leiden_0.4")
    sc.tl.leiden(adata, resolution = 1.4, key_added = "leiden_1.4")
    
    sc.pl.umap(adata, color=['leiden_0.4', 'leiden_0.6', 'leiden_1.0','leiden_1.4'], return_fig=True).savefig('./figs/mouse_67/leiden.png')
    sc.tl.dendrogram(adata, groupby = "leiden_0.6")
    sc.pl.dendrogram(adata, groupby = "leiden_0.6", return_fig=True).savefig('./figs/mouse_67/leiden_groupby.png')
    
    
    sc.tl.louvain(adata, key_added = "louvain_1.0") # default resolution in 1.0
    sc.tl.louvain(adata, resolution = 0.6, key_added = "louvain_0.6")
    sc.tl.louvain(adata, resolution = 0.4, key_added = "louvain_0.4")
    sc.tl.louvain(adata, resolution = 1.4, key_added = "louvain_1.4")

    sc.pl.umap(adata, color=['louvain_0.4', 'louvain_0.6', 'louvain_1.0','louvain_1.4'], return_fig=True).savefig('./figs/mouse_67/louvain.png')
    
    sc.tl.dendrogram(adata, groupby = "louvain_0.6")
    sc.pl.dendrogram(adata, groupby = "louvain_0.6", return_fig=True).savefig('./figs/mouse_67/louvain_groupby.png')
    '''
    
    	
    
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pl.pca(adata, color='region_name', components = ['1,2','3,4','5,6','7,8'], ncols=2, return_fig=True).savefig('./figs/human_88/pca.png')
    
    
    
    sc.pp.neighbors(adata, n_pcs = 30, n_neighbors = 20)
    sc.tl.umap(adata)
    
    sc.pl.umap(adata, color=['region_name'], return_fig=True).savefig('./figs/human_88/umap.png')
    
    print(adata)
   

    # extract pca coordinates
    X_pca = adata.obsm['X_pca'] 

    # kmeans with k=5
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X_pca) 
    adata.obs['kmeans5'] = kmeans.labels_.astype(str)

    # kmeans with k=11
    kmeans = KMeans(n_clusters=16, random_state=0).fit(X_pca) 
    adata.obs['kmeans16'] = kmeans.labels_.astype(str)

    # kmeans with k=20
    kmeans = KMeans(n_clusters=30, random_state=0).fit(X_pca) 
    adata.obs['kmeans30'] = kmeans.labels_.astype(str)
    
     # kmeans with k=67
    kmeans = KMeans(n_clusters=88, random_state=0).fit(X_pca) 
    adata.obs['kmeans88'] = kmeans.labels_.astype(str)

    sc.pl.umap(adata, color=['kmeans5', 'kmeans16', 'kmeans30', 'kmeans88'], return_fig=True).savefig('./figs/human_88/k-means.png')
    
    print(adata)
    #print(adata.obs['kmeans88'])
    
    # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate the values
    x_vals = adata.obs['mri_voxel_x'].values
    y_vals = adata.obs['mri_voxel_y'].values
    z_vals = adata.obs['mri_voxel_z'].values
    color_region = adata.obs['region_name'].values
    
    unique_region_list = list(set(color_region))
    color_list = []
    for region in color_region:
        color_list.append(unique_region_list.index(region))
    # Plot the values
    #ax.scatter(x_vals, y_vals, z_vals, c = 'b', marker='o')
    #ax.set_xlabel('X-axis')
    #ax.set_ylabel('Y-axis')
    #ax.set_zlabel('Z-axis')

    #plt.savefig('./figs/human_88/mri_xyy.png')
    
    

    
    # axes instance
    fig = plt.figure(figsize=(10,10))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # get colormap from seaborn
    cmap = ListedColormap(sns.color_palette(n_colors=88, as_cmap=True))

    # plot
    sc = ax.scatter(x_vals, y_vals, z_vals, s=10, c=color_list, marker='o', cmap=cmap, alpha=1)
    ax.set_xlabel('MRI X')
    ax.set_ylabel('MRI Y')
    ax.set_zlabel('MRI Z')

    # legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

    # save
    plt.savefig("./figs/human_88/scatter_xyz_region_name.png", bbox_inches='tight')
    
    
    
    
    color_list_str = list(adata.obs['kmeans88'].values)
    color_list = [int(x) for x in color_list_str]
    
    # axes instance
    fig = plt.figure(figsize=(10,10))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # get colormap from seaborn
    cmap = ListedColormap(sns.color_palette(n_colors=88, as_cmap=True))

    # plot
    sc = ax.scatter(x_vals, y_vals, z_vals, s=10, c=color_list, marker='o', cmap=cmap, alpha=1)
    ax.set_xlabel('MRI X')
    ax.set_ylabel('MRI Y')
    ax.set_zlabel('MRI Z')

    # legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

    # save
    plt.savefig("./figs/human_88/scatter_xyz_kmeans_88.png", bbox_inches='tight')
    
    
    #sc.pl.scatter(adata, x=['mri_voxel_x'], y=['mri_voxel_y'], color=['region_name'], save='./figs/human_88/mri_xy_region.png')
    #sc.pl.scatter(adata, x=['mri_voxel_x'], y=['mri_voxel_z'], color=['region_name'], save='./figs/human_88/mri_xz_region.png')
    #sc.pl.scatter(adata, x=['mri_voxel_y'], y=['mri_voxel_z'], color=['region_name'], save='./figs/human_88/mri_yz_region.png')
    
    
    
    
    
    	
