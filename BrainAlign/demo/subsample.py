# -- coding: utf-8 --
# @Time : 2024/2/1 18:13
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : subsample.py
# @Description: This file is subsampling the origin data
import scanpy as sc
import numpy as np

def obs_key_wise_subsampling(adata, obs_key, N):
    '''
    Subsample each class to same cell numbers (N). Classes are given by obs_key pointing to categorical in adata.obs.
    '''
    counts = adata.obs[obs_key].value_counts()
    # subsample indices per group defined by obs_key
    indices = [np.random.choice(adata.obs_names[adata.obs[obs_key]==group], size=N, replace=True).unique() for group in counts.index]
    selection = np.hstack(np.array(indices))
    return adata[selection].copy()

if __name__ == '__main__':
    mouse_h5ad_file = 'G:/backup/CAME/brain_mouse_2020sa/mouse_2020sa_64regions.h5ad'
    mouse_adata = sc.read_h5ad(mouse_h5ad_file)
    #mouse_adata = sc.pp.subsample(mouse_adata, fraction=0.1, copy=True)


    target_cells = 20

    adatas = [mouse_adata[mouse_adata.obs['region_name'].isin([clust])] for clust in mouse_adata.obs['region_name'].cat.categories]

    for dat in adatas:
        if dat.n_obs > target_cells:
            sc.pp.subsample(dat, fraction=0.1)

    adata_downsampled = adatas[0].concatenate(*adatas[1:])

    print(adata_downsampled)
    print(adata_downsampled.obs['region_name'].value_counts())

    adata_downsampled.write_h5ad("./mouse_2020sa_64regions_demo.h5ad")