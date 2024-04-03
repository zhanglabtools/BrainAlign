import os.path
import sys
sys.path.append('../')
import came_origin as came
from came_origin import pipeline, pp, pl
import warnings
warnings.filterwarnings("ignore")
import logging
import os
#import sys
#sys.path.append('../')
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt  # optional
import seaborn as sns  # optional

import scanpy as sc
from scipy import sparse

import networkx as nx
import torch
from heco_utils import threshold_quantile
import logging
from analysis_utils.logger import setup_logger

import anndata as ad

try:
    import matplotlib as mpl
    mpl.use('agg')
except Exception as e:
    print(f"An error occurred when setting matplotlib backend ({e})")



def get_common_clusters(adata_dict, df_varmap_1v1, save_path, 
                        clustering_resolution = 3, 
                        cluster_name = 'cluster_region',
                        n_top_genes = 5000, embedding_size = 128):
    '''
        Get commone clusters for two species
    '''
    k = 0
    species_2 = list(adata_dict.keys())[1]
    species_1 = list(adata_dict.keys())[0]
    column1 = 'Gene name'
    column2 = species_2 +' gene name'
    df_varmap_1v1 = df_varmap_1v1[df_varmap_1v1['Gene name'].notna() & df_varmap_1v1[species_2 +' gene name'].notna()]

    #df_varmap_1v1 = df_varmap_1v1.drop_duplicates(subset='Gene name', keep="first")
    #df_varmap_1v1 = df_varmap_1v1.drop_duplicates(subset=column2, keep="first")

    species_name_list = list(adata_dict.keys())
    column_dict = {species_name_list[0]: 'Gene name', species_name_list[1]: species_2 +' gene name'}
    for species_id, adata in adata_dict.items():
        df_varmap_1v1 = df_varmap_1v1[df_varmap_1v1[column_dict[species_id]].isin(adata.var_names)]
    print(df_varmap_1v1.shape)
    print(df_varmap_1v1)
    for species_id, adata in adata_dict.items():
        if k == 0:
            gene_set = df_varmap_1v1[column1]#.values
            print(adata)
            print(adata.var_names)
            adata_temp = adata[:, gene_set].copy()
            #print(len(adata.var_names))
            X = adata_temp.X
            adata_concat = ad.AnnData(X = X)
            adata_concat.obs_names = adata.obs_names
            adata_concat.obs['species_id'] = [species_id] * len(adata.obs_names)
            #print(len(gene_set))
            #print(adata_concat.X.shape)
            #break
            adata_concat.var_names = list(gene_set)
            var_names = list(gene_set)
        else:
            gene_set = df_varmap_1v1[column2]#.values
            adata_temp = adata[:, gene_set].copy()
            X = adata_temp.X
            adata_1 = ad.AnnData(X = X)
            adata_1.obs_names = adata.obs_names
            adata_1.var_names = var_names
            adata_1.obs['species_id'] = [species_id] * len(adata.obs_names)

            adata_concat = ad.concat([adata_concat, adata_1], axis=0)
        k += 1

    sc.pp.combat(adata_concat, key='species_id', covariates=None, inplace=True)

    if len(adata_concat.var_names) < n_top_genes:
        n_top_genes = len(adata_concat.var_names)
    
    sc.pp.highly_variable_genes(adata_concat, flavor="seurat_v3", n_top_genes=n_top_genes)

    print(f'One-to-one homologous gene number = {adata_concat.n_vars}')

    adata_concat = adata_concat[:, adata_concat.var['highly_variable']]
    sc.tl.pca(adata_concat, svd_solver='arpack', n_comps=embedding_size)

    sc.pp.neighbors(adata_concat, n_neighbors=30, metric='cosine', use_rep='X_pca')
    sc.tl.leiden(adata_concat, resolution = clustering_resolution, key_added = cluster_name)

    ## Plot UMAP
    color_1 = '#5D8AEF'#'#4472C4'
    color_2 = '#FE1613'#'#C94799'#'#ED7D31'
    umap_neighbor = 30

    fig_format = 'jpg'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig_dpi = 400
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
    
    
    palette = {species_1:color_1, species_2:color_2}

    sc.pp.neighbors(adata_concat, n_neighbors=umap_neighbor, metric='cosine',
                use_rep='X_pca')
    sc.tl.umap(adata_concat)
    
    with plt.rc_context({"figure.figsize": (4, 2), "figure.dpi": (fig_dpi)}):
        sc.pl.umap(adata_concat, color=[cluster_name], return_fig=True, size=2,
                   legend_loc='right margin').savefig(
            save_path + 'umap_cluster_name.' + fig_format, format=fig_format)
    rcParams["figure.subplot.left"] = 0.2
    rcParams["figure.subplot.right"] = 0.9

    rcParams["figure.subplot.left"] = 0.1
    rcParams["figure.subplot.right"] = 0.68#0.6545
    with plt.rc_context({"figure.figsize": (4, 3), "figure.dpi": (fig_dpi)}):
        fg = sc.pl.umap(adata_concat, color=['species_id'], return_fig=True, legend_loc='right margin', title='', size=2, palette=palette)
        plt.title('')
        fg.savefig(save_path + 'umap_species.' + fig_format, format=fig_format)

    
    #############################
    cluster_dict = {k:v for k,v in zip(adata_concat.obs_names, adata_concat.obs[cluster_name])}

    for species_id, adata in adata_dict.items():
        adata.obs[cluster_name] = [cluster_dict[x] for x in adata.obs_names]
        adata_dict[species_id] = adata

    
    return adata_dict



def run_came_homo_random(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CAME.visible_device
    path_rawdata1 = cfg.CAME.path_rawdata1
    path_rawdata2 = cfg.CAME.path_rawdata2
    resdir, logger = run_came(cfg)
    

    cfg.CAME.embedding_path = resdir + 'adt_hidden_cell.h5ad'

    homo_corr(resdir, logger, cfg)

    random_corr(resdir, logger, cfg)

    plot_homo_random(resdir, logger, cfg)

    ttest(resdir, logger, cfg)



def run_came(cfg):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s-%(lineno)d-%(funcName)s(): '
               '%(levelname)s\n %(message)s')

    # came.__test1__(10, batch_size=None, reverse=False)
    # came.__test2__(10, batch_size=2048)

    dsnames = (cfg.CAME.species_name_list[0], cfg.CAME.species_name_list[1])  # the dataset names, set by user
    dsn1, dsn2 = dsnames
    time_tag = came.make_nowtime_tag(brackets=False)
    resdir = cfg.CAME.ROOT + f'{dsn1}-{dsn2}/'  # set by user

    #path_rawdata1 = '../Brain_ST_human_mouse/data/mouse_brain_region_67_sparse_no_threshold.h5ad'
    #path_rawdata2 = './brain_human_mouse/human_brain_region_88_sparse.h5ad'

    path_varmap = cfg.CAME.path_varmap
    path_varmap_1v1 =  cfg.CAME.path_varmap_1v1

    # ========= load data =========
    df_varmap = pd.read_csv(path_varmap)
    df_varmap_1v1 = pd.read_csv(path_varmap_1v1) if path_varmap_1v1 else came.pp.take_1v1_matches(df_varmap)

    

    adata_raw1 = sc.read_h5ad(cfg.CAME.path_rawdata1)
    # sc.pp.neighbors(adata_raw1, n_neighbors=cfg.ANALYSIS.mouse_umap_neighbor, metric='cosine', use_rep='X')
    # sc.tl.leiden(adata_raw1, resolution=3, key_added='cluster_region')

    adata_raw2 = sc.read_h5ad(cfg.CAME.path_rawdata2)
    # sc.pp.neighbors(adata_raw2, n_neighbors=cfg.ANALYSIS.human_umap_neighbor, metric='cosine', use_rep='X')
    # sc.tl.leiden(adata_raw2, resolution=3, key_added='cluster_region')

    

    adata_dict = {cfg.CAME.species_name_list[0]:adata_raw1, cfg.CAME.species_name_list[1]:adata_raw2}
    adata_dict = get_common_clusters(adata_dict, df_varmap_1v1, save_path = resdir + 'preclustering/', 
                        clustering_resolution = cfg.CAME.preclustering_resolution, 
                        cluster_name = 'cluster_region',
                        n_top_genes = 5000, embedding_size=cfg.CAME.embedding_size)
    adata_raw1 = adata_dict[cfg.CAME.species_name_list[0]]
    adata_raw2 = adata_dict[cfg.CAME.species_name_list[1]]
    
    adatas = [adata_raw1, adata_raw2]

    key_class1 = cfg.CAME.annotation_name[0]#'cluster_region'#'region_name'  # set by user
    key_class2 = cfg.CAME.annotation_name[1]#'cluster_region'#'region_name'  # set by user

    #ROOT = '/home1/zhangbiao/CAME/brain_mouse_human_no_threshold_sparse/'
    

    logger = setup_logger("Running CAME and analysis", resdir, if_train=True)

    logger.info("Running with config:\n{}".format(cfg))

    figdir = resdir + 'figs/'
    came.check_dirs(figdir)  # check and make the directory

    sc.pp.filter_genes(adata_raw1, min_cells=1)
    sc.pp.filter_genes(adata_raw2, min_cells=1)

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
    n_epochs = cfg.TRAINING.n_epochs

    # The training batch size
    # When the GPU memory is limited, set 4096 or more if possible.
    batch_size = cfg.TRAINING.batch_size

    # The number of epochs to skip for checkpoint backup
    n_pass = cfg.TRAINING.n_pass

    # Whether to use the single-cell networks
    use_scnets = True

    # The number of top DEGs to take as the node-features of each cells.
    # You set it 70-100 for distant species pairs.
    ntop_deg = cfg.TRAINING.ntop_deg

    # The number of top DEGs to take as the graph nodes, which can be directly displayed on the UMAP plot.
    ntop_deg_nodes = cfg.TRAINING.ntop_deg_nodes
    # The source of the node genes; use both DEGs and HVGs by default
    node_source = 'deg,hvg'

    # Whether to take into account the non-1v1 variables as the node features.
    keep_non1v1_feats = True

    key_class1 = cfg.CAME.learning_label[0]# 'cluster_region'  # 'cluster_region'#'region_name'  # set by user
    key_class2 = cfg.CAME.learning_label[1]#'cluster_region'  # 'region_name'  # 'cluster_region'#'region_name'  # set by user

    came_inputs, (adata1, adata2) = pipeline.preprocess_unaligned(
        adatas,
        key_class=key_class1,
        use_scnets=use_scnets,
        ntop_deg=ntop_deg,
        ntop_deg_nodes=ntop_deg_nodes,
        node_source=node_source,
        n_top_genes=cfg.CAME.n_top_genes,
        do_normalize=cfg.CAME.do_normalize
    )

    logger.info('came_results:'.format(came_inputs))

    if cfg.CAME.sparse == True:
        #print('Mouse features: ', came_inputs['adatas'][0])
        X = came_inputs['adatas'][0].X.toarray()
        print(type(X))
        came_inputs['adatas'][0].X = sparse.csr_matrix(threshold_quantile(X, quantile_gene=cfg.CAME.quantile_gene, quantile_sample=cfg.CAME.quantile_sample))

        #print('Human features: ', came_inputs['adatas'][1])
        X = came_inputs['adatas'][1].X.toarray()
        print(type(X))
        came_inputs['adatas'][1].X = sparse.csr_matrix(threshold_quantile(X, quantile_gene=cfg.CAME.quantile_gene, quantile_sample=cfg.CAME.quantile_sample))

    outputs = pipeline.main_for_unaligned(
        **came_inputs,
        df_varmap=df_varmap,
        df_varmap_1v1=df_varmap_1v1,
        dataset_names=dsnames,
        key_class1=key_class1,
        key_class2=key_class2,
        do_normalize=cfg.CAME.do_normalize,
        keep_non1v1_feats=keep_non1v1_feats,
        n_epochs=n_epochs,
        resdir=resdir,
        n_pass=n_pass,
        batch_size=batch_size,
        plot_results=True,
        device_id=cfg.CAME.visible_device,
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
    print(y_true.shape)
    print(y_pred.shape)
    print(type(y_true[0]))
    print(type(y_pred[0]))

    # ax, contmat = pl.plot_contingency_mat(y_true, y_pred, norm_axis=1, order_rows=False, order_cols=False,)
    # pl._save_with_adjust(ax.figure, figdir / 'contingency_mat.png')
    # ax.figure

    name_label = 'celltype'
    cols_anno = ['celltype', 'predicted'][:]
    df_probs = obs[list(classes)]

    gs = pl.wrapper_heatmap_scores(df_probs.iloc[obs_ids2], obs.iloc[obs_ids2], ignore_index=True,
                                   col_label='celltype', col_pred='predicted',
                                   n_subsample=50,  # sampling 50 cells for each group
                                   cmap_heat='magma_r',
                                   fp=figdir + 'heatmap_probas.pdf')

    gs.figure

    # further analysis
    hidden_list = came.load_hidden_states(resdir + 'hidden_list.h5')
    # hidden_list  # a list of dicts
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
    obs.to_csv(resdir + 'obs.csv')
    adt.write(resdir + 'adt_hidden_cell.h5ad')

    adata1.obsm['X_umap'] = obs_umap[obs_ids1]
    adata2.obsm['X_umap'] = obs_umap[obs_ids2]

    # umap of mouse
    #sc.pl.umap(adata1, color='celltype', save=f'-ctype{ftype}')

    # Umap of genes
    sc.pp.neighbors(gadt, n_neighbors=15, metric='cosine', use_rep='X')

    # gadt = pp.make_adata(h_dict['gene'], obs=dpair.var.iloc[:, :2], assparse=False, ignore_index=True)
    sc.tl.umap(gadt)
    sc.pl.umap(gadt, color='dataset')

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

    df_var_links = came.weight_linked_vars(
        gadt.X, dpair._vv_adj, names=dpair.get_vnode_names(),
        matric='cosine', index_names=dsnames, )

    gadt1.write(resdir + 'adt_hidden_gene1.h5ad')
    gadt2.write(resdir + 'adt_hidden_gene2.h5ad')

    # Gene-expression-profiles (for each cell type) on gene UMAP
    # averaged expressions
    avg_expr1 = pp.group_mean_adata(adatas[0], groupby=key_class1, features=dpair.vnode_names1, use_raw=True)
    avg_expr2 = pp.group_mean_adata(adatas[1], groupby=key_class2 if key_class2 else 'predicted',
                                    features=dpair.vnode_names2, use_raw=True)

    plkwds = dict(cmap='RdYlBu_r', vmax=2.5, vmin=-1.5, do_zscore=True,
                  axscale=3, ncols=5, with_cbar=True)

    fig1, axs1 = pl.adata_embed_with_values(gadt1, avg_expr1, fp=figdir + f'umap_exprAvgs-{dsnames[0]}-all.png',
                                            **plkwds)
    fig2, axs2 = pl.adata_embed_with_values(gadt2, avg_expr2, fp=figdir + f'umap_exprAvgs-{dsnames[1]}-all.png',
                                            **plkwds)

    # Abstracted graph
    norm_ov = ['max', 'zs', None][1]
    cut_ov = 0.

    groupby_var = 'module'
    obs_labels1, obs_labels2 = adt.obs['celltype'][dpair.obs_ids1], \
                               adt.obs['celltype'][dpair.obs_ids2]
    var_labels1, var_labels2 = gadt1.obs[groupby_var], gadt2.obs[groupby_var]

    return resdir, logger


def homo_corr(resdir, logger, cfg):

    # No label order version
    '''
    Step 1: Compute average embedding of every region in two species, use two dict to store;
    Step 2: Compute similarity matrix, use np array to store;
    Step 3: Heatmap.

    '''
    # Read ordered labels
    species1 = cfg.CAME.species_name_list[0]
    species2 = cfg.CAME.species_name_list[1]

    #expression_human_path = '../brain_human_mouse/human_brain_region_88_sparse_with3d.h5ad'
    adata_human = sc.read_h5ad(cfg.CAME.path_rawdata2)
    print(adata_human)

    #expression_mouse_path = '../../Brain_ST_human_mouse/data/mouse_brain_region_67_sagittal.h5ad'
    adata_mouse = sc.read_h5ad(cfg.CAME.path_rawdata1)
    print(adata_mouse)

    path_human = cfg.CAME.embedding_path

    adata_human_mouse = sc.read_h5ad(path_human)
    print(adata_human_mouse)
    # Step 1
    embedding_len = 128

    mouse_annotation = cfg.CAME.annotation_name[0]
    human_annotation = cfg.CAME.annotation_name[1]

    adata_human_mouse.obs_names = adata_human_mouse.obs['original_name']
    adata_human_embedding = adata_human_mouse[adata_human.obs_names]
    adata_human_embedding.obs[human_annotation] = adata_human.obs[human_annotation]
    for sample_id in adata_human_embedding.obs_names:
        adata_human_embedding[sample_id].obs[human_annotation] = adata_human[sample_id].obs[human_annotation]

    adata_mouse_embedding = adata_human_mouse[adata_mouse.obs_names]
    adata_mouse_embedding.obs[mouse_annotation] = adata_mouse.obs[mouse_annotation]
    for sample_id in adata_mouse_embedding.obs_names:
        adata_mouse_embedding[sample_id].obs[mouse_annotation] = adata_mouse[sample_id].obs[mouse_annotation]
    #

    # umap of embedding
    sc.set_figure_params(dpi_save=200)

    sc.pp.neighbors(adata_mouse_embedding, n_neighbors=15, metric='cosine', use_rep='X')
    sc.tl.umap(adata_mouse_embedding)
    # sc.pl.umap(adt, color=['dataset', 'celltype'], ncols=1)
    ftype = ['.svg', ''][1]
    #sc.pl.umap(adata_mouse_embedding, color='dataset', save=f'-dataset{ftype}')
    sc.pl.umap(adata_mouse_embedding, color=mouse_annotation, save=f'-umap_mouse{ftype}')

    # umap of embedding
    sc.set_figure_params(dpi_save=200)

    sc.pp.neighbors(adata_human_embedding, n_neighbors=15, metric='cosine', use_rep='X')
    sc.tl.umap(adata_human_embedding)
    # sc.pl.umap(adt, color=['dataset', 'celltype'], ncols=1)
    ftype = ['.svg', ''][1]
    # sc.pl.umap(adata_mouse_embedding, color='dataset', save=f'-dataset{ftype}')
    sc.pl.umap(adata_human_embedding, color=human_annotation, save= f'-umap_human{ftype}', size=25)

    human_mouse_homo_region = pd.read_csv(cfg.CAME.human_mouse_homo_region)

    print(human_mouse_homo_region)
    home_region_dict = OrderedDict()
    for x, y in zip(human_mouse_homo_region[species2], human_mouse_homo_region[species1].values):
        home_region_dict[x] = y

    k = 0

    human_correlation_dict = {f'{species2}_region_list': [], 'mean': [], 'std': []}
    mouse_correlation_dict = {f'{species1}_region_list': [], 'mean': [], 'std': []}
    human_mouse_correlation_dict = {f'{species2}_region_list': [], f'{species1}_region_list': [], 'mean': [], 'std': []}

    distance_type = 'correlation'

    save_path_root = resdir + 'figs/Hiercluster_came_embedding_{}/'.format(distance_type)

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    with open(save_path_root + 'homo_region_dict.pkl', 'wb') as f:
        pickle.dump(home_region_dict, f)

    for human_region, mouse_region in home_region_dict.items():

        save_path = resdir + 'figs/Hiercluster_came_embedding_{}/human_{}_mouse_{}/'.format(distance_type,
                                                                                              human_region,
                                                                                              mouse_region)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        #logger.info('human_region: %s', human_region)
        adata_human_embedding_region = adata_human_embedding[
            adata_human_embedding.obs[human_annotation].isin([human_region])]

        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        # plt.figure(figsize=(16, 16))
        # sns.clustermap(adata_human_embedding_region.X, metric=distance_type).fig.suptitle(
        #     'Human - {}'.format(human_region))
        # plt.savefig(save_path + 'human_{}.png'.format(human_region))
        # plt.show()

        #logger.info('mouse_region: %s ', mouse_region)
        adata_mouse_embedding_region = adata_mouse_embedding[adata_mouse_embedding.obs[mouse_annotation] == mouse_region]
        # plt.figure(figsize=(16, 16))
        # sns.clustermap(adata_mouse_embedding_region.X, metric=distance_type).fig.suptitle(
        #     'Mouse - {}'.format(mouse_region))
        # plt.savefig(save_path + 'mouse_{}.png'.format(mouse_region))
        # plt.show()

        # ---------human corr---------------------
        human_df = pd.DataFrame(adata_human_embedding_region.X).T
        human_corr = human_df.corr()
        #logger.info('human corr shape:'.format(human_corr.shape))
        mean, std = human_corr.mean().mean(), human_corr.stack().std()
        #logger.info('mean = {}, std = {}'.format(mean, std))
        human_correlation_dict[f'{species2}_region_list'].append(human_region)
        human_correlation_dict['mean'].append(mean)
        human_correlation_dict['std'].append(std)
        # ---------mouse corr---------------------
        mouse_df = pd.DataFrame(adata_mouse_embedding_region.X).T
        mouse_corr = mouse_df.corr()
        #logger.info('mouse corr shape:'.format(mouse_corr.shape))
        mean, std = mouse_corr.mean().mean(), mouse_corr.stack().std()
        #logger.info('mean = {}, std = {}'.format(mean, std))
        mouse_correlation_dict[f'{species1}_region_list'].append(mouse_region)
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
        #logger.info('mean = {}, std = {}'.format(mean, std))
        human_mouse_correlation_dict[f'{species2}_region_list'].append(human_region)
        human_mouse_correlation_dict[f'{species1}_region_list'].append(mouse_region)
        human_mouse_correlation_dict['mean'].append(mean)
        human_mouse_correlation_dict['std'].append(std)

        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        # plt.figure(figsize=(16, 16))
        k += 1
        print('{}-th region finished!'.format(k))
        # plt.show()

    with open(save_path_root + 'human_mouse_correlation_dict.pkl', 'wb') as f:
        pickle.dump(human_mouse_correlation_dict, f)
    with open(save_path_root + 'human_correlation_dict.pkl', 'wb') as f:
        pickle.dump(human_correlation_dict, f)
    with open(save_path_root + 'mouse_correlation_dict.pkl', 'wb') as f:
        pickle.dump(mouse_correlation_dict, f)


def random_corr(resdir, logger, cfg):

    # No label order version
    '''
    Step 1: Compute average embedding of every region in two species, use two dict to store;
    Step 2: Compute similarity matrix, use np array to store;
    Step 3: Heatmap.

    '''
    species1 = cfg.CAME.species_name_list[0]
    species2 = cfg.CAME.species_name_list[1]
    # Read ordered labels
    mouse_annotation = cfg.CAME.annotation_name[0]
    human_annotation = cfg.CAME.annotation_name[1]
    human_88_labels = pd.read_csv(cfg.CAME.path_labels_2)
    mouse_67_labels = pd.read_csv(cfg.CAME.path_labels_1)

    #expression_human_path = '../brain_human_mouse/human_brain_region_88_sparse_with3d.h5ad'
    adata_human = sc.read_h5ad(cfg.CAME.path_rawdata2)
    print(adata_human)

    # expression_mouse_path = '../../Brain_ST_human_mouse/data/mouse_brain_region_67_sparse_no_threshold.h5ad'
    adata_mouse = sc.read_h5ad(cfg.CAME.path_rawdata1)
    print(adata_mouse)

    # path_human = '../brain_mouse_human_no_threshold_sparse/Baron_mouse-Baron_human-(10-12 15.16.30)/adt_hidden_cell.h5ad'

    adata_human_mouse = sc.read_h5ad(cfg.CAME.embedding_path)
    print(adata_human_mouse)
    # Step 1
    embedding_len = 128

    adata_human_mouse.obs_names = adata_human_mouse.obs['original_name']
    adata_human_embedding = adata_human_mouse[adata_human.obs_names]
    adata_human_embedding.obs[human_annotation] = adata_human.obs[human_annotation]
    for sample_id in adata_human_embedding.obs_names:
        adata_human_embedding[sample_id].obs[human_annotation] = adata_human[sample_id].obs[human_annotation]

    adata_mouse_embedding = adata_human_mouse[adata_mouse.obs_names]
    adata_mouse_embedding.obs[mouse_annotation] = adata_mouse.obs[mouse_annotation]
    for sample_id in adata_mouse_embedding.obs_names:
        adata_mouse_embedding[sample_id].obs[mouse_annotation] = adata_mouse[sample_id].obs[mouse_annotation]
    #
    human_mouse_homo_region = pd.read_csv(cfg.CAME.human_mouse_homo_region)

    print(human_mouse_homo_region)
    home_region_dict = OrderedDict()
    mouse_region_list = human_mouse_homo_region[species1].values
    # random.shuffle(mouse_region_list)
    for x, y in zip(human_mouse_homo_region[species2].values, mouse_region_list):
        home_region_dict[x] = y

    # home_region_dict = OrderedDict()
    human_88_labels_list = list(human_88_labels[human_annotation])
    mouse_67_labels_list = list(mouse_67_labels[mouse_annotation])
    # human_88_labels_list = human_mouse_homo_region['Human'].values
    # mouse_67_labels_list = human_mouse_homo_region['Mouse'].values
    # for x in human_88_labels_list:
    #    for y in mouse_67_labels_list:
    #        if x in home_region_dict_real.keys() and home_region_dict_real[x] != y:
    #            home_region_dict[x] = y

    k = 0

    human_correlation_dict = {f'{species2}_region_list': [], 'mean': [], 'std': []}
    mouse_correlation_dict = {f'{species1}_region_list': [], 'mean': [], 'std': []}
    human_mouse_correlation_dict = {f'{species2}_region_list': [], f'{species1}_region_list': [], 'mean': [], 'std': []}

    distance_type = 'correlation'

    save_path_root = resdir + 'figs/random_Hiercluster_came_embedding_{}/'.format(distance_type)
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

                #logger.info('human_region:%s', human_region)
                adata_human_embedding_region = adata_human_embedding[
                    adata_human_embedding.obs[human_annotation] == human_region]
                # print(adata_human_embedding_region)
                if min(adata_human_embedding_region.X.shape) <= 1:
                    continue

                # color_map = sns.color_palette("coolwarm", as_cmap=True)
                # plt.figure(figsize=(16, 16))
                '''
                sns.clustermap(adata_human_embedding_region.X, metric=distance_type).fig.suptitle('Human - {}'.format(human_region))
                plt.savefig(save_path + 'human_{}.png'.format(human_region))
                '''
                # plt.show()

                #logger.info('mouse_region: %s', mouse_region)
                adata_mouse_embedding_region = adata_mouse_embedding[
                    adata_mouse_embedding.obs[mouse_annotation] == mouse_region]
                if min(adata_mouse_embedding_region.X.shape) <= 1:
                    continue

                # ---------human corr---------------------
                human_df = pd.DataFrame(adata_human_embedding_region.X).T
                human_corr = human_df.corr()
                #logger.info('human corr shape:'.format(human_corr.shape))
                mean, std = human_corr.mean().mean(), human_corr.stack().std()
                #logger.info('mean = {}, std = {}'.format(mean, std))
                human_correlation_dict[f'{species2}_region_list'].append(human_region)
                human_correlation_dict['mean'].append(mean)
                human_correlation_dict['std'].append(std)
                # ---------mouse corr---------------------
                mouse_df = pd.DataFrame(adata_mouse_embedding_region.X).T
                mouse_corr = mouse_df.corr()
                #logger.info('mouse corr shape:'.format(mouse_corr.shape))
                mean, std = mouse_corr.mean().mean(), mouse_corr.stack().std()
                #logger.info('mean = {}, std = {}'.format(mean, std))
                mouse_correlation_dict[f'{species1}_region_list'].append(mouse_region)
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
                #print('mean', mean, 'std', std)
                human_mouse_correlation_dict[f'{species2}_region_list'].append(human_region)
                human_mouse_correlation_dict[f'{species1}_region_list'].append(mouse_region)
                human_mouse_correlation_dict['mean'].append(mean)
                human_mouse_correlation_dict['std'].append(std)

                # color_map = sns.color_palette("coolwarm", as_cmap=True)
                # plt.figure(figsize=(16, 16))
                k += 1
                print('{}-th region finished!'.format(k))
                # plt.show()

    with open(save_path_root + 'human_mouse_correlation_dict.pkl', 'wb') as f:
        pickle.dump(human_mouse_correlation_dict, f)
    with open(save_path_root + 'human_correlation_dict.pkl', 'wb') as f:
        pickle.dump(human_correlation_dict, f)
    with open(save_path_root + 'mouse_correlation_dict.pkl', 'wb') as f:
        pickle.dump(mouse_correlation_dict, f)


import matplotlib.pyplot as plt

import re, seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from collections import OrderedDict
from matplotlib import rcParams

'''
TINY_SIZE = 12
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

plt.rc('font', size=12)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams["legend.frameon"] = False
'''


import pickle


def plot_homo_random(resdir, logger, cfg):

    # No label order version
    '''
    Step 1:load human and mouse cross expression data of homologous regions, and random regions
    Step 2: plot bar, mean and std
    '''
    # Read ordered labels
    species1 = cfg.CAME.species_name_list[0]
    species2 = cfg.CAME.species_name_list[1]

    homo_region_data_path = resdir + 'figs/Hiercluster_came_embedding_correlation/'
    with open(homo_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
        human_mouse_correlation_dict = pickle.load(f)


    home_len = len(human_mouse_correlation_dict['mean'])
    home_random_type = ['homologous'] * home_len
    human_mouse_correlation_dict['type'] = home_random_type
    #data_df = pd.DataFrame.from_dict(human_mouse_correlation_dict)

    random_region_data_path =  resdir +'figs/random_Hiercluster_came_embedding_correlation/'
    with open(random_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
        random_human_mouse_correlation_dict = pickle.load(f)

    random_len = len(random_human_mouse_correlation_dict['mean'])
    home_random_type = ['random'] * random_len
    random_human_mouse_correlation_dict['type'] = home_random_type
    concat_dict = {}
    for k,v in random_human_mouse_correlation_dict.items():
        concat_dict[k] = human_mouse_correlation_dict[k] + random_human_mouse_correlation_dict[k]
    data_df = pd.DataFrame.from_dict(concat_dict)


    save_path =  resdir + 'figs/homo_random/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    #my_pal = {"homo region": (0/255,149/255,182/255), "random":(178/255, 0/255, 32/255)}
    my_pal = {"homologous": (0 / 255, 149 / 255, 182 / 255), "random": (178 / 255, 0 / 255, 32 / 255)}
    #sns.set_theme(style="whitegrid")
    #tips = sns.load_dataset("tips")

    plt.figure(figsize=(8,8))
    ax = sns.boxplot(x="type", y="mean", data=data_df, order=["homologous", "random"], palette = my_pal)
    plt.savefig(save_path + 'mean.svg')
    plt.show()

    plt.figure(figsize=(8, 8))
    ax = sns.boxplot(x="type", y="std", data=data_df, order=["homologous", "random"], palette=my_pal)
    plt.savefig(save_path + 'std.svg')
    plt.show()


    homo_region_data_path =  resdir + 'figs/Hiercluster_came_embedding_correlation/'
    with open(homo_region_data_path + 'human_correlation_dict.pkl', 'rb') as f:
        human_correlation_dict = pickle.load(f)

    with open(homo_region_data_path + 'mouse_correlation_dict.pkl', 'rb') as f:
        mouse_correlation_dict = pickle.load(f)

    human_mouse_dict_mean = {species2:[], species1:[]}
    human_mouse_dict_std = {species2:[], species1:[]}

    human_mouse_dict_mean[species2] = human_correlation_dict['mean']
    human_mouse_dict_mean[species1] = mouse_correlation_dict['mean']

    human_mouse_dict_std[species2] = human_correlation_dict['std']
    human_mouse_dict_std[species1] = mouse_correlation_dict['std']

    sns.jointplot(data=pd.DataFrame.from_dict(human_mouse_dict_mean), x=species2, y=species1, kind="reg")
    plt.savefig(save_path + 'mean_human_mouse.svg')
    plt.show()

    sns.jointplot(data=pd.DataFrame.from_dict(human_mouse_dict_std), x=species2, y=species1, kind="reg")
    plt.savefig(save_path + 'std_human_mouse.svg')
    plt.show()


def ttest(resdir, logger, cfg):

    # No label order version
    '''
    Step 1:load human and mouse cross expression data of homologous regions, and random regions
    Step 2: plot bar, mean and std
    '''
    # Read ordered labels
    species1 = cfg.CAME.species_name_list[0]
    species2 = cfg.CAME.species_name_list[1]

    homo_region_data_path = resdir + 'figs/Hiercluster_came_embedding_correlation/'
    with open(homo_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
        human_mouse_correlation_dict = pickle.load(f)

    home_len = len(human_mouse_correlation_dict['mean'])
    home_random_type = ['homologous'] * home_len
    human_mouse_correlation_dict['type'] = home_random_type
    # data_df = pd.DataFrame.from_dict(human_mouse_correlation_dict)

    random_region_data_path = resdir + 'figs/random_Hiercluster_came_embedding_correlation/'
    with open(random_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
        random_human_mouse_correlation_dict = pickle.load(f)

    random_len = len(random_human_mouse_correlation_dict['mean'])
    home_random_type = ['random'] * random_len
    random_human_mouse_correlation_dict['type'] = home_random_type
    concat_dict = {}
    for k, v in random_human_mouse_correlation_dict.items():
        concat_dict[k] = human_mouse_correlation_dict[k] + random_human_mouse_correlation_dict[k]
    data_df = pd.DataFrame.from_dict(concat_dict)

    save_path = resdir + 'figs/homo_random/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # my_pal = {"homo region": (0/255,149/255,182/255), "random":(178/255, 0/255, 32/255)}
    my_pal = {"homologous": (0 / 255, 149 / 255, 182 / 255), "random": (178 / 255, 0 / 255, 32 / 255)}
    # sns.set_theme(style="whitegrid")
    # tips = sns.load_dataset("tips")

    print(data_df)

    random_df = data_df[data_df['type'] == 'random']
    print(random_df)
    mean_random_list = random_df['mean'].values
    mean_r = np.mean(mean_random_list)
    std_r = np.std(mean_random_list)

    homologous_df = data_df[data_df['type'] == 'homologous']
    print(homologous_df)
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

    with open(save_path+'t_test_result.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(stats.ttest_ind(
            mean_homo_list,
            mean_random_list,
            equal_var=False
        ))
        sys.stdout = original_stdout  # Reset the standard output to its original value

