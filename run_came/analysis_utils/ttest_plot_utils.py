import os.path
import came
from came import pipeline, pp, pl

import logging
import os
import sys
sys.path.append('../')
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

try:
    import matplotlib as mpl
    mpl.use('agg')
except Exception as e:
    print(f"An error occurred when setting matplotlib backend ({e})")



def load_part_regions(cfg):
    adata_raw1 = sc.read_h5ad(cfg.CAME.path_rawdata1)
    adata_raw2 = sc.read_h5ad(cfg.CAME.path_rawdata2)




def run_came_homo_random(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CAME.visible_device
    path_rawdata1 = cfg.CAME.path_rawdata1
    path_rawdata2 = cfg.CAME.path_rawdata2
    resdir, logger = run_came(cfg)

    cfg.CAME.embedding_path = resdir + 'adt_hidden_cell.h5ad'

    homo_corr(resdir, logger, cfg)

    random_corr(resdir, logger, cfg)

    plot_homo_random(resdir, logger)

    ttest(resdir, logger)



def run_came(cfg):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s-%(lineno)d-%(funcName)s(): '
               '%(levelname)s\n %(message)s')

    # came.__test1__(10, batch_size=None, reverse=False)
    # came.__test2__(10, batch_size=2048)
    dsnames = ('Baron_mouse', 'Baron_human')  # the dataset names, set by user
    dsn1, dsn2 = dsnames

    #path_rawdata1 = '../Brain_ST_human_mouse/data/mouse_brain_region_67_sparse_no_threshold.h5ad'
    #path_rawdata2 = './brain_human_mouse/human_brain_region_88_sparse.h5ad'

    path_varmap = cfg.CAME.path_varmap
    path_varmap_1v1 =  cfg.CAME.path_varmap_1v1

    # ========= load data =========
    df_varmap = pd.read_csv(path_varmap)
    df_varmap_1v1 = pd.read_csv(path_varmap_1v1) if path_varmap_1v1 else came.pp.take_1v1_matches(df_varmap)

    adata_raw1 = sc.read_h5ad(cfg.CAME.path_rawdata1)

    sc.tl.leiden(adata_raw1, resolution=9, key_added='cluster_region')

    adata_raw2 = sc.read_h5ad(cfg.CAME.path_rawdata2)
    sc.tl.leiden(adata_raw2, resolution=9, key_added='cluster_region')

    adatas = [adata_raw1, adata_raw2]

    key_class1 = 'region_name'#'cluster_region'#'region_name'  # set by user
    key_class2 = 'region_name'#'cluster_region'#'region_name'  # set by user

    #ROOT = '/home1/zhangbiao/CAME/brain_mouse_human_no_threshold_sparse/'
    time_tag = came.make_nowtime_tag(brackets=False)
    resdir = cfg.CAME.ROOT + f'{dsn1}-{dsn2}-{time_tag}/'  # set by user

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
    n_epochs = 300

    # The training batch size
    # When the GPU memory is limited, set 4096 or more if possible.
    batch_size = 4096

    # The number of epochs to skip for checkpoint backup
    n_pass = 50

    # Whether to use the single-cell networks
    use_scnets = True

    # The number of top DEGs to take as the node-features of each cells.
    # You set it 70-100 for distant species pairs.
    ntop_deg = 70

    # The number of top DEGs to take as the graph nodes, which can be directly displayed on the UMAP plot.
    ntop_deg_nodes = 50
    # The source of the node genes; use both DEGs and HVGs by default
    node_source = 'deg,hvg'

    # Whether to take into account the non-1v1 variables as the node features.
    keep_non1v1_feats = True

    key_class1 = 'cluster_region'  # 'cluster_region'#'region_name'  # set by user
    key_class2 = 'cluster_region'  # 'region_name'  # 'cluster_region'#'region_name'  # set by user

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

    sp1, sp2 = 'mouse', 'human'
    g = came.make_abstracted_graph(
        obs_labels1, obs_labels2,
        var_labels1, var_labels2,
        avg_expr1, avg_expr2,
        df_var_links,
        tags_obs=(f'{sp1} ', f'{sp2} '),
        tags_var=(f'{sp1} module ', f'{sp2} module '),
        cut_ov=cut_ov,
        norm_mtd_ov=norm_ov, )

    # visualization
    fp_abs = figdir + f'abstracted_graph-{groupby_var}-cut{cut_ov}-{norm_ov}.pdf'
    ax = pl.plot_multipartite_graph(g, edge_scale=10, figsize=(9, 7.5), alpha=0.5, fp=fp_abs)  # nodelist=nodelist,

    ax.figure

    came.save_pickle(g, resdir + 'abs_graph.pickle')

    return resdir, logger


def homo_corr(resdir, logger, cfg):

    # No label order version
    '''
    Step 1: Compute average embedding of every region in two species, use two dict to store;
    Step 2: Compute similarity matrix, use np array to store;
    Step 3: Heatmap.

    '''
    # Read ordered labels

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

    adata_human_mouse.obs_names = adata_human_mouse.obs['original_name']
    adata_human_embedding = adata_human_mouse[adata_human.obs_names]
    adata_human_embedding.obs['region_name'] = adata_human.obs['region_name']
    for sample_id in adata_human_embedding.obs_names:
        adata_human_embedding[sample_id].obs['region_name'] = adata_human[sample_id].obs['region_name']

    adata_mouse_embedding = adata_human_mouse[adata_mouse.obs_names]
    adata_mouse_embedding.obs['region_name'] = adata_mouse.obs['region_name']
    for sample_id in adata_mouse_embedding.obs_names:
        adata_mouse_embedding[sample_id].obs['region_name'] = adata_mouse[sample_id].obs['region_name']
    #

    # umap of embedding
    sc.set_figure_params(dpi_save=200)

    sc.pp.neighbors(adata_mouse_embedding, n_neighbors=15, metric='cosine', use_rep='X')
    sc.tl.umap(adata_mouse_embedding)
    # sc.pl.umap(adt, color=['dataset', 'celltype'], ncols=1)
    ftype = ['.svg', ''][1]
    #sc.pl.umap(adata_mouse_embedding, color='dataset', save=f'-dataset{ftype}')
    sc.pl.umap(adata_mouse_embedding, color='region_name', save=f'-umap_mouse{ftype}')

    # umap of embedding
    sc.set_figure_params(dpi_save=200)

    sc.pp.neighbors(adata_human_embedding, n_neighbors=15, metric='cosine', use_rep='X')
    sc.tl.umap(adata_human_embedding)
    # sc.pl.umap(adt, color=['dataset', 'celltype'], ncols=1)
    ftype = ['.svg', ''][1]
    # sc.pl.umap(adata_mouse_embedding, color='dataset', save=f'-dataset{ftype}')
    sc.pl.umap(adata_human_embedding, color='region_name', save= f'-umap_human{ftype}', size=25)

    human_mouse_homo_region = pd.read_csv(cfg.CAME.human_mouse_homo_region)

    print(human_mouse_homo_region)
    home_region_dict = OrderedDict()
    for x, y in zip(human_mouse_homo_region['Human'], human_mouse_homo_region['Mouse'].values):
        home_region_dict[x] = y

    k = 0

    human_correlation_dict = {'human_region_list': [], 'mean': [], 'std': []}
    mouse_correlation_dict = {'mouse_region_list': [], 'mean': [], 'std': []}
    human_mouse_correlation_dict = {'human_region_list': [], 'mouse_region_list': [], 'mean': [], 'std': []}

    distance_type = 'correlation'

    save_path_root = resdir + 'figs/human_88/Hiercluster_came_embedding_{}/'.format(distance_type)

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    with open(save_path_root + 'homo_region_dict.pkl', 'wb') as f:
        pickle.dump(home_region_dict, f)

    for human_region, mouse_region in home_region_dict.items():

        save_path = resdir + 'figs/human_88/Hiercluster_came_embedding_{}/human_{}_mouse_{}/'.format(distance_type,
                                                                                              human_region,
                                                                                              mouse_region)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        logger.info('human_region: %s', human_region)
        adata_human_embedding_region = adata_human_embedding[
            adata_human_embedding.obs['region_name'].isin([human_region])]

        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        # plt.figure(figsize=(16, 16))
        sns.clustermap(adata_human_embedding_region.X, metric=distance_type).fig.suptitle(
            'Human - {}'.format(human_region))
        plt.savefig(save_path + 'human_{}.png'.format(human_region))
        # plt.show()

        logger.info('mouse_region: %s ', mouse_region)
        adata_mouse_embedding_region = adata_mouse_embedding[adata_mouse_embedding.obs['region_name'] == mouse_region]
        # plt.figure(figsize=(16, 16))
        sns.clustermap(adata_mouse_embedding_region.X, metric=distance_type).fig.suptitle(
            'Mouse - {}'.format(mouse_region))
        plt.savefig(save_path + 'mouse_{}.png'.format(mouse_region))
        # plt.show()

        # ---------human corr---------------------
        human_df = pd.DataFrame(adata_human_embedding_region.X).T
        human_corr = human_df.corr()
        logger.info('human corr shape:'.format(human_corr.shape))
        mean, std = human_corr.mean().mean(), human_corr.stack().std()
        logger.info('mean = {}, std = {}'.format(mean, std))
        human_correlation_dict['human_region_list'].append(human_region)
        human_correlation_dict['mean'].append(mean)
        human_correlation_dict['std'].append(std)
        # ---------mouse corr---------------------
        mouse_df = pd.DataFrame(adata_mouse_embedding_region.X).T
        mouse_corr = mouse_df.corr()
        logger.info('mouse corr shape:'.format(mouse_corr.shape))
        mean, std = mouse_corr.mean().mean(), mouse_corr.stack().std()
        logger.info('mean = {}, std = {}'.format(mean, std))
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
        logger.info('mean = {}, std = {}'.format(mean, std))
        human_mouse_correlation_dict['human_region_list'].append(human_region)
        human_mouse_correlation_dict['mouse_region_list'].append(mouse_region)
        human_mouse_correlation_dict['mean'].append(mean)
        human_mouse_correlation_dict['std'].append(std)

        # color_map = sns.color_palette("coolwarm", as_cmap=True)
        # plt.figure(figsize=(16, 16))
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


def random_corr(resdir, logger, cfg):

    # No label order version
    '''
    Step 1: Compute average embedding of every region in two species, use two dict to store;
    Step 2: Compute similarity matrix, use np array to store;
    Step 3: Heatmap.

    '''
    # Read ordered labels
    human_88_labels = pd.read_csv(cfg.CAME.path_human_labels)
    mouse_67_labels = pd.read_csv(cfg.CAME.path_mouse_labels)

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
    adata_human_embedding.obs['region_name'] = adata_human.obs['region_name']
    for sample_id in adata_human_embedding.obs_names:
        adata_human_embedding[sample_id].obs['region_name'] = adata_human[sample_id].obs['region_name']

    adata_mouse_embedding = adata_human_mouse[adata_mouse.obs_names]
    adata_mouse_embedding.obs['region_name'] = adata_mouse.obs['region_name']
    for sample_id in adata_mouse_embedding.obs_names:
        adata_mouse_embedding[sample_id].obs['region_name'] = adata_mouse[sample_id].obs['region_name']
    #
    human_mouse_homo_region = pd.read_csv(cfg.CAME.human_mouse_homo_region)

    print(human_mouse_homo_region)
    home_region_dict = OrderedDict()
    mouse_region_list = human_mouse_homo_region['Mouse'].values
    # random.shuffle(mouse_region_list)
    for x, y in zip(human_mouse_homo_region['Human'].values, mouse_region_list):
        home_region_dict[x] = y

    # home_region_dict = OrderedDict()
    human_88_labels_list = list(human_88_labels['region_name'])
    mouse_67_labels_list = list(mouse_67_labels['region_name'])
    # human_88_labels_list = human_mouse_homo_region['Human'].values
    # mouse_67_labels_list = human_mouse_homo_region['Mouse'].values
    # for x in human_88_labels_list:
    #    for y in mouse_67_labels_list:
    #        if x in home_region_dict_real.keys() and home_region_dict_real[x] != y:
    #            home_region_dict[x] = y

    k = 0

    human_correlation_dict = {'human_region_list': [], 'mean': [], 'std': []}
    mouse_correlation_dict = {'mouse_region_list': [], 'mean': [], 'std': []}
    human_mouse_correlation_dict = {'human_region_list': [], 'mouse_region_list': [], 'mean': [], 'std': []}

    distance_type = 'correlation'

    save_path_root = resdir + 'figs/human_88/random_Hiercluster_came_embedding_{}/'.format(distance_type)
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
                '''
                save_path = '../figs_all_genes_came/human_88/random_Hiercluster_came_embedding_{}/human_{}_mouse_{}/'.format(distance_type, human_region, mouse_region)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                '''

                logger.info('human_region:%s', human_region)
                adata_human_embedding_region = adata_human_embedding[
                    adata_human_embedding.obs['region_name'] == human_region]
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

                logger.info('mouse_region: %s', mouse_region)
                adata_mouse_embedding_region = adata_mouse_embedding[
                    adata_mouse_embedding.obs['region_name'] == mouse_region]
                if min(adata_mouse_embedding_region.X.shape) <= 1:
                    continue
                # if max(adata_mouse_embedding_region.X.shape) >= 4500:
                #    continue
                # print(adata_mouse_embedding_region)
                # plt.figure(figsize=(16, 16))
                '''
                sns.clustermap(adata_mouse_embedding_region.X, metric=distance_type).fig.suptitle('Mouse - {}'.format(mouse_region))
                plt.savefig(save_path + 'mouse_{}.png'.format(mouse_region))
                '''
                # plt.show()

                # ---------human corr---------------------
                human_df = pd.DataFrame(adata_human_embedding_region.X).T
                human_corr = human_df.corr()
                logger.info('human corr shape:'.format(human_corr.shape))
                mean, std = human_corr.mean().mean(), human_corr.stack().std()
                logger.info('mean = {}, std = {}'.format(mean, std))
                human_correlation_dict['human_region_list'].append(human_region)
                human_correlation_dict['mean'].append(mean)
                human_correlation_dict['std'].append(std)
                # ---------mouse corr---------------------
                mouse_df = pd.DataFrame(adata_mouse_embedding_region.X).T
                mouse_corr = mouse_df.corr()
                logger.info('mouse corr shape:'.format(mouse_corr.shape))
                mean, std = mouse_corr.mean().mean(), mouse_corr.stack().std()
                logger.info('mean = {}, std = {}'.format(mean, std))
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
                '''
                sns.clustermap(Var_Corr, metric=distance_type).fig.suptitle('Human-{} and Mouse-{}'.format(human_region, mouse_region))
                #plt.title('Human-{} and Mouse-{}'.format(human_region, mouse_region), loc='right')
                plt.savefig(save_path + 'human_{}_mouse_{}.png'.format(human_region, mouse_region))
                '''

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


def plot_homo_random(resdir, logger):

    # No label order version
    '''
    Step 1:load human and mouse cross expression data of homologous regions, and random regions
    Step 2: plot bar, mean and std
    '''
    # Read ordered labels

    homo_region_data_path = resdir + 'figs/human_88/Hiercluster_came_embedding_correlation/'
    with open(homo_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
        human_mouse_correlation_dict = pickle.load(f)


    home_len = len(human_mouse_correlation_dict['mean'])
    home_random_type = ['homologous'] * home_len
    human_mouse_correlation_dict['type'] = home_random_type
    #data_df = pd.DataFrame.from_dict(human_mouse_correlation_dict)

    random_region_data_path =  resdir +'figs/human_88/random_Hiercluster_came_embedding_correlation/'
    with open(random_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
        random_human_mouse_correlation_dict = pickle.load(f)

    random_len = len(random_human_mouse_correlation_dict['mean'])
    home_random_type = ['random'] * random_len
    random_human_mouse_correlation_dict['type'] = home_random_type
    concat_dict = {}
    for k,v in random_human_mouse_correlation_dict.items():
        concat_dict[k] = human_mouse_correlation_dict[k] + random_human_mouse_correlation_dict[k]
    data_df = pd.DataFrame.from_dict(concat_dict)


    save_path =  resdir + 'figs/human_88/homo_random/'
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


    homo_region_data_path =  resdir + 'figs/human_88/Hiercluster_came_embedding_correlation/'
    with open(homo_region_data_path + 'human_correlation_dict.pkl', 'rb') as f:
        human_correlation_dict = pickle.load(f)

    with open(homo_region_data_path + 'mouse_correlation_dict.pkl', 'rb') as f:
        mouse_correlation_dict = pickle.load(f)

    human_mouse_dict_mean = {'Human':[], 'Mouse':[]}
    human_mouse_dict_std = {'Human':[], 'Mouse':[]}

    human_mouse_dict_mean['Human'] = human_correlation_dict['mean']
    human_mouse_dict_mean['Mouse'] = mouse_correlation_dict['mean']

    human_mouse_dict_std['Human'] = human_correlation_dict['std']
    human_mouse_dict_std['Mouse'] = mouse_correlation_dict['std']

    sns.jointplot(data=pd.DataFrame.from_dict(human_mouse_dict_mean), x="Human", y="Mouse", kind="reg")
    plt.savefig(save_path + 'mean_human_mouse.svg')
    plt.show()

    sns.jointplot(data=pd.DataFrame.from_dict(human_mouse_dict_std), x="Human", y="Mouse", kind="reg")
    plt.savefig(save_path + 'std_human_mouse.svg')
    plt.show()


def ttest(resdir, logger):

    # No label order version
    '''
    Step 1:load human and mouse cross expression data of homologous regions, and random regions
    Step 2: plot bar, mean and std
    '''
    # Read ordered labels

    homo_region_data_path = resdir + 'figs/human_88/Hiercluster_came_embedding_correlation/'
    with open(homo_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
        human_mouse_correlation_dict = pickle.load(f)

    home_len = len(human_mouse_correlation_dict['mean'])
    home_random_type = ['homologous'] * home_len
    human_mouse_correlation_dict['type'] = home_random_type
    # data_df = pd.DataFrame.from_dict(human_mouse_correlation_dict)

    random_region_data_path = resdir + 'figs/human_88/random_Hiercluster_came_embedding_correlation/'
    with open(random_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
        random_human_mouse_correlation_dict = pickle.load(f)

    random_len = len(random_human_mouse_correlation_dict['mean'])
    home_random_type = ['random'] * random_len
    random_human_mouse_correlation_dict['type'] = home_random_type
    concat_dict = {}
    for k, v in random_human_mouse_correlation_dict.items():
        concat_dict[k] = human_mouse_correlation_dict[k] + random_human_mouse_correlation_dict[k]
    data_df = pd.DataFrame.from_dict(concat_dict)

    save_path = resdir + 'figs/human_88/homo_random/'
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

