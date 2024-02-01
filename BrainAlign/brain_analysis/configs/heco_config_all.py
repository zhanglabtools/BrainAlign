# -- coding: utf-8 --
# @Time : 2022/10/15 16:11
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : heco_config.py

from yacs.config import CfgNode as CN

# --------------------------------------------------------------
# Config of model
# --------------------------------------------------------------
_C = CN()

_C.CAME = CN()
_C.CAME.path_rawdata1 = '../../../../Brain_ST_human_mouse/data/mouse_brain_region_67_sparse_no_threshold.h5ad'
_C.CAME.path_rawdata2 = '../../../../CAME/brain_human_mouse/human_brain_region_88_sparse.h5ad'
_C.CAME.ROOT = '../../../../CAME/analysis_results/Dense_Baron_mouse-Baron_human-10-24_11.37.58/'
_C.CAME.figdir = '../../../../CAME/analysis_results/Dense_Baron_mouse-Baron_human-10-24_11.37.58/figs/' #
_C.CAME.embedding_dim = 128

_C.CAME.homo_region_file_path = '../../../../CAME/brain_human_mouse/MouseHumanMatches_H88M67.csv'
_C.CAME.labels_dir = '../../../../CAME/brain_human_mouse/'

_C.HECO = CN()
# Could be pca or came
_C.HECO.embedding_type = 'pca'
_C.HECO.embedding_pca_dim = 30
_C.HECO.DATA_PATH = './data/'
_C.HECO.dataset = 'mouse_human'
_C.HECO.result_save_path = './results/2022-11-04_22-37-26'
_C.HECO.embeddings_file_path = _C.HECO.result_save_path + "/embeds/"

_C.HECO.if_threshold = True
_C.HECO.pruning_method = 'std'  # top, std, quantile
_C.HECO.pruning_std_times_sm = 3.5
_C.HECO.pruning_std_times_vh = 3.2

_C.HECO.sm_gene_top = 100
_C.HECO.vh_gene_top = 20
_C.HECO.sm_sample_top = 5
_C.HECO.vh_sample_top = 5

_C.HECO.target_sum = 1 # None

_C.HECO.NODE_TYPE_NUM = 4
_C.HECO.S = 72968
_C.HECO.S_sample_rate = [0.2]
_C.HECO.M = 2578
_C.HECO.M_sample_rate = [5, 2]
_C.HECO.H = 3326
_C.HECO.H_sample_rate = [0.5, 0.5]
_C.HECO.V = 3682
_C.HECO.V_sample_rate = [2]

_C.HECO.DEG_batch_key = None
_C.HECO.DEG_n_top_genes = 2000

_C.HECO.positive_sample_number = 5000


_C.HECO.fig_format = 'png'
# --------------------------------------------------------------
# Config of INPUT
# --------------------------------------------------------------
_C.HOMO_RANDOM = CN()

# if use all the non-homogeneous regions as back ground, default 'all', else will only use 35 regions in each species
_C.HOMO_RANDOM.random_field = 'all' # or all
# config if plot all the cross species correlation heatmap, default false; Require large memory if True.
_C.HOMO_RANDOM.random_plot = False # config

