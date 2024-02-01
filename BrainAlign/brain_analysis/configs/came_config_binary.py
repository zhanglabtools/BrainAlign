# -- coding: utf-8 --
# @Time : 2022/10/15 16:11
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : heco_config.py
import time
from yacs.config import CfgNode as CN

# --------------------------------------------------------------
# Config of model
# --------------------------------------------------------------
_C = CN()

_C.CAME = CN()
_C.CAME = CN()
_C.CAME.path_rawdata1 = '../../../../CAME/brain_mouse_2020sa/6regions_mouse_2020sa_64regions.h5ad'
_C.CAME.path_rawdata2 = '../../../../Brain_ST_human_mouse/data/6regions_human_brain_region_88_sparse_with3d.h5ad'
_C.CAME.ROOT = '../../../../CAME/analysis_results/mouse_2020sa/Baron_mouse-Baron_human-12-22_09.28.19/'
_C.CAME.figdir = '../../../../CAME/analysis_results/mouse_2020sa/Baron_mouse-Baron_human-12-22_09.28.19/figs/'

_C.CAME.homo_region_file_path = '../../../../CAME/brain_human_mouse/MouseHumanMatches_H88M67_6regions_2020sa.csv'  # '../../../../CAME/brain_human_mouse/MouseHumanMatches_H88M67.csv'
_C.CAME.labels_dir = '../../../../CAME/brain_human_mouse/'
_C.CAME.labels_mouse_file = 'mouse_67_label_6regions_2020sa.csv'
_C.CAME.labels_human_file = 'human_88_label_6regions.csv'

_C.CAME.parent_labels_mouse_file = 'mouse_parent_region_list_15.csv'
_C.CAME.parent_labels_human_file = 'human_16_label.csv'

_C.BrainAlign = CN()
# Could be pca or came

_C.BrainAlign.dsnames = ['Mouse', 'Human']

_C.BrainAlign.normalize_before_pca = None #'default'  # None represent no normalization
_C.BrainAlign.normalize_before_pca_target_sum = None
_C.BrainAlign.embedding_type = 'pca' # pca_sample_pass_gene, came
_C.BrainAlign.embedding_pca_dim = 30

_C.BrainAlign.homo_region_num = 10

_C.BrainAlign.dataset = 'mouse_human_binary'
_C.BrainAlign.mouse_dataset = '2020sa'
_C.BrainAlign.method = 'srrsc' # heco
_C.BrainAlign.result_save_folder = './results_{}/'.format(_C.BrainAlign.homo_region_num)
_C.BrainAlign.experiment_time = time.strftime("%Y-%m-%d_%H-%M-%S")#
_C.BrainAlign.result_save_path = _C.BrainAlign.result_save_folder + _C.BrainAlign.experiment_time
_C.BrainAlign.embeddings_file_path = _C.BrainAlign.result_save_path + "/embeds/"
_C.BrainAlign.DATA_PATH = _C.BrainAlign.result_save_path + '/data/'

_C.BrainAlign.normalize_before_pruning_method = 'default'
_C.BrainAlign.pruning_target_sum = None # None
_C.BrainAlign.pruning_normalize_axis = 0

_C.BrainAlign.normalize_before_pruning_method_1 = None
_C.BrainAlign.pruning_target_sum_1 = None # None
_C.BrainAlign.pruning_normalize_axis_1 = 0
_C.BrainAlign.normalize_before_pruning_method_2 = 'default'
_C.BrainAlign.pruning_target_sum_2 = None # None
_C.BrainAlign.pruning_normalize_axis_2 = 0

_C.BrainAlign.normalize_scale = True

_C.BrainAlign.if_threshold = True
_C.BrainAlign.pruning_method = 'std'  # top, std, quantile
_C.BrainAlign.pruning_std_times_sm = 3 #3.6#2.9
_C.BrainAlign.pruning_std_times_vh = 2.3 #2.8

_C.BrainAlign.sm_gene_top = 5#100
_C.BrainAlign.vh_gene_top = 5#20
_C.BrainAlign.sm_sample_top = 5
_C.BrainAlign.vh_sample_top = 5

_C.BrainAlign.NODE_TYPE_NUM = 2
_C.BrainAlign.S = 25431 # 21749 + 3682
_C.BrainAlign.S_sample_rate = [1]
_C.BrainAlign.M = 3615#4035 + 6507
_C.BrainAlign.M_sample_rate = [3]

_C.BrainAlign.binary_S = 21749 #
_C.BrainAlign.binary_M = 1709#4035
_C.BrainAlign.binary_H = 1906#6507
_C.BrainAlign.binary_V = 3682


_C.BrainAlign.DEG_batch_key = None
_C.BrainAlign.DEG_n_top_genes = 5000

_C.BrainAlign.positive_sample_number = 5000

_C.BrainAlign.fig_format = 'png'
_C.BrainAlign.fig_dpi = 500

_C.ANALYSIS = CN()
_C.ANALYSIS.cut_ov = 0

_C.ANALYSIS.umap_neighbor = 30 #30
_C.ANALYSIS.mouse_umap_neighbor = 30 # 40
_C.ANALYSIS.human_umap_neighbor = 30

_C.ANALYSIS.umap_marker_size = 15 # 40
_C.ANALYSIS.mouse_umap_marker_size = 15 # 40
_C.ANALYSIS.human_umap_marker_size = 15 # 40

_C.ANALYSIS.genes_umap_neighbor = 15 #30

_C.ANALYSIS.umap_homo_random = 30#10

_C.SRRSC_args = CN()
_C.SRRSC_args.save_emb = True
_C.SRRSC_args.dataset = _C.BrainAlign.dataset
_C.SRRSC_args.if_pretrained = False
_C.SRRSC_args.pretrained_model_path = "./results/2022-11-30_17-18-47/"
_C.SRRSC_args.save_path = "./results/" + _C.BrainAlign.experiment_time+'/'#"../data/{}/results/".format(_C.BrainAlign_args.dataset)+_C.BrainAlign.experiment_time+'/'
_C.SRRSC_args.data_path = "./results/"+_C.BrainAlign.experiment_time+'/data/'#"../data/{}/results/".format(_C.BrainAlign_args.dataset)+_C.BrainAlign.experiment_time+'/data/'


_C.TRAIN = CN()
_C.TRAIN.if_pretrain = False

# --------------------------------------------------------------
# Config of INPUT
# --------------------------------------------------------------
_C.HOMO_RANDOM = CN()

# if use all the non-homogeneous regions as back ground, default 'all', else will only use 35 regions in each species
_C.HOMO_RANDOM.random_field = 'all' # or all
# config if plot all the cross species correlation heatmap, default false; Require large memory if True.
_C.HOMO_RANDOM.random_plot = False # config


