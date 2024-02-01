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
_C.CAME.path_rawdata1 = '../../../../Brain_ST_human_mouse/data/mouse_brain_region_67_sagittal.h5ad'#'../../../../Brain_ST_human_mouse/data/mouse_brain_region_67_sagittal.h5ad'
_C.CAME.path_rawdata2 = '../../../../Brain_ST_human_mouse/data/human_brain_region_88_sparse_with3d.h5ad'#'../../../../Brain_ST_human_mouse/data/human_brain_region_88_sparse_with3d.h5ad'
#_C.CAME.ROOT = '../../../../CAME/brain_mouse_human_sagittal/Baron_mouse-Baron_human-(10-13_15.26.12)/'
#_C.CAME.figdir = '../../../../CAME/analysis_results/Baron_mouse-Baron_human-(10-13_15.26.12)/figs/' #
_C.CAME.ROOT = '../../../../CAME/analysis_results/Baron_mouse-Baron_human-(11-25_09.48.43)/'
_C.CAME.figdir = '../../../../CAME/analysis_results/Baron_mouse-Baron_human-(11-25_09.48.43)/figs/'
_C.CAME.embedding_dim = 128

_C.CAME.homo_region_file_path = '../../../../CAME/brain_human_mouse/MouseHumanMatches_H88M67_10regions.csv'#'../../../../CAME/brain_human_mouse/MouseHumanMatches_H88M67.csv'
_C.CAME.labels_dir = '../../../../CAME/brain_human_mouse/'
_C.CAME.labels_mouse_file = 'mouse_67_label.csv'
_C.CAME.labels_human_file = 'human_88_label.csv'

_C.HECO = CN()
# Could be pca or came

_C.HECO.dsnames = ['Mouse', 'Human']

_C.HECO.normalize_before_pca = None #'default'  # None represent no normalization
_C.HECO.normalize_before_pca_target_sum = None
_C.HECO.embedding_type = 'pca' # pca_sample_pass_gene, came
_C.HECO.embedding_pca_dim = 30

_C.HECO.homo_region_num = 10

_C.HECO.dataset = 'mouse_human_binary'
_C.HECO.mouse_dataset = '2006nature' # 2020sa
_C.HECO.result_save_folder = './results_{}/'.format(_C.HECO.homo_region_num)
_C.HECO.experiment_time = time.strftime("%Y-%m-%d_%H-%M-%S")#
_C.HECO.result_save_path = _C.HECO.result_save_folder + _C.HECO.experiment_time
_C.HECO.embeddings_file_path = _C.HECO.result_save_path + "/embeds/"
_C.HECO.DATA_PATH = _C.HECO.result_save_path + '/data/'

_C.HECO.normalize_before_pruning_method = 'default'
_C.HECO.pruning_target_sum = None # None
_C.HECO.pruning_normalize_axis = 0
_C.HECO.normalize_before_pruning_method_1 = 'default'
_C.HECO.pruning_target_sum_1 = None # None
_C.HECO.pruning_normalize_axis_1 = 0
_C.HECO.normalize_before_pruning_method_2 = 'default'
_C.HECO.pruning_target_sum_2 = None # None
_C.HECO.pruning_normalize_axis_2 = 0

_C.HECO.normalize_scale = True

_C.HECO.if_threshold = True
_C.HECO.pruning_method = 'std'  # top, std, quantile
_C.HECO.pruning_std_times_sm = 3 #3.6#2.9
_C.HECO.pruning_std_times_vh = 2.3 #2.8

_C.HECO.sm_gene_top = 5#100
_C.HECO.vh_gene_top = 5#20
_C.HECO.sm_sample_top = 5
_C.HECO.vh_sample_top = 5

_C.HECO.NODE_TYPE_NUM = 2
_C.HECO.S = 25431 # 21749 + 3682
_C.HECO.S_sample_rate = [1]
_C.HECO.M = 3615#4035 + 6507
_C.HECO.M_sample_rate = [3]

_C.HECO.binary_S = 21749 #
_C.HECO.binary_M = 1709#4035
_C.HECO.binary_H = 1906#6507
_C.HECO.binary_V = 3682


_C.HECO.DEG_batch_key = None
_C.HECO.DEG_n_top_genes = 5000

_C.HECO.positive_sample_number = 5000

_C.HECO.fig_format = 'png'

_C.ANALYSIS = CN()
_C.ANALYSIS.cut_ov = 0

_C.ANALYSIS.umap_neighbor = 30
_C.ANALYSIS.mouse_umap_neighbor = 40
_C.ANALYSIS.human_umap_neighbor = 20


_C.HECO_args = CN()
_C.HECO_args.save_emb = True
_C.HECO_args.turn = 0
_C.HECO_args.dataset = _C.HECO.dataset
_C.HECO_args.target_node = "S" # S, M, H, V
_C.HECO_args.if_pretrained = False
_C.HECO_args.pretrained_model_path = "./results/2022-11-30_17-18-47/"
_C.HECO_args.save_path = "./results/" + _C.HECO.experiment_time+'/'#"../data/{}/results/".format(_C.HECO_args.dataset)+_C.HECO.experiment_time+'/'
_C.HECO_args.data_path = "./results/"+_C.HECO.experiment_time+'/data/'#"../data/{}/results/".format(_C.HECO_args.dataset)+_C.HECO.experiment_time+'/data/'
_C.HECO_args.ratio = [20, 40, 60]
_C.HECO_args.gpu = 0
_C.HECO_args.seed = 53
_C.HECO_args.hidden_dim = 128
_C.HECO_args.nb_epochs = 1000
# The parameters of evaluation
_C.HECO_args.eva_lr = 0.01
_C.HECO_args.eva_wd = 0
# The parameters of learning process
_C.HECO_args.patience = 30
_C.HECO_args.lr = 0.0005
_C.HECO_args.l2_coef = 0
# model-specific parameters
_C.HECO_args.tau = 0.9
_C.HECO_args.feat_drop = 0.4
_C.HECO_args.attn_drop = 0.35
_C.HECO_args.sample_rate = [6]
_C.HECO_args.lam = 0.5

_C.HECO_args.type_num = [25431, 3615]
_C.HECO_args.nei_num = 1


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


