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

_C.CAME.homo_region_file_path = '../../../../CAME/brain_human_mouse/MouseHumanMatches_H88M67_all.csv'  # '../../../../CAME/brain_human_mouse/MouseHumanMatches_H88M67.csv'
_C.CAME.labels_dir = '../../../../CAME/brain_mouse_2020sa/'
_C.CAME.labels_mouse_file = 'mouse_region_list_64.csv'
_C.CAME.labels_human_file = 'human_88_label.csv'

_C.CAME.parent_labels_mouse_file = 'mouse_parent_region_list_15.csv'
_C.CAME.parent_labels_human_file = 'human_16_label.csv'

_C.CAME.path_varmap = '../../../../CAME/brain_human_mouse/gene_matches_mouse2human.csv'
_C.CAME.path_varmap_1v1 = '../../../../CAME/brain_human_mouse/gene_matches_1v1_mouse2human.csv'

_C.BrainAlign = CN()
# Could be pca or came

# palette gene file
_C.BrainAlign.palette_mouse_file = '../../../../CAME/brain_mouse_2020sa/mouse_gene_palette/genes-list-266.tsv'
_C.BrainAlign.palette_human_file = '../../../../CAME/brain_mouse_2020sa/human_gene_palette/gene-list.tsv'

_C.BrainAlign.used_data_path = '../../../used_data/'

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

_C.BrainAlign.node_relation = 'knn'#'spatial' # knn
_C.BrainAlign.spatial_node_neighbor = 1

_C.BrainAlign.DEG_batch_key = None
_C.BrainAlign.DEG_n_top_genes = 5000

_C.BrainAlign.positive_sample_number = 5000

_C.BrainAlign.fig_format = 'png'
_C.BrainAlign.fig_dpi = 500

_C.ANALYSIS = CN()
_C.ANALYSIS.cut_ov = 0

_C.ANALYSIS.umap_neighbor = 15 #30
_C.ANALYSIS.mouse_umap_neighbor = 15 # 40
_C.ANALYSIS.human_umap_neighbor = 15

_C.ANALYSIS.umap_marker_size = 15 # 40
_C.ANALYSIS.mouse_umap_marker_size = 15 # 40
_C.ANALYSIS.human_umap_marker_size = 15 # 40

_C.ANALYSIS.genes_umap_neighbor = 15 #30

_C.ANALYSIS.umap_homo_random = 30#10

_C.ANALYSIS.cluster_markergenes = 20#10

_C.ANALYSIS.sample_cluster_num = 30
#_C.ANALYSIS.method_list = ['CAME', 'Seurat', 'LIGER', 'Harmony', 'Scanorama', 'BBKNN', 'SAMap', 'BrainAlign']

_C.SRRSC_args = CN()
_C.SRRSC_args.save_emb = True
_C.SRRSC_args.dataset = _C.BrainAlign.dataset
_C.SRRSC_args.if_pretrained = False
_C.SRRSC_args.pretrained_model_path = "./results/2022-11-30_17-18-47/"
_C.SRRSC_args.save_path = "./results/" + _C.BrainAlign.experiment_time+'/'#"../data/{}/results/".format(_C.BrainAlign_args.dataset)+_C.BrainAlign.experiment_time+'/'
_C.SRRSC_args.data_path = "./results/"+_C.BrainAlign.experiment_time+'/data/'#"../data/{}/results/".format(_C.BrainAlign_args.dataset)+_C.BrainAlign.experiment_time+'/data/'

_C.SRRSC_args.gpu_num = '0'
_C.SRRSC_args.model = 'SubHIN'
_C.SRRSC_args.dataset = 'mouse_human_binary'
_C.SRRSC_args.nb_epochs = 100
_C.SRRSC_args.lr = 0.001
_C.SRRSC_args.patience = 50
_C.SRRSC_args.hid_units = 256
_C.SRRSC_args.hid_units2 = 128
_C.SRRSC_args.out_ft = 128
_C.SRRSC_args.drop_prob = 0.0
_C.SRRSC_args.lamb = 0.5 # coefficient for the losses in node task
_C.SRRSC_args.lamb_lp = 1.0 # coefficient for the losses in link task
_C.SRRSC_args.margin = 0.8 # coefficient for the margin loss
_C.SRRSC_args.isBias = False
_C.SRRSC_args.isAtt = False # if use attention
_C.SRRSC_args.isLP = False
_C.SRRSC_args.isSemi = False

_C.SRRSC_args.train_ratio = 0.85
_C.SRRSC_args.validation_ratio = 0.12
_C.SRRSC_args.test_ratio = 0.03

_C.SRRSC_args.seed = 100

_C.SRRSC_args.lr_stepsize = 5
_C.SRRSC_args.lr_gamma = 0.9

_C.SRRSC_args.device = 'cpu'
#_C.SRRSC_args.node2id = {}
_C.SRRSC_args.labels = None   # {node_type: labels} refer to the sequence of [n, node_cnt[node_type]]
#_C.SRRSC_args.nt_rel = None   # {note_type: [rel1, rel2]}
#_C.SRRSC_args.node_cnt = None  # {note_type: nb}
_C.SRRSC_args.node_type = None
_C.SRRSC_args.ft_size = None
#_C.SRRSC_args.node_size = None
_C.SRRSC_args.rel_types = None

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


