# -- coding: utf-8 --
# @Time : 2022/10/15 11:20
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : homo_random_config.py


from yacs.config import CfgNode as CN

# --------------------------------------------------------------
# Config of model
# --------------------------------------------------------------
_C = CN()

_C.CAME = CN()
_C.CAME.path_rawdata1 = '../../Brain_ST_human_mouse/data/mouse_brain_region_67_sagittal.h5ad'#'../../Brain_ST_human_mouse/data/mouse_brain_region_67_sparse_no_threshold.h5ad'
_C.CAME.path_rawdata2 = '../brain_human_mouse/human_brain_region_88_sparse.h5ad'

_C.CAME.path_labels_1 = '../brain_human_mouse/mouse_67_label.csv'
_C.CAME.path_labels_2 = '../brain_human_mouse/human_88_label_origin.csv'

_C.CAME.embedding_path = None

_C.CAME.path_varmap = '../came/sample_data/gene_matches_mouse2human.csv'
_C.CAME.path_varmap_1v1 = '../came/sample_data/gene_matches_1v1_mouse2human.csv'

_C.CAME.human_mouse_homo_region = '../brain_human_mouse/MouseHumanMatches_H88M67.csv'

_C.CAME.species_name_list = ['Mouse', 'Human']
_C.CAME.annotation_name = ['region_name', 'region_name']

_C.CAME.learning_label = ['region_name', 'region_name']

_C.CAME.n_top_genes = 5000
_C.CAME.do_normalize = [True, True]

_C.CAME.sparse = False
_C.CAME.quantile_gene= 0.5
_C.CAME.quantile_sample = 0.99

_C.CAME.embedding_size = 128

_C.CAME.preclustering_resolution = 3


_C.TRAINING = CN()
_C.TRAINING.n_epochs = 300

# The training batch size
# When the GPU memory is limited, set 4096 or more if possible.
_C.TRAINING.batch_size = 2048
# The number of epochs to skip for checkpoint backup
_C.TRAINING.n_pass = 50
# The number of top DEGs to take as the node-features of each cells.
# You set it 70-100 for distant species pairs.
_C.TRAINING.ntop_deg = 70

# The number of top DEGs to take as the graph nodes, which can be directly displayed on the UMAP plot.
_C.TRAINING.ntop_deg_nodes = 50


_C.ANALYSIS = CN()
_C.ANALYSIS.cut_ov = 0
_C.ANALYSIS.umap_neighbor = 20 #30
_C.ANALYSIS.mouse_umap_neighbor = 20 #40
_C.ANALYSIS.human_umap_neighbor = 20


_C.CAME.ROOT = '../analysis_results/'

_C.CAME.visible_device = '0'


_C.PROCESS = CN()
_C.PROCESS.path_rawdata1 = '../../Brain_ST_human_mouse/data/mouse_brain_region_67_sagittal.h5ad'
_C.PROCESS.path_rawdata2 = '../../Brain_ST_human_mouse/data/human_brain_region_88_sparse_with3d.h5ad'

_C.PROCESS.path_mouse_labels = '../brain_human_mouse/mouse_67_label_10regions.csv'
_C.PROCESS.path_human_labels = '../brain_human_mouse/human_88_label_10regions.csv'

_C.PROCESS.path_rawdata1_part = '../../Brain_ST_human_mouse/data/10regions_mouse_brain_region_67_sagittal.h5ad'
_C.PROCESS.path_rawdata2_part = '../../Brain_ST_human_mouse/data/10regions_human_brain_region_88_sparse_with3d.h5ad'


# --------------------------------------------------------------
# Config of INPUT
# --------------------------------------------------------------
_C.HOMO_RANDOM = CN()


