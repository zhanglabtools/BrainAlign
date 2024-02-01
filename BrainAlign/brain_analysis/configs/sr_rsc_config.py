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

_C.BrainAlign = CN()

_C.BrainAlign.species_list = ('Bonobo', 'Chimpanzee', 'Human', 'Macaque')
_C.BrainAlign.species_color = ('#CD4537', '#28AF60', '#ED7D31', '#8A4AA2')

# The adata file order should be consistant with the species name order
_C.BrainAlign.path_rawdata_list = tuple('../gb_2020/bulk/adatas/'+ species +'.h5ad' for species in _C.BrainAlign.species_list)
_C.BrainAlign.do_normalize = [False, False, False, False]
_C.BrainAlign.key_class_list = ('region_name', 'region_name', 'region_name', 'region_name')   # set by user

# gene multi-multi homology relations, and the homology file order should be consistant with the species name order
_C.BrainAlign.path_varmap_list = ('../gb_2020/bulk/homologous_genes/bonobo_homologous2_chimpanzee_human_macaque.tsv',
                                  '../gb_2020/bulk/homologous_genes/chimpanzee_homologous2_bonobo_human_macaque.tsv',
                                  '../gb_2020/bulk/homologous_genes/human_homologous2_bonobo_chimpanzee_macaque.tsv',
                                  '../gb_2020/bulk/homologous_genes/macaque_homologous2_bonobo_chimpanzee_human.tsv')
#

_C.Preprocess = CN()
_C.Preprocess.n_top_genes = 1000 # int
_C.Preprocess.nneigh_scnet = 5 # int
_C.Preprocess.nneigh_clust = 30 # int

_C.Preprocess.key_clust = 'clust_lbs' # str
# _C.Preprocess.deg_cuts = {} # dict with keys 'cut_padj', 'cut_pts', and 'cut_logfc'
_C.Preprocess.ext_feats = None # Optional[Sequence[Sequence]],
# A tuple of lists of variable names. Extra variables (genes) to be added to the auto-selected ones as the **observation(cell)-node features**.

_C.Preprocess.ext_nodes = None # Optional[Sequence[Sequence]],
# A tuple of two lists of variable names. Extra variables (genes) to be added to the auto-selected ones as the **variable(gene)-nodes**.

_C.Preprocess.norm_target_sum = None #float, the scale factor for library-size normalization

_C.Preprocess.sparse = False
_C.Preprocess.quantile_gene= 0.5
_C.Preprocess.quantile_sample = 0.99
_C.Preprocess.embedding_size = 128
# Whether to use the single-cell networks
_C.Preprocess.use_scnets = True
# The number of top DEGs to take as the node-features of each cells.
# You set it 70-100 for distant species pairs.
_C.Preprocess.ntop_deg = 70
# The number of top DEGs to take as the graph nodes, which can be directly displayed on the UMAP plot.
_C.Preprocess.ntop_deg_nodes = 50
# The source of the node genes; use both DEGs and HVGs by default
_C.Preprocess.node_source = 'deg,hvg'

_C.Preprocess.n_pcs = 30 # the number of PCs for computing the single-cell-network

_C.Preprocess.union_var_nodes = True # whether to take the union of the variable-nodes
_C.Preprocess.union_node_feats = True # whether to take the union of the observation-node-features

_C.Preprocess.with_single_vnodes = True # whether to include the varibales (node) that are ocurred in only one of the datasets
_C.Preprocess.keep_non1v1_feats = True # whether to take into account the non-1v1 variables as the node features.
_C.Preprocess.col_weight = None # str, A column in ``df_varmap`` specifying the weights between homologies.

_C.Preprocess.if_feature_pca = True#False # whether do pca on node input features
_C.Preprocess.feature_pca_dim = 128 # pca dimension on node input features


_C.BrainAlign.normalize_before_pca = None #'default'  # None represent no normalization
_C.BrainAlign.normalize_before_pca_target_sum = None
_C.BrainAlign.embedding_type = 'pca' # pca_sample_pass_gene, came
_C.BrainAlign.embedding_pca_dim = 30

_C.BrainAlign.homo_region_num = 10

_C.BrainAlign.dataset = 'gb_2020'
_C.BrainAlign.method = 'srrsc' # heco
_C.BrainAlign.result_save_folder = './results_{}_{}genes/'.format(_C.BrainAlign.dataset, int(_C.Preprocess.n_top_genes))
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
_C.BrainAlign.S_list = []
_C.BrainAlign.S_sample_rate = [1]
_C.BrainAlign.M = 3615#4035 + 6507
_C.BrainAlign.M_list = []
_C.BrainAlign.M_sample_rate = [3]

_C.BrainAlign.binary_S = 21749 #
_C.BrainAlign.binary_M = 1709#4035
_C.BrainAlign.binary_H = 1906#6507
_C.BrainAlign.binary_V = 3682

_C.BrainAlign.node_relation = 'knn'#'spatial' # knn
_C.BrainAlign.spatial_node_neighbor = 5

_C.BrainAlign.DEG_batch_key = None
_C.BrainAlign.DEG_n_top_genes = 1000

_C.BrainAlign.positive_sample_number = 5000

_C.BrainAlign.fig_format = 'png'
_C.BrainAlign.fig_dpi = 500

_C.ANALYSIS = CN()
_C.ANALYSIS.cut_ov = 0

_C.ANALYSIS.umap_neighbor = 15 #30
_C.ANALYSIS.mouse_umap_neighbor = 15 # 40
_C.ANALYSIS.human_umap_neighbor = 15
_C.ANALYSIS.min_dist = 0.5

#_C.ANALYSIS.sample_umap_markersize = 0

_C.ANALYSIS.genes_umap_neighbor = 15 #30
_C.ANALYSIS.genes_min_dist = 0.5

_C.ANALYSIS.umap_homo_random = 30#10

_C.ANALYSIS.cluster_markergenes = 20#10
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
_C.SRRSC_args.lr = 0.02
_C.SRRSC_args.patience = 50
_C.SRRSC_args.hid_units = 256
_C.SRRSC_args.hid_units2 = 128
_C.SRRSC_args.out_ft = 128
_C.SRRSC_args.drop_prob = 0.5
_C.SRRSC_args.lamb = 0.5 # coefficient for the losses in node task
_C.SRRSC_args.lamb_lp = 1.0 # coefficient for the losses in link task
_C.SRRSC_args.margin = 0.8 # coefficient for the margin loss
_C.SRRSC_args.isBias = True
_C.SRRSC_args.isAtt = True # if use attention
_C.SRRSC_args.isLP = False
_C.SRRSC_args.isSemi = False

_C.SRRSC_args.train_ratio = 0.5
_C.SRRSC_args.validation_ratio = 0.45
_C.SRRSC_args.test_ratio = 0.05

_C.SRRSC_args.seed = 100

_C.SRRSC_args.lr_stepsize = 10
_C.SRRSC_args.lr_gamma = 0.5

_C.SRRSC_args.device = 'cpu'
_C.SRRSC_args.labels = None   # {node_type: labels} refer to the sequence of [n, node_cnt[node_type]]
_C.SRRSC_args.node_type = None
_C.SRRSC_args.ft_size = None
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


