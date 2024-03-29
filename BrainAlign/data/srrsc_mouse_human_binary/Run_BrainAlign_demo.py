# -- coding: utf-8 --
# @Time : 2022/10/18 16:19
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : run_analysis.py

import sys
sys.path.append('../../../')
#from BrainAlign.code.utils import set_params
from BrainAlign.brain_analysis.configs import sr_rsc_config_binary
from BrainAlign.brain_analysis.data_utils import load_dimensions_binary
from BrainAlign.brain_analysis import process
from BrainAlign.SR_RSC import main_sr_rsc
from BrainAlign.brain_analysis.analysis import load_srrsc_embeddings

from BrainAlign.brain_analysis.analysis_main import alignment_STs, alignment_STs_analysis,\
    gene_comparison
from BrainAlign.brain_analysis.analysis_anatomical import anatomical_STs_analysis
from BrainAlign.brain_analysis.analysis_genomic import genomic_analysis
from BrainAlign.brain_analysis.process import get_spatial_relation
from BrainAlign.brain_analysis.analysis_spatial import spatial_analysis
import time
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    cfg = sr_rsc_config_binary._C
    cfg.SRRSC_args.if_pretrained = False
    #cfg.freeze()
    cfg.CAME.path_rawdata1 = '../../../BrainAlign/demo/mouse_2020sa_64regions_demo.h5ad'
    cfg.CAME.path_rawdata2 = '../../../BrainAlign/demo/human_brain_region_88_sparse_with3d.h5ad'
    cfg.CAME.ROOT = '../../../run_came/analysis_results/mouse_2020sa/Baron_mouse-Baron_human-02-02_00.13.38/'#Baron_mouse-Baron_human-05-23_21.33.44/ #Baron_mouse-Baron_human-01-03_19.40.04
    cfg.CAME.figdir = '../../../run_came/analysis_results/mouse_2020sa/Baron_mouse-Baron_human-02-02_00.13.38/figs/'

    cfg.CAME.homo_region_file_path = '../../../run_came/brain_human_mouse/MouseHumanMatches_H88M67_all.csv'  # '../../../../CAME/brain_human_mouse/MouseHumanMatches_H88M67.csv'
    cfg.CAME.labels_dir = '../../../run_came/brain_mouse_2020sa/'
    cfg.CAME.labels_mouse_file = 'mouse_region_list_64.csv'
    cfg.CAME.labels_human_file = 'human_88_label.csv'

    cfg.CAME.parent_labels_mouse_file = 'mouse_parent_region_list_15.csv'
    cfg.CAME.parent_labels_human_file = 'human_16_label.csv'

    cfg.BrainAlign.homo_region_num = 20

    cfg.BrainAlign.embedding_type = 'came'
    cfg.BrainAlign.embedding_pca_dim = 30


    cfg.BrainAlign.dataset = 'mouse_human_binary'
    cfg.BrainAlign.result_save_folder = './results_{}_1000genes_all_came_selfloop/'.format(cfg.BrainAlign.homo_region_num)
    cfg.BrainAlign.experiment_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    cfg.BrainAlign.result_save_path = cfg.BrainAlign.result_save_folder + cfg.BrainAlign.experiment_time
    cfg.BrainAlign.embeddings_file_path = cfg.BrainAlign.result_save_path + "/embeds/"
    cfg.BrainAlign.DATA_PATH = cfg.BrainAlign.result_save_path + '/data/'

    cfg.SRRSC_args.save_path = "./results_{}_1000genes_all_came_selfloop/".format(
        cfg.BrainAlign.homo_region_num) + cfg.BrainAlign.experiment_time + '/'  # "../data/{}/results/".format(_C.HECO_args.dataset)+_C.HECO.experiment_time+'/'
    cfg.SRRSC_args.data_path = "./results_{}_1000genes_all_came_selfloop/".format(cfg.BrainAlign.homo_region_num) + cfg.BrainAlign.experiment_time + '/data/'

    S, M, binary_S, binary_M, binary_H, binary_V = load_dimensions_binary(cfg.CAME.ROOT + 'datapair_init.pickle')

    cfg.BrainAlign.S = S
    cfg.BrainAlign.M = M

    cfg.BrainAlign.binary_S = binary_S
    cfg.BrainAlign.binary_M = binary_M
    cfg.BrainAlign.binary_H = binary_H
    cfg.BrainAlign.binary_V = binary_V

    cfg.BrainAlign.node_relation = 'spatial'  # 'spatial' # knn
    cfg.BrainAlign.spatial_node_neighbor = 5

    cfg.ANALYSIS.umap_neighbor = 30  # 30
    cfg.ANALYSIS.mouse_umap_neighbor = 30  # 40
    cfg.ANALYSIS.human_umap_neighbor = 30

    cfg.ANALYSIS.umap_marker_size = 5  # 40
    cfg.ANALYSIS.mouse_umap_marker_size = 5  # 40
    cfg.ANALYSIS.human_umap_marker_size = 10  # 40

    cfg.ANALYSIS.genes_umap_neighbor = 15#15

    cfg.ANALYSIS.sample_cluster_num = 60

    cfg.ANALYSIS.cut_ov = 0

    cfg.SRRSC_args.lr = 0.02
    cfg.SRRSC_args.patience = 30

    cfg.SRRSC_args.out_ft = 128

    cfg.SRRSC_args.drop_prob = 0.7
    cfg.SRRSC_args.lamb = 0.5  # coefficient for the losses in node task
    cfg.SRRSC_args.lamb_lp = 1.0  # coefficient for the losses in link task
    cfg.SRRSC_args.margin = 0.5 #0.8

    cfg.BrainAlign.fig_format = 'png'

    cfg.SRRSC_args.isAtt = True#True#True
    cfg.SRRSC_args.isBias = True

    cfg.SRRSC_args.lamb = 0.5  # coefficient for the losses in node task
    cfg.SRRSC_args.lamb_lp = 1.0  # coefficient for the losses in link task
    cfg.SRRSC_args.margin = 0.8  # coefficient for the margin loss

    cfg.SRRSC_args.train_ratio = 0.5
    cfg.SRRSC_args.validation_ratio = 0.45
    cfg.SRRSC_args.test_ratio = 0.05

    cfg.SRRSC_args.lr_stepsize = 10
    cfg.SRRSC_args.lr_gamma = 0.5

    cfg.SRRSC_args.nb_epochs = 100

    print('--------------------------Config:', cfg)
    #print('--------------------------Config:', cfg)
    logger = process.get_srrsc_input(cfg)
    print('Training BrainAlign...')
    main_sr_rsc.run_srrsc(cfg, logger)

    load_srrsc_embeddings(cfg)  # (21749, 128)

    print('Analysis of BrainAlign embeddings...')
    #alignment_STs(cfg)
    alignment_STs_analysis_obj = alignment_STs_analysis(cfg)
    #alignment_STs_analysis_obj.forward()
    alignment_STs_analysis_obj.experiment_1_cross_species_clustering()
    alignment_STs_analysis_obj.experiment_2_umap_evaluation()
    alignment_STs_analysis_obj.experiment_2_umap_diagram()
    alignment_STs_analysis_obj.experiment_3_homo_random()
    alignment_STs_analysis_obj.experiment_3_1_homo_random_beforeAlign()
    # self.experiment_3_homo_random()
    alignment_STs_analysis_obj.experiment_4_cross_species_genes_analysis()
    alignment_STs_analysis_obj.experiment_5_genes_homo_random()
    # alignment_STs_analysis_obj.experiment_6_brain_region_classfier()
    # alignment_STs_analysis_obj.experiment_7_align_cross_species()
    # alignment_STs_analysis_obj.experiment_7_1_align_cross_species_split()
    # alignment_STs_analysis_obj.experiment_7_2_align_cross_species_alignementscore()
    #
    # alignment_STs_analysis_obj.experiment_7_3_beforealign_cross_species_split()
    #
    # alignment_STs_analysis_obj.experiment_8_cross_evaluation_aligment_cluster()
    # alignment_STs_analysis_obj.experiment_8_1_cross_aligment_cluster_merge_lines()
    # alignment_STs_analysis_obj.experiment_9_name_clusters()
    # alignment_STs_analysis_obj.experiment_9_1_region_cluster_heatmap()
    # alignment_STs_analysis_obj.experiment_10_check_alignment()
    # alignment_STs_analysis_obj.experiment_10_1_check_alignment_plot()
    # alignment_STs_analysis_obj.experiment_11_paga()
    #
    #
    # #alignment_STs_analysis_obj.forward()
    # genomic_analysis_obj = genomic_analysis(cfg)
    # genomic_analysis_obj.experiment_1_th_clusters_deg()
    # genomic_analysis_obj.experiment_2_clusters_deg()
    # genomic_analysis_obj.experiment_2_1_homoregions_deg()
    #
    # genomic_analysis_obj.experiment_2_3_homoregions_deg_proportion()
    #
    # genomic_analysis_obj.experiment_2_1_homoregions_deg_homo()
    #
    # genomic_analysis_obj.experiment_2_2_homoregions_multiple_deg()
    #
    # genomic_analysis_obj.experiment_3_regions_gene_module()
    # genomic_analysis_obj.experiment_3_1_regions_gene_module_analysis()
    # genomic_analysis_obj.experiment_4_clusters_gene_module()
    # genomic_analysis_obj.experiment_5_deg_homo_distribution()
    # genomic_analysis_obj.experiment_5_1_gene_module_homogenes_ratio()
    #
    # genomic_analysis_obj.experiment_6_regions_gene_module_average_exp()
    # genomic_analysis_obj.experiment_6_1_regions_gene_module_cross()
    # genomic_analysis_obj.experiment_7_clusters_gene_module_average_exp()
    # genomic_analysis_obj.experiment_7_1_clusters_gene_module_cross()
    # genomic_analysis_obj.experiment_8_homoregions_go_terms_overlap()
    # genomic_analysis_obj.experiment_8_1_plot_heatmap()
    #
    # genomic_analysis_obj.experiment_9_clusters_go_terms_overlap()
    # genomic_analysis_obj.experiment_9_1_plot_heatmap()
    # genomic_analysis_obj.experiment_10_binary_clustering()
    # genomic_analysis_obj.experiment_10_1_binary_clustering_multiple_genesets()
    #
    # #gene_comparison(cfg)
    # anatomical_STs_analysis_obj = anatomical_STs_analysis(cfg)
    # anatomical_STs_analysis_obj.forward()
    # spatial_analysis_obj = spatial_analysis(cfg)
    # spatial_analysis_obj.experiment_1_spatial_hippocampal()
    # #spatial_analysis_obj.forward()
    # spatial_analysis_obj.experiment_2_spatial_isocortex()
    # spatial_analysis_obj.experiment_3_spatial_clusters()



