# -- coding: utf-8 --
# @Time : 2022/12/14 17:06
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : run_srrsc_6regions.py


# -- coding: utf-8 --
# @Time : 2022/10/18 16:19
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : run_analysis.py
import sys
sys.path.append('../../../')
from BrainAlign.code.utils import set_params
from BrainAlign.brain_analysis.configs import sr_rsc_config_binary
from BrainAlign.brain_analysis.data_utils import load_dimensions_binary
from BrainAlign.brain_analysis import process
from BrainAlign.SR_RSC import main_sr_rsc
from BrainAlign.brain_analysis.analysis import load_srrsc_embeddings, \
    homo_corr, \
    random_corr,\
    plot_homo_random,\
    ttest_homo_random, \
    cross_species_binary_clustering, \
    umap_seperate, \
    spatial_alignment, \
    spatial_alignment_mouse, \
    umap_genes_seperate, \
    cross_species_genes_clustering, \
    load_homo_genes, \
    brain_region_classfier,\
    align_cross_species,\
    genes_homo_random_corr,\
    cross_evaluation_aligment, \
    cross_evaluation_aligment_cluster

from BrainAlign.brain_analysis.analysis_main import alignment_STs_analysis,\
    gene_comparison

from BrainAlign.brain_analysis.analysis_anatomical import anatomical_STs_analysis

from BrainAlign.brain_analysis.analysis_genomic import genomic_analysis

from BrainAlign.brain_analysis.process import get_spatial_relation

from BrainAlign.brain_analysis.analysis_spatial import spatial_analysis

import time

import warnings
warnings.filterwarnings("ignore")
import anndata as ad

if __name__ == '__main__':
    cfg = sr_rsc_config_binary._C
    #cfg.BrainAlign.DATA_PATH = './data/'
    #cfg.BrainAlign_args.data_path = "./data/"

    cfg.SRRSC_args.if_pretrained = False
    #cfg.freeze()
    cfg.CAME.path_rawdata1 = './Data/Mouse.h5ad'
    cfg.CAME.path_rawdata2 = './Data/Macaque.h5ad'

    adata_1 = ad.read_h5ad(cfg.CAME.path_rawdata1)
    print(adata_1)
    adata_1.obs['region_name'] = adata_1.obs['gene_area']
    adata_1 = adata_1[adata_1.obs['annotation'].isin(['Mouse_CA1', 'Mouse_CA2', 'Mouse_CA3', 'Mouse_DG'])]
    adata_1.write_h5ad(cfg.CAME.path_rawdata1)
    print(adata_1.obs['gene_area'].unique())
    adata_2 = ad.read_h5ad(cfg.CAME.path_rawdata2)
    adata_2.obs['region_name'] = adata_2.obs['gene_area']
    adata_1 = adata_1[adata_1.obs['annotation'].isin(['Macaque_CA1', 'Macaque_CA2', 'Macaque_CA3', 'Macaque_DG'])]
    adata_2.write_h5ad(cfg.CAME.path_rawdata2)
    print(adata_2)
    print(adata_2.obs['gene_area'].unique())


    cfg.CAME.ROOT = '../../../../CAME/analysis_results/mouse_2020sa/Baron_mouse-Baron_human-06-03_14.38.27/'#Baron_mouse-Baron_human-05-23_21.33.44/ #Baron_mouse-Baron_human-01-03_19.40.04
    cfg.CAME.figdir = '../../../../CAME/analysis_results/mouse_2020sa/Baron_mouse-Baron_human-06-03_14.38.27/figs/'

    cfg.CAME.homo_region_file_path = './Data/MouseHumanMatches_H88M67_all.csv'  # '../../../../CAME/brain_human_mouse/MouseHumanMatches_H88M67.csv'
    cfg.CAME.labels_dir = '../../../../CAME/brain_mouse_2020sa/'
    cfg.CAME.labels_mouse_file = 'mouse_region_list_64.csv'
    cfg.CAME.labels_human_file = 'human_88_label.csv'


    cfg.BrainAlign.homo_region_num = 4

    cfg.BrainAlign.embedding_type = 'came'
    cfg.BrainAlign.embedding_pca_dim = 30


    cfg.BrainAlign.dataset = 'mouse_human_binary'
    cfg.BrainAlign.result_save_folder = './results_{}_1000genes_all_came_selfloop/'.format(cfg.BrainAlign.homo_region_num)
    cfg.BrainAlign.experiment_time = '2023-06-23_20-31-14'#'2023-06-02_21-34-15' #'2023-05-11_10-08-35'#'2023-05-10_20-36-57'#time.strftime("%Y-%m-%d_%H-%M-%S")#'2023-04-30_23-00-05'#'2023-04-20_11-02-23'#'2023-02-12_10-57-19'#time.strftime("%Y-%m-%d_%H-%M-%S")  #'2023-01-04_10-15-59'
    cfg.BrainAlign.result_save_path = cfg.BrainAlign.result_save_folder + cfg.BrainAlign.experiment_time
    cfg.BrainAlign.embeddings_file_path = cfg.BrainAlign.result_save_path + "/embeds/"
    cfg.BrainAlign.DATA_PATH = cfg.BrainAlign.result_save_path + '/data/'

    cfg.SRRSC_args.save_path = "./results_{}_1000genes_all_came_selfloop/".format(
        cfg.BrainAlign.homo_region_num) + cfg.BrainAlign.experiment_time + '/'  # "../data/{}/results/".format(_C.HECO_args.dataset)+_C.HECO.experiment_time+'/'
    cfg.SRRSC_args.data_path = "./results_{}_1000genes_all_came_selfloop/".format(cfg.BrainAlign.homo_region_num) + cfg.BrainAlign.experiment_time + '/data/'

    S, M, binary_S, binary_M, binary_H, binary_V = load_dimensions_binary(cfg.CAME.ROOT + 'datapair_init.pickle')

    cfg.BrainAlign.S = S  # 21749 + 3682
    cfg.BrainAlign.M = M  # 4035 + 6507

    cfg.BrainAlign.binary_S = binary_S  # ,
    cfg.BrainAlign.binary_M = binary_M  # 1034  # 4035
    cfg.BrainAlign.binary_H = binary_H  # 1004  # 6507
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
    #process.get_srrsc_input(cfg)
    print('Training BrainAlign...')
    #main_sr_rsc.run_srrsc(cfg)

    #get_spatial_relation(cfg)

    print('Analysis of BrainAlign embeddings...')


    #load_srrsc_embeddings(cfg)  # (21749, 128)
    #alignment_STs(cfg)
    #alignment_STs_analysis_obj = alignment_STs_analysis(cfg)
    #alignment_STs_analysis_obj.forward()
    #alignment_STs_analysis_obj.experiment_1_cross_species_clustering()
    #alignment_STs_analysis_obj.experiment_2_umap_evaluation()
    #alignment_STs_analysis_obj.experiment_2_umap_diagram()
    #alignment_STs_analysis_obj.experiment_3_homo_random()
    # self.experiment_3_homo_random()
    #alignment_STs_analysis_obj.experiment_4_cross_species_genes_analysis()
    #alignment_STs_analysis_obj.experiment_5_genes_homo_random()
    # alignment_STs_analysis_obj.experiment_6_brain_region_classfier()
    #alignment_STs_analysis_obj.experiment_7_align_cross_species()
    #alignment_STs_analysis_obj.experiment_7_1_align_cross_species_split()
    # alignment_STs_analysis_obj.experiment_8_cross_evaluation_aligment_cluster()
    # alignment_STs_analysis_obj.experiment_8_1_cross_aligment_cluster_merge_lines()
    # alignment_STs_analysis_obj.experiment_9_name_clusters()
    #alignment_STs_analysis_obj.experiment_9_1_region_cluster_heatmap()
    #alignment_STs_analysis_obj.experiment_10_check_alignment()
    #alignment_STs_analysis_obj.experiment_10_1_check_alignment_plot()
    #alignment_STs_analysis_obj.experiment_11_paga()

    # alignment_STs_analysis_obj.experiment_14_clusters_alignment()
    #
    # alignment_STs_analysis_obj.experiment_15_regions_alignment()
    # alignment_STs_analysis_obj.experiment_15_1_isocortex_clusters(if_load=True)
    #
    # alignment_STs_analysis_obj.experiment_16_thalamus_clusters(if_load=True)
    # alignment_STs_analysis_obj.experiment_17_hypothalamus_clusters(if_load=True)
    # alignment_STs_analysis_obj.experiment_18_pons_clusters(if_load=True)

    #alignment_STs_analysis_obj.experiment_18_pons_marker_gene()

    #alignment_STs_analysis_obj.experiment_19_Midbrain_hypothalamus(if_load=True, topk=30)

    #alignment_STs_analysis_obj.experiment_7_align_cross_species()
    #
    #alignment_STs_analysis_obj.experiment_4_cross_species_genes_analysis()
    #alignment_STs_analysis_obj.experiment_4_1_genes_statistics()
    #alignment_STs_analysis_obj.experiment_8_cross_evaluation_aligment_cluster()
    #alignment_STs_analysis_obj.experiment_9_name_clusters()
    #alignment_STs_analysis_obj.experiment_10_check_alignment()

    #alignment_STs_analysis_obj.forward()
    #genomic_analysis_obj = genomic_analysis(cfg)
    #genomic_analysis_obj.experiment_1_th_clusters_deg()
    #genomic_analysis_obj.experiment_2_clusters_deg()
    #genomic_analysis_obj.experiment_2_1_homoregions_deg()
    #genomic_analysis_obj.experiment_3_regions_gene_module()
    #genomic_analysis_obj.experiment_3_1_regions_gene_module_analysis()
    #genomic_analysis_obj.experiment_4_clusters_gene_module()
    #genomic_analysis_obj.experiment_5_deg_homo_distribution()
    #genomic_analysis_obj.experiment_6_regions_gene_module_average_exp()
    #genomic_analysis_obj.experiment_6_1_regions_gene_module_cross()
    #genomic_analysis_obj.experiment_7_clusters_gene_module_average_exp()
    #genomic_analysis_obj.experiment_7_1_clusters_gene_module_cross()
    #genomic_analysis_obj.experiment_8_homoregions_go_terms_overlap()
    #genomic_analysis_obj.experiment_8_1_plot_heatmap()
    #genomic_analysis_obj.experiment_8_2_homoregions_go_terms_cross()

    #genomic_analysis_obj.experiment_9_clusters_go_terms_overlap()

    #genomic_analysis_obj.experiment_9_1_plot_heatmap()

    #genomic_analysis_obj.experiment_10_binary_clustering()

    #gene_comparison(cfg)
    #anatomical_STs_analysis_obj = anatomical_STs_analysis(cfg)
    #anatomical_STs_analysis_obj.forward()
    #spatial_analysis_obj = spatial_analysis(cfg)
    #spatial_analysis_obj.experiment_1_spatial_hippocampal()
    #spatial_analysis_obj.forward()
    #spatial_analysis_obj.experiment_2_spatial_isocortex()
    #spatial_analysis_obj.experiment_3_spatial_clusters()



