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
from BrainAlign.brain_analysis.configs import came_config_binary
from BrainAlign.brain_analysis import process
from BrainAlign.SR_RSC import main_sr_rsc
from BrainAlign.brain_analysis.analysis import load_came_embeddings, \
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
    load_homo_genes
import time

if __name__ == '__main__':
    cfg = came_config_binary._C
    #cfg.HECO.DATA_PATH = './data/'
    #cfg.HECO_args.data_path = "./data/"

    cfg.SRRSC_args.if_pretrained = False
    #cfg.freeze()
    print('Process of heco embeddings...')

    cfg.CAME.path_rawdata1 = '../../../../CAME/brain_mouse_2020sa/mouse_2020sa_64regions.h5ad'
    cfg.CAME.path_rawdata2 = '../../../../Brain_ST_human_mouse/data/human_brain_region_88_sparse_with3d.h5ad'
    cfg.CAME.ROOT = '../../../../CAME/analysis_results/mouse_2020sa/Baron_mouse-Baron_human-01-17_12.20.22/'
    cfg.CAME.figdir = '../../../../CAME/analysis_results/mouse_2020sa/Baron_mouse-Baron_human-01-17_12.20.22/figs/'

    cfg.CAME.homo_region_file_path = '../../../../CAME/brain_human_mouse/MouseHumanMatches_H88M67_all.csv'  # '../../../../CAME/brain_human_mouse/MouseHumanMatches_H88M67.csv'
    cfg.CAME.labels_dir = '../../../../CAME/brain_mouse_2020sa/'
    cfg.CAME.labels_mouse_file = 'mouse_region_list_64.csv'
    cfg.CAME.labels_human_file = 'human_88_label.csv'

    cfg.HECO.homo_region_num = 22

    cfg.HECO.embedding_type = 'came'
    cfg.HECO.embedding_pca_dim = 30


    cfg.HECO.dataset = 'mouse_human_binary'
    cfg.HECO.result_save_folder = './results_{}_2000genes_all_came/'.format(cfg.HECO.homo_region_num)
    cfg.HECO.experiment_time = time.strftime("%Y-%m-%d_%H-%M-%S")  #'2023-01-04_10-15-59'
    cfg.HECO.result_save_path = cfg.HECO.result_save_folder + cfg.HECO.experiment_time
    cfg.HECO.embeddings_file_path = cfg.HECO.result_save_path + "/embeds/"
    cfg.HECO.DATA_PATH = cfg.HECO.result_save_path + '/data/'

    cfg.SRRSC_args.save_path = "./results_{}_2000genes_all_came/".format(
        cfg.HECO.homo_region_num) + cfg.HECO.experiment_time + '/'  # "../data/{}/results/".format(_C.HECO_args.dataset)+_C.HECO.experiment_time+'/'
    cfg.SRRSC_args.data_path = "./results_{}_2000genes_all_came/".format(cfg.HECO.homo_region_num) + cfg.HECO.experiment_time + '/data/'

    cfg.HECO.pruning_method = 'std'  # top, std, quantile
    cfg.HECO.pruning_std_times_sm = 3.6 # 2.9
    cfg.HECO.pruning_std_times_vh = 2.6

    cfg.HECO.sm_gene_top = 2  # 100
    cfg.HECO.vh_gene_top = 2  # 20
    cfg.HECO.sm_sample_top = 3
    cfg.HECO.vh_sample_top = 3

    cfg.HECO.S = 37735  # 21749 + 3682
    cfg.HECO.M = 6722  # 4035 + 6507

    cfg.HECO.binary_S = 34053  # ,
    cfg.HECO.binary_M = 3491  # 1034  # 4035
    cfg.HECO.binary_H = 3231  # 1004  # 6507
    cfg.HECO.binary_V = 3682

    cfg.SRRSC_args.lr = 0.005

    cfg.HECO.fig_format = 'svg'

    print('--------------------------Config:', cfg)
    print('Analysis of came embeddings...')
    # args = set_params(cfg.HECO.dataset)
    # load_heco_embeddings(cfg)
    load_came_embeddings(cfg) # (21749, 128)
    homo_corr(cfg)
    random_corr(cfg)
    plot_homo_random(cfg)
    ttest_homo_random(cfg)
    cross_species_binary_clustering(cfg)
    umap_seperate(cfg)
    spatial_alignment(cfg)
    load_homo_genes(cfg)
    spatial_alignment_mouse(cfg)
    #umap_genes_seperate(cfg)
    cross_species_genes_clustering(cfg)



