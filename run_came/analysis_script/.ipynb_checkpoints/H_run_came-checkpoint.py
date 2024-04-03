# -- coding: utf-8 --
# @Time : 2024/02/01 11:30
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : run_came.py
import sys
sys.path.append('../')

from analysis_utils import ttest_plot_utils
from analysis_utils import homo_random_config as config
import os


if __name__ == '__main__':

    cfg = config._C
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #cfg.CAME.visible_device
    #cfg.CAME.n_top_genes = 1000
    cfg.CAME.visible_device = '-1'
    n_top_genes_list = [2000]
    #quantile_gene_list = [0.8]
    #quantile_sample_list = [0.9]
    #cfg.CAME.quantile_gene = quantile_gene_list[0]
    #cfg.CAME.quantile_sample = quantile_sample_list[0]
    #for n_top_genes in n_top_genes_list:
    cfg.CAME.n_top_genes = n_top_genes_list[0]
    cfg.CAME.sparse = False
    cfg.CAME.do_normalize = [True, True]
    cfg.CAME.ROOT = '../analysis_results/mouse_macaque_hippocampus/'
    cfg.CAME.path_rawdata1 = '../../BrainAlign/data/srrsc_mouse_macaque_hippocampus/Data/Mouse.h5ad'
    cfg.CAME.path_rawdata2 = '../../BrainAlign/data/srrsc_mouse_macaque_hippocampus/Data/Macaque.h5ad'

    cfg.CAME.path_mouse_labels = '../../BrainAlign/data/srrsc_mouse_macaque_hippocampus/Data/mouse_region_list.csv'
    cfg.CAME.path_human_labels = '../../BrainAlign/data/srrsc_mouse_macaque_hippocampus/Data/macaque_region_list.csv'

    cfg.CAME.human_mouse_homo_region = '../../BrainAlign/data/srrsc_mouse_macaque_hippocampus/Data/mouse_macaque_homo_region.csv'
    #    ttest_plot_utils.run_came_homo_random(cfg)

    cfg.PROCESS.path_rawdata1 = cfg.CAME.path_rawdata1
    cfg.PROCESS.path_rawdata2 = cfg.CAME.path_rawdata2

    #cfg.PROCESS.path_mouse_labels = '../brain_human_mouse/mouse_67_label_10regions.csv'
    #cfg.PROCESS.path_human_labels = '../brain_human_mouse/human_88_label_10regions.csv'

    #cfg.PROCESS.path_rawdata1_part = '../../Brain_ST_human_mouse/data/10regions_mouse_brain_region_67_sagittal.h5ad'
    #cfg.PROCESS.path_rawdata2_part = '../../Brain_ST_human_mouse/data/10regions_human_brain_region_88_sparse_with3d.h5ad'
    ttest_plot_utils.run_came_homo_random(cfg)

