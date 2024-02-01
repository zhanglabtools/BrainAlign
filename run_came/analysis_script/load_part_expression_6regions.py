# -- coding: utf-8 --
# @Time : 2022/12/1 18:38
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : load_part_expression.py
import sys

import pandas as pd

sys.path.append('..')
from analysis_utils import ttest_plot_utils
from analysis_utils import homo_random_config as config
import os
import scanpy as sc

if __name__ == "__main__":
    cfg = config._C

    cfg.PROCESS.path_mouse_labels = '../brain_human_mouse/mouse_67_label_6regions.csv'
    cfg.PROCESS.path_human_labels = '../brain_human_mouse/human_88_label_6regions.csv'

    cfg.PROCESS.path_rawdata1_part = '../../Brain_ST_human_mouse/data/6regions_mouse_brain_region_67_sagittal.h5ad'
    cfg.PROCESS.path_rawdata2_part = '../../Brain_ST_human_mouse/data/6regions_human_brain_region_88_sparse_with3d.h5ad'

    mouse_all_h5ad = sc.read_h5ad(cfg.PROCESS.path_rawdata1)
    mouse_region_list = set(list(pd.read_csv(cfg.PROCESS.path_mouse_labels)['region_name']))
    print(mouse_region_list)
    mouse_h5ad_part = mouse_all_h5ad[mouse_all_h5ad.obs['region_name'].isin(mouse_region_list)]
    print(mouse_h5ad_part)
    mouse_h5ad_part.write_h5ad(cfg.PROCESS.path_rawdata1_part)

    human_all_h5ad = sc.read_h5ad(cfg.PROCESS.path_rawdata2)
    human_region_list = set(list(pd.read_csv(cfg.PROCESS.path_human_labels)['region_name']))
    print(human_region_list)
    human_h5ad_part = human_all_h5ad[human_all_h5ad.obs['region_name'].isin(human_region_list)]
    print(human_h5ad_part)
    human_h5ad_part.write_h5ad(cfg.PROCESS.path_rawdata2_part)
