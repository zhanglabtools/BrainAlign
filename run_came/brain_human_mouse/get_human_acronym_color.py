# -- coding: utf-8 --
# @Time : 2023/3/6 19:55
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : get_human_acronym_color.py
# @Description: This file is used to ...

import pandas as pd

if __name__ == '__main__':
    human_structure_df = pd.read_csv('./human_query.csv')
    name_list = human_structure_df['name']
    acronym_list = human_structure_df['acronym']
    acronym_dict = {k:v for k,v in zip(name_list,acronym_list)}
    color_list = human_structure_df['color_hex_triplet']
    color_dict = {k:v for k,v in zip(name_list,color_list)}


    human_88_label_df = pd.read_csv('human_88_label_origin.csv', index_col=0)

    region_name_list = human_88_label_df['region_name']

    human_88_label_df['acronym'] = [acronym_dict[r] for r in region_name_list]
    human_88_label_df['color_hex_triplet'] = ['#'+color_dict[r] for r in region_name_list]

    human_88_label_df.to_csv('./human_88_labels.csv')

