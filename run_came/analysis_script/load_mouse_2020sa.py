# -- coding: utf-8 --
# @Time : 2022/12/17 14:02
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : load_mouse_2020sa.py
import numpy as np
import pandas as pd

import difflib
from collections import Counter
import json
import scanpy as sc, anndata as ad
from scipy.sparse import csr_matrix

def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

import json
from treelib import Node, Tree

tree = Tree()
tree.create_node('Mouse brain', 'Mouse brain')

def add_node(tree_dict, key_input):
    #print(type(tree_dict[key_input]))
    if tree_dict['children'] == []:
        return
    #elif isinstance(tree_dict['children'], list):
    #    for key in tree_dict['children']:
    #        tree.create_node(key, key, parent=key_input)
    else:
        for dict in tree_dict['children']:
            tree.create_node(dict['name'], dict['name'], parent=key_input)
            add_node(dict, dict['name'])

def list_parent_region():
    meta_data_path = '../brain_mouse_2020sa/meta_table_v1.tsv'
    meta_df = pd.read_csv(meta_data_path, sep='\t')
    # region_label = meta_df['region_name']
    print(meta_df.columns)
    meta_df = meta_df.set_index('spotID')
    print('meta_df.index', meta_df.index)
    coordinate_path = '../brain_mouse_2020sa/spot_coordinates.txt'
    coordinate_df = pd.read_csv(coordinate_path, sep='\t', index_col=None, header=None)
    coordinate_df.columns = ['spotID', 'x_grid', 'y_grid', 'z_grid']
    print(coordinate_df.head(5))
    print(coordinate_df.shape)
    coordinate_df = coordinate_df.set_index('spotID')

    expression_path = '../brain_mouse_2020sa/expr_normalized_table.tsv'
    expression_df = pd.read_csv(expression_path, sep='\t', index_col=None)
    expression_df.index.name = 'spotID'
    print('expression_df.head(5)', expression_df.head(5))
    print('expression_df.shape', expression_df.shape)
    print('expression_df.iloc[:, 0]', expression_df.iloc[:, 0])
    print('expression_df.index', expression_df.index)

    print('expression_df first gene:', expression_df['Lbx1'])

    meta_df_part = meta_df.loc[expression_df.index]

    region_label = meta_df_part['ABA_parent']
    print('parent_region_labels:', set(region_label))

def load_part_data():
    meta_data_path = '../brain_mouse_2020sa/meta_table_v1.tsv'
    meta_df = pd.read_csv(meta_data_path, sep='\t')
    #region_label = meta_df['region_name']
    print(meta_df.columns)
    meta_df = meta_df.set_index('spotID')
    print('meta_df.index', meta_df.index)

    coordinate_path = '../brain_mouse_2020sa/spot_coordinates.txt'
    coordinate_df = pd.read_csv(coordinate_path, sep='\t',index_col=None, header=None)
    coordinate_df.columns = ['spotID', 'x_grid', 'y_grid', 'z_grid']
    print(coordinate_df.head(5))
    print(coordinate_df.shape)
    coordinate_df = coordinate_df.set_index('spotID')

    expression_path = '../brain_mouse_2020sa/expr_normalized_table.tsv'
    expression_df = pd.read_csv(expression_path, sep='\t', index_col=None)
    expression_df.index.name = 'spotID'
    print('expression_df.head(5)', expression_df.head(5))
    print('expression_df.shape', expression_df.shape)
    print('expression_df.iloc[:, 0]', expression_df.iloc[:, 0])
    print('expression_df.index', expression_df.index)

    print('expression_df first gene:',expression_df['Lbx1'])


    mouse_anndata = ad.AnnData(csr_matrix(expression_df.values))
    mouse_anndata.obs_names = expression_df.index
    mouse_anndata.var_names = expression_df.columns

    meta_df_part = meta_df.loc[expression_df.index]
    region_label = meta_df_part['region_name']

    coordinate_df = coordinate_df.loc[expression_df.index]

    print('expression_df.index', expression_df.index)

    mouse_anndata.obs['region_name'] = pd.Categorical(region_label)

    print(mouse_anndata.obs['region_name'])

    mouse_anndata.obs['x_grid'] = coordinate_df['x_grid']
    mouse_anndata.obs['y_grid'] = coordinate_df['y_grid']
    mouse_anndata.obs['z_grid'] = coordinate_df['z_grid']

    print(mouse_anndata)
    #mouse_anndata.write_h5ad('../brain_mouse_2020sa/mouse_2020sa_64regions.h5ad')

    region_label = meta_df_part['ABA_parent']
    print('parent_region_labels:', set(region_label))
    mouse_anndata.obs['parent_region_name'] = pd.Categorical(region_label)
    mouse_anndata.write_h5ad('../brain_mouse_2020sa/mouse_2020sa_64regions.h5ad')
    #mouse_anndata.write_h5ad('../brain_mouse_2020sa/mouse_2020sa_15regions.h5ad')


def load_part_data_6regions():

    path_mouse_labels = '../brain_human_mouse/mouse_67_label_6regions_2020sa.csv'
    path_rawdata1 = '../brain_mouse_2020sa/mouse_2020sa_64regions.h5ad'


    path_rawdata1_part = '../brain_mouse_2020sa/6regions_mouse_2020sa_64regions.h5ad'

    mouse_all_h5ad = sc.read_h5ad(path_rawdata1)
    mouse_region_list = set(list(pd.read_csv(path_mouse_labels)['region_name']))
    print(mouse_region_list)
    mouse_h5ad_part = mouse_all_h5ad[mouse_all_h5ad.obs['region_name'].isin(mouse_region_list)]
    print(mouse_h5ad_part)
    mouse_h5ad_part.write_h5ad(path_rawdata1_part)


def load_part_data_raw():
    meta_data_path = '../brain_mouse_2020sa/meta_table_v1.tsv'
    meta_df = pd.read_csv(meta_data_path, sep='\t')
    #region_label = meta_df['region_name']
    print(meta_df.columns)
    meta_df = meta_df.set_index('spotID')
    print('meta_df.index', meta_df.index)

    coordinate_path = '../brain_mouse_2020sa/spot_coordinates.txt'
    coordinate_df = pd.read_csv(coordinate_path, sep='\t',index_col=None, header=None)
    coordinate_df.columns = ['spotID', 'x_grid', 'y_grid', 'z_grid']
    print(coordinate_df.head(5))
    print(coordinate_df.shape)
    coordinate_df = coordinate_df.set_index('spotID')

    expression_path = '../brain_mouse_2020sa/expr_raw_table.tsv'
    expression_df = pd.read_csv(expression_path, sep='\t', index_col=None)
    expression_df.index.name = 'spotID'
    print('expression_df.head(5)', expression_df.head(5))
    print('expression_df.shape', expression_df.shape)
    print('expression_df.iloc[:, 0]', expression_df.iloc[:, 0])
    print('expression_df.index', expression_df.index)

    print('expression_df first gene:',expression_df['Lbx1'])


    mouse_anndata = ad.AnnData(csr_matrix(expression_df.values))
    mouse_anndata.obs_names = expression_df.index
    mouse_anndata.var_names = expression_df.columns

    meta_df_part = meta_df.loc[expression_df.index]
    region_label = meta_df_part['region_name']

    coordinate_df = coordinate_df.loc[expression_df.index]

    print('expression_df.index', expression_df.index)

    mouse_anndata.obs['region_name'] = pd.Categorical(region_label)

    print(mouse_anndata.obs['region_name'])

    mouse_anndata.obs['x_grid'] = coordinate_df['x_grid']
    mouse_anndata.obs['y_grid'] = coordinate_df['y_grid']
    mouse_anndata.obs['z_grid'] = coordinate_df['z_grid']

    print(mouse_anndata)
    #mouse_anndata.write_h5ad('../brain_mouse_2020sa/mouse_2020sa_64regions_raw.h5ad')

    region_label = meta_df_part['ABA_parent']
    mouse_anndata.obs['parent_region_name'] = pd.Categorical(region_label)
    mouse_anndata.write_h5ad('../brain_mouse_2020sa/mouse_2020sa_64regions_raw.h5ad')



def assign_region_name():
    meta_data_path = '../brain_mouse_2020sa/meta_table.tsv'
    meta_df = pd.read_csv(meta_data_path, sep='\t', index_col=None)
    print(meta_df.shape)
    # Process brain regions
    brain_region_list = meta_df['ABA_name'].values
    parent_region_list = meta_df['ABA_parent'].values
    '''
    simple_brain_region_list = []
    for x in brain_region_list:
        if len(x.split(',')) > 2:
            simple_brain_region_list.append(x.strip(', '+x.split(',')[-1]))
        else:
            simple_brain_region_list.append(x.split(',')[0])
    '''
    simple_brain_region_list = [x.split(',')[0] for x in brain_region_list]
    #print(simple_brain_region_list)
    uniqe_brain_region = set(simple_brain_region_list)
    print(uniqe_brain_region)
    print(len(uniqe_brain_region))


    parent_brain_region_list = meta_df['ABA_parent'].values
    uniqe_parent_brain_region_list = set(parent_brain_region_list)
    print(uniqe_parent_brain_region_list)
    print(len(uniqe_parent_brain_region_list))

    brain_ascronym_list = meta_df['ABA_acronym'].values
    #brain_region_list = meta_df['ABA_acronym'].values

    express_data_path = '../brain_mouse_2020sa/expr_normalized_table.tsv'

    region_ascronym_df = pd.read_csv('../brain_mouse_2020sa/mouse_69_label_acronym.csv', sep=',')
    print(region_ascronym_df)
    region_ascronym_dict = {k:v for k,v in zip(region_ascronym_df['region_name'], region_ascronym_df['acronym'])}
    ascronym_region_dict = {k:v for k,v in zip(region_ascronym_df['acronym'], region_ascronym_df['region_name'])}
    ascronym_region_list = list(region_ascronym_df['acronym'])
    name_region_list = list(region_ascronym_df['region_name'])

    # load the anatomical tree
    print('Build the anatomical tree of Mouse...')
    tree_path = '../brain_mouse_2020sa/mouse_structure.json'
    f = open(tree_path, 'rb')
    tree_dict = json.load(f)
    tree_dict_new = tree_dict['msg'][0]['children']

    for dict in tree_dict_new:
        tree.create_node(dict['name'], dict['name'], parent='Mouse brain')
        add_node(dict, dict['name'])

    # read assign region list
    assign_region_df = pd.read_csv('../brain_mouse_2020sa/mouse_region_list.csv', sep=',')
    assign_region_list = assign_region_df['region_name'].values
    assign_region_acronym_list = assign_region_df['acronym'].values

    # init region and acronym list
    output_region_list = []
    output_region_acronym_list = []
    output_region_color_list = []
    output_parent_region_color_list = []


    # Assign a proper brain label
    for region, parent_brain_region in zip(brain_region_list, parent_brain_region_list):
        spot_region = None
        spot_region_acronym = None

        for assign_region,assign_region_acronym in zip(assign_region_list, assign_region_acronym_list):
            tree_node_list = [tree[node].tag for node in tree.subtree(assign_region).expand_tree(mode=Tree.DEPTH)]
            for region_node in tree_node_list:
                if region_node == region:
                    spot_region = assign_region
                    spot_region_acronym = assign_region_acronym

        if spot_region == None:
            if region == 'Olfactory areas' and parent_brain_region == 'Olfactory areas':
                spot_region = 'Olfactory areas-other'
                spot_region_acronym = 'OLF-other'
            elif parent_brain_region == 'Undefined areas':
                spot_region = 'Undefined areas'
                spot_region_acronym = 'Undefined areas'
            elif region == 'Cerebellum' and parent_brain_region == 'Cerebellum':
                spot_region = 'Cerebellum'
                spot_region_acronym = 'Cerebellum'
            elif region == 'Barringtons nucleus':
                spot_region = 'Barringtons nucleus'
                spot_region_acronym = 'Barringtons nucleus'

        if parent_brain_region == 'Thalamus':
            spot_region = 'Thalamus'
            spot_region_acronym = 'TH'

        if spot_region == None:
            print('Unassigned regions:', region, parent_brain_region)
        output_region_list.append(spot_region)
        output_region_acronym_list.append(spot_region_acronym)

    meta_df['region_name'] = output_region_list
    meta_df['ascronym'] = output_region_acronym_list
    meta_df.index.name = 'spotID'
    meta_df.to_csv('../brain_mouse_2020sa/meta_table_v1.tsv', sep='\t')

    print('len(brain_ascronym_69_list):', len(output_region_acronym_list))
    print(Counter(output_region_acronym_list))
    print('len of Counter:', len(Counter(output_region_acronym_list)))
    print('len(brain_region_69_list)', len(output_region_list))
    print(Counter(output_region_list))
    print('len of Counter:', len(Counter(output_region_list)))

def test_mouse_2020sa_nan():
    path_rawdata1_part = '../brain_mouse_2020sa/6regions_mouse_2020sa_64regions.h5ad'
    mouse_all_h5ad = sc.read_h5ad(path_rawdata1_part)
    print(mouse_all_h5ad.X)

if __name__ == '__main__':
    #assign_region_name()
    load_part_data()
    #list_parent_region()
    #load_part_data_6regions()
    #test_mouse_2020sa_nan()