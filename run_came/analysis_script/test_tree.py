# -- coding: utf-8 --
# @Time : 2022/12/19 21:16
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : test_tree.py
# @Description: This file is used to ...
import pandas as pd
from treelib import Node, Tree

tree = Tree()
tree.create_node('root', 'root')

def add_node(tree_dict, key_input):
    if tree_dict[key_input]['children'] == []:
        return
    elif isinstance(tree_dict[key_input]['children'], list):
        for key in tree_dict[key_input]['children']:
            tree.create_node(key, key, parent=key_input)
    else:
        for key in tree_dict[key_input]['children'].keys():
            tree.create_node(key, key, parent=key_input)
            add_node(tree_dict[key_input]['children'], key)

if __name__ == '__main__':


    dict_ = {"2": {'parent': "1"}, "1": {'parent': None}, "3": {'parent': "2"}}
    tree_dict = {"0": {'name':'n0','children': {'0-1':{'name':'n0-1', 'children':['n0-1-0']}}},
                 "1": {'name':'n1','children': []},
                 "2": {'name':'n2', 'children': {'n2-0':{'name':'n2-0', 'children':{'n2-0-0':{'name':'n2-0-0', 'children':[]}}}}}}
    added = set()
    #tree = Tree()
    for key in tree_dict.keys():
        tree.create_node(key, key, parent='root')
        add_node(tree_dict, key)

    tree.show()
    print(tree.depth())
    print(tree.subtree('0-1').depth())
    #new_tree = tree.expand_tree(filter=lambda x:(tree.depth()-tree.subtree(x).depth())!=2)
    print([tree[node].tag for node in tree.subtree('0-1').expand_tree(mode=Tree.DEPTH)])
    #new_tree.show()
    '''
    region_69_df = pd.read_csv('../brain_mouse_2020sa/mouse_69_label_acronym.csv', sep=',')
    region_69_list = region_69_df['region_name']

    region_new_df = pd.read_csv('../brain_mouse_2020sa/mouse_region_list.csv')
    region_list = region_new_df['region_name']

    for region in region_69_list:
        if not region in set(region_list):
            print(region)
    '''

