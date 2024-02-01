# -- coding: utf-8 --
# @Time : 2022/12/14 13:30
# @Author : Biao Zhang
# @Email : littlebiao@outlook.com
# @File : test_input_data.py
import pickle
import numpy as np

if __name__ == '__main__':
    '''
    edges_path = './dataset/acm/edges.pkl'
    file = open(edges_path, 'rb')
    load_data = pickle.load(file)
    print('edges type: ', type(load_data))
    #print('edges shape: ', np.shape(load_data))
    print('edges: ', load_data)

    
    labels_path = './dataset/acm/labels.pkl'
    file = open(labels_path, 'rb')
    load_data = pickle.load(file)
    print('labels type: ', type(load_data))
    print('labels: ', load_data)
    print('labels 0', load_data[0].shape)
    print('labels 1', load_data[1].shape)
    print('labels 2', load_data[2].shape)
    


    meta_data_path = './dataset/acm/meta_data.pkl'
    file = open(meta_data_path, 'rb')
    load_data = pickle.load(file)
    print('meta_data type: ', type(load_data))
    print('meta_data keys: ', load_data.keys())
    print('meta_data: ', load_data)
    '''

    node_features_path = './dataset/acm/node_features.pkl'
    file = open(node_features_path, 'rb')
    load_data = pickle.load(file)
    print('node_features type: ', type(load_data))
    print('node_features shape: ', np.shape(load_data))






