import os
import numpy as np
np.random.seed(0)
from collections import defaultdict, Counter
# import pickle5 as pickle
import pickle
import torch
import torch.nn as nn
import scipy.sparse as sp
import sys
sys.path.append('../../')
from BrainAlign.SR_RSC.utils import process
#import sys

class embedder:
    def __init__(self, args, rewrite=False, useMP2vec=False):
        args.sparse = True
        if args.gpu_num == "cpu":
            args.device = "cpu"
        else:
            args.device = str(torch.device("cuda:"+ args.gpu_num if torch.cuda.is_available() else "cpu"))

        path = args.data_path
        adj_norm = True
        norm = True

        with open(path+'/meta_data.pkl', 'rb') as f:
            data = pickle.load(f)
        idx ={}
        #print(data.keys())
        #print(data['node2gid'])
        for t in data['t_info'].keys():
            idx[t] = torch.LongTensor([i for p, i in data['node2gid'].items() if p.startswith(t)])
        node2id = data['node2gid']

        with open(path+'/labels.pkl', 'rb') as f:
            labels = pickle.load(f)
        if args.isLP:
            with open(path+'/lp_edges.pkl', 'rb') as f:
                edges = pickle.load(f)
            with open(path+'/lp_test.pkl', 'rb') as f:
                lp_test = pickle.load(f)
            self.node1 = lp_test[:,0]
            self.node2 = lp_test[:,1]
            self.lp_label = lp_test[:,2]
        else:
            with open(path+'/edges.pkl', 'rb') as f:
                edges = pickle.load(f)

        # added code by biao
        # if args.with_spatial:
        #     with open(path+'/edges_spatial.pkl', 'rb') as f:
        #         edges_spatial = pickle.load(f)

        node_rel = defaultdict(list)
        rel_types = set()
        real_adj = {}
        neighbors = defaultdict(set)
        if args.model in ['SubHIN','SubHIN2']:
            subgraph = {}
            subgraph_nb = {}
            edge_index = np.array([[],[]])
            
            for rel in edges:
#                 if (rel=='b-l') or (rel == 'l-b'): continue
                if (rel == 'citing'):
                    s, t = 'p', 'p'
                    vu = 'cited'
                elif (rel == 'cited'):
                    s, t = 'p', 'p'
                    vu = 'citing'
                else:
                    s, t = rel.split('-')
                    vu = t + '-' + s
                node_rel[s].append(rel)
                rel_types.add(rel)
                x,y = edges[rel].nonzero()
                for i,j in zip(x,y):
                    neighbors[i].add(j)     
            for nt, rels in node_rel.items():
                rel_list = []
                nb_neighbor = []
                for rel in rels:
                    if (rel == 'citing') or (rel == 'cited'):
                        s, t = 'p', 'p'
                    else:
                        s, t = rel.split('-')
                    e = edges[rel][idx[s],:][:,idx[t]]
                    nb = e.sum(1)
                    nb_neighbor.append(torch.FloatTensor(nb))
                    if adj_norm:
                        e = process.normalize_adj(e)
                    e = process.sparse_to_tuple(e)  # coords, values, shape
                    rel_list.append(torch.sparse_coo_tensor(torch.LongTensor(e[0]),torch.FloatTensor(e[1]), torch.Size(e[2])))
                    edge_index = np.concatenate([edge_index, e[0]],-1)
                    
                subgraph[nt] = rel_list
                subgraph_nb[nt] = nb_neighbor
        else:
            subgraph = defaultdict(dict)
            subgraph_nb = defaultdict(dict)
            edge_index = np.array([[],[]])

            for rel in edges:
#                 if (rel=='b-l') or (rel == 'l-b'): continue
                if (rel == 'citing'):
                    s, t = 'p', 'p'
                    vu = 'cited'
                elif (rel == 'cited'):
                    s, t = 'p', 'p'
                    vu = 'citing'
                else:
                    s, t = rel.split('-')
                    vu = t + '-' + s
                node_rel[s].append(rel)

                if vu not in rel_types:
                    rel_types.add(rel)
                
                x,y = edges[rel].nonzero()
                for i,j in zip(x,y):
                    neighbors[i].add(j)
                
                e = edges[rel][idx[s],:][:,idx[t]]
                nb = e.sum(1)
                e = process.sparse_to_tuple(e)  # coords, values, shape
                
                subgraph[s][rel] = torch.sparse_coo_tensor(torch.LongTensor(e[0]),torch.FloatTensor(e[1]), torch.Size(e[2]))
                subgraph_nb[s][rel] = torch.FloatTensor(nb)
                edge_index = np.concatenate([edge_index, e[0]],-1)

                
        neighbors_list = []
        for i in range(len(node2id)):
            if len(neighbors[i]) == 0:
                print('Node %s has no neighbor'%(str(i)))
#                 sys.exit()
            neighbors_list.append(neighbors[i].union(set([i])))
            
        if useMP2vec:
            with open("dataset/"+args.dataset+"mp_emb.pkl", "rb") as f:
                features = torch.FloatTensor(pickle.load(f))
            ft = features.shape[1]
            self.features = features
        else:
            with open(path+"/node_features.pkl", "rb") as f:
                features = pickle.load(f)
            ft = features.shape[1]
            padding_idx = features.shape[0]
            self.features = process.preprocess_features(features, norm=norm)  # {node_type: [0 || node_features]}
        
        self.graph = subgraph
        self.neighbor_list = neighbors_list
        self.graph_nb_neighbor = subgraph_nb
        #args.node2id = node2id
        #args.labels = labels   # {node_type: labels} refer to the sequence of [n, node_cnt[node_type]]
        #args.nt_rel = node_rel   # {note_type: [rel1, rel2]}
        #args.node_cnt = idx  # {note_type: nb}
        #args.node_type = list(args.node_cnt)
        #args.ft_size = ft
        #args.node_size = len(node2id)
        #args.rel_types = rel_types

        self.args_node2id = node2id
        args.labels = labels  # {node_type: labels} refer to the sequence of [n, node_cnt[node_type]]
        self.args_nt_rel = node_rel  # {note_type: [rel1, rel2]}
        self.args_node_cnt = idx  # {note_type: nb}
        self.args_node_type = list(self.args_node_cnt)
        self.args_ft_size = ft
        self.args_node_size = len(node2id)
        self.args_rel_types = rel_types

        self.args = args
        self.edge_index = edge_index
        self.args.nb_edge = self.edge_index.shape[1]
        if args.isSemi:
            self.args.n_label = data['n_class']
            
            self.args.trX, trY = np.array(labels[0])[:,0], np.array(labels[0])[:,1]
            self.args.trY = process.indices_to_one_hot(trY, self.args.n_label)
        
        print("Dataset: %s"%args.dataset)
        print("node_type num_node:")
        for t, num in self.args_node_cnt.items():
            print("\n%s\t %s"%(t, len(num)))
        print("Graph prepared!")
        print("Model setup:")
        print("learning rate: %s"%args.lr)
        
        print("model: %s" % args.model)
        if args.gpu_num == "cpu":
            print("use cpu")
        else:
            print("use cuda")
        if args.isAtt:
            print("use attention")
        else:
            print("use mean pool")
        if args.isLP:
            print("task: link prediction")
        else:
            print("task: cluster and classification")

