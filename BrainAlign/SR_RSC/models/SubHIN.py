import os.path

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import sys

from torch.optim.lr_scheduler import StepLR

sys.path.append('../../')

from BrainAlign.SR_RSC.embedder import embedder
from BrainAlign.SR_RSC.layers import *
from BrainAlign.SR_RSC.evaluation import evaluation_metrics
import pickle as pkl
import logging
#import torch.optim.lr_scheduler.StepLR


class SubHIN(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self, logger):
        # input data
        self.features = self.features.to(self.args.device)
        self.graph = {t: [m.to(self.args.device) for m in ms] for t, ms in self.graph.items()}
        # Build the model
        model = modeler(self.args, self.args_nt_rel, self.args_ft_size, self.args_node_cnt, self.args_node_size).to(self.args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr)
        scheduler = StepLR(optimizer, step_size=self.args.lr_stepsize, gamma=self.args.lr_gamma)
        cnt_wait = 0; best = 1e9; best_acc = 0
        
        file = ''
        if self.args.isLP:
            file += '.lp'
        if self.args.isSemi:
            file += '.semi'
        file += '.npy'
        for epoch in range(self.args.nb_epochs):
            logger.info("Epoch {}...".format(epoch))
            model.train()
            optimizer.zero_grad()

            embs = model(self.graph, self.features) 
            loss = model.loss2(embs, self.features, self.graph)
            loss.backward()
            optimizer.step()
   
            train_loss = loss.item()
    
            # validation
            test_out = embs.detach().cpu().numpy()
            ev = evaluation_metrics(test_out, self.args.labels, logger)
            acc = ev.val_acc

            scheduler.step()
            
            if (train_loss < best):# and (acc >= best_acc):
                best = train_loss
                best_acc = acc
                cnt_wait = 0
                logger.info("Epoch {}, loss {:.5}, valacc {:.5}".format(epoch, train_loss, best_acc))
                outs = embs.detach().cpu().numpy()
                if self.args.isLP:
                    ev.evaluation_lp(self.node1, self.node2, self.lp_label)
                else:
                    ev.evalutation()

            else:
                cnt_wait += 1

            if cnt_wait == self.args.patience or epoch >= self.args.nb_epochs-1:
                if not os.path.exists(self.args.save_path + "/embeds/"):
                    os.makedirs(self.args.save_path + "/embeds/")
                f = open(self.args.save_path + "/embeds/" + "node" + ".pkl", "wb")
                pkl.dump(outs, f)
                f.close()
                break
        return self.args.save_path + "/embeds/"


class modeler(nn.Module):
    def __init__(self, args, args_nt_rel, args_ft_size, args_node_cnt, args_node_size):
        super().__init__()
        self.args = args
        self.marginloss = nn.MarginRankingLoss(self.args.margin)
        self.b_xent = nn.BCEWithLogitsLoss()
        self.bnn = nn.ModuleDict()
        self.disc2 = nn.ModuleDict()
        self.fc = nn.ModuleDict()

        self.semanticatt = nn.ModuleDict()

        self.args_nt_rel = args_nt_rel
        self.args_ft_size = args_ft_size
        self.args_node_cnt = args_node_cnt
        self.args_node_size = args_node_size
            
        for t, rels in self.args_nt_rel.items():  # {note_type: [rel1, rel2]}

            if self.args.isSemi:
                self.fc[t+'1'] = FullyConnect(args.out_ft, args.n_label, drop_prob=self.args.drop_prob)
            
            self.fc[t] = FullyConnect(args.hid_units2+self.args_ft_size, args.out_ft, drop_prob=self.args.drop_prob)
            self.disc2[t] = Discriminator(self.args_ft_size,args.out_ft)

            for rel in rels: 
                self.bnn['0'+rel] = GCN(self.args_ft_size, args.hid_units, act=nn.LeakyReLU(), isBias=args.isBias)
                self.bnn['1'+rel] = GCN(args.hid_units, args.hid_units2, act=nn.LeakyReLU(), isBias=args.isBias)

            self.semanticatt['0'+t] = SemanticAttention(args.hid_units, args.hid_units//4)
            self.semanticatt['1'+t] = SemanticAttention(args.hid_units2, args.hid_units2//4)

        # newly added
        self.last_fc = FullyConnect(args.out_ft * 2, args.out_ft, drop_prob=self.args.drop_prob)


    def forward(self, graph, features):
        totalLoss = 0.0
        reg_loss = 0.0
        embs1 = torch.zeros((self.args_node_size, self.args.hid_units)).to(self.args.device)
        embs2 = torch.zeros((self.args_node_size, self.args.out_ft)).to(self.args.device)
        for n, rels in self.args_nt_rel.items():   # p: [[Nei(p-a), Nei(p-c)]]  (Np, Nprel)
            vec = []
            for j, rel in enumerate(rels):
                if (rel == 'citing') or (rel == 'cited'):
                    t = 'p'
                else:
                    t = rel.split('-')[1]
                    
                mean_neighbor = torch.spmm(graph[n][j], features[self.args_node_cnt[t]])
                v = self.bnn['0'+rel](mean_neighbor)
                vec.append(v)  # (1, Nt, ft)

            vec = torch.stack(vec, 0)  # (rel_size, Nt, ft_size)
            
            if self.args.isAtt:
                v_summary = self.semanticatt['0'+n](vec.view(-1, self.args.hid_units), len(rels))
            else:
                v_summary = torch.mean(vec, 0)  # (Nt, hd)

            embs1[self.args_node_cnt[n]] = v_summary
        
        for n, rels in self.args_nt_rel.items():   # p: [[Nei(p-a), Nei(p-c)]]  (Np, Nprel)
            vec = []
            for j, rel in enumerate(rels):
                if (rel == 'citing') or (rel == 'cited'):
                    t = 'p'
                else:
                    t = rel.split('-')[-1]

                mean_neighbor = torch.spmm(graph[n][j], embs1[self.args_node_cnt[t]])
                v = self.bnn['1'+rel](mean_neighbor)  
                vec.append(v)  # (1, Nt, ft)

            vec = torch.stack(vec, 0)  # (rel_size, Nt, ft_size)     
            if self.args.isAtt:
                v_summary = self.semanticatt['1'+n](vec.view(-1, self.args.hid_units2), len(rels))
            else:
                v_summary = torch.mean(vec, 0)  # (Nt, hd)

            v_cat = torch.hstack((v_summary, features[self.args_node_cnt[n]]))
            v_summary = self.fc[n](v_cat)
            
            embs2[self.args_node_cnt[n]] = v_summary

        return embs2
    
            
    def loss2(self, embs2, features, graph):
        totalLoss = 0.0
        embs = torch.zeros((self.args_node_size, self.args.out_ft)).to(self.args.device)
        if self.args.isLP:
            coef = self.args.lamb_lp
        else:
            coef = self.args.lamb

        for n, rels in self.args_nt_rel.items():
            
            nb = len(self.args_node_cnt[n])
            ones = torch.ones(nb).to(self.args.device)
            zeros = torch.zeros(nb).to(self.args.device)
            lbl = torch.cat((ones, zeros), 0).squeeze()

            shuf_index = torch.randperm(nb).to(self.args.device)

            vec = embs2[self.args_node_cnt[n]]
            fvec = vec[shuf_index]
            a = nn.Softmax()(features[self.args_node_cnt[n]])

            logits_pos = self.disc2[n](a,vec)
            logits_neg = self.disc2[n](a,fvec)
            logits = torch.hstack((logits_pos,logits_neg))

            totalLoss += 1.0*self.b_xent(logits, lbl)

            for j, rel in enumerate(rels):                
                if (rel == 'citing') or (rel == 'cited'):
                    t = 'p'
                else:
                    t = rel.split('-')[-1]
              
                mean_neighbor = torch.spmm(graph[n][j], embs2[self.args_node_cnt[t]])

                logits_pos = (vec*mean_neighbor).sum(-1).view(-1)
                logits_neg = (fvec*mean_neighbor).sum(-1).view(-1)
                
                totalLoss += coef*self.marginloss(torch.sigmoid(logits_pos), torch.sigmoid(logits_neg), ones)

                # 2-hop proximity
                logits = []
                for k, nr in enumerate(self.args_nt_rel[t]):
                    if (nr == 'citing') or (nr == 'cited'):
                        tt = 'p'
                    else:
                        tt = nr.split('-')[-1]
                    nmn = torch.spmm(graph[t][k], embs2[self.args_node_cnt[tt]])
                    nmn = torch.spmm(graph[n][j], nmn)

                    logits_pos = (vec*nmn).sum(-1).view(-1)
                    logits_neg = (fvec*nmn).sum(-1).view(-1)
                    totalLoss += (1-coef)*self.marginloss(torch.sigmoid(logits_pos), torch.sigmoid(logits_neg), ones)

        if self.args.isSemi:
            
            criterion = nn.CrossEntropyLoss()
            outs = torch.zeros((self.args_node_size, self.args.n_label)).to(self.args.device)
            for n, rels in self.args_nt_rel.items():
                outs[self.args_node_cnt[n]] = self.fc[t+'1']( embs2[self.args_node_cnt[n]])#(vec)
            trY = torch.argmax(self.args.trY.to(self.args.device), dim=1)
            totalLoss += criterion(outs[self.args.trX], trY)
            
        return totalLoss

            
            
