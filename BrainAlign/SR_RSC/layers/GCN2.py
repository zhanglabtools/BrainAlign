import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5, isBias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(nfeat, nhid))
        nn.init.xavier_uniform_(self.weight)
        if isBias:
            self.bias = nn.Parameter(torch.empty(nhid))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        self.dropout = dropout
        self.act = nn.ReLU()


    def forward(self, adj, x):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)


class DGCN(nn.Module):
    def __init__(self, v_in_ft, u_in_ft, out_ft, act=nn.PReLU(), drop_prob=0.5, isBias=False):
        super().__init__()

        self.v_gc1 = GCN(nfeat=v_in_ft,
                        nhid=out_ft,
                        dropout=drop_prob)
        self.v_gc2 = GCN(nfeat=out_ft,
                        nhid=out_ft,
                        dropout=drop_prob)

        self.u_gc1 = GCN(nfeat=u_in_ft,
                        nhid=out_ft,
                        dropout=drop_prob)
        self.u_gc2 = GCN(nfeat=out_ft,
                        nhid=out_ft,
                        dropout=drop_prob)
        self.u_fc = nn.Linear(out_ft + u_in_ft, out_ft)
        nn.init.xavier_uniform_(self.u_fc.weight.data)
        self.v_fc = nn.Linear(out_ft + v_in_ft, out_ft)
        nn.init.xavier_uniform_(self.v_fc.weight.data)
        self.u_fc2 = nn.Linear(out_ft , out_ft)
        nn.init.xavier_uniform_(self.u_fc.weight.data)
        self.v_fc2 = nn.Linear(out_ft , out_ft)
        nn.init.xavier_uniform_(self.v_fc.weight.data)

        self.act = act
        self.drop_prob = drop_prob
        self.isBias = isBias

    def forward(self, uv_adj, vu_adj, ufea, vfea):
        # emb (batch_size, ft)
#         u = F.dropout(ufea, self.drop_prob, training=self.training)
#         v = F.dropout(vfea, self.drop_prob, training=self.training)
        
        vu = self.u_gc1(vu_adj, ufea)
        uv = self.v_gc1(uv_adj, vfea)

        uv2 = self.v_gc2(uv_adj, vu)
        vu2 = self.u_gc2(vu_adj, uv)

        Hv = torch.cat((vu2, vfea), dim=1)
        Hu = torch.cat((uv2, ufea), dim=1)

        Hv = nn.ReLU()(self.v_fc(Hv))  #  (batch_size, d)
        Hu = nn.ReLU()(self.u_fc(Hu))  #  (batch_size, d)
        Hv = self.v_fc2(Hv)
        Hu = self.u_fc2(Hu)

        return self.act(Hu), self.act(Hv)
