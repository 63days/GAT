import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import GraphConvolutionLayer, GraphAttentionLayer, SparseGraphConvolutionLayer, SparseGraphAttentionLayer


# TODO step 1.
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolutionLayer(nfeat, nhid, dropout)
        self.gc2 = GraphConvolutionLayer(nhid, nclass, dropout)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


# TODO step 2.
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.ga1s = [GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True) for _ in range(nheads)]
        for i, ga in enumerate(self.ga1s):
            self.add_module('ga1_{}'.format(i), ga)
        self.ga2 = GraphAttentionLayer(nhid * nheads, nclass, dropout, alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([ga1(x, adj) for ga1 in self.ga1s], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.ga2(x, adj))
        return F.log_softmax(x, dim=1)


# TODO step 3.
class SpGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SpGCN, self).__init__()
        self.gc1 = SparseGraphConvolutionLayer(nfeat, nhid, dropout)
        self.gc2 = SparseGraphConvolutionLayer(nhid, nclass, dropout)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.ga1s = [SparseGraphAttentionLayer(nfeat,
                                               nhid,
                                               dropout=dropout,
                                               alpha=alpha,
                                               concat=True) for _ in range(nheads)]
        for i, ga1 in enumerate(self.ga1s):
            self.add_module('ga1_{}'.format(i), ga1)

        self.ga2 = SparseGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.ga1s], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.ga2(x, adj))
        return F.log_softmax(x, dim=1)