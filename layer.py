import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# TODO step 1.
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, input, adj):  # AXW
        output = torch.mm(input, self.weight)  # [N, hidden dim]
        output = torch.mm(adj, output)  # [N, hidden dim]
        output = self.dropout(output)
        return output


# TODO step 2.
class GraphAttentionLayer(nn.Module):
    """multihead attention """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.out_features = out_features
        self.dropout = dropout
        self.concat = concat
        self.linear = nn.Linear(in_features, out_features)
        self.attn = nn.Linear(out_features * 2, 1)
        self.leaky = nn.LeakyReLU(alpha)

    def forward(self, input, adj):
        N = input.size(0)
        h = self.linear(input)
        concated = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=-1).view(N, N, self.out_features * 2)

        energy = self.attn(concated).squeeze(-1)
        energy = self.leaky(energy)
        zero_vec = -9e15 * torch.ones_like(energy)
        masked = torch.where(adj > 0, energy, zero_vec)
        masked_attn_score = F.softmax(masked, -1)
        masked_attn_score = F.dropout(masked_attn_score, p=self.dropout, training=self.training)

        h_prime = torch.matmul(masked_attn_score, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


# TODO step 3.
class SparsemmFunction(torch.autograd.Function):
    """ for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class Sparsemm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SparsemmFunction.apply(indices, values, shape, b)


class SparseGraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(SparseGraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

        self.spmm = Sparsemm()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, input, adj):  # AXW
        N = input.size(0)
        edge = adj.nonzero().t()
        edge_v = torch.ones(edge.size(1)).to(device)
        output = torch.mm(input, self.weight)  # [N, hidden dim]
        output = self.spmm(edge, edge_v, torch.Size([N, N]), output)  # [N, hidden dim]
        output = self.dropout(output)
        return output


class SparseGraphAttentionLayer(nn.Module):
    """multihead attention """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SparseGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.spmm = Sparsemm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        assert not torch.isnan(h).any()

        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()

        e_rowsum = self.spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))

        edge_e = self.dropout(edge_e)

        h_prime = self.spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()

        h_prime = h_prime.div(e_rowsum)

        assert not torch.isnan(h_prime).any()

        if self.concat:

            return F.elu(h_prime)
        else:

            return h_prime