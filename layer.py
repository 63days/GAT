
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

# TODO step 1. 
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(GraphConvolutionLayer,self).__init__()
        pass
    
    def forward(self, input, adj):
        pass

# TODO step 2. 
class GraphAttentionLayer(nn.Module):
    """multihead attention """ 
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        pass 
    
    def forward(self, input, adj):
        pass
        

# TODO step 3.
class SparsemmFunction(torch.autograd.Function):
    """ for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass

class Sparsemm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SparsemmFunction.apply(indices, values, shape, b)


class SparseGraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(SparseGraphConvolutionLayer,self).__init__()
        pass
    def forward(self, input, adj):
        pass

class SparseGraphAttentionLayer(nn.Module):
    """multihead attention """ 
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SparseGraphAttentionLayer, self).__init__()
        pass 
    
    def forward(self, input, adj):
        pass