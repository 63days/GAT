# GAT / GCN
Pytorch Implementations of Graph Attention Network(GAT) and Graph Convolution Network(GCN)
![image](https://user-images.githubusercontent.com/37788686/97774053-18346d00-1b98-11eb-91d1-af98189df894.png)

## Introduction
![image](https://user-images.githubusercontent.com/37788686/98549556-4c1d3a00-22de-11eb-8741-eaff06af5554.png)

Most machine learning algorthims assume that input data exists in Euclidean space. But data such as social networks, molecular structures, relationships between objects, etc.., are basically graphically represented. For example, the problem of classifying nodes(such as documents) in a graph(such as citation network) can be framed as graph-based learning. Convolution is very efficient for learning that takes into account the data structure. So GCN is the network that combines convolution mechanisms with graph data. Furthermore, GAT is a network that adds attention mechanisms to the GCN.

In GCN, graphs are converted into vector-type data to apply convolution. Graph is represented as G=(A, X). A is adjacent matrix and X is nodes feature matrix. The most basic convolution is defined as follows: sigmoid(AXW). W is learnable weight matrix. GCN tunes this W matrix. Simply, GAT is GCN + Attention network.

## Datasets
|       | Cora | Citeseer |
| ----- | ---- | -------- |
| Nodes | 2708 | 3327     |
| Edges | 5429 | 4732     |
| Features/Node | 1433 | 3703 |
| Classes | 7 | 6 |

Citeseer is spare graph structured data.

## Results
**\<Cora dataset>**
|       | Training Time | Loss | Acc |
| ----- | ------------- | ---- | --- |
| GCN   | 2s | 0.9790 | 81.6% |
| GAT   | 17m | 0.6724 | 83.8% |
| spGCN | 13s | 0.9215 | 81.4% |
| spGAT | 1m30s | 0.6745 | 84.7% | 

**\<Siteseer dataset>**
|       | Training Time | Loss | Acc |
| ----- | ------------- | ---- | --- |
| GCN   | 2s | 1.2088 | 60% |
| GAT   | 21m | 1.1907 | 59.1% |
| spGCN | 17s | 1.602 | 58.3% |
| spGAT | 1m47s | 1.1591 | 59.2% | 

GAT shows higher performance than GCN.

## To train & test
### GCN
`python3 train.py --model gcn`
### GAT
`python3 train.py --model gat`
### SpGCN
`python3 train.py --model spgcn`
### SpGAT
`python3 train.py --model spgat`

## Code Explanation
* layer.py
```python
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
```
GraphConvolutionLayer is similar with CNN. This takes feature matrix and computes like convolution layer. torch.mm() is matrix multiply function. In pytorch library, CNN is implemented by torch.mm().
```python
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
```
GraphAttentionLayer is GraphConvolutionLayer+Attention. I implements attention as Bahdanau attention(using FC). I uses mask to select only adjacent nodes at each node.

## References
[1] [Graph Attention Network](https://arxiv.org/pdf/1710.10903.pdf)  

[2] https://github.com/tkipf/pygcn

[3] https://github.com/Diego999/pyGAT
