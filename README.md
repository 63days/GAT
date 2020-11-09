# GAT / GCN
Pytorch Implementations of Graph Attention Network(GAT) and Graph Convolution Network(GCN)
![image](https://user-images.githubusercontent.com/37788686/97774053-18346d00-1b98-11eb-91d1-af98189df894.png)

## Introduction
![image](https://user-images.githubusercontent.com/37788686/98549556-4c1d3a00-22de-11eb-8741-eaff06af5554.png)

Most machine learning algorthims assume that input data exists in Euclidean space. But data such as social networks, molecular structures, relationships between objects, etc.., are basically graphically represented. For example, the problem of classifying nodes(such as documents) in a graph(such as citation network) can be framed as graph-based learning. Convolution is very efficient for learning that takes into account the data structure. So GCN is the network that combines convolution mechanisms with graph data. Furthermore, GAT is a network that adds attention mechanisms to the GCN.

In GCN, graphs are converted into vector-type data to apply convolution. Graph is represented as G=(A, X). A is adjacent matrix and X is nodes feature matrix. The most basic convolution is defined as follows: sigmoid(AXW). W is learnable weight matrix. GCN tunes this W matrix. Simply, GAT is GCN + Attention network.




## Results

|      | GCN    | GAT    |
| ---- | ------ | ------ |
| Loss | 0.9790 | 0.6724 |
| Acc  | 0.8160 | 0.8385 |

GAT shows higher performance than GCN.

## To train
### GCN
`python3 train.py --model gcn`
### GAT
`python3 train.py --model gat`
### SpGCN
`python3 train.py --model spgcn`
### SpGAT
`python3 train.py --model spgat`

## References
[1] [Graph Attention Network](https://arxiv.org/pdf/1710.10903.pdf)  

[2] https://github.com/tkipf/pygcn

[3] https://github.com/Diego999/pyGAT
