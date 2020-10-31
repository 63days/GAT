# GAT / GCN
Pytorch Implementations of Graph Attention Network and Graph Convolution Network
![image](https://user-images.githubusercontent.com/37788686/97774053-18346d00-1b98-11eb-91d1-af98189df894.png)

## What is the GAT and GCN
Most machine learning algorithms assume that input data exists in Euclidean space. But data such as relationships between objects, social networks, relational databases, molecular structures, etc., are basically graphically represented. GCN(Graph Convolutional Network) and GAT(Graph Attention Network) is used in these graphical data.


## Results

|      | GCN    | GAT    |
| ---- | ------ | ------ |
| Loss | 0.9790 | 0.6724 |
| Acc  | 0.8160 | 0.8385 |

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
