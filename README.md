# GAT / GCN
Pytorch Implementations of Graph Attention Network and Graph Convolution Network

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