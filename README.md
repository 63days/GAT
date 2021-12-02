# Pytorch Implementation of GCN & GAT
Pytorch Implementations of [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) and [Graph Attention Networks](https://arxiv.org/abs/1710.10903).

![image](https://user-images.githubusercontent.com/37788686/97774053-18346d00-1b98-11eb-91d1-af98189df894.png)
Overview of GAT.

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

GAT achieves better performances compared to GCN.

## Usage
`python3 train.py --model {gcn, gat, spgcn, spgat}`


## References
https://github.com/tkipf/pygcn
https://github.com/Diego999/pyGAT
