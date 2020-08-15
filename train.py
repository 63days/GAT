import os
import time
import random
import argparse
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm

from utils import accuracy, load_data
from model import GCN, GAT, SpGCN, SpGAT


def main(args):
    device = torch.device('cuda' if args.cuda else 'cpu')

    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)

    if args.model == 'gcn':
        model = GCN(nfeat=features.size(1),
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout)

        print('Model: GCN')

    elif args.model == 'gat':
        model = GAT(nfeat=features.size(1),
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout,
                    alpha=args.alpha,
                    nheads=args.n_heads)
        print('Model: GAT')

    elif args.model == 'spgcn':
        model = SpGCN(nfeat=features.size(1),
                      nhid=args.hidden,
                      nclass=labels.max().item() + 1,
                      dropout=args.dropout)
        print('Model: SpGCN')

    elif args.model == 'spgat':
        model = SpGAT(nfeat=features.size(1),
                      nhid=args.hidden,
                      nclass=labels.max().item() + 1,
                      dropout=args.dropout,
                      alpha=args.alpha,
                      nheads=args.n_heads)
        print('Model: SpGAT')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        adj = adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        model.cuda()
        print(device)

    def train(epoch):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            model.eval()
            output = model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        #         print('Epoch: {:04d}'.format(epoch + 1),
        #               'loss_train: {:.4f}'.format(loss_train.item()),
        #               'acc_train: {:.4f}'.format(acc_train.item()),
        #               'loss_val: {:.4f}'.format(loss_val.item()),
        #               'acc_val: {:.4f}'.format(acc_val.item()))
        pbar.set_description('| epoch: {:4d} | loss_train: {:.4f} | acc_train: {:.4f} |'
                             ' loss_val: {:.4f} | acc_val: {:.4f}'.format(
            epoch + 1, loss_train.item(), acc_train.item(), loss_val.item(), acc_val.item()))
        return loss_train.item(), loss_val.item()

    def test():
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

    losses = {}
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        loss_train, loss_val = train(epoch)

        if epoch % 10 == 0:
            if len(losses) == 0:
                losses['train'] = [loss_train]
                losses['val'] = [loss_val]

            else:
                losses['train'].append(loss_train)
                losses['val'].append(loss_val)

    f, ax = plt.subplots()

    train_loss = ax.plot(losses['train'], label='Train Loss')
    val_loss = ax.plot(losses['val'], label='Validation Loss')

    ax.legend()
    ax.set_xlabel('Epoch / 10')
    ax.set_ylabel('Loss')

    plt.savefig('results/loss_{}_{}.png'.format(args.model, args.dataset), dpi=300)

    print('Optimization Finished!')

    test()


if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--save_every', type=int, default=10, help='Save every n epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=10, help='patience')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer'], help='Dataset to train.')
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'spgcn', 'spgat'],
                        help='Model to train.')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        ## TODO
    main(args)