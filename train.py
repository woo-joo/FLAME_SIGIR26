import argparse
from time import time
from copy import deepcopy

import torch
from torch import optim

from utils import set_seed, get_dataloader
from metrics import metrics, metric_funcs
from models.flame import FLAME



def train(train_loader, model, optimizer, epoch, device):
    start = time()

    model.train()
    model.anneal(epoch)
    losses = []
    for seq, pos in train_loader:
        seq, pos = seq.to(device), pos.to(device)

        optimizer.zero_grad()

        seq_repr = model(seq)
        loss = model.compute_loss(seq_repr, pos)

        loss.backward()
        optimizer.step()

        losses.append(loss)
    loss = sum(losses) / len(losses)

    end = time()


    print(f'[Epoch {epoch:>3}] Train ({(end - start):>6.2f}s) | Loss    : {loss:>7.4f}')



@torch.no_grad()
def eval(split, eval_loader, model, rating_matrix, device):
    start = time()

    model.eval()
    ranks = []
    for user, seq, pos in eval_loader:
        user, seq, pos = user.to(device), seq.to(device), pos.to(device)

        logit = model.predict(seq)
        logit[rating_matrix[user.cpu()-1].toarray() == 1] = 0

        rank = torch.argsort(torch.argsort(logit, descending=True))
        rank = rank[torch.arange(len(pos)).to(device), pos-1]
        ranks.append(rank)
    rank = torch.cat(ranks)
    results = [100 * metric_funcs[metric](rank).item() for metric in metrics]

    end = time()


    print(f'            {split:<4}  ({(end - start):>6.2f}s) | {metrics[0]:<8}: {results[0]:>5.2f}')
    for metric, result in zip(metrics[1:], results[1:]):
        print(f'                            | {metric:<8}: {result:>5.2f}')

    return results[-1]



def main(args):
    device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    set_seed(args.seed)


    train_loader, val_loader, test_loader = get_dataloader(args)
    model = FLAME(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


    best = {'result': 0.0, 'epoch': 0, 'state_dict': None}
    for epoch in range(1, args.num_epoch+1):
        if epoch > best['epoch'] + args.patience:
            print('Early Stopped!!!')
            break

        train(train_loader, model, optimizer, epoch, device)
        result = eval('Val', val_loader, model, args.val_rating_matrix, device)

        if best['result'] < result:
            best = {'result': result, 'epoch': epoch, 'state_dict': deepcopy(model.state_dict())}
    print()

    print(f'Loading best model from epoch {best['epoch']}...')
    model.load_state_dict(best['state_dict'])
    eval('Test', test_loader, model, args.test_rating_matrix, device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['amazon-toys', 'amazon-beauty', 'amazon-games', 'amazon-sports',
                                 'yelp', 'ml-1m'])
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--std', type=float, default=0.02)
    parser.add_argument('--num_latent', type=int, default=64)
    parser.add_argument('--num_head', type=int, default=2)
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--max_len', type=int, default=50)

    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--lambda_0', type=float, default=1e-0)
    parser.add_argument('--lambda_R', type=float, default=1e-5)

    args = parser.parse_args()

    main(args)
