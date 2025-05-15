import os
import random
from time import time
from importlib import import_module

import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import DataLoader

from samplers import TrainSampler, EvalSampler



def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



def read_data(args):
    data = {}
    num_user, num_item = 0, 0
    lines = open(f'datasets/{args.dataset}.txt').readlines()
    for line in lines:
        ids = line.split()
        user, items = int(ids[0]), list(map(int, ids[1:]))

        data[user] = items[-args.max_len-3:]

        num_user = max(num_user, user)
        num_item = max(num_item, max(items))

    args.data = data
    args.num_user = num_user
    args.num_item = num_item



def make_rating_matrix(args, split):
    row, col = [], []
    for user, items in args.data.items():
        if split == 'Val': items = items[:-2]
        if split == 'Test': items = items[:-1]
        row += [user-1] * len(items)
        col += [item-1 for item in items]
    rating_matrix = csr_matrix(([1] * len(row), (row, col)),
                               shape=(args.num_user, args.num_item))

    setattr(args, f'{split.lower()}_rating_matrix', rating_matrix)



def get_dataloader(args):
    print(f'Preparing dataloader for {args.dataset}...')

    start = time()


    read_data(args)
    make_rating_matrix(args, 'Val')
    make_rating_matrix(args, 'Test')

    train_set = TrainSampler(args)
    val_set = EvalSampler('Val', args)
    test_set = EvalSampler('Test', args)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)


    finish = time()

    print('Done...')
    print('Consumed Time: {:>3.2f}s'.format(finish - start), end='\n\n')
    
    return train_loader, val_loader, test_loader
