import torch
from torch.utils.data import Dataset



class TrainSampler(Dataset):
    def __init__(self, args):
        self.seqs, self.users, self.slices = {}, [], []
        for user, items in args.data.items():
            seq = [0] * (args.max_len - 1) + items[:-2]
            self.seqs[user] = seq

            for i in range(len(seq) - args.max_len):
                self.users.append(user)
                self.slices.append(slice(i, i + args.max_len + 1))

    def __getitem__(self, index):
        user, slice = self.users[index], self.slices[index]
        seq = torch.tensor(self.seqs[user][slice])
        seq, pos = seq[:-1], seq[-1]

        return seq, pos

    def __len__(self): return len(self.slices)



class EvalSampler(Dataset):
    def __init__(self, split, args):
        self.seqs, self.users = [], []
        for user, items in args.data.items():
            if split == 'Val': items = items[:-1]
            if split == 'Test': items = items
            seq = [0] * (args.max_len + 1 - len(items)) + items[-args.max_len-1:]
            self.seqs.append(seq)
            self.users.append(user)

    def __getitem__(self, index):
        user = self.users[index]
        seq = torch.tensor(self.seqs[index])
        seq, pos = seq[:-1], seq[-1]

        return user, seq, pos

    def __len__(self): return len(self.seqs)
