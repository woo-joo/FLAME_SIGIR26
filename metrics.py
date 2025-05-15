import torch



def hr_k(rank, k):
    return torch.mean((rank < k).float())

def ndcg_k(rank, k):
    return torch.mean(torch.where(rank < k, 1 / torch.log2(rank+2), 0.0))



metrics = ['HR@1', 'HR@5', 'HR@10', 'HR@20', 'NDCG@5', 'NDCG@10', 'NDCG@20']

metric_funcs = {'HR@1'    : lambda rank: hr_k(rank, 1),
                'HR@5'    : lambda rank: hr_k(rank, 5),
                'HR@10'   : lambda rank: hr_k(rank, 10),
                'HR@20'   : lambda rank: hr_k(rank, 20),
                'NDCG@5'  : lambda rank: ndcg_k(rank, 5),
                'NDCG@10' : lambda rank: ndcg_k(rank, 10),
                'NDCG@20' : lambda rank: ndcg_k(rank, 20)}
