import torch
from torch import nn

from models.sasrec import SASRec



class FLAME(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.max_len = args.max_len
        self.tau = args.tau
        self.lambda_0 = args.lambda_0
        self.lambda_R = args.lambda_R
        self.R = args.num_epoch

        self.learnable = SASRec(args)
        self.frozen = SASRec(args)

        self.softmax = nn.Softmax(dim=-1)
        self.ce_loss = nn.CrossEntropyLoss()

        self.load_frozen(args)


    def load_frozen(self, args):
        state_dict = torch.load(f'frozen_weights/{args.dataset}.pt')
        self.frozen.load_state_dict(state_dict)
        for param in self.frozen.parameters():
            param.requires_grad = False


    def anneal(self, epoch):
        self.lambda_r = self.lambda_0 * ((self.lambda_R / self.lambda_0) ** (epoch / self.R))


    def get_seq_emb(self, seq):
        seq_emb_l = self.learnable.get_seq_emb(seq)
        seq_emb_f = self.frozen.get_seq_emb(seq)
        seq_emb = torch.cat((seq_emb_l, seq_emb_f), dim=1)

        return seq_emb


    def get_attn_mask(self, seq):
        seq_mask = (seq > 0).long()[:, None]
        seq_mask = seq_mask[..., None, :] * seq_mask[..., :, None]
        seq_mask = seq_mask.repeat(1, 1, 2, 2)

        causal_mask = torch.ones(1, 1, self.max_len, self.max_len)
        causal_mask = torch.tril(causal_mask)
        causal_mask = causal_mask.repeat(1, 1, 2, 2)
        causal_mask = causal_mask.to(seq.device)

        attn_mask = (seq_mask * causal_mask).float()
        attn_mask = (1.0 - attn_mask) * (-10000.0)

        return attn_mask


    def forward(self, seq):
        seq_repr = self.learnable(seq)

        seq_emb = self.get_seq_emb(seq)
        attn_mask = self.get_attn_mask(seq)

        seq_repr_l = self.learnable.seq_encoder(seq_emb, attn_mask)
        seq_repr_f = self.frozen.seq_encoder(seq_emb, attn_mask)

        seq_repr_ll, seq_repr_fl = seq_repr_l[:, self.max_len - 1], seq_repr_l[:, -1]
        seq_repr_lf, seq_repr_ff = seq_repr_f[:, self.max_len - 1], seq_repr_f[:, -1]

        return seq_repr, seq_repr_ll, seq_repr_fl, seq_repr_lf, seq_repr_ff


    def predict(self, seq):
        seq_repr = self.learnable(seq)
        logit = seq_repr @ self.learnable.item_emb.weight[1:].T

        return logit


    def compute_info_nce_loss(self, z_i, z_j):
        B, _ = z_i.shape

        z = torch.cat((z_i, z_j), dim=0)
        logit = z @ z.T / self.tau
        logit.fill_diagonal_(-10000.0)

        label_i = torch.arange(B) + B
        label_j = torch.arange(B) + 0
        label = torch.cat((label_i, label_j)).to(logit.device)

        info_nce_loss = self.ce_loss(logit, label)

        return info_nce_loss


    def compute_loss(self, seq_repr, pos):
        seq_repr, seq_aug_reprs = seq_repr[0], seq_repr[1:]

        logit = seq_repr @ self.learnable.item_emb.weight[1:].T
        rec_loss = self.ce_loss(logit, pos-1)

        ws, cl_losses = [], []
        for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            ws.append(-torch.mean(torch.sum(seq_aug_reprs[i] * seq_aug_reprs[j], dim=-1)))
            cl_losses.append(self.compute_info_nce_loss(seq_aug_reprs[i], seq_aug_reprs[j]))
        ws, cl_losses = torch.stack(ws), torch.stack(cl_losses)
        cl_loss = torch.sum(self.softmax(ws) * cl_losses)

        return rec_loss + self.lambda_r * cl_loss
