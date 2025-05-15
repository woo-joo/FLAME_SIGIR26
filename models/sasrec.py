import torch
from torch import nn

from models.modules import Encoder



class SASRec(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.std = args.std
        self.max_len = args.max_len

        self.item_emb = nn.Embedding(args.num_item+1, args.num_latent, padding_idx=0)
        self.pos_emb = nn.Embedding(args.max_len, args.num_latent)

        self.layernorm = nn.LayerNorm(args.num_latent, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)

        self.seq_encoder = Encoder(args)

        self.apply(self.init_weight)


    def init_weight(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight.data, mean=0.0, std=self.std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight.data)
            nn.init.zeros_(module.bias.data)
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.bias.data)


    def get_seq_emb(self, seq):
        item_emb = self.item_emb(seq)

        pos_idx = torch.arange(self.max_len).to(seq.device)
        pos_emb = self.pos_emb(pos_idx)

        seq_emb = item_emb + pos_emb
        seq_emb = self.layernorm(seq_emb)
        seq_emb = self.dropout(seq_emb)

        return seq_emb


    def get_attn_mask(self, seq):
        seq_mask = (seq > 0).long()[:, None]
        seq_mask = seq_mask[..., None, :] * seq_mask[..., :, None]

        causal_mask = torch.ones(1, 1, self.max_len, self.max_len)
        causal_mask = torch.tril(causal_mask)
        causal_mask = causal_mask.to(seq.device)

        attn_mask = (seq_mask * causal_mask).float()
        attn_mask = (1.0 - attn_mask) * (-10000.0)

        return attn_mask


    def forward(self, seq):
        seq_emb = self.get_seq_emb(seq)
        attn_mask = self.get_attn_mask(seq)
        seq_repr = self.seq_encoder(seq_emb, attn_mask)[:, -1]

        return seq_repr
