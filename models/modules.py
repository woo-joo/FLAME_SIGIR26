import torch
from torch import nn



class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.w1 = nn.Linear(args.num_latent, 4 * args.num_latent)
        self.activation = nn.GELU()
        self.w2 = nn.Linear(4 * args.num_latent, args.num_latent)

        self.dropout = nn.Dropout(args.dropout)
        self.layernorm = nn.LayerNorm(args.num_latent, eps=1e-12)


    def forward(self, x):
        y = self.w2(self.activation(self.w1(x)))
        y = self.layernorm(x + self.dropout(y))

        return y



class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_head = args.num_head
        self.head_size = args.num_latent // args.num_head
        self.num_latent = args.num_latent
        self.max_len = args.max_len

        self.w_Q = nn.Linear(args.num_latent, args.num_latent)
        self.w_K = nn.Linear(args.num_latent, args.num_latent)
        self.w_V = nn.Linear(args.num_latent, args.num_latent)

        self.w = nn.Linear(args.num_latent, args.num_latent)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm = nn.LayerNorm(args.num_latent, eps=1e-12)


    def split_head(self, x):
        new_shape = x.shape[:2] + (self.num_head, self.head_size)
        x = x.view(*new_shape)
        x = torch.permute(x, (0, 2, 1, 3))

        return x
    

    def concat_head(self, x):
        x = torch.permute(x, (0, 2, 1, 3)).contiguous()
        new_shape = x.shape[:2] + (self.num_latent,)
        x = x.view(*new_shape)

        return x


    def forward(self, x, attn_mask):
        Q = self.split_head(self.w_Q(x))
        K = self.split_head(self.w_K(x))
        V = self.split_head(self.w_V(x))

        attn_score = torch.matmul(Q, torch.transpose(K, -1, -2))
        attn_score = attn_score / (self.head_size ** 0.5)
        attn_score = attn_score + attn_mask

        attn_prob = self.softmax(attn_score)
        attn_prob = self.dropout(attn_prob)

        context = torch.matmul(attn_prob, V)
        context = self.concat_head(context)

        y = self.w(context)
        y = self.layernorm(x + self.dropout(y))

        return y



class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.mhas = nn.ModuleList([MultiHeadAttention(args)
                                   for _ in range(args.num_layer)])
        self.ffns = nn.ModuleList([FeedForward(args)
                                   for _ in range(args.num_layer)])


    def forward(self, x, attn_mask):
        for mha, ffn in zip(self.mhas, self.ffns):
            x = mha(x, attn_mask)
            x = ffn(x)
        
        return x
