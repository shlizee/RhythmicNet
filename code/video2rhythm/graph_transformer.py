"""
RhythmicNet stage 1 Video2Rhythm code of "How does it sound?: Generation of Rhythmic
Soundtracks for Human Movement Videos" 
Copyright (c) 2021-2022 University of Washington. Developed in UW NeuroAI Lab by Xiulong Liu.
"""
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from third_party.mmskeleton.st_gcn import Model as ST_GCN
import math



class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 2
    n_head = 2
    n_embd = 64

# n_layer | n_head | n_embd |
#   2     |    2   |  32    |

class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config, add_self_sim=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.add_self_sim = add_self_sim
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, x, selfsim=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.add_self_sim:
            att = (q @ k.transpose(-2, -1) + selfsim) * (1.0 / math.sqrt(k.size(-1)))
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class RelativeGlobalAttention(torch.nn.Module):
    """
    from Music Transformer ( Huang et al, 2018 )
    [paper link](https://arxiv.org/pdf/1809.04281.pdf)
    """

    def __init__(self, h=2, d=64, add_emb=False, max_seq=300, add_self_sim=False, **kwargs):
        super().__init__()
        self.len_k = None
        self.add_self_sim = add_self_sim
        self.max_seq = max_seq
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = torch.nn.Linear(self.d, self.d)
        self.Wk = torch.nn.Linear(self.d, self.d)
        self.Wv = torch.nn.Linear(self.d, self.d)
        self.fc = torch.nn.Linear(d, d)
        self.additional = add_emb
        self.E = nn.Parameter(torch.rand([self.max_seq, int(self.dh)]))

        # self.E = torch.randn([self.max_seq, int(self.dh)], requires_grad=False)
        if self.additional:
            self.Radd = None

    def forward(self, inputs, selfsim=None, mask=None, **kwargs):
        """
        :param inputs: a list of tensors. i.e) [Q, K, V]
        :param mask: mask tensor
        :param kwargs:
        :return: final tensor ( output of attention )
        """
        q = inputs
        q = self.Wq(q)
        q = torch.reshape(q, (q.size(0), q.size(1), self.h, -1))
        q = q.permute(0, 2, 1, 3)  # batch, h, seq, dh

        k = inputs
        k = self.Wk(k)
        k = torch.reshape(k, (k.size(0), k.size(1), self.h, -1))
        k = k.permute(0, 2, 1, 3)

        v = inputs
        v = self.Wv(v)
        v = torch.reshape(v, (v.size(0), v.size(1), self.h, -1))
        v = v.permute(0, 2, 1, 3)

        self.len_k = k.size(2)
        self.len_q = q.size(2)

        E = self._get_left_embedding(self.len_q).to(q.device)
        QE = torch.einsum('bhld,md->bhlm', [q, E])
        QE = self._qe_masking(QE)
        Srel = self._skewing(QE)

        Kt = k.permute(0, 1, 3, 2)
        QKt = torch.matmul(q, Kt)
        if self.add_self_sim:
            QKt += selfsim
        logits = QKt + Srel
        logits = logits / math.sqrt(self.dh)

        if mask is not None:
            logits += (mask.to(torch.int64) * -1e9).to(logits.dtype)

        attention_weights = F.softmax(logits, -1)
        attention = torch.matmul(attention_weights, v)

        out = attention.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), -1, self.d))

        out = self.fc(out)
        return out  # , attention_weights

    def _get_left_embedding(self, len_q):
        starting_point = max(0, self.max_seq - len_q)
        e = self.E[starting_point:, :]
        return e

    def _skewing(self, tensor: torch.Tensor):
        padded = F.pad(tensor, [1, 0, 0, 0, 0, 0, 0, 0])
        reshaped = torch.reshape(padded, shape=[padded.size(0), padded.size(1), padded.size(-1), padded.size(-2)])
        Srel = reshaped[:, :, 1:, :]
        if self.len_k > self.len_q:
            Srel = F.pad(Srel, [0, 0, 0, 0, 0, 0, 0, self.len_k - self.len_q])
        elif self.len_k < self.len_q:
            Srel = Srel[:, :, :, :self.len_k]

        return Srel

    def _qe_masking(self, qe):
        mask = self.sequence_mask(
            torch.arange(qe.size()[-1] - 1, qe.size()[-1] - qe.size()[-2] - 1, -1).to(qe.device),
            qe.size()[-1])
        mask = ~mask.to(mask.device)
        return mask.to(qe.dtype) * qe

    def sequence_mask(self, length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)




class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config, use_rel_attn=False, add_self_sim=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.add_self_sim = add_self_sim
        if not use_rel_attn:
            self.attn = SelfAttention(config)
        else:
            self.attn = RelativeGlobalAttention(h=config.n_head, d=config.n_embd, add_emb=False, max_seq=300,
                                                add_self_sim=add_self_sim)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, selfsim=None):
        if self.add_self_sim:
            x = x + self.attn(self.ln1(x), selfsim)
        else:
            x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, use_rel_attn, add_self_sim): #, num_period_class=10
        super().__init__()

        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.add_self_sim = add_self_sim
        self.use_rel_attn = use_rel_attn
        self.blocks = nn.Sequential(*[Block(config, use_rel_attn=use_rel_attn, add_self_sim=add_self_sim)
                                      for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.period_head = nn.Linear(config.n_embd, num_period_class)
        self.block_size = config.block_size
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, selfsim=None):
        b, t, c = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        token_embedding = idx

        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embedding + position_embeddings)
        if self.add_self_sim:
            for block in self.blocks:
                x = block(x, selfsim)
        else:
            x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits


class SkelGraphTransformer(nn.Module):
    def __init__(self, use_rel_attn, add_self_sim, vocab_size=1, block_size=300, input_size=34):
        super(SkelGraphTransformer, self).__init__()
        self.config = GPT1Config(vocab_size=vocab_size, block_size=block_size)
        self.add_self_sim = add_self_sim
        self.skel_encoder = ST_GCN()
        self.gpt = GPT(self.config, use_rel_attn=use_rel_attn, add_self_sim=add_self_sim)
        self.input_size = input_size
        if self.add_self_sim:
            self.conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)

    def forward(self, skel_input):
        skel_latent = self.skel_encoder(skel_input)
        if self.add_self_sim:
            selfsim = torch.sum(skel_latent * skel_latent, dim=-1, keepdim=True) - \
                      2 * torch.matmul(skel_latent,skel_latent.permute(0,2,1)) + \
                      torch.sum(skel_latent.permute(0, 2, 1) * skel_latent.permute(0, 2, 1), dim=1, keepdim=True)
            norm_selfsim = torch.softmax(selfsim, dim=-1)
            selfsim_feat = self.conv(norm_selfsim.unsqueeze(1))
            logits = self.gpt(skel_latent, selfsim_feat)
        else:
            logits = self.gpt(skel_latent)
        return logits


if __name__ == "__main__":
    model = SkelGraphTransformer(use_rel_attn=True, add_self_sim=True)
    model.cuda()
    B = 32
    C = 2
    T = 300
    V = 17
    M = 1
    skel_inp = torch.rand(B, C, T, V, M)
    out = model(skel_inp.cuda())
    print(out.shape)